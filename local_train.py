"""
AGI-NOW Local Training - 1 GB Grokipedia on RTX 5070
=====================================================
Fast vectorized data pipeline (no Python loops per sample).
Checkpoints every epoch. Scaled model for 12 GB VRAM.
"""
import torch
import torch.nn as nn
import time
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_gpt_association import AssociationBinaryGPT, generate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE, 'checkpoints')
DATA_DIR = os.path.join(BASE, 'training_data')
os.makedirs(CKPT_DIR, exist_ok=True)

# --- Model config (scaled for 5070 12 GB VRAM, ~10M params) ---
MODEL_CONFIG = dict(
    context_bytes=128,
    embed_dim=512,
    num_memories=1024,
    num_hopfield=768,
    num_hops=3,
    dropout=0.08,
)

# --- Training config ---
BATCH_SIZE = 2048
STEPS_PER_EPOCH = 5000      # ~10M samples per epoch
EPOCHS = 1000
LR = 0.001
WARMUP_EPOCHS = 5


def load_text_bytes():
    """Load all training .txt files into a single byte tensor on CPU."""
    print("  Loading training data...", end=" ", flush=True)
    t0 = time.perf_counter()
    all_bytes = []
    file_count = 0
    for fpath in sorted(glob.glob(os.path.join(DATA_DIR, '*.txt'))):
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        for ch in text:
            v = ord(ch) & 0x7F
            if v >= 10:  # skip control chars except newline
                all_bytes.append(v)
        all_bytes.append(10)  # newline between files
        file_count += 1

    tensor = torch.tensor(all_bytes, dtype=torch.long)
    elapsed = time.perf_counter() - t0
    mb = len(all_bytes) / 1024 / 1024
    print(f"{file_count} files | {mb:.1f} MB | {len(all_bytes):,} bytes | {elapsed:.1f}s")
    return tensor


def build_batch(text_bytes, batch_size, ctx_len):
    """
    Build a random training batch using vectorized torch ops.
    No Python loops over samples - this is FAST.
    """
    N = len(text_bytes)

    # Random character positions to predict
    positions = torch.randint(ctx_len, N, (batch_size,))

    # Context windows via fancy indexing
    offsets = torch.arange(-ctx_len, 0)
    idx = positions.unsqueeze(1) + offsets.unsqueeze(0)  # (B, ctx_len)
    ctx = text_bytes[idx]  # (B, ctx_len)

    # Target bytes
    targets = text_bytes[positions]  # (B,)

    # Random bit position per sample
    bit_pos = torch.randint(0, 8, (batch_size,))

    # Target bit
    y = ((targets >> (7 - bit_pos)) & 1).float()

    # Partial bits (bits 0..bit_pos-1 of target byte)
    partial = torch.zeros(batch_size, 7)
    for b in range(7):
        mask = bit_pos > b
        partial[mask, b] = ((targets[mask] >> (7 - b)) & 1).float()

    # Instinct features (vectorized, 10 features matching EnglishInstinct)
    instinct = torch.zeros(batch_size, 10)

    # F0: bit-1 probability prior
    bit_probs = torch.tensor([0.02, 0.52, 0.55, 0.48, 0.50, 0.50, 0.52, 0.48])
    instinct[:, 0] = bit_probs[bit_pos]

    # F1: is not first bit
    instinct[:, 1] = (bit_pos > 0).float()

    # F2-F3: partial bit pattern checks
    has3 = bit_pos >= 3
    if has3.any():
        instinct[has3, 2] = ((partial[has3, 0] == 0) & (partial[has3, 1] == 1) & (partial[has3, 2] == 1)).float()
        instinct[has3, 3] = ((partial[has3, 0] == 0) & (partial[has3, 1] == 0) & (partial[has3, 2] == 1)).float()
    instinct[~has3, 2] = 0.5
    instinct[~has3, 3] = 0.1

    # F4: last char is space
    last_byte = ctx[:, -1]
    instinct[:, 4] = (last_byte == 32).float()

    # F5: last char is sentence ender
    instinct[:, 5] = ((last_byte == 46) | (last_byte == 33) | (last_byte == 63) | (last_byte == 10)).float()

    # F6: bigram frequency proxy
    instinct[:, 6] = torch.clamp(last_byte.float() / 127.0, 0, 1) * 0.5

    # F7: word length proxy
    instinct[:, 7] = ((ctx[:, -1] != 32).float() * 0.3 +
                       (ctx[:, -2] != 32).float() * 0.2 +
                       (ctx[:, -3] != 32).float() * 0.1)

    # F8: consonant proxy
    lc = last_byte | 32
    is_vowel = ((lc == 97) | (lc == 101) | (lc == 105) | (lc == 111) | (lc == 117))
    is_letter = (lc >= 97) & (lc <= 122)
    instinct[:, 8] = (is_letter & ~is_vowel).float() * 0.7 + (~is_letter).float() * 0.3

    # F9: sentence start (space after period)
    instinct[:, 9] = ((ctx[:, -1] == 32) & (ctx[:, -2] == 46)).float()

    return (ctx.to(DEVICE), bit_pos.to(DEVICE), partial.to(DEVICE),
            instinct.to(DEVICE), y.to(DEVICE))


def get_lr(epoch):
    if epoch < WARMUP_EPOCHS:
        return LR * (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
    return LR * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())


def train():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

    print()
    print("=" * 62)
    print("  AGI-NOW - Binary Association Model")
    print("  Training on 1 GB Grokipedia")
    print("=" * 62)
    print()
    print(f"  GPU: {gpu} ({gpu_mem:.0f} GB)")
    print(f"  Device: {DEVICE}")
    print()

    # Load data
    text_bytes = load_text_bytes()

    # Create model
    model = AssociationBinaryGPT(**MODEL_CONFIG).to(DEVICE)
    params = model.count_params()
    print(f"  Model: {params:,} parameters")
    print(f"  Context: {MODEL_CONFIG['context_bytes']} bytes")
    print(f"  Embed: {MODEL_CONFIG['embed_dim']}")
    print(f"  Memories: {MODEL_CONFIG['num_memories']}")
    print(f"  Hopfield: {MODEL_CONFIG['num_hopfield']}")
    print(f"  Hops: {MODEL_CONFIG['num_hops']}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')  # Mixed precision for speed

    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0
    latest_ckpt = os.path.join(CKPT_DIR, 'local_latest.pt')
    if os.path.exists(latest_ckpt):
        print("  Resuming from checkpoint...", end=" ", flush=True)
        ckpt = torch.load(latest_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0)
        print(f"epoch {start_epoch}, best acc {best_acc:.1f}%")

    print()
    samples_per_epoch = STEPS_PER_EPOCH * BATCH_SIZE
    print(f"  Training: {EPOCHS} epochs | {STEPS_PER_EPOCH} steps/epoch | "
          f"{samples_per_epoch:,} samples/epoch")
    print(f"  Batch: {BATCH_SIZE} | LR: {LR} | Warmup: {WARMUP_EPOCHS} epochs")
    print(f"  Checkpoints: every epoch -> {CKPT_DIR}/")
    print(f"  Data: {len(text_bytes):,} bytes ({len(text_bytes)/1024/1024:.0f} MB)")
    print()
    print("-" * 62)
    print()

    model.train()
    t0 = time.perf_counter()

    try:
        for epoch in range(start_epoch, EPOCHS):
            lr = get_lr(epoch)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            epoch_t0 = time.perf_counter()

            for step in range(STEPS_PER_EPOCH):
                # Build batch (vectorized, fast)
                x_b, x_bp, x_p, x_i, y = build_batch(
                    text_bytes, BATCH_SIZE, MODEL_CONFIG['context_bytes']
                )

                # Forward with mixed precision
                with torch.amp.autocast('cuda'):
                    logits = model(x_b, x_bp, x_p, x_i)
                    loss = criterion(logits, y)

                # Backward
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                total_correct += (preds == y).sum().item()
                total_samples += BATCH_SIZE

            # Epoch stats
            acc = total_correct / total_samples * 100
            avg_loss = total_loss / STEPS_PER_EPOCH
            best_acc = max(best_acc, acc)
            epoch_time = time.perf_counter() - epoch_t0
            total_time = time.perf_counter() - t0

            print(f"  Epoch {epoch+1:4d}/{EPOCHS}  "
                  f"loss={avg_loss:.4f}  acc={acc:.1f}%  best={best_acc:.1f}%  "
                  f"lr={lr:.6f}  {epoch_time:.0f}s  "
                  f"[{total_time/3600:.1f}h total]")
            sys.stdout.flush()

            # Save checkpoint every epoch
            ckpt_data = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'loss': avg_loss,
                'config': MODEL_CONFIG,
            }

            # Always save latest (for resume)
            torch.save(ckpt_data, latest_ckpt)

            # Save numbered checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                numbered = os.path.join(CKPT_DIR, f'local_epoch_{epoch+1:05d}.pt')
                torch.save(ckpt_data, numbered)

            # Save model weights for inference
            torch.save(model.state_dict(), os.path.join(BASE, 'agi_model.pt'))

            # Generate sample every 10 epochs
            if (epoch + 1) % 10 == 0:
                model.eval()
                for seed in ['The ', 'Hello ', 'In the beginning ']:
                    text = generate(model, seed, 60, 0.42)
                    print(f"    [GEN] {seed}{text[:50]}")
                model.train()
                print()
                sys.stdout.flush()

    except KeyboardInterrupt:
        print(f"\n  Stopped at epoch {epoch+1}. Saving...")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
            'config': MODEL_CONFIG,
        }, latest_ckpt)
        torch.save(model.state_dict(), os.path.join(BASE, 'agi_model.pt'))

    elapsed = time.perf_counter() - t0
    print(f"\nDone. Best: {best_acc:.1f}% | Time: {elapsed/3600:.1f}h")
    print(f"Model: {os.path.join(BASE, 'agi_model.pt')}")


if __name__ == '__main__':
    train()
