"""
AGI-NOW B200 Pod Training - 2B Parameter Binary Association Model
==================================================================
Runs directly on RunPod pod with B200 GPU.
Saves checkpoints to network volume every epoch.
If no training data found, runs the scraper first.

Usage on pod:
    cd /workspace/BinaryAIConsensus
    pip install torch numpy requests
    python b200_train.py
"""
import torch
import torch.nn as nn
import time
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_gpt_association import DeepAssociationBinaryGPT, generate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = os.path.dirname(os.path.abspath(__file__))

# Auto-detect paths (RunPod network volume or local)
if os.path.exists('/workspace'):
    CKPT_DIR = '/workspace/checkpoints'
    DATA_DIR = os.path.join(BASE, 'training_data')
    if not os.path.isdir(DATA_DIR) or len(os.listdir(DATA_DIR)) < 10:
        DATA_DIR = '/workspace/training_data'
else:
    CKPT_DIR = os.path.join(BASE, 'checkpoints')
    DATA_DIR = os.path.join(BASE, 'training_data')

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- 2B Model config ---
MODEL_CONFIG = dict(
    context_bytes=128,
    embed_dim=2048,
    num_memories=1024,
    num_hopfield=1024,
    num_hops=3,
    num_blocks=22,
    dropout=0.05,
)

# --- Training config ---
BATCH_SIZE = 4096          # B200 has 192 GB, can handle large batches
STEPS_PER_EPOCH = 5000     # ~20M samples per epoch
EPOCHS = 1000
LR = 0.0003                # Lower LR for larger model
WARMUP_EPOCHS = 10


def load_text_bytes():
    """Load all training .txt files into a single byte tensor."""
    print("  Loading training data...", end=" ", flush=True)
    t0 = time.perf_counter()
    all_bytes = []
    file_count = 0
    for fpath in sorted(glob.glob(os.path.join(DATA_DIR, '*.txt'))):
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        for ch in text:
            v = ord(ch) & 0x7F
            if v >= 10:
                all_bytes.append(v)
        all_bytes.append(10)
        file_count += 1

    tensor = torch.tensor(all_bytes, dtype=torch.long)
    elapsed = time.perf_counter() - t0
    mb = len(all_bytes) / 1024 / 1024
    print(f"{file_count} files | {mb:.1f} MB | {elapsed:.1f}s")
    return tensor


def maybe_scrape_data():
    """If no training data, scrape Grokipedia."""
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    if len(files) >= 100:
        return  # Already have data

    print("  No training data found. Running Grokipedia scraper...")
    print("  Target: 1 GB")
    print()

    # Import and run scraper
    scraper_path = os.path.join(BASE, 'scrape_grokipedia.py')
    if os.path.exists(scraper_path):
        import subprocess
        subprocess.run([sys.executable, '-u', scraper_path], cwd=BASE)
    else:
        print("  ERROR: scrape_grokipedia.py not found!")
        print("  Upload training data to:", DATA_DIR)
        sys.exit(1)


def build_batch(text_bytes, batch_size, ctx_len):
    """Vectorized batch building - no Python loops over samples."""
    N = len(text_bytes)

    positions = torch.randint(ctx_len, N, (batch_size,))
    offsets = torch.arange(-ctx_len, 0)
    idx = positions.unsqueeze(1) + offsets.unsqueeze(0)
    ctx = text_bytes[idx]

    targets = text_bytes[positions]
    bit_pos = torch.randint(0, 8, (batch_size,))
    y = ((targets >> (7 - bit_pos)) & 1).float()

    partial = torch.zeros(batch_size, 7)
    for b in range(7):
        mask = bit_pos > b
        partial[mask, b] = ((targets[mask] >> (7 - b)) & 1).float()

    instinct = torch.zeros(batch_size, 10)
    bit_probs = torch.tensor([0.02, 0.52, 0.55, 0.48, 0.50, 0.50, 0.52, 0.48])
    instinct[:, 0] = bit_probs[bit_pos]
    instinct[:, 1] = (bit_pos > 0).float()
    has3 = bit_pos >= 3
    if has3.any():
        instinct[has3, 2] = ((partial[has3, 0] == 0) & (partial[has3, 1] == 1) & (partial[has3, 2] == 1)).float()
        instinct[has3, 3] = ((partial[has3, 0] == 0) & (partial[has3, 1] == 0) & (partial[has3, 2] == 1)).float()
    instinct[~has3, 2] = 0.5
    instinct[~has3, 3] = 0.1
    last_byte = ctx[:, -1]
    instinct[:, 4] = (last_byte == 32).float()
    instinct[:, 5] = ((last_byte == 46) | (last_byte == 33) | (last_byte == 63) | (last_byte == 10)).float()
    instinct[:, 6] = torch.clamp(last_byte.float() / 127.0, 0, 1) * 0.5
    instinct[:, 7] = ((ctx[:, -1] != 32).float() * 0.3 +
                       (ctx[:, -2] != 32).float() * 0.2 +
                       (ctx[:, -3] != 32).float() * 0.1)
    lc = last_byte | 32
    is_vowel = ((lc == 97) | (lc == 101) | (lc == 105) | (lc == 111) | (lc == 117))
    is_letter = (lc >= 97) & (lc <= 122)
    instinct[:, 8] = (is_letter & ~is_vowel).float() * 0.7 + (~is_letter).float() * 0.3
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
    print("=" * 66)
    print("  AGI-NOW - Deep Binary Association Model")
    print("  2 BILLION PARAMETERS | B200 Training")
    print("=" * 66)
    print()
    print(f"  GPU: {gpu} ({gpu_mem:.0f} GB VRAM)")
    print(f"  Device: {DEVICE}")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Checkpoint dir: {CKPT_DIR}")
    print()

    # Ensure training data exists
    maybe_scrape_data()

    # Load data
    text_bytes = load_text_bytes()
    if len(text_bytes) < 1000:
        print("  ERROR: Not enough training data!")
        sys.exit(1)

    # Create 2B model
    print("  Creating 2B model...", end=" ", flush=True)
    model = DeepAssociationBinaryGPT(**MODEL_CONFIG).to(DEVICE)
    params = model.count_params()
    print(f"{params:,} parameters")
    print()
    print(f"  Architecture:")
    print(f"    Context:    {MODEL_CONFIG['context_bytes']} bytes")
    print(f"    Embed dim:  {MODEL_CONFIG['embed_dim']}")
    print(f"    Blocks:     {MODEL_CONFIG['num_blocks']}")
    print(f"    Memories:   {MODEL_CONFIG['num_memories']} per block")
    print(f"    Hopfield:   {MODEL_CONFIG['num_hopfield']} per block")
    print(f"    Hops:       {MODEL_CONFIG['num_hops']} per block")
    print(f"    Total hops: {MODEL_CONFIG['num_blocks'] * MODEL_CONFIG['num_hops']} deep")
    print()

    # Model size
    model_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    print(f"  Model size:   {model_mb:.0f} MB ({model_mb/1024:.1f} GB)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')

    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0
    latest_ckpt = os.path.join(CKPT_DIR, 'b200_latest.pt')
    if os.path.exists(latest_ckpt):
        print(f"  Resuming from checkpoint...", end=" ", flush=True)
        ckpt = torch.load(latest_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0)
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        print(f"epoch {start_epoch}, best acc {best_acc:.1f}%")

    print()
    samples_per_epoch = STEPS_PER_EPOCH * BATCH_SIZE
    print(f"  Training:")
    print(f"    Epochs:     {EPOCHS}")
    print(f"    Steps/epoch: {STEPS_PER_EPOCH}")
    print(f"    Batch size: {BATCH_SIZE}")
    print(f"    Samples/epoch: {samples_per_epoch:,}")
    print(f"    LR:         {LR}")
    print(f"    Warmup:     {WARMUP_EPOCHS} epochs")
    print(f"    Data:       {len(text_bytes):,} bytes ({len(text_bytes)/1024/1024:.0f} MB)")
    print()
    print("=" * 66)
    print()

    # Warmup forward pass (compiles CUDA kernels, can take a minute)
    model.train()
    print("  Compiling CUDA kernels (first forward pass)...", flush=True)
    warmup_b = build_batch(text_bytes, 32, MODEL_CONFIG['context_bytes'])
    with torch.amp.autocast('cuda'):
        _ = model(warmup_b[0], warmup_b[1], warmup_b[2], warmup_b[3])
    del warmup_b
    torch.cuda.empty_cache()
    vram_used = torch.cuda.memory_allocated() / 1e9
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  Ready. VRAM: {vram_used:.1f} / {vram_total:.0f} GB", flush=True)
    print()

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
                x_b, x_bp, x_p, x_i, y = build_batch(
                    text_bytes, BATCH_SIZE, MODEL_CONFIG['context_bytes']
                )

                with torch.amp.autocast('cuda'):
                    logits = model(x_b, x_bp, x_p, x_i)
                    loss = criterion(logits, y)

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

                # Log step 1 immediately, then every 100 steps
                if step == 0 or (step + 1) % 100 == 0:
                    step_acc = total_correct / total_samples * 100
                    step_loss = total_loss / (step + 1)
                    elapsed = time.perf_counter() - epoch_t0
                    steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
                    eta = (STEPS_PER_EPOCH - step - 1) / max(steps_per_sec, 0.01)
                    print(f"    step {step+1:5d}/{STEPS_PER_EPOCH}  "
                          f"loss={step_loss:.4f}  acc={step_acc:.1f}%  "
                          f"{steps_per_sec:.1f} steps/s  ETA={eta:.0f}s",
                          flush=True)

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
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'loss': avg_loss,
                'config': MODEL_CONFIG,
            }

            torch.save(ckpt_data, latest_ckpt)

            # Numbered checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                numbered = os.path.join(CKPT_DIR, f'b200_epoch_{epoch+1:05d}.pt')
                torch.save(ckpt_data, numbered)
                print(f"    [SAVED] {numbered}")

            # Model weights for inference
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'agi_model_2b.pt'))

            # Generate every 10 epochs
            if (epoch + 1) % 10 == 0:
                model.eval()
                for seed in ['The ', 'Hello ', 'Science is ']:
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
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
            'config': MODEL_CONFIG,
        }, latest_ckpt)

    elapsed = time.perf_counter() - t0
    print()
    print("=" * 66)
    print(f"  Done. Best: {best_acc:.1f}% | Time: {elapsed/3600:.1f}h")
    print(f"  Checkpoints: {CKPT_DIR}/")
    print("=" * 66)


if __name__ == '__main__':
    train()
