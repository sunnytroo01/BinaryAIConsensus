"""
AGI-NOW B200 Pod Training - 2B Parameter Binary Association Model
==================================================================
1. First run: scrapes 1 GB from Grokipedia, saves to network volume
2. Future runs: skips scraping, uses existing data
3. Always resumes from latest checkpoint if one exists
4. Checkpoints saved to network volume (survives pod restarts)

Usage:
    cd /workspace/BinaryAIConsensus && git pull && python -u b200_train.py
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

# Network volume paths (persist across pod restarts)
CKPT_DIR = '/workspace/checkpoints' if os.path.exists('/workspace') else os.path.join(BASE, 'checkpoints')
DATA_DIR = '/workspace/training_data' if os.path.exists('/workspace') else os.path.join(BASE, 'training_data')
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
BATCH_SIZE = 32768         # B200 has 192 GB VRAM - use it
STEPS_PER_EPOCH = 5000     # 5000 steps x 32768 batch = 163M samples per epoch
EPOCHS = 1000
LR = 0.0003
WARMUP_EPOCHS = 10


# =====================================================================
# STEP 1: GET TRAINING DATA (scrape once, reuse forever)
# =====================================================================
def ensure_training_data():
    """
    Check if we already have training data on the network volume.
    If yes: skip (instant).
    If no: scrape 1 GB from Grokipedia (one time only).
    """
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    total_size = sum(os.path.getsize(f) for f in files) if files else 0

    if total_size >= 100 * 1024 * 1024:  # Already have 100+ MB
        print(f"  [DATA] Found {len(files)} articles ({total_size/1024/1024:.0f} MB) - using existing data")
        return

    print(f"  [DATA] No training data found at {DATA_DIR}")
    print(f"  [DATA] Scraping 1 GB from Grokipedia (one-time, saves to network volume)...")
    print()

    # Copy scraper to work with network volume DATA_DIR
    scraper_path = os.path.join(BASE, 'scrape_grokipedia.py')
    if not os.path.exists(scraper_path):
        print("  ERROR: scrape_grokipedia.py not found in repo!")
        sys.exit(1)

    # Patch the scraper to use our DATA_DIR
    with open(scraper_path, 'r') as f:
        code = f.read()
    code = code.replace(
        "OUTPUT_DIR = os.path.join(BASE_DIR, 'training_data')",
        f"OUTPUT_DIR = '{DATA_DIR}'"
    )
    patched = os.path.join(BASE, '_scraper_tmp.py')
    with open(patched, 'w') as f:
        f.write(code)

    import subprocess
    subprocess.run([sys.executable, '-u', patched], cwd=BASE)
    os.remove(patched)

    # Verify
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    total_size = sum(os.path.getsize(f) for f in files) if files else 0
    print(f"  [DATA] Scraping complete: {len(files)} articles ({total_size/1024/1024:.0f} MB)")
    print()


# =====================================================================
# STEP 2: LOAD DATA INTO MEMORY
# =====================================================================
def load_text_bytes():
    """Load all .txt files into one big byte tensor."""
    print("  [LOAD] Reading files...", end=" ", flush=True)
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
    print(f"done! {file_count} files, {mb:.0f} MB, {elapsed:.1f}s")
    return tensor


# =====================================================================
# STEP 3: BUILD TRAINING BATCHES (vectorized, no slow Python loops)
# =====================================================================
def build_batch(text_bytes, batch_size, ctx_len):
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


# =====================================================================
# STEP 4: TRAIN
# =====================================================================
def train():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

    print()
    print("=" * 70)
    print("  AGI-NOW  |  2 BILLION PARAM  |  Binary Association Model")
    print("=" * 70)
    print()
    print(f"  GPU:         {gpu} ({gpu_mem:.0f} GB)")
    print(f"  Data:        {DATA_DIR}")
    print(f"  Checkpoints: {CKPT_DIR}")
    print()

    # --- Data ---
    ensure_training_data()
    text_bytes = load_text_bytes()

    # --- Model ---
    print(f"  [MODEL] Creating 2B model...", end=" ", flush=True)
    model = DeepAssociationBinaryGPT(**MODEL_CONFIG).to(DEVICE)
    params = model.count_params()
    print(f"{params:,} params")
    model_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    print(f"  [MODEL] Size: {model_gb:.1f} GB | {MODEL_CONFIG['num_blocks']} blocks | "
          f"{MODEL_CONFIG['num_blocks'] * MODEL_CONFIG['num_hops']} total hops")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')

    # --- Resume from checkpoint ---
    start_epoch = 0
    best_acc = 0
    latest_ckpt = os.path.join(CKPT_DIR, 'b200_latest.pt')
    if os.path.exists(latest_ckpt):
        print(f"  [RESUME] Loading checkpoint...", end=" ", flush=True)
        ckpt = torch.load(latest_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0)
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        print(f"resuming from epoch {start_epoch} (best acc: {best_acc:.1f}%)")
    else:
        print(f"  [START] Fresh training from epoch 1")

    # --- Print training plan ---
    print()
    print(f"  ---- TRAINING PLAN ----")
    print(f"  Epochs:          {start_epoch + 1} to {EPOCHS}")
    print(f"  Steps per epoch: {STEPS_PER_EPOCH:,}")
    print(f"  Batch size:      {BATCH_SIZE:,}")
    print(f"  Samples/epoch:   {STEPS_PER_EPOCH * BATCH_SIZE:,}")
    print(f"  Training data:   {len(text_bytes):,} bytes ({len(text_bytes)/1024/1024:.0f} MB)")
    print(f"  Learning rate:   {LR} (cosine decay, {WARMUP_EPOCHS} epoch warmup)")
    print(f"  Saving to:       {latest_ckpt}")
    print()

    # --- Warmup CUDA ---
    print("  [CUDA] Compiling kernels (first forward pass)...", flush=True)
    model.train()
    wb = build_batch(text_bytes, 32, MODEL_CONFIG['context_bytes'])
    with torch.amp.autocast('cuda'):
        model(wb[0], wb[1], wb[2], wb[3])
    del wb
    torch.cuda.empty_cache()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  [CUDA] Ready! Using {vram:.1f} / {gpu_mem:.0f} GB VRAM")
    print()
    print("=" * 70)
    print("  TRAINING STARTED - logs every 100 steps, checkpoint every epoch")
    print("=" * 70)
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

                # --- Progress log (every 100 steps) ---
                if step == 0 or (step + 1) % 100 == 0:
                    acc_now = total_correct / total_samples * 100
                    loss_now = total_loss / (step + 1)
                    elapsed = time.perf_counter() - epoch_t0
                    speed = (step + 1) / elapsed if elapsed > 0 else 0
                    eta_sec = (STEPS_PER_EPOCH - step - 1) / max(speed, 0.01)
                    eta_min = eta_sec / 60

                    # Simple progress bar
                    pct = (step + 1) / STEPS_PER_EPOCH
                    bar_len = 20
                    filled = int(bar_len * pct)
                    bar = '=' * filled + '-' * (bar_len - filled)

                    print(f"    [{bar}] {pct*100:5.1f}%  "
                          f"step {step+1}/{STEPS_PER_EPOCH}  "
                          f"loss={loss_now:.4f}  acc={acc_now:.1f}%  "
                          f"speed={speed:.1f}/s  ETA={eta_min:.1f}min",
                          flush=True)

            # --- End of epoch ---
            acc = total_correct / total_samples * 100
            avg_loss = total_loss / STEPS_PER_EPOCH
            best_acc = max(best_acc, acc)
            epoch_time = time.perf_counter() - epoch_t0
            total_time = time.perf_counter() - t0

            print()
            print(f"  >>> EPOCH {epoch+1} COMPLETE <<<")
            print(f"      Accuracy:  {acc:.1f}%  (best ever: {best_acc:.1f}%)")
            print(f"      Loss:      {avg_loss:.4f}")
            print(f"      Time:      {epoch_time/60:.1f} min this epoch, {total_time/3600:.1f}h total")
            print(f"      Saving checkpoint...", end=" ", flush=True)

            # --- Save checkpoint ---
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
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'agi_model_2b.pt'))
            print("saved!")

            # Numbered checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                numbered = os.path.join(CKPT_DIR, f'b200_epoch_{epoch+1:05d}.pt')
                torch.save(ckpt_data, numbered)
                print(f"      Milestone: {numbered}")

            # Generate sample text every 10 epochs
            if (epoch + 1) % 10 == 0:
                model.eval()
                print(f"      --- Sample text ---")
                for seed in ['The ', 'Hello ', 'Science is ']:
                    text = generate(model, seed, 60, 0.42)
                    print(f"      \"{seed}{text[:50]}\"")
                model.train()

            print()
            sys.stdout.flush()

    except KeyboardInterrupt:
        print(f"\n  [STOPPED] Saving checkpoint at epoch {epoch+1}...")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
            'config': MODEL_CONFIG,
        }, latest_ckpt)
        print(f"  [STOPPED] Saved. Run again to resume from epoch {epoch+1}.")

    elapsed = time.perf_counter() - t0
    print()
    print("=" * 70)
    print(f"  TRAINING COMPLETE")
    print(f"  Best accuracy: {best_acc:.1f}%")
    print(f"  Total time:    {elapsed/3600:.1f} hours")
    print(f"  Checkpoints:   {CKPT_DIR}/")
    print(f"  Run again to resume from where you left off.")
    print("=" * 70)


if __name__ == '__main__':
    train()
