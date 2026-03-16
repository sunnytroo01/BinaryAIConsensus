"""
AGI-NOW Big Training - 10x model on 5070
Binary Association | Checkpoints every 5 epochs | Generates samples
"""
import torch, torch.nn as nn, time, random, os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training_data import DEVICE, TRAINING_TEXT, build_dataset
from binary_gpt_association import AssociationBinaryGPT, generate

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

def train():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f'GPU: {gpu}')
    print(f'Data: {len(TRAINING_TEXT):,} chars ({len(TRAINING_TEXT)/1024:.0f} KB)')
    print()

    # Same architecture as 30-epoch run, resuming
    model = AssociationBinaryGPT(
        context_bytes=64, embed_dim=256,
        num_memories=512, num_hopfield=384,
        num_hops=3, dropout=0.08,
    ).to(DEVICE)

    params = model.count_params()
    print(f'Parameters: {params:,}')
    print()

    # Check for existing checkpoint
    ckpts = sorted([f for f in os.listdir(CKPT_DIR) if f.startswith('big_') and f.endswith('.pt')])
    start_epoch = 0
    best_acc = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    if ckpts:
        latest = os.path.join(CKPT_DIR, ckpts[-1])
        print(f'Resuming from {ckpts[-1]}...')
        ckpt = torch.load(latest, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0)
        print(f'Resuming from epoch {start_epoch}, best acc: {best_acc:.1f}%')
        print()

    criterion = nn.BCEWithLogitsLoss()
    ctx = model.context_bytes
    EPOCHS = 30000
    BATCH = 512
    CHUNK = 150000

    # Split into chunks
    chunks = []
    for i in range(0, len(TRAINING_TEXT) - ctx, CHUNK - ctx):
        end = min(i + CHUNK, len(TRAINING_TEXT))
        chunks.append(TRAINING_TEXT[i:end])
    if not chunks:
        chunks = [TRAINING_TEXT]

    print(f'Chunks: {len(chunks)} | Epochs: {EPOCHS} | Batch: {BATCH}')
    print(f'Checkpoints: every 5 epochs')
    print()

    warmup = 5
    def get_lr(epoch):
        if epoch < warmup:
            return 0.001 * (epoch + 1) / warmup
        progress = (epoch - warmup) / max(EPOCHS - warmup, 1)
        return 0.001 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    t0 = time.perf_counter()

    try:
        for epoch in range(start_epoch, EPOCHS):
            random.shuffle(chunks)
            model.train()
            total_loss = total_correct = total_samples = total_batches = 0

            lr = get_lr(epoch)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            for chunk in chunks:
                X_b, X_bp, X_p, X_i, Y = build_dataset(chunk, ctx)
                N = X_b.shape[0]
                if N < 10: continue

                perm = torch.randperm(N, device=DEVICE)
                for s in range(0, N, BATCH):
                    e = min(s + BATCH, N)
                    idx = perm[s:e]
                    logits = model(X_b[idx], X_bp[idx], X_p[idx], X_i[idx])
                    loss = criterion(logits, Y[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    total_correct += (preds == Y[idx]).sum().item()
                    total_batches += 1

                total_samples += N
                del X_b, X_bp, X_p, X_i, Y
                torch.cuda.empty_cache()

            acc = total_correct / max(total_samples, 1) * 100
            avg_loss = total_loss / max(total_batches, 1)
            best_acc = max(best_acc, acc)
            elapsed = time.perf_counter() - t0

            # Log every epoch
            print(f'  Epoch {epoch+1:3d}/{EPOCHS}  loss={avg_loss:.4f}  acc={acc:.1f}%  '
                  f'lr={lr:.6f}  time={elapsed:.0f}s  [{total_samples:,} samples]')
            sys.stdout.flush()

            # Checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                ckpt_path = os.path.join(CKPT_DIR, f'big_epoch_{epoch+1:05d}.pt')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'loss': avg_loss,
                }, ckpt_path)
                torch.save(model.state_dict(), os.path.join(BASE, 'agi_model.pt'))
                mb = os.path.getsize(ckpt_path) / 1024 / 1024
                print(f'    [SAVED] {ckpt_path} ({mb:.0f} MB)')

                # Generate sample every 10 epochs
                model.eval()
                for seed in ['The Moon is ', 'Hello ', 'The meaning of life ']:
                    text = generate(model, seed, 80, 0.42)
                    print(f'    [GEN] {seed}{text[:60]}')
                model.train()
                print()
                sys.stdout.flush()

    except KeyboardInterrupt:
        print(f'\n  Stopped at epoch {epoch+1}. Saving...')
        torch.save(model.state_dict(), os.path.join(BASE, 'agi_model.pt'))

    elapsed = time.perf_counter() - t0
    print(f'\nDone. Best: {best_acc:.1f}% | Time: {elapsed:.0f}s')
    print(f'Model: {os.path.join(BASE, "agi_model.pt")}')

if __name__ == '__main__':
    train()
