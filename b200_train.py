"""
AGI-NOW B200 Training - 2B params on 1 GB Grokipedia
=====================================================
Data comes from the git repo (training_data/).
Checkpoints save to /workspace/checkpoints/ (network volume).
Auto-resumes from last checkpoint.
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

# Data from git repo, checkpoints on network volume
DATA_DIR = os.path.join(BASE, 'training_data')
CKPT_DIR = '/workspace/checkpoints' if os.path.exists('/workspace') else os.path.join(BASE, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

LATEST_CKPT = os.path.join(CKPT_DIR, 'b200_latest.pt')

# 2B model
MODEL_CONFIG = dict(
    context_bytes=128, embed_dim=2048, num_memories=1024,
    num_hopfield=1024, num_hops=3, num_blocks=22, dropout=0.05,
)

# Training
BATCH = 32768
STEPS = 5000
EPOCHS = 1000
LR = 0.0003
WARMUP = 10


def load_data():
    print("  Loading data...", end=" ", flush=True)
    t0 = time.perf_counter()
    out = []
    n = 0
    for fp in sorted(glob.glob(os.path.join(DATA_DIR, '*.txt'))):
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            for ch in f.read():
                v = ord(ch) & 0x7F
                if v >= 10:
                    out.append(v)
        out.append(10)
        n += 1
    t = torch.tensor(out, dtype=torch.long)
    print(f"{n} files, {len(out)/1024/1024:.0f} MB, {time.perf_counter()-t0:.1f}s")
    return t


def batch(data, bs, ctx):
    N = len(data)
    pos = torch.randint(ctx, N, (bs,))
    off = torch.arange(-ctx, 0)
    cx = data[pos.unsqueeze(1) + off.unsqueeze(0)]
    tgt = data[pos]
    bp = torch.randint(0, 8, (bs,))
    y = ((tgt >> (7 - bp)) & 1).float()
    pa = torch.zeros(bs, 7)
    for b in range(7):
        m = bp > b
        pa[m, b] = ((tgt[m] >> (7 - b)) & 1).float()
    ins = torch.zeros(bs, 10)
    bpr = torch.tensor([0.02, 0.52, 0.55, 0.48, 0.50, 0.50, 0.52, 0.48])
    ins[:, 0] = bpr[bp]
    ins[:, 1] = (bp > 0).float()
    h3 = bp >= 3
    if h3.any():
        ins[h3, 2] = ((pa[h3,0]==0)&(pa[h3,1]==1)&(pa[h3,2]==1)).float()
        ins[h3, 3] = ((pa[h3,0]==0)&(pa[h3,1]==0)&(pa[h3,2]==1)).float()
    ins[~h3, 2] = 0.5
    ins[~h3, 3] = 0.1
    lb = cx[:, -1]
    ins[:, 4] = (lb == 32).float()
    ins[:, 5] = ((lb==46)|(lb==33)|(lb==63)|(lb==10)).float()
    ins[:, 6] = torch.clamp(lb.float()/127, 0, 1)*0.5
    ins[:, 7] = (cx[:,-1]!=32).float()*0.3+(cx[:,-2]!=32).float()*0.2+(cx[:,-3]!=32).float()*0.1
    lc = lb|32
    ins[:, 8] = ((lc>=97)&(lc<=122)&~((lc==97)|(lc==101)|(lc==105)|(lc==111)|(lc==117))).float()*0.7+(~((lc>=97)&(lc<=122))).float()*0.3
    ins[:, 9] = ((cx[:,-1]==32)&(cx[:,-2]==46)).float()
    return cx.to(DEVICE), bp.to(DEVICE), pa.to(DEVICE), ins.to(DEVICE), y.to(DEVICE)


def main():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    mem = torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0

    print()
    print("=" * 70)
    print("  AGI-NOW  |  2B PARAMS  |  ASSOCIATION IS ALL YOU NEED")
    print("=" * 70)
    print(f"  GPU: {gpu} ({mem:.0f} GB)  |  Checkpoints: {CKPT_DIR}")
    print()

    data = load_data()
    if len(data) < 1000:
        print("  ERROR: No training data! Run: git pull")
        sys.exit(1)

    print(f"  Creating model...", end=" ", flush=True)
    model = DeepAssociationBinaryGPT(**MODEL_CONFIG).to(DEVICE)
    print(f"{model.count_params():,} params, "
          f"{sum(p.numel()*p.element_size() for p in model.parameters())/1024**3:.1f} GB")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')

    ep0, best = 0, 0.0
    if os.path.exists(LATEST_CKPT):
        print(f"  Resuming...", end=" ", flush=True)
        ck = torch.load(LATEST_CKPT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        ep0 = ck['epoch'] + 1
        best = ck.get('best_acc', 0)
        if 'scaler' in ck: scaler.load_state_dict(ck['scaler'])
        print(f"epoch {ep0}, best {best:.1f}%")

    print(f"  Plan: epochs {ep0+1}-{EPOCHS}, {STEPS} steps/ep, batch {BATCH}, "
          f"{STEPS*BATCH:,} samples/ep, data {len(data)/1024/1024:.0f} MB")
    print()

    # Warm up CUDA (one tiny forward pass)
    model.train()
    wb = batch(data, 32, 128)
    with torch.amp.autocast('cuda'): model(wb[0], wb[1], wb[2], wb[3])
    del wb; torch.cuda.empty_cache()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}/{mem:.0f} GB  |  GO!")
    print("=" * 70)
    print()

    t0 = time.perf_counter()

    try:
        for ep in range(ep0, EPOCHS):
            # LR schedule
            if ep < WARMUP:
                lr = LR * (ep + 1) / WARMUP
            else:
                p = (ep - WARMUP) / max(EPOCHS - WARMUP, 1)
                lr = LR * 0.5 * (1 + torch.cos(torch.tensor(p * 3.14159)).item())
            for g in opt.param_groups: g['lr'] = lr

            tl, tc, ts = 0.0, 0, 0
            et0 = time.perf_counter()

            for s in range(STEPS):
                xb, xbp, xp, xi, y = batch(data, BATCH, 128)
                with torch.amp.autocast('cuda'):
                    logits = model(xb, xbp, xp, xi)
                    loss = loss_fn(logits, y)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                tl += loss.item()
                tc += ((torch.sigmoid(logits) > 0.5).float() == y).sum().item()
                ts += BATCH

                if s == 0 or (s+1) % 100 == 0:
                    a = tc/ts*100
                    l = tl/(s+1)
                    spd = (s+1)/(time.perf_counter()-et0)
                    eta = (STEPS-s-1)/max(spd, 0.01)/60
                    pct = (s+1)/STEPS*100
                    bar = '=' * int(pct/5) + '-' * (20 - int(pct/5))
                    print(f"    [{bar}] {pct:5.1f}%  step {s+1}/{STEPS}  "
                          f"loss={l:.4f}  acc={a:.1f}%  {spd:.1f}/s  ~{eta:.0f}min left", flush=True)

            # Epoch done
            acc = tc/ts*100
            avg = tl/STEPS
            best = max(best, acc)
            et = time.perf_counter() - et0
            tt = time.perf_counter() - t0

            print(f"\n  EPOCH {ep+1} DONE  |  acc={acc:.1f}%  best={best:.1f}%  "
                  f"loss={avg:.4f}  |  {et/60:.1f}min  |  {tt/3600:.1f}h total")

            # Save
            ckpt = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
                     'scaler': scaler.state_dict(), 'epoch': ep, 'best_acc': best,
                     'loss': avg, 'config': MODEL_CONFIG}
            torch.save(ckpt, LATEST_CKPT)
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'agi_model_2b.pt'))
            print(f"  SAVED to {LATEST_CKPT}")

            if (ep+1) % 10 == 0:
                p = os.path.join(CKPT_DIR, f'b200_ep{ep+1:04d}.pt')
                torch.save(ckpt, p)
                print(f"  MILESTONE {p}")
                model.eval()
                for seed in ['The ', 'Hello ', 'Science is ']:
                    print(f"  GEN: \"{seed}{generate(model, seed, 60, 0.42)[:50]}\"")
                model.train()
            print()

    except KeyboardInterrupt:
        print(f"\n  STOPPED at epoch {ep+1}. Saving...")
        torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                     'scaler': scaler.state_dict(), 'epoch': ep, 'best_acc': best,
                     'config': MODEL_CONFIG}, LATEST_CKPT)
        print(f"  Saved. Run again to resume from epoch {ep+1}.")

    print(f"\n  Done. Best: {best:.1f}% | {(time.perf_counter()-t0)/3600:.1f}h")


if __name__ == '__main__':
    main()
