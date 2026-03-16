"""
ASSOCIATION IS ALL YOU NEED v2 - Binary Edition
=================================================
Predicts next BIT (vocab=2) using ASSOCIATIVE MEMORY.

v2 UPGRADES:
  1. Local pattern detection (1D convolutions find bigrams/trigrams
     BEFORE memory lookup - the model sees "th" not just "t","h")
  2. Multi-hop retrieval (query memory -> refine -> query again,
     like thinking harder about a memory)
  3. Gated memory fusion (learn which memory bank to trust)
  4. Larger, better-structured Hopfield patterns

THE FUNDAMENTAL DIFFERENCE FROM ATTENTION:
  Attention: "Which parts of my input should I focus on?"
  Association: "What stored memory does this input remind me of?"

Trained on ~4 KB. GPU accelerated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from training_data import DEVICE, EnglishInstinct


# -- Local Pattern Detector ------------------------------------------------
class LocalPatternDetector(nn.Module):
    """
    1D convolutions over the byte sequence to detect local patterns
    (bigrams, trigrams, common word fragments) BEFORE memory lookup.

    This is crucial: without this, the model sees individual bytes
    but misses that "t" + "h" together = a very common pattern.
    """

    def __init__(self, embed_dim, num_filters=128):
        super().__init__()
        # Bigram detector (kernel=2)
        self.conv2 = nn.Conv1d(embed_dim, num_filters, kernel_size=2, padding=1)
        # Trigram detector (kernel=3)
        self.conv3 = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        # 4-gram detector (kernel=4) - catches common words
        self.conv4 = nn.Conv1d(embed_dim, num_filters, kernel_size=4, padding=2)

        self.norm = nn.LayerNorm(num_filters * 3)
        self.proj = nn.Linear(num_filters * 3, embed_dim)

    def forward(self, x):
        """x: (B, seq_len, embed_dim)"""
        # Conv1d expects (B, C, L)
        xt = x.transpose(1, 2)

        c2 = F.gelu(self.conv2(xt))[:, :, :x.shape[1]]  # trim to seq_len
        c3 = F.gelu(self.conv3(xt))[:, :, :x.shape[1]]
        c4 = F.gelu(self.conv4(xt))[:, :, :x.shape[1]]

        # Concatenate all n-gram features
        combined = torch.cat([c2, c3, c4], dim=1)  # (B, filters*3, seq_len)
        combined = combined.transpose(1, 2)          # (B, seq_len, filters*3)

        combined = self.norm(combined)
        return self.proj(combined)  # (B, seq_len, embed_dim)


# -- Associative Memory Bank -----------------------------------------------
class AssociativeMemoryBank(nn.Module):
    """
    Stores learned patterns and retrieves the closest match.
    Each memory slot has a KEY (what it matches) and VALUE (what it returns).
    """

    def __init__(self, num_slots, key_dim, value_dim):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(num_slots, key_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(num_slots, value_dim) * 0.02)
        self.scale = key_dim ** 0.5

    def forward(self, query):
        """query: (B, key_dim) -> retrieved: (B, value_dim)"""
        sim = torch.matmul(query, self.keys.T) / self.scale
        weights = F.softmax(sim, dim=-1)
        return torch.matmul(weights, self.values)


# -- Multi-Hop Retrieval ---------------------------------------------------
class MultiHopMemory(nn.Module):
    """
    Query memory, then use the result to REFINE the query,
    then query again. Like thinking harder about a memory.

    Hop 1: "This reminds me of something..."
    Hop 2: "Oh wait, that reminds me of something MORE specific..."
    Hop 3: "Yes! Now I remember exactly."
    """

    def __init__(self, dim, num_slots, num_hops=3):
        super().__init__()
        self.hops = num_hops
        self.memories = nn.ModuleList([
            AssociativeMemoryBank(num_slots, dim, dim)
            for _ in range(num_hops)
        ])
        # Refinement: combine query + retrieved to form next query
        self.refiners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
            )
            for _ in range(num_hops - 1)
        ])

    def forward(self, query):
        """Multi-hop: query -> retrieve -> refine -> retrieve -> ..."""
        current_query = query
        all_retrieved = []

        for i in range(self.hops):
            retrieved = self.memories[i](current_query)
            all_retrieved.append(retrieved)

            if i < self.hops - 1:
                # Refine query using what we retrieved
                combined = torch.cat([current_query, retrieved], dim=-1)
                current_query = self.refiners[i](combined) + query  # residual

        # Sum all hops (each hop adds more specific information)
        return sum(all_retrieved)


# -- Modern Hopfield Layer -------------------------------------------------
class HopfieldLayer(nn.Module):
    """
    Modern continuous Hopfield network.
    Stores binary-like attractor patterns and does
    one-step pattern completion via softmax.
    """

    def __init__(self, input_dim, pattern_dim, num_patterns, dropout=0.1):
        super().__init__()
        self.patterns = nn.Parameter(torch.randn(num_patterns, pattern_dim) * 0.02)
        self.query_proj = nn.Linear(input_dim, pattern_dim)
        self.value_proj = nn.Linear(pattern_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = pattern_dim ** 0.5

    def forward(self, x):
        query = self.query_proj(x)
        sim = torch.matmul(query, self.patterns.T) / self.scale
        attn = F.softmax(sim, dim=-1)
        retrieved = torch.matmul(attn, self.patterns)
        out = self.value_proj(retrieved)
        return self.dropout(self.norm(out))


# -- Gated Fusion -----------------------------------------------------------
class GatedFusion(nn.Module):
    """
    Learns which memory source to trust for each prediction.
    Instead of just concatenating, it gates each source.
    """

    def __init__(self, dim, num_sources):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * num_sources, num_sources),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(dim * num_sources, dim)

    def forward(self, *sources):
        """sources: each (B, dim)"""
        cat = torch.cat(sources, dim=-1)       # (B, dim * num_sources)
        gates = self.gate(cat)                  # (B, num_sources)

        # Apply gates to each source
        gated = []
        for i, src in enumerate(sources):
            gated.append(src * gates[:, i:i+1])

        merged = torch.cat(gated, dim=-1)
        return self.proj(merged)


# -- Association Block (for stacking) ----------------------------------------
class AssociationBlock(nn.Module):
    """One block of associative processing. Stack these for depth."""

    def __init__(self, embed_dim, num_memories, num_hopfield, num_hops, dropout):
        super().__init__()
        self.multi_hop = MultiHopMemory(embed_dim, num_memories, num_hops)
        self.hopfield = HopfieldLayer(embed_dim, embed_dim, num_hopfield, dropout)
        self.direct = AssociativeMemoryBank(num_memories, embed_dim, embed_dim)
        self.fusion = GatedFusion(embed_dim, 3)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        m = self.multi_hop(x)
        h = self.hopfield(x)
        d = self.direct(x)
        mem = self.fusion(m, h, d)
        x = self.norm1(x + mem)
        x = self.norm2(x + self.ffn(x))
        return x


# -- Deep Association Model (2B) --------------------------------------------
class DeepAssociationBinaryGPT(nn.Module):
    """
    "Association Is All You Need" - Deep Edition (2B params).

    Stacks 22 AssociationBlocks, each with its own memory banks.
    Early blocks learn low-level patterns, later blocks learn
    high-level semantic associations.

    Same philosophy: REMEMBER patterns, don't just process sequences.
    """

    def __init__(self, context_bytes=128, embed_dim=2048,
                 num_memories=1024, num_hopfield=1024,
                 num_hops=3, num_blocks=22, dropout=0.05):
        super().__init__()
        self.context_bytes = context_bytes

        # Byte + position embeddings
        self.byte_embed = nn.Embedding(128, embed_dim)
        self.pos_embed = nn.Embedding(context_bytes, embed_dim)

        # Local pattern detection
        self.pattern_detector = LocalPatternDetector(embed_dim, num_filters=embed_dim // 4)

        # Initial projection: combine embeddings + patterns into query
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Stacked association blocks - the core
        self.blocks = nn.ModuleList([
            AssociationBlock(embed_dim, num_memories, num_hopfield, num_hops, dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        # Bit prediction head
        head_in = embed_dim + 16 + 7 + EnglishInstinct.FEATURE_SIZE
        self.bit_pos_embed = nn.Embedding(8, 16)
        self.head = nn.Sequential(
            nn.Linear(head_in, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, ctx_bytes, bit_pos, partial, instinct):
        positions = torch.arange(self.context_bytes, device=ctx_bytes.device)

        # 1. Embed bytes + positions
        x = self.byte_embed(ctx_bytes) + self.pos_embed(positions)

        # 2. Detect local patterns
        patterns = self.pattern_detector(x)

        # 3. Compress to query (last position)
        combined = torch.cat([x[:, -1, :], patterns[:, -1, :]], dim=-1)
        query = self.input_proj(combined)

        # 4. Deep associative retrieval (22 blocks of memory lookups)
        for block in self.blocks:
            query = block(query)

        query = self.final_norm(query)

        # 5. Predict bit
        bit_emb = self.bit_pos_embed(bit_pos)
        final = torch.cat([query, bit_emb, partial, instinct], dim=1)
        return self.head(final).squeeze(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# -- Association Model v2 (small, ~10M) ------------------------------------
class AssociationBinaryGPT(nn.Module):
    """
    "Association Is All You Need" v2.

    Architecture:
      1. Embed bytes + detect local patterns (bigrams/trigrams)
      2. Encode sequence into query vector
      3. Multi-hop memory retrieval (think harder, remember better)
      4. Hopfield pattern completion
      5. Gated fusion of all memory sources
      6. Predict next bit

    The model REMEMBERS patterns, not just processes sequences.
    """

    def __init__(self, context_bytes=64, embed_dim=192,
                 num_memories=384, num_hopfield=256,
                 num_hops=3, dropout=0.12):
        super().__init__()
        self.context_bytes = context_bytes

        # Byte + position embeddings
        self.byte_embed = nn.Embedding(128, embed_dim)
        self.pos_embed = nn.Embedding(context_bytes, embed_dim)

        # Local pattern detection (bigrams, trigrams, 4-grams)
        self.pattern_detector = LocalPatternDetector(embed_dim, num_filters=96)

        # Encode: combine raw embeddings + detected patterns -> query
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Multi-hop associative memory (3 hops of refinement)
        self.multi_hop = MultiHopMemory(embed_dim, num_memories, num_hops=num_hops)

        # Hopfield pattern completion
        self.hopfield = HopfieldLayer(embed_dim, embed_dim, num_hopfield, dropout)

        # Direct memory (simple single-hop for fast retrieval)
        self.direct_memory = AssociativeMemoryBank(num_memories, embed_dim, embed_dim)

        # Gated fusion of 3 memory sources
        self.fusion = GatedFusion(embed_dim, num_sources=3)

        # Bit prediction head
        head_in = embed_dim + 16 + 7 + EnglishInstinct.FEATURE_SIZE
        self.bit_pos_embed = nn.Embedding(8, 16)
        self.head = nn.Sequential(
            nn.Linear(head_in, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, ctx_bytes, bit_pos, partial, instinct):
        B = ctx_bytes.shape[0]
        positions = torch.arange(self.context_bytes, device=ctx_bytes.device)

        # 1. Embed bytes
        x = self.byte_embed(ctx_bytes) + self.pos_embed(positions)  # (B, 64, dim)

        # 2. Detect local patterns (bigrams, trigrams)
        patterns = self.pattern_detector(x)  # (B, 64, dim)

        # 3. Combine last position embeddings + patterns into query
        combined = torch.cat([x[:, -1, :], patterns[:, -1, :]], dim=-1)
        query = self.encoder(combined)  # (B, dim)

        # 4. ASSOCIATIVE RETRIEVAL (the core)
        multi_hop_result = self.multi_hop(query)       # Deep recall
        hopfield_result = self.hopfield(query)          # Pattern completion
        direct_result = self.direct_memory(query)       # Fast recall

        # 5. Gated fusion: learn which memory to trust
        fused = self.fusion(multi_hop_result, hopfield_result, direct_result)

        # 6. Predict bit
        bit_emb = self.bit_pos_embed(bit_pos)
        final = torch.cat([fused, bit_emb, partial, instinct], dim=1)
        return self.head(final).squeeze(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# -- Training ---------------------------------------------------------------
def train(model, text, epochs=200, batch_size=1024, lr=0.001):
    from training_data import build_dataset
    X_b, X_bp, X_p, X_i, Y = build_dataset(text, model.context_bytes)
    N = X_b.shape[0]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Warmup + cosine decay
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # Label smoothing: prevents overconfidence, helps generalization
    criterion = nn.BCEWithLogitsLoss()

    print(f"  [DATA]  {N:,} samples from {len(text):,} chars ({len(text)/1024:.1f} KB)")
    print(f"  [TRAIN] {epochs} epochs | warmup: {warmup_epochs} | device: {DEVICE}")
    print()

    model.train()
    t0 = time.perf_counter()
    best_acc = 0

    for epoch in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        total_loss = correct = batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]
            logits = model(X_b[idx], X_bp[idx], X_p[idx], X_i[idx])
            loss = criterion(logits, Y[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == Y[idx]).sum().item()
            batches += 1

        scheduler.step()
        acc = correct / N * 100
        best_acc = max(best_acc, acc)
        elapsed = time.perf_counter() - t0

        if (epoch + 1) % 20 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1:3d}/{epochs}  "
                  f"loss={total_loss/batches:.4f}  acc={acc:.1f}%  "
                  f"lr={lr_now:.6f}  time={elapsed:.1f}s")

    elapsed = time.perf_counter() - t0
    print(f"\n  [DONE] {elapsed:.1f}s | Best: {best_acc:.1f}%\n")
    return best_acc


def train_chunked(model, text, epochs=50, batch_size=1024, lr=0.001, chunk_size=50000):
    """
    Train on large text by processing in chunks.
    Only one chunk's data is in GPU memory at a time.
    Each epoch cycles through all chunks.
    """
    import random
    from training_data import build_dataset

    # Split text into overlapping chunks
    ctx = model.context_bytes
    chunks = []
    for start in range(0, len(text) - ctx, chunk_size - ctx):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
    if not chunks:
        chunks = [text]

    total_chars = len(text)
    print(f"  [DATA]  {total_chars:,} chars ({total_chars/1024:.1f} KB) in {len(chunks)} chunks")
    print(f"  [TRAIN] {epochs} epochs | {len(chunks)} chunks | device: {DEVICE}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup_epochs = min(5, epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    t0 = time.perf_counter()
    best_acc = 0

    for epoch in range(epochs):
        random.shuffle(chunks)
        total_loss = total_correct = total_samples = batches = 0

        for chunk in chunks:
            X_b, X_bp, X_p, X_i, Y = build_dataset(chunk, ctx)
            N = X_b.shape[0]
            if N < 10:
                continue

            perm = torch.randperm(N, device=DEVICE)
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx = perm[start:end]
                logits = model(X_b[idx], X_bp[idx], X_p[idx], X_i[idx])
                loss = criterion(logits, Y[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                total_correct += (preds == Y[idx]).sum().item()
                batches += 1

            total_samples += N
            # Free chunk data from GPU
            del X_b, X_bp, X_p, X_i, Y
            torch.cuda.empty_cache()

        scheduler.step()
        acc = total_correct / max(total_samples, 1) * 100
        best_acc = max(best_acc, acc)
        elapsed = time.perf_counter() - t0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1:3d}/{epochs}  "
                  f"loss={total_loss/max(batches,1):.4f}  acc={acc:.1f}%  "
                  f"lr={lr_now:.6f}  time={elapsed:.1f}s  "
                  f"[{total_samples:,} samples]")

    elapsed = time.perf_counter() - t0
    print(f"\n  [DONE] {elapsed:.1f}s | Best: {best_acc:.1f}%\n")
    return best_acc


# -- Generation -------------------------------------------------------------
def generate(model, seed="The ", num_chars=100, temperature=0.42):
    """Generate with repetition penalty to avoid loops."""
    model.eval()
    ctx_len = model.context_bytes
    all_bytes = [ord(' ')] * max(0, ctx_len - len(seed))
    all_bytes += [ord(c) & 0x7F for c in seed]
    generated = []

    # Track recent 3-grams for repetition penalty
    recent_trigrams = {}

    with torch.no_grad():
        for char_i in range(num_chars):
            ctx = all_bytes[-ctx_len:]
            ctx_text = ''.join(chr(b) if 10 <= b < 127 else ' ' for b in ctx)
            byte_bits = []

            # Check if we're about to repeat a pattern
            rep_boost = 0.0
            if len(generated) >= 3:
                tri = ''.join(generated[-3:])
                count = recent_trigrams.get(tri, 0)
                if count > 0:
                    rep_boost = min(count * 0.15, 0.4)  # boost randomness

            for bit_pos in range(8):
                partial = [0.0] * 7
                for j in range(min(bit_pos, 7)):
                    partial[j] = float(byte_bits[j])
                inst = EnglishInstinct.get_features(ctx_text, bit_pos, byte_bits)

                x_b = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
                x_bp = torch.tensor([bit_pos], dtype=torch.long, device=DEVICE)
                x_p = torch.tensor([partial], dtype=torch.float32, device=DEVICE)
                x_i = torch.tensor([inst], dtype=torch.float32, device=DEVICE)

                logit = model(x_b, x_bp, x_p, x_i)
                # Add randomness if repeating
                adj_temp = temperature + rep_boost
                prob = torch.sigmoid(logit / adj_temp).item()
                byte_bits.append(1 if torch.rand(1).item() < prob else 0)

            byte_val = 0
            for bit in byte_bits:
                byte_val = (byte_val << 1) | bit
            char = chr(byte_val) if 10 <= byte_val < 127 else ' '
            all_bytes.append(byte_val)
            generated.append(char)

            # Track trigrams
            if len(generated) >= 3:
                tri = ''.join(generated[-3:])
                recent_trigrams[tri] = recent_trigrams.get(tri, 0) + 1

    return ''.join(generated)


# -- Main -------------------------------------------------------------------
def main():
    from training_data import TRAINING_TEXT
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print()
    print("=" * 62)
    print("  ASSOCIATION IS ALL YOU NEED v2 - Binary Edition")
    print("  Next-bit prediction with multi-hop associative memory")
    print("=" * 62)
    print()
    print(f"  GPU: {gpu}")
    print()
    print("  v2 UPGRADES:")
    print("  - Local pattern detection (conv bigrams/trigrams)")
    print("  - Multi-hop retrieval (think harder, remember better)")
    print("  - Gated fusion (learn which memory to trust)")
    print("  - Hopfield pattern completion")
    print()
    print(f"  Training data: {len(TRAINING_TEXT):,} chars ({len(TRAINING_TEXT)/1024:.1f} KB)")
    print(f"  GPT-2 used:    40,000,000,000 chars (40 GB)")
    print(f"  Ratio:         1 : {40_000_000_000 // len(TRAINING_TEXT):,}")
    print()

    model = AssociationBinaryGPT(
        context_bytes=64,
        embed_dim=192,
        num_memories=384,
        num_hopfield=256,
        num_hops=3,
        dropout=0.12,
    ).to(DEVICE)

    print(f"  [MODEL] Context:       64 chars")
    print(f"  [MODEL] Embed dim:     192")
    print(f"  [MODEL] Memory slots:  384 (per bank, 3 hops + direct)")
    print(f"  [MODEL] Hopfield:      256 patterns")
    print(f"  [MODEL] Memory hops:   3 (iterative refinement)")
    print(f"  [MODEL] Parameters:    {model.count_params():,}")
    print(f"  [MODEL] Vocab:         2 (0 and 1)")
    print(f"  [MODEL] Core:          MULTI-HOP ASSOCIATIVE MEMORY")
    print()

    print("-" * 62)
    train(model, TRAINING_TEXT, epochs=400, batch_size=512, lr=0.0008)
    print("-" * 62)
    print()

    # Generate
    seeds = ["The ", "I ", "She ", "The cat ", "Good ", "He went to ",
             "In the ", "Once upon ", "The old "]

    print("  Generating text one BIT at a time (8 bits = 1 char):")
    print()
    for seed in seeds:
        text = generate(model, seed, num_chars=80, temperature=0.42)
        print(f"  \"{seed}\"")
        print(f"   -> {seed}{text}")
        print()

    # Interactive
    print("-" * 62)
    print("  Interactive. |chars=N |temp=T  Type 'quit' to exit.")
    print("-" * 62)
    print()
    while True:
        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw or raw.lower() in ("quit", "exit", "q"):
            break

        num_c, temp, seed = 120, 0.50, raw
        if "|" in raw:
            parts = raw.split("|")
            seed = parts[0].strip()
            for p in parts[1:]:
                p = p.strip()
                if p.startswith("chars="): num_c = int(p[6:])
                elif p.startswith("temp="): temp = float(p[5:])

        t0 = time.perf_counter()
        text = generate(model, seed, num_c, temp)
        elapsed = time.perf_counter() - t0
        print(f"\n  >> {seed}{text}")
        print(f"     [{num_c*8:,} bits | {elapsed:.2f}s | "
              f"ASSOCIATION v2 | {len(TRAINING_TEXT)/1024:.1f}KB training | vocab=2]\n")

    print("\n" + "=" * 62)
    print("  Association v2 - memory is all you need.")
    print("=" * 62)


if __name__ == "__main__":
    main()
