"""Shared training data and utilities for both architectures."""

import torch

import os as _os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ALL training data from training_data/ folder
def _load_training_files():
    base = _os.path.dirname(__file__)
    extra = ""
    # Load from training_data/ folder (the main corpus)
    td_dir = _os.path.join(base, "training_data")
    if _os.path.isdir(td_dir):
        for fname in sorted(_os.listdir(td_dir)):
            if fname.endswith('.txt'):
                path = _os.path.join(td_dir, fname)
                with open(path, 'r', encoding='utf-8') as f:
                    extra += "\n" + f.read()
    # Also load legacy files if they exist
    for fname in ["training_earth.txt", "training_english.txt", "training_knowledge.txt"]:
        path = _os.path.join(base, fname)
        if _os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                extra += "\n" + f.read()
    return extra

_BASE_TEXT = """The cat sat on the mat. The dog ran in the park.
I like to read books every day. She likes to write stories.
The sun is bright and warm today. The moon shines at night.
He went to the store to buy some food. They went to the beach to swim.
The wind blows through the tall trees. Rain falls from the dark clouds.
She said hello to her old friend. He waved goodbye to his brother.
A good day starts with a warm smile. The children played in the yard.
The old man told a long story. The young girl listened with great care.
The quick brown fox jumps over the lazy dog. Pack my box with five dozen jugs.
Good morning to you. How are you today. I am fine thank you very much.
"""

# Lazy-loaded: only computed when first accessed
_TRAINING_TEXT = None

def _get_training_text():
    global _TRAINING_TEXT
    if _TRAINING_TEXT is None:
        _TRAINING_TEXT = _BASE_TEXT + _load_training_files()
    return _TRAINING_TEXT


class _LazyText:
    """Allows `TRAINING_TEXT` to work like a string but loads lazily."""
    def __getattr__(self, name):
        return getattr(_get_training_text(), name)
    def __len__(self):
        return len(_get_training_text())
    def __str__(self):
        return _get_training_text()
    def __repr__(self):
        return repr(_get_training_text())
    def __getitem__(self, key):
        return _get_training_text()[key]
    def __contains__(self, item):
        return item in _get_training_text()
    def __add__(self, other):
        return _get_training_text() + other
    def __radd__(self, other):
        return other + _get_training_text()
    def __iter__(self):
        return iter(_get_training_text())
    def __format__(self, fmt):
        return format(_get_training_text(), fmt)


TRAINING_TEXT = _LazyText()


class EnglishInstinct:
    """Built-in English knowledge. No training data needed."""

    CHAR_FREQ = {
        ' ': 0.18, 'e': 0.105, 't': 0.075, 'a': 0.068, 'o': 0.063,
        'i': 0.058, 'n': 0.057, 's': 0.053, 'h': 0.050, 'r': 0.049,
        'd': 0.035, 'l': 0.033, 'c': 0.023, 'u': 0.023, 'm': 0.020,
        'w': 0.019, 'f': 0.018, 'g': 0.017, 'y': 0.016, 'p': 0.016,
        'b': 0.013, 'v': 0.008, 'k': 0.006, '.': 0.012, '\n': 0.008,
    }

    BIGRAMS = {
        'th': 37, 'he': 33, 'in': 29, 'er': 28, 'an': 27, 'on': 23,
        'en': 22, 'at': 21, 'es': 20, 'ed': 20, 'or': 19, 'te': 19,
        'of': 18, 'it': 18, 'is': 18, 'al': 17, 'ar': 16, 'st': 16,
        'to': 16, 'nt': 15, 'ng': 15, 'se': 14, 'ha': 14, 'ou': 13,
        're': 13, 'nd': 13, 'e ': 25, 's ': 20, 'd ': 18, ' t': 30,
        ' a': 20, ' s': 15, ' i': 13, ' w': 12, ' h': 11, ' o': 10,
    }

    FEATURE_SIZE = 10

    @classmethod
    def get_features(cls, context_text, bit_pos, partial_bits):
        f = []
        bit_1_probs = [0.02, 0.52, 0.55, 0.48, 0.50, 0.50, 0.52, 0.48]
        f.append(bit_1_probs[bit_pos])
        f.append(0.0 if bit_pos == 0 else 1.0)

        if len(partial_bits) >= 3:
            f.append(1.0 if partial_bits[:3] == [0, 1, 1] else 0.0)
            f.append(1.0 if partial_bits[:3] == [0, 0, 1] else 0.0)
        else:
            f.extend([0.5, 0.1])

        if context_text:
            last = context_text[-1]
            f.append(1.0 if last == ' ' else 0.0)
            f.append(1.0 if last in '.!?\n' else 0.0)
        else:
            f.extend([0.0, 0.0])

        if len(context_text) >= 1:
            pair = context_text[-1].lower()
            score = max((v for k, v in cls.BIGRAMS.items()
                         if k[0] == pair), default=0) / 40.0
            f.append(score)
        else:
            f.append(0.0)

        wlen = 0
        for c in reversed(context_text):
            if c == ' ':
                break
            wlen += 1
        f.append(min(wlen / 8.0, 1.0))

        if context_text and context_text[-1].lower() in 'bcdfghjklmnpqrstvwxyz':
            f.append(0.7)
        else:
            f.append(0.3)

        if len(context_text) >= 2 and context_text[-1] == ' ' and context_text[-2] == '.':
            f.append(1.0)
        else:
            f.append(0.0)

        return f


def text_to_bits(text):
    bits = []
    for ch in text:
        b = ord(ch) & 0x7F
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def bits_to_text(bits):
    out = []
    for i in range(0, len(bits) - 7, 8):
        v = 0
        for j in range(8):
            v = (v << 1) | bits[i + j]
        out.append(chr(v) if 10 <= v < 127 else ' ')
    return ''.join(out)


def build_dataset(text, context_bytes=32):
    """Build byte-aware training samples with instinct features."""
    text_bytes = [ord(c) & 0x7F for c in text]

    X_bytes = []
    X_bpos = []
    X_partial = []
    X_inst = []
    Y = []

    for char_idx in range(context_bytes, len(text_bytes)):
        ctx = text_bytes[char_idx - context_bytes:char_idx]
        ctx_text = text[char_idx - context_bytes:char_idx]
        target_byte = text_bytes[char_idx]
        target_bits = [(target_byte >> (7 - i)) & 1 for i in range(8)]

        for bit_pos in range(8):
            X_bytes.append(ctx)
            X_bpos.append(bit_pos)

            partial = [0.0] * 7
            for j in range(min(bit_pos, 7)):
                partial[j] = float(target_bits[j])
            X_partial.append(partial)

            partial_list = target_bits[:bit_pos]
            inst = EnglishInstinct.get_features(ctx_text, bit_pos, partial_list)
            X_inst.append(inst)
            Y.append(target_bits[bit_pos])

    return (
        torch.tensor(X_bytes, dtype=torch.long, device=DEVICE),
        torch.tensor(X_bpos, dtype=torch.long, device=DEVICE),
        torch.tensor(X_partial, dtype=torch.float32, device=DEVICE),
        torch.tensor(X_inst, dtype=torch.float32, device=DEVICE),
        torch.tensor(Y, dtype=torch.float32, device=DEVICE),
    )
