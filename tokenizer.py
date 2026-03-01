# =============================================================================
# tokenizer.py — Word-level tokenizer with vocabulary management
# =============================================================================
from __future__ import annotations

import re
import json
import os
from collections import Counter
from typing import List, Dict, Optional

from config import cfg


# ── Helper ────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Lowercase and keep only alphanumeric characters + spaces."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ── Tokenizer class ───────────────────────────────────────────────────────────

class WordTokenizer:
    """
    Simple whitespace-based word tokenizer.

    Special tokens
    --------------
    <PAD>  – padding token (index 0)
    <UNK>  – unknown / out-of-vocab token (index 1)
    <BOS>  – beginning of sentence (index 2)
    <EOS>  – end of sentence (index 3)
    """

    PAD, UNK, BOS, EOS = "<PAD>", "<UNK>", "<BOS>", "<EOS>"
    SPECIALS = [PAD, UNK, BOS, EOS]

    def __init__(self, max_seq_len: int = cfg.MAX_SEQ_LEN,
                 min_freq: int = cfg.MIN_FREQ):
        self.max_seq_len = max_seq_len
        self.min_freq    = min_freq

        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._built = False

    # ── Build ──────────────────────────────────────────────────────────────

    def build_vocab(self, sentences: List[str]) -> None:
        """Build vocabulary from a list of raw sentences."""
        counter: Counter = Counter()
        for sent in sentences:
            tokens = _clean(sent).split()
            counter.update(tokens)

        # Initialise with special tokens first
        self.word2idx = {tok: i for i, tok in enumerate(self.SPECIALS)}

        # Add words that meet minimum frequency
        for word, freq in counter.most_common():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)

        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self._built   = True
        print(f"[Tokenizer] Vocabulary built: {len(self.word2idx)} tokens "
              f"(min_freq={self.min_freq})")

    # ── Encode / Decode ────────────────────────────────────────────────────

    def encode(self, text: str,
               add_special_tokens: bool = True) -> List[int]:
        """
        Convert a raw sentence to a zero-padded list of indices.

        The output length is always `max_seq_len`.
        """
        if not self._built:
            raise RuntimeError("Vocabulary not built. Call build_vocab() first.")

        tokens  = _clean(text).split()
        unk_idx = self.word2idx[self.UNK]
        ids     = [self.word2idx.get(t, unk_idx) for t in tokens]

        if add_special_tokens:
            ids = [self.word2idx[self.BOS]] + ids + [self.word2idx[self.EOS]]

        # Truncate
        ids = ids[:self.max_seq_len]

        # Pad
        pad_idx = self.word2idx[self.PAD]
        ids += [pad_idx] * (self.max_seq_len - len(ids))

        return ids

    def decode(self, ids: List[int],
               skip_special: bool = True) -> str:
        """Convert a list of indices back to a readable string."""
        tokens = [self.idx2word.get(i, self.UNK) for i in ids]
        if skip_special:
            tokens = [t for t in tokens if t not in self.SPECIALS]
        return " ".join(tokens)

    # ── Serialisation ──────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> None:
        path = path or cfg.VOCAB_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "max_seq_len": self.max_seq_len,
            "min_freq":    self.min_freq,
            "word2idx":    self.word2idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[Tokenizer] Vocabulary saved → {path}")

    @classmethod
    def load(cls, path: Optional[str] = None) -> "WordTokenizer":
        path = path or cfg.VOCAB_PATH
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok            = cls(max_seq_len=payload["max_seq_len"],
                             min_freq=payload["min_freq"])
        tok.word2idx   = payload["word2idx"]
        tok.idx2word   = {int(i): w for w, i in payload["word2idx"].items()}
        tok._built     = True
        print(f"[Tokenizer] Vocabulary loaded from {path} "
              f"({len(tok.word2idx)} tokens)")
        return tok

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    @property
    def pad_idx(self) -> int:
        return self.word2idx[self.PAD]
