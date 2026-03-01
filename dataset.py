# =============================================================================
# dataset.py — PyTorch Dataset + DataLoader helpers
# =============================================================================

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Tuple

from tokenizer import WordTokenizer
from config import cfg


class IntentDataset(Dataset):
    """
    Reads a JSON file with the format:
    [
       {"text": "book a flight to paris", "intent": "book_flight"},
       ...
    ]

    Parameters
    ----------
    data       : list of {"text": ..., "intent": ...} dicts
    tokenizer  : fitted WordTokenizer
    label2idx  : mapping from intent string → integer
    """

    def __init__(self, data: List[Dict],
                 tokenizer: WordTokenizer,
                 label2idx: Dict[str, int]):
        self.data      = data
        self.tokenizer = tokenizer
        self.label2idx = label2idx

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item   = self.data[idx]
        ids    = self.tokenizer.encode(item["text"])
        label  = self.label2idx[item["intent"]]

        # Compute actual (non-padding) length for pack_padded_sequence
        pad_id = self.tokenizer.pad_idx
        length = sum(1 for i in ids if i != pad_id)
        length = max(length, 1)    # at least 1

        return (
            torch.tensor(ids,    dtype=torch.long),
            torch.tensor(label,  dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )


# ── Factory helpers ────────────────────────────────────────────────────────────

def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_label_maps(data: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build label2idx and idx2label from raw data."""
    intents   = sorted({item["intent"] for item in data})
    label2idx = {intent: i for i, intent in enumerate(intents)}
    idx2label = {i: intent for intent, i in label2idx.items()}
    return label2idx, idx2label


def get_dataloaders(
    data_path:   str            = cfg.DATA_PATH,
    batch_size:  int            = cfg.BATCH_SIZE,
    val_split:   float          = 0.15,
    tokenizer:   WordTokenizer  | None = None,
) -> Tuple[DataLoader, DataLoader, WordTokenizer, Dict, Dict]:
    """
    Full pipeline:
      1. Load JSON data
      2. Build / use tokenizer
      3. Build label maps
      4. Split into train / val
      5. Return DataLoaders + metadata

    Returns
    -------
    train_loader, val_loader, tokenizer, label2idx, idx2label
    """
    data      = load_json(data_path)
    label2idx, idx2label = build_label_maps(data)

    # Build tokenizer if not provided
    if tokenizer is None:
        tokenizer = WordTokenizer()
        tokenizer.build_vocab([item["text"] for item in data])

    dataset = IntentDataset(data, tokenizer, label2idx)

    # Train / val split
    val_size   = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.SEED),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    print(f"[Dataset] total={len(dataset)} | "
          f"train={len(train_ds)} | val={len(val_ds)} | "
          f"classes={len(label2idx)}")

    return train_loader, val_loader, tokenizer, label2idx, idx2label
