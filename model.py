# =============================================================================
# model.py — BiLSTM encoder + projection head + classification head
# =============================================================================
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder.

    Input : (batch, seq_len)  — token indices
    Output: (batch, hidden_size * 2)  — mean-pooled BiLSTM representation
    """

    def __init__(self, vocab_size: int, embedding_dim: int,
                 hidden_size: int, num_layers: int,
                 dropout: float, pad_idx: int):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : (batch, seq_len) — padded token ids
        lengths : (batch,) — actual lengths before padding (optional)

        Returns
        -------
        pooled : (batch, hidden_size * 2)
        """
        # Embedding
        emb = self.dropout(self.embedding(x))          # (B, T, E)

        if lengths is not None:
            # Pack for efficiency (avoids LSTM computing over padding)
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out_packed, batch_first=True
            )                                           # (B, T, 2H)
        else:
            out, _ = self.lstm(emb)                    # (B, T, 2H)

        # Mean pooling over time dimension
        pooled = out.mean(dim=1)                       # (B, 2H)
        return pooled


class ProjectionHead(nn.Module):
    """
    MLP projection head used for supervised contrastive learning.
    Maps encoder output → L2-normalised embedding in lower-dim space.
    """

    def __init__(self, in_dim: int, proj_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.net(x)                        # (B, proj_dim)
        return F.normalize(projected, p=2, dim=-1)     # L2 normalise


class ClassificationHead(nn.Module):
    """
    Linear head: encoder → num_classes logits.
    """

    def __init__(self, in_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                             # (B, num_classes)


class IntentClassifier(nn.Module):
    """
    Full model:
        tokens  →  BiLSTMEncoder  →  encoder_out
                                   ├─→  ProjectionHead  →  z  (contrastive)
                                   └─→  ClassificationHead →  logits (CE)

    Parameters
    ----------
    vocab_size  : size of the learned embedding vocabulary
    num_classes : number of intent classes
    pad_idx     : index of the <PAD> token in the vocabulary
    """

    def __init__(self, vocab_size: int, num_classes: int, pad_idx: int,
                 embedding_dim: int = cfg.EMBEDDING_DIM,
                 hidden_size:   int = cfg.HIDDEN_SIZE,
                 num_layers:    int = cfg.NUM_LAYERS,
                 dropout:       float = cfg.DROPOUT,
                 proj_dim:      int = cfg.PROJ_DIM):
        super().__init__()

        encoder_out_dim = hidden_size * 2   # bidirectional

        self.encoder = BiLSTMEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=pad_idx,
        )

        self.projection = ProjectionHead(
            in_dim=encoder_out_dim,
            proj_dim=proj_dim,
            dropout=dropout,
        )

        self.classifier = ClassificationHead(
            in_dim=encoder_out_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Store for convenience
        self.num_classes  = num_classes
        self.hidden_size  = hidden_size
        self.proj_dim     = proj_dim

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier/Orthogonal initialisation for stable training."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor,
                lengths=None
                ) -> tuple:
        """
        Returns
        -------
        logits : (B, num_classes)   — for cross-entropy
        z      : (B, proj_dim)      — L2-normalised, for contrastive loss
        """
        enc    = self.encoder(x, lengths)   # (B, 2H)
        logits = self.classifier(enc)       # (B, C)
        z      = self.projection(enc)       # (B, proj_dim)
        return logits, z

    def predict(self, x: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Convenience: returns predicted class indices. (B,)"""
        logits, _ = self.forward(x, lengths)
        return logits.argmax(dim=-1)

    # ── Pretty summary ─────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __str__(self) -> str:
        return (
            f"IntentClassifier | "
            f"params={self.count_parameters():,} | "
            f"classes={self.num_classes}"
        )
