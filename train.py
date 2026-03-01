# =============================================================================
# train.py — Full training loop with logging + early stopping + checkpointing
# =============================================================================

import os
import json
import time
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import cfg
from dataset import get_dataloaders
from model import IntentClassifier
from losses import CombinedLoss


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


# ── Metrics ────────────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer,
              device, train: bool = True) -> dict:
    model.train(train)
    total_loss = total_ce = total_sup = total_acc = 0.0
    n_batches  = 0

    with torch.set_grad_enabled(train):
        for tokens, labels, lengths in loader:
            tokens  = tokens.to(device)
            labels  = labels.to(device)
            lengths = lengths.to(device)

            logits, z = model(tokens, lengths)

            loss, parts = criterion(logits, z, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()

            total_loss += parts["total"]
            total_ce   += parts["ce"]
            total_sup  += parts["contrastive"]
            total_acc  += accuracy(logits, labels)
            n_batches  += 1

    n = max(n_batches, 1)
    return {
        "loss":        total_loss / n,
        "ce":          total_ce   / n,
        "contrastive": total_sup  / n,
        "accuracy":    total_acc  / n,
    }


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch: int,
                    val_acc: float, label2idx: dict,
                    idx2label: dict) -> None:
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
    path = os.path.join(cfg.MODEL_SAVE_DIR, cfg.MODEL_NAME)
    torch.save({
        "epoch":       epoch,
        "val_acc":     val_acc,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "label2idx":   label2idx,
        "idx2label":   idx2label,
        "model_cfg": {
            "vocab_size":     model.encoder.embedding.num_embeddings,
            "num_classes":    model.num_classes,
            "pad_idx":        model.encoder.embedding.padding_idx,
            "embedding_dim":  cfg.EMBEDDING_DIM,
            "hidden_size":    cfg.HIDDEN_SIZE,
            "num_layers":     cfg.NUM_LAYERS,
            "dropout":        cfg.DROPOUT,
            "proj_dim":       cfg.PROJ_DIM,
        },
    }, path)
    print(f"  ✓ Checkpoint saved → {path}  (val_acc={val_acc:.4f})")


def load_checkpoint(path: str = None, device: str = cfg.DEVICE):
    """Load model from checkpoint. Returns (model, meta_dict)."""
    from tokenizer import WordTokenizer
    path = path or os.path.join(cfg.MODEL_SAVE_DIR, cfg.MODEL_NAME)
    ckpt = torch.load(path, map_location=device)

    mcfg  = ckpt["model_cfg"]
    model = IntentClassifier(**mcfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    print(f"[Checkpoint] Loaded from {path}  (epoch={ckpt['epoch']}, "
          f"val_acc={ckpt['val_acc']:.4f})")
    return model, ckpt


# ── Main training function ─────────────────────────────────────────────────────

def train() -> None:
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)

    print("=" * 60)
    print("  Intent Classifier — Training")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, tokenizer, label2idx, idx2label = \
        get_dataloaders(cfg.DATA_PATH, cfg.BATCH_SIZE)

    # Save vocabulary for inference
    tokenizer.save(cfg.VOCAB_PATH)

    # Save label maps
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
    with open(os.path.join(cfg.MODEL_SAVE_DIR, "label_maps.json"), "w") as f:
        json.dump({"label2idx": label2idx, "idx2label": idx2label}, f, indent=2)

    # ── Model ─────────────────────────────────────────────────────────────
    model = IntentClassifier(
        vocab_size=tokenizer.vocab_size,
        num_classes=len(label2idx),
        pad_idx=tokenizer.pad_idx,
    ).to(device)

    print(f"\n{model}\n")

    # ── Optimizer / Scheduler / Loss ───────────────────────────────────────
    optimizer = Adam(model.parameters(),
                     lr=cfg.LEARNING_RATE,
                     weight_decay=cfg.WEIGHT_DECAY)

    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    criterion = CombinedLoss(num_classes=len(label2idx))

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc = 0.0
    patience_cnt = 0
    history      = []

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()

        train_metrics = run_epoch(model, train_loader, criterion, optimizer,
                                  device, train=True)
        val_metrics   = run_epoch(model, val_loader,   criterion, optimizer,
                                  device, train=False)

        scheduler.step(val_metrics["accuracy"])
        elapsed = time.time() - t0

        # Log
        if epoch % cfg.LOG_EVERY == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}/{cfg.EPOCHS} | "
                f"T loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['accuracy']:.4f} | "
                f"V loss={val_metrics['loss']:.4f} "
                f"acc={val_metrics['accuracy']:.4f} | "
                f"{elapsed:.1f}s"
            )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val":   val_metrics,
        })

        # Checkpoint best
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_cnt = 0
            save_checkpoint(model, optimizer, epoch,
                            best_val_acc, label2idx, idx2label)
        else:
            patience_cnt += 1

        # Early stopping
        if patience_cnt >= cfg.PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    print(f"\n✓ Training complete. Best val accuracy: {best_val_acc:.4f}")

    # Save training history
    hist_path = os.path.join(cfg.MODEL_SAVE_DIR, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"✓ History saved → {hist_path}")


if __name__ == "__main__":
    train()
