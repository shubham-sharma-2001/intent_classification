# Intent Classifier – BiLSTM + SupCon Loss

Production-ready custom intent classification model built from scratch with PyTorch.  
No pretrained weights. No external NLP libraries. CPU-friendly.

---

## Project Structure

```
intent_classifier/
├── config.py          # Central hyperparameters
├── tokenizer.py       # Word-level tokenizer (build / encode / save / load)
├── model.py           # BiLSTM encoder + projection head + classifier
├── losses.py          # Supervised Contrastive Loss + Combined Loss
├── dataset.py         # PyTorch Dataset & DataLoader factory
├── train.py           # Training loop (logging, early stopping, checkpointing)
├── predict.py         # Inference (single / batch / interactive / CLI)
├── requirements.txt   # torch>=2.0.0
├── data/
│   └── sample_data.json   # 80 labelled examples across 8 intents
└── checkpoints/            # Created after training
    ├── intent_classifier.pt
    ├── vocab.json
    ├── label_maps.json
    └── history.json
```

---

## Architecture

```
Input tokens (B, T)
      │
  Embedding (trainable, dim=128)
      │
  BiLSTM  (2 layers, hidden=128, bidirectional)
      │
  Mean-pool  →  encoder_out  (B, 256)
      ├──────────────────────────────────────────────────────┐
      │                                                      │
  ClassificationHead                                 ProjectionHead
  Linear(256 → C) + Dropout                         Linear(256→256)→ReLU→Linear(256→64)
      │                                                      │
  logits (B, C)                                     L2-normalised z (B, 64)
      │                                                      │
  CrossEntropyLoss                            SupervisedContrastiveLoss (τ=0.07)
      └──────────┬───────────────────────────────────────────┘
                 │
          Combined Loss = 0.5 * CE  +  0.5 * SupCon
```

---

## Quick Start

### 1 — Setup (uv + venv)

```powershell
# Install uv (once)
pip install uv

# Create venv and install torch
python -m uv venv .venv
python -m uv pip install -r requirements.txt --python .venv\Scripts\python.exe
```

### 2 — Train

```powershell
.venv\Scripts\python.exe train.py
```

Expected output:
```
Intent Classifier — Training
[Tokenizer] Vocabulary built: 221 tokens
[Dataset] total=80 | train=68 | val=12 | classes=8
IntentClassifier | params=772,040 | classes=8

Epoch 001/30 | T loss=3.15 acc=0.23 | V loss=2.21 acc=0.08 | 0.1s
...
✓ Checkpoint saved → checkpoints\intent_classifier.pt
✓ Training complete. Best val accuracy: X.XXXX
```

> **Tip**: Accuracy improves significantly with more data.  
> Each intent needs at least 50–200 examples for good generalisation.

### 3 — Predict (CLI)

```powershell
# Single text
.venv\Scripts\python.exe predict.py --text "remind me at 7am"

# Batch file (one sentence per line)
.venv\Scripts\python.exe predict.py --file my_sentences.txt

# Interactive REPL
.venv\Scripts\python.exe predict.py
```

---

## Dataset Format

Plain JSON array — one dict per example:

```json
[
  {"text": "book a flight to paris",      "intent": "book_flight"},
  {"text": "what is the weather today",   "intent": "get_weather"},
  {"text": "play some relaxing music",    "intent": "play_music"}
]
```

- `text`   — raw query string (lowercasing / cleaning done automatically)
- `intent` — class label string (anything, model auto-builds label maps)

---

## Save & Load Model

### Save (automatic during training)

`train.py` saves the best checkpoint to `checkpoints/intent_classifier.pt`  
on every validation accuracy improvement.

Manual save example:
```python
from train import save_checkpoint
save_checkpoint(model, optimizer, epoch=10, val_acc=0.95,
                label2idx=label2idx, idx2label=idx2label)
```

### Load for inference

```python
from predict import IntentPredictor

predictor = IntentPredictor(
    checkpoint_path="checkpoints/intent_classifier.pt",  # optional, uses default
    vocab_path="checkpoints/vocab.json",                 # optional
)

result = predictor.predict("book a flight to london")
print(result["intent"])     # → book_flight
print(result["confidence"]) # → 0.9712

# Batch
results = predictor.predict_batch(["play jazz", "set alarm 6am"], top_k=3)
```

### Load raw model weights

```python
from train import load_checkpoint
model, ckpt = load_checkpoint("checkpoints/intent_classifier.pt")
label2idx   = ckpt["label2idx"]
idx2label   = ckpt["idx2label"]
```

---

## Adding New Intents

1. Add new examples to `data/sample_data.json`
2. Re-run `train.py` — the model auto-discovers all intents

No code changes required.

---

## Key Hyperparameters (`config.py`)

| Parameter        | Default | Notes                              |
|------------------|---------|------------------------------------|
| `EMBEDDING_DIM`  | 128     | Trainable embedding size           |
| `HIDDEN_SIZE`    | 128     | BiLSTM hidden size (per direction) |
| `NUM_LAYERS`     | 2       | BiLSTM layers                      |
| `DROPOUT`        | 0.3     | Applied to embedding + classifier  |
| `PROJ_DIM`       | 64      | Contrastive projection dim         |
| `ALPHA`          | 0.5     | Weight for SupCon loss             |
| `BETA`           | 0.5     | Weight for CrossEntropy            |
| `TEMPERATURE`    | 0.07    | Contrastive temperature τ          |
| `LEARNING_RATE`  | 0.001   | Adam LR                            |
| `BATCH_SIZE`     | 32      | Training batch size                |
| `PATIENCE`       | 7       | Early stopping patience            |

---

## Constraints Met

- ✅ CPU-only (no CUDA required)
- ✅ No pretrained / public models
- ✅ BiLSTM encoder with trainable embeddings
- ✅ Supervised Contrastive Loss + CrossEntropy
- ✅ L2 normalisation before contrastive calculation
- ✅ Adam optimiser with L2 weight decay
- ✅ Early stopping + LR scheduler
- ✅ Future-expandable via JSON dataset
- ✅ `uv` virtual environment
