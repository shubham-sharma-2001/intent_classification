# =============================================================================
# config.py — Central configuration for IntentClassifier
# =============================================================================

class Config:
    # ── Data ─────────────────────────────────────────────────────────────────
    DATA_PATH       = "data/sample_data.json"
    MODEL_SAVE_DIR  = "checkpoints"
    MODEL_NAME      = "intent_classifier.pt"
    VOCAB_PATH      = "checkpoints/vocab.json"

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    MAX_SEQ_LEN     = 32        # pad / truncate all sequences to this length
    MIN_FREQ        = 1         # minimum word frequency to keep in vocab

    # ── Model ─────────────────────────────────────────────────────────────────
    EMBEDDING_DIM   = 128       # trainable embedding size
    HIDDEN_SIZE     = 128       # BiLSTM hidden size (per direction)
    NUM_LAYERS      = 2         # number of BiLSTM layers
    DROPOUT         = 0.2       # reduced for small dataset
    PROJ_DIM        = 64        # projection head output dim (for contrastive)

    # ── Training ─────────────────────────────────────────────────────
    EPOCHS          = 100
    BATCH_SIZE      = 16        # 175 train samples → ~10 batches/epoch
    LEARNING_RATE   = 1e-3
    WEIGHT_DECAY    = 1e-5
    GRAD_CLIP       = 5.0

    # ── Loss weights ──────────────────────────────────────────────────────────
    ALPHA           = 0.3       # weight for supervised contrastive loss
    BETA            = 0.7       # weight for cross-entropy loss (dominant)
    TEMPERATURE     = 0.1       # contrastive temperature

    # ── Misc ──────────────────────────────────────────────────────────────────
    SEED            = 42
    DEVICE          = "cpu"     # force CPU; change to "cuda" if available
    LOG_EVERY       = 10
    PATIENCE        = 20        # generous patience for trading fine-tuning


# singleton-style access
cfg = Config()
