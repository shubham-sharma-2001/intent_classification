# =============================================================================
# predict.py — Inference script (single text or batch)
# =============================================================================
from __future__ import annotations

import os
import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

from config import cfg
from tokenizer import WordTokenizer
from train import load_checkpoint


class IntentPredictor:
    """
    High-level inference interface.

    Usage
    -----
    >>> predictor = IntentPredictor()
    >>> result = predictor.predict("book a flight to London")
    >>> print(result)
    {'intent': 'book_flight', 'confidence': 0.97, 'top_k': [...]}
    """

    def __init__(self,
                 checkpoint_path: str | None = None,
                 vocab_path:      str | None = None,
                 device:          str        = cfg.DEVICE):
        self.device = torch.device(device)

        # ── Load tokenizer ─────────────────────────────────────────────
        vocab_path = vocab_path or cfg.VOCAB_PATH
        self.tokenizer = WordTokenizer.load(vocab_path)

        # ── Load model + label maps ────────────────────────────────────
        self.model, ckpt = load_checkpoint(checkpoint_path, device)
        self.label2idx   = ckpt["label2idx"]
        self.idx2label   = {int(k): v for k, v in ckpt["idx2label"].items()}

    # ── Public API ─────────────────────────────────────────────────────────

    def predict(self, text: str, top_k: int = 3) -> Dict:
        """
        Predict intent for a single text.

        Returns
        -------
        dict with keys:
            intent      — top predicted intent string
            confidence  — probability (softmax) of top intent
            top_k       — list of (intent, confidence) for top k classes
        """
        results = self.predict_batch([text], top_k=top_k)
        return results[0]

    def predict_batch(self, texts: List[str],
                      top_k: int = 3) -> List[Dict]:
        """
        Predict intents for a list of texts.
        """
        self.model.eval()

        tokens_list = [self.tokenizer.encode(t) for t in texts]
        tokens = torch.tensor(tokens_list, dtype=torch.long,
                              device=self.device)

        # Compute real lengths
        pad_id  = self.tokenizer.pad_idx
        lengths = torch.tensor(
            [max(1, sum(1 for i in ids if i != pad_id))
             for ids in tokens_list],
            dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            logits, _ = self.model(tokens, lengths)
            probs     = F.softmax(logits, dim=-1)

        results = []
        for i in range(len(texts)):
            prob_i   = probs[i]
            top_vals, top_idxs = prob_i.topk(min(top_k, len(self.idx2label)))

            results.append({
                "text":       texts[i],
                "intent":     self.idx2label[top_idxs[0].item()],
                "confidence": round(top_vals[0].item(), 4),
                "top_k": [
                    {
                        "intent":     self.idx2label[idx.item()],
                        "confidence": round(val.item(), 4),
                    }
                    for val, idx in zip(top_vals, top_idxs)
                ],
            })

        return results


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Intent Classifier — Inference")
    parser.add_argument("--text",       type=str, help="Single text to classify")
    parser.add_argument("--file",       type=str, help="Path to .txt file (one sentence per line)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--vocab",      type=str, default=None)
    parser.add_argument("--top_k",      type=int, default=3)
    args = parser.parse_args()

    predictor = IntentPredictor(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
    )

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        results = predictor.predict_batch(texts, top_k=args.top_k)
        for r in results:
            print(f"[{r['intent']:20s}] ({r['confidence']:.4f})  →  {r['text']}")

    elif args.text:
        r = predictor.predict(args.text, top_k=args.top_k)
        print(f"\nText      : {r['text']}")
        print(f"Intent    : {r['intent']}")
        print(f"Confidence: {r['confidence']:.4f}")
        print(f"Top-{args.top_k}    :")
        for entry in r["top_k"]:
            print(f"  {entry['intent']:20s}  {entry['confidence']:.4f}")

    else:
        # Interactive mode
        print("= Intent Classifier — Interactive Mode =")
        print("Type 'quit' to exit.\n")
        while True:
            text = input(">> ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            r = predictor.predict(text, top_k=args.top_k)
            print(f"   Intent: {r['intent']}  ({r['confidence']:.4f})")
            for entry in r["top_k"][1:]:
                print(f"       ↳ {entry['intent']}  ({entry['confidence']:.4f})")
            print()


if __name__ == "__main__":
    main()
