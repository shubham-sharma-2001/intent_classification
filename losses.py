# =============================================================================
# losses.py — Supervised Contrastive Loss (SupCon)
# =============================================================================
#
# Reference:
#   Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
#   https://arxiv.org/abs/2004.11362
#
# Key design details
# ------------------
# • Embeddings are expected to be L2-normalised before being passed in
#   (the ProjectionHead already does this).
# • We use the "one positive per anchor" formulation: for each anchor, all
#   samples sharing the same label are treated as positives.
# • Temperature τ is a hyper-parameter (default from cfg.TEMPERATURE).
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Parameters
    ----------
    temperature : float
        Scaling temperature τ. Smaller → sharper distribution.

    Usage
    -----
    >>> loss_fn = SupervisedContrastiveLoss(temperature=0.07)
    >>> loss = loss_fn(embeddings, labels)
    """

    def __init__(self, temperature: float = cfg.TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : (B, D)  — L2-normalised projection vectors
        labels     : (B,)    — integer class labels

        Returns
        -------
        loss : scalar tensor
        """
        batch_size = embeddings.size(0)

        # Safety: re-normalise in case caller forgot
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        # ── Similarity matrix ─────────────────────────────────────────────
        # (B, B) cosine similarities, scaled by temperature
        # Clamp to [-1/τ, 1/τ] for numerical safety before exp
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature
        sim = sim.clamp(-50.0, 50.0)

        # ── Mask out self-similarities on the diagonal ────────────────────
        self_mask   = torch.eye(batch_size, dtype=torch.bool,
                                device=embeddings.device)

        # ── Positive mask: same label, different sample ───────────────────
        labels_row = labels.unsqueeze(1)           # (B, 1)
        labels_col = labels.unsqueeze(0)           # (1, B)
        pos_mask   = (labels_row == labels_col) & ~self_mask   # (B, B)

        # ── Skip anchors that have no positive pair in the batch ──────────
        num_positives = pos_mask.sum(dim=1).float()   # (B,)
        valid         = num_positives > 0
        if not valid.any():
            # Edge-case: single sample per class in this batch
            return embeddings.sum() * 0.0           # zero with grad

        # ── Numerically stable log-sum-exp over non-self elements ─────────
        sim_no_self = sim.masked_fill(self_mask, -1e9)
        # log(sum_j exp(sim_ij))  via logsumexp
        log_denom   = torch.logsumexp(sim_no_self, dim=1)   # (B,)

        # ── Sum of log-prob for each positive pair ────────────────────────
        # log p(i,j) = sim_ij - log_denom_i
        log_probs         = sim - log_denom.unsqueeze(1)    # (B, B)
        pos_log_prob_sum  = (log_probs * pos_mask).sum(dim=1)   # (B,)

        loss_per_anchor = -pos_log_prob_sum / num_positives.clamp(min=1)

        # Average only over valid anchors
        loss = loss_per_anchor[valid].mean()
        return loss


class CombinedLoss(nn.Module):
    """
    Weighted combination of:
      - Supervised Contrastive Loss  (alpha)
      - Cross-Entropy Loss           (beta)

    total = alpha * L_contrastive + beta * L_ce
    """

    def __init__(self, num_classes: int,
                 alpha: float = cfg.ALPHA,
                 beta:  float = cfg.BETA,
                 temperature: float = cfg.TEMPERATURE):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

        self.contrastive_loss = SupervisedContrastiveLoss(temperature)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self,
                logits:     torch.Tensor,    # (B, C)
                embeddings: torch.Tensor,    # (B, D) — L2-normalised
                labels:     torch.Tensor     # (B,)
                ) -> tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        total_loss : scalar
        components : dict with keys 'ce', 'contrastive', 'total'
        """
        l_ce   = self.ce_loss(logits, labels)
        l_sup  = self.contrastive_loss(embeddings, labels)
        total  = self.beta * l_ce + self.alpha * l_sup

        return total, {
            "ce":          l_ce.item(),
            "contrastive": l_sup.item(),
            "total":       total.item(),
        }
