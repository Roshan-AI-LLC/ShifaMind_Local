"""
training/loss.py

Multi-objective loss used across all three phases:

    L_total = λ_dx · FocalLoss(logits, dx_labels)
            + λ_align · CoOccAlignLoss(concept_scores, dx_probs, co_occ)
            + λ_concept · BCE(concept_logits, concept_labels)

FocalLoss replaces plain BCE for the diagnosis head to address the severe
class imbalance in multi-label ICD-10 coding: most label matrix entries are 0,
so plain BCE overwhelms the gradient with easy negatives. Focal loss
down-weights easy examples via (1 - p_t)^γ and up-weights the positive class
via α, directly improving recall.

Alignment loss (Phase 1):
    For each sample, the *expected* concept activation pattern given the model's
    predicted label probabilities is:
        expected_concept[b, c] = Σ_k  dx_probs[b, k] · co_occ[c, k]
    We minimise MSE(concept_scores, expected_concept).  This is mathematically
    sound: when the model predicts label k strongly, concept c should activate
    proportional to how often concept c appears with label k in the training data.
    Previously the loss was |P(dx_i) - P(concept_j)| over all 50×111 pairs —
    a meaningless cross-product that adds pure noise to the gradient.

Phase 3 uses a higher λ_dx (2.0) to focus on diagnosis quality.
All other phases use the defaults from config.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary focal loss for multi-label classification.

        FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Works element-wise on raw logits (same interface as BCEWithLogitsLoss).

    Args:
        gamma : focusing exponent — 0 reduces to weighted BCE, 2 is typical.
        alpha : weight on the positive class (0.75 means positives get 3×
                more gradient weight than negatives).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Element-wise BCE (no reduction)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t: model's estimated probability for the *true* class
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1.0 - probs) * (1.0 - targets)

        # α_t: per-element positive / negative weight
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        # Focal modulation: down-weights confident correct predictions
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma

        return (focal_weight * bce).mean()


class MultiObjectiveLoss(nn.Module):
    """
    Args:
        lambda_dx      : weight on diagnosis focal loss
        lambda_align   : weight on co-occurrence alignment loss
        lambda_concept : weight on concept BCE loss
        focal_gamma    : γ for the diagnosis FocalLoss
        focal_alpha    : α for the diagnosis FocalLoss
        co_occ_matrix  : optional float tensor [num_concepts, num_classes] of
                         P(concept | label) values from training data.
                         When provided, the alignment loss becomes a semantically
                         meaningful MSE between observed concept activations and
                         the expected concept pattern given predicted label probs.
                         When None (Phase 2/3), lambda_align should be 0.
    """

    def __init__(
        self,
        lambda_dx: float,
        lambda_align: float,
        lambda_concept: float,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        co_occ_matrix: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.lambda_dx      = lambda_dx
        self.lambda_align   = lambda_align
        self.lambda_concept = lambda_concept
        self.focal          = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.bce            = nn.BCEWithLogitsLoss()

        # Register co-occurrence matrix as a non-trainable buffer so it moves
        # automatically with .to(device) and is excluded from optimizer updates.
        # Shape: [num_concepts, num_classes]
        if co_occ_matrix is not None:
            self.register_buffer("co_occ", co_occ_matrix.float())
        else:
            self.co_occ = None

    def forward(self, outputs: dict, dx_labels: torch.Tensor, concept_labels: torch.Tensor):
        """
        Returns:
            total_loss  : scalar tensor (backward-able)
            components  : dict with float values for logging
        """
        # ── Diagnosis focal loss ─────────────────────────────────────────────────
        loss_dx = self.focal(outputs["logits"], dx_labels)

        # ── Alignment loss ───────────────────────────────────────────────────────
        dx_probs   = torch.sigmoid(outputs["logits"])   # [B, num_classes]
        concept_sc = outputs["concept_scores"]           # [B, num_concepts]

        if self.co_occ is not None and self.lambda_align > 0.0:
            # expected_concept[b, c] = Σ_k  dx_probs[b, k] * co_occ[c, k]
            #                        = dx_probs @ co_occ.T  (matrix multiply)
            # Divided by num_classes to keep magnitudes ~comparable to concept_sc.
            co_occ = self.co_occ.to(dx_probs.device)
            expected_concept = torch.mm(dx_probs, co_occ.T) / dx_probs.shape[1]
            loss_align = F.mse_loss(concept_sc, expected_concept)
        else:
            # Fallback: zero (lambda_align=0 in Phase 2/3 anyway)
            loss_align = dx_probs.new_tensor(0.0)

        # ── Concept loss ─────────────────────────────────────────────────────────
        if "concept_logits" in outputs:
            loss_concept = self.bce(outputs["concept_logits"], concept_labels)
        else:
            concept_logits = torch.logit(concept_sc.clamp(1e-7, 1.0 - 1e-7))
            loss_concept   = self.bce(concept_logits, concept_labels)

        total = (
            self.lambda_dx        * loss_dx
            + self.lambda_align   * loss_align
            + self.lambda_concept * loss_concept
        )

        components = {
            "total"  : total.item(),
            "dx"     : loss_dx.item(),
            "align"  : loss_align.item(),
            "concept": loss_concept.item(),
        }

        return total, components
