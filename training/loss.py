"""
training/loss.py

Multi-objective loss used across all three phases:

    L_total = λ_dx · BCE(logits, dx_labels)
            + λ_align · |σ(logits) - concept_scores|_mean
            + λ_concept · BCE(concept_logits, concept_labels)

Phase 3 uses a higher λ_dx (2.0) to focus on diagnosis quality.
All other phases use the defaults from config.
"""
import torch
import torch.nn as nn


class MultiObjectiveLoss(nn.Module):
    """
    Args:
        lambda_dx      : weight on diagnosis BCE loss
        lambda_align   : weight on dx–concept alignment loss
        lambda_concept : weight on concept BCE loss
    """

    def __init__(
        self,
        lambda_dx: float,
        lambda_align: float,
        lambda_concept: float,
    ) -> None:
        super().__init__()
        self.lambda_dx      = lambda_dx
        self.lambda_align   = lambda_align
        self.lambda_concept = lambda_concept
        self.bce            = nn.BCEWithLogitsLoss()

    def forward(self, outputs: dict, dx_labels: torch.Tensor, concept_labels: torch.Tensor):
        """
        Returns:
            total_loss  : scalar tensor (backward-able)
            components  : dict with float values for logging
        """
        # Diagnosis loss
        loss_dx = self.bce(outputs["logits"], dx_labels)

        # Alignment loss: |P(dx) - P(concept)|  averaged over all pairs
        dx_probs     = torch.sigmoid(outputs["logits"])
        concept_sc   = outputs["concept_scores"]
        loss_align   = torch.abs(dx_probs.unsqueeze(-1) - concept_sc.unsqueeze(1)).mean()

        # Concept loss — use raw logits when available, else derive them
        if "concept_logits" in outputs:
            loss_concept = self.bce(outputs["concept_logits"], concept_labels)
        else:
            concept_logits = torch.logit(concept_sc.clamp(1e-7, 1.0 - 1e-7))
            loss_concept   = self.bce(concept_logits, concept_labels)

        total = (
            self.lambda_dx      * loss_dx
            + self.lambda_align * loss_align
            + self.lambda_concept * loss_concept
        )

        components = {
            "total"  : total.item(),
            "dx"     : loss_dx.item(),
            "align"  : loss_align.item(),
            "concept": loss_concept.item(),
        }

        return total, components
