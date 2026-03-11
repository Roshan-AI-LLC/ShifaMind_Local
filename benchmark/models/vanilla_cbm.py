"""
benchmark/models/vanilla_cbm.py

Vanilla CBM — Standard Concept Bottleneck Model (joint training mode).
Reference: Koh et al. (2020) "Concept Bottleneck Models." ICML 2020.
           https://github.com/yewsiang/ConceptBottleneck

─────────────────────────────────────────────────────────────────────────────
WHAT "ADDITIVE BOTTLENECK" MEANS (and why it differs from ShifaMind)
─────────────────────────────────────────────────────────────────────────────
Vanilla CBM:  ŷ_k  =  Σ_i w_ki · σ(c_i)  +  b_k

Each diagnosis score is a WEIGHTED SUM (linear combination) of concept
probabilities.  The diagnosis head is a single nn.Linear(C, K) applied to
the concept probability vector.  This is "additive" because the contribution
of each concept to each diagnosis is simply w_ki × σ(c_i): concepts add up
independently with no interaction.

ShifaMind Phase 1: uses BioClinicalBERT cross-attention over a learned
concept embedding matrix.  Each concept embedding E_i ∈ R^768 attends over
the full BERT hidden state sequence, producing a 768-dim concept context
vector that is then fused back into the BERT representation before the
diagnosis head.  This is "multiplicative" in spirit: concept relevance is
computed via softmax attention (exp(Q·K/√d)), which is a non-linear,
multiplicative interaction between the BERT hidden states and the concept
embeddings — fundamentally different from the additive linear combination.

The gap between Vanilla CBM and ShifaMind Phase 1 in the table isolates
the benefit of this cross-attention concept fusion.

─────────────────────────────────────────────────────────────────────────────
TRAINING MODE — Joint (recommended by Koh et al.)
─────────────────────────────────────────────────────────────────────────────
The paper reports three modes: independent, sequential, joint.
Joint achieves the best label accuracy while maintaining concept accuracy,
and is the recommended mode for comparison.

Joint loss:
    L = α · BCE(concept_logits, concept_labels)
      + β · BCE(diag_logits, diag_labels)

where diag_logits are computed from the PREDICTED concept probabilities
(not ground-truth).  BERT + concept_head + diag_head are all trained
end-to-end.

α = 0.5, β = 1.0  (following the paper's hyperparams for medical datasets).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# Loss weights (Koh et al. 2020, Table 3 — medical setting)
ALPHA_CONCEPT = 0.5   # concept supervision weight
BETA_DIAG     = 1.0   # diagnosis supervision weight


class VanillaCBM(nn.Module):
    """
    Vanilla Concept Bottleneck Model — joint training mode.

    Information path:
        text → BERT → [CLS] → concept_head → sigmoid(concept_probs)
                                            → diag_head → diagnosis_logits

    The ONLY information reaching diag_head is the 111-dim concept
    probability vector — no skip connection, no BERT hidden states,
    no embedding matrix.  Fully interpretable: each diagnosis score is
    a linear combination of named concept probabilities.

    Args:
        bert_model_name : HuggingFace model name
        num_concepts    : concept bottleneck size (111 in ShifaMind)
        num_labels      : output codes (50)
        hidden_size     : BERT hidden dimension (768)
        dropout         : dropout on [CLS] before concept_head
    """

    def __init__(
        self,
        bert_model_name: str   = "emilyalsentzer/Bio_ClinicalBERT",
        num_concepts:    int   = 111,
        num_labels:      int   = 50,
        hidden_size:     int   = 768,
        dropout:         float = 0.1,
    ) -> None:
        super().__init__()
        self.num_concepts = num_concepts
        self.num_labels   = num_labels

        self.bert         = AutoModel.from_pretrained(bert_model_name)
        self.dropout      = nn.Dropout(dropout)

        # Concept predictor: [CLS] → concept logits
        self.concept_head = nn.Linear(hidden_size, num_concepts)

        # Diagnosis predictor: concept_probs → diag logits (additive bottleneck)
        # NO hidden layer, NO activation — strictly linear as in the paper.
        self.diag_head = nn.Linear(num_concepts, num_labels, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.concept_head.weight)
        nn.init.zeros_(self.concept_head.bias)
        # Diag head: zero init — starts at neutral prediction, learns from concepts
        nn.init.zeros_(self.diag_head.weight)
        nn.init.zeros_(self.diag_head.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> dict:
        """
        Joint forward pass.

        Returns:
            logits          [B, K]   — diagnosis logits (from concept probs only)
            concept_logits  [B, C]   — concept logits (before sigmoid)
            concept_scores  [B, C]   — concept probabilities (after sigmoid)
        """
        bert_out       = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls            = self.dropout(bert_out.last_hidden_state[:, 0, :])  # [B, H]

        # Concept bottleneck
        concept_logits = self.concept_head(cls)                              # [B, C]
        concept_probs  = torch.sigmoid(concept_logits)                       # [B, C]

        # Additive linear combination: ŷ_k = Σ_i w_ki * σ(c_i) + b_k
        # No interaction between concepts — purely additive
        logits         = self.diag_head(concept_probs)                       # [B, K]

        return {
            "logits"        : logits,
            "concept_logits": concept_logits,
            "concept_scores": concept_probs,
        }

    # ------------------------------------------------------------------
    def joint_loss(
        self,
        outputs:        dict,
        diag_labels:    torch.Tensor,
        concept_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Joint CBM loss (Koh et al. eq. 1):
            L = α · BCE(concept_logits, concept_labels)
              + β · BCE(diag_logits, diag_labels)

        Call this from the training loop instead of a plain BCE criterion.
        """
        loss_concept = F.binary_cross_entropy_with_logits(
            outputs["concept_logits"], concept_labels
        )
        loss_diag = F.binary_cross_entropy_with_logits(
            outputs["logits"], diag_labels
        )
        return ALPHA_CONCEPT * loss_concept + BETA_DIAG * loss_diag

    # ------------------------------------------------------------------
    def freeze_concept_head(self) -> None:
        """Freeze BERT + concept_head (for intervention experiments)."""
        for p in self.bert.parameters():
            p.requires_grad = False
        for p in self.concept_head.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
