"""
benchmark/models/vanilla_cbm.py

Vanilla CBM — Standard two-stage Concept Bottleneck Model.
Reference: Koh et al. (2020) "Concept Bottleneck Models." ICML 2020.

Architecture:
  Stage 1 — Concept predictor (trained with concept supervision):
    BioClinicalBERT → [CLS] → Linear(768, num_concepts) → sigmoid
  Stage 2 — Diagnosis predictor (trained on predicted concepts, BERT frozen):
    concept_logits → Linear(num_concepts, num_labels) → sigmoid

Training:
  1. Fine-tune BERT + concept_head for epochs_stage1 (using concept BCE loss).
  2. Freeze BERT + concept_head; train diagnosis_head for epochs_stage2
     using predicted concept scores as input (not ground-truth concepts).

This is the "pure" CBM: interpretable by design (each diagnosis prediction
is a linear combination of concept probabilities), with no cross-attention,
no GAT, no RAG.  Comparing it to ShifaMind Phase 1 reveals the specific
benefit of ShifaMind's cross-attention concept fusion.
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class VanillaCBM(nn.Module):
    """
    Two-stage Concept Bottleneck Model.

    Args:
        bert_model_name : HuggingFace BERT model name
        num_concepts    : concept bottleneck dimension (111 in ShifaMind)
        num_labels      : output ICD-10 codes (50)
        hidden_size     : BERT hidden size (768)
        dropout         : dropout on BERT [CLS]
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
        self.bert         = AutoModel.from_pretrained(bert_model_name)
        self.dropout      = nn.Dropout(dropout)
        self.concept_head = nn.Linear(hidden_size, num_concepts)
        self.diag_head    = nn.Linear(num_concepts, num_labels)

        nn.init.xavier_uniform_(self.concept_head.weight)
        nn.init.zeros_(self.concept_head.bias)
        nn.init.xavier_uniform_(self.diag_head.weight)
        nn.init.zeros_(self.diag_head.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> dict:
        """
        Full forward pass: BERT → concept logits → diagnosis logits.

        Returns:
            "logits"         : [B, num_labels]
            "concept_logits" : [B, num_concepts]
            "concept_scores" : [B, num_concepts]  (sigmoid of concept_logits)
        """
        out           = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls           = self.dropout(out.last_hidden_state[:, 0, :])   # [B, H]
        concept_logits = self.concept_head(cls)                         # [B, C]
        concept_scores = torch.sigmoid(concept_logits)
        logits         = self.diag_head(concept_scores)                 # [B, K]
        return {
            "logits"         : logits,
            "concept_logits" : concept_logits,
            "concept_scores" : concept_scores,
        }

    # ------------------------------------------------------------------
    def freeze_stage1(self) -> None:
        """Freeze BERT + concept_head — call before Stage 2 training."""
        for p in self.bert.parameters():
            p.requires_grad = False
        for p in self.concept_head.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze everything (e.g. for joint fine-tuning after Stage 2)."""
        for p in self.parameters():
            p.requires_grad = True
