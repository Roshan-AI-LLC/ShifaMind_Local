"""
benchmark/models/plm_icd.py

PLM-ICD — Pre-trained Language Model for ICD coding.
Reference: Huang et al. (2022) "PLM-ICD: Automatic ICD Coding with
           Pretrained Language Models." ACL BioNLP 2022.

Architecture (faithful to the paper, Section 3):
  • BioClinicalBERT encoder → all token hidden states H ∈ R^{B×L×768}.
  • Per-label attention (identical mechanism to CAML but on BERT features):
      α_k = softmax(H u_k)      — label-k attention weights [B, L]
      v_k = H^T α_k             — label-k document vector   [B, 768]
      ŷ_k = w_k^T v_k + b_k    — label-k logit             [B]
    where u_k ∈ R^768 and w_k ∈ R^768 are learned per-label vectors.
  • Padding positions are masked to -inf before softmax so the model
    cannot attend to [PAD] tokens.
  • Binary cross-entropy loss — no focal loss, no label smoothing.

Why not CLS+Linear?
  The PLM-ICD paper explicitly proposes label-wise attention over token
  features, NOT [CLS] pooling.  CLS+Linear is a much weaker baseline
  that misrepresents PLM-ICD in the comparison table.

Differences from the full PLM-ICD paper (acknowledged adaptations):
  • The paper processes full MIMIC notes (thousands of tokens) with
    chunk-and-concatenate.  Here we truncate to 512 tokens (same as
    ShifaMind) — same constraint applies to all models in this benchmark.
  • The paper's backbone is RoBERTa-base; we use BioClinicalBERT to keep
    the encoder consistent with ShifaMind for a fair within-benchmark
    comparison.
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class PLMICD(nn.Module):
    """
    PLM-ICD: per-label attention over BERT token hidden states.

    Args:
        bert_model_name : HuggingFace model name
        num_labels      : number of output codes (50)
        hidden_size     : BERT hidden size (768)
        dropout         : dropout on BERT hidden states before attention
    """

    def __init__(
        self,
        bert_model_name: str   = "emilyalsentzer/Bio_ClinicalBERT",
        num_labels:      int   = 50,
        hidden_size:     int   = 768,
        dropout:         float = 0.1,
    ) -> None:
        super().__init__()
        self.bert    = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)

        # Per-label attention vectors U ∈ R^{K × H}
        # u_k is the query for label k's attention over token features.
        self.label_attn = nn.Linear(hidden_size, num_labels, bias=False)

        # Per-label classifier W ∈ R^{K × H}, b ∈ R^K
        # Reused as both the attention key projector and the scorer
        # (same "tied weight" design as in CAML and the PLM-ICD paper).
        self.output = nn.Linear(hidden_size, num_labels)

        nn.init.xavier_uniform_(self.label_attn.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> dict:
        """
        Args:
            input_ids      : [B, L]
            attention_mask : [B, L]  — 1 for real tokens, 0 for [PAD]
        Returns:
            dict with "logits" : [B, num_labels]
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H   = self.dropout(out.last_hidden_state)          # [B, L, H]

        # Per-label attention scores: [B, L, K]
        alpha = self.label_attn(H)                         # [B, L, K]

        # Mask padding positions — prevents the model attending to [PAD]
        if attention_mask is not None:
            pad_mask = (attention_mask == 0).unsqueeze(-1)  # [B, L, 1]
            alpha    = alpha.masked_fill(pad_mask, float("-inf"))

        alpha = torch.softmax(alpha, dim=1)                # [B, L, K]

        # Label-specific document vectors: v_k = Σ_t α_tk * h_t
        v = torch.bmm(alpha.transpose(1, 2), H)            # [B, K, H]
        v = self.dropout(v)

        # Per-label logits: ŷ_k = w_k · v_k + b_k
        # output.weight is [K, H] → element-wise product + sum across H
        logits = (v * self.output.weight.unsqueeze(0)).sum(-1) + self.output.bias  # [B, K]

        return {"logits": logits}
