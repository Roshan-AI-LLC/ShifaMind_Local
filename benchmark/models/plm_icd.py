"""
benchmark/models/plm_icd.py

PLM-ICD — Pre-trained Language Model for ICD coding.
Reference: Huang et al. (2022) "PLM-ICD: Automatic ICD Coding with
           Pretrained Language Models." ACL ClinicalNLP 2022.
           https://github.com/MiuLab/PLM-ICD

Architecture (laat mode — default in paper experiments):
  • BERT encoder WITHOUT pooling layer (add_pooling_layer=False).
  • LAAT-style label attention over all token hidden states:
      1. first_linear  H → H, no bias  (tanh activation)
      2. second_linear H → K, no bias  (per-label attention scores)
      3. Softmax over sequence dim, weighted sum → [B, K, H]
      4. third_linear  H → K  (per-label element-wise classifier)
  This is strictly faithful to Huang et al. 2022 / Vu et al. 2020.

Note: The full PLM-ICD paper chunks documents into 128-token windows
(up to 24 chunks = 3072 tokens) and max-pools logits across chunks
(laat-split mode).  Here we apply LAAT attention over the 512-token
BERT window — equivalent to the single-chunk laat mode.
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class PLMICD(nn.Module):
    """
    PLM-ICD: BERT with LAAT label attention for multi-label ICD classification.

    Args:
        bert_model_name : HuggingFace model name
        num_labels      : number of output codes (50)
        hidden_size     : BERT hidden size (768)
        dropout         : dropout on BERT token representations
    """

    def __init__(
        self,
        bert_model_name: str   = "emilyalsentzer/Bio_ClinicalBERT",
        num_labels:      int   = 50,
        hidden_size:     int   = 768,
        dropout:         float = 0.1,
    ) -> None:
        super().__init__()
        # No pooling layer — we need all token representations, not just [CLS]
        self.bert    = AutoModel.from_pretrained(bert_model_name, add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout)

        # LAAT attention layers (Huang et al. 2022 / Vu et al. 2020)
        # first_linear  : projects hidden states before computing attention scores
        # second_linear : maps to per-label attention logits (weight = label queries)
        # third_linear  : per-label element-wise classifier on weighted representations
        self.first_linear  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.second_linear = nn.Linear(hidden_size, num_labels,  bias=False)
        self.third_linear  = nn.Linear(hidden_size, num_labels)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.first_linear.weight)
        nn.init.xavier_uniform_(self.second_linear.weight)
        nn.init.xavier_uniform_(self.third_linear.weight)
        nn.init.zeros_(self.third_linear.bias)

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
            attention_mask : [B, L]
        Returns:
            dict with "logits" : [B, num_labels]
        """
        # BERT token representations — all positions, no CLS pooling
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H   = self.dropout(out.last_hidden_state)    # [B, L, H]

        # LAAT label attention
        weights     = torch.tanh(self.first_linear(H))        # [B, L, H]
        att_weights = self.second_linear(weights)              # [B, L, K]

        # Mask padding positions before softmax
        if attention_mask is not None:
            pad_mask    = (~attention_mask.bool()).unsqueeze(-1)          # [B, L, 1]
            att_weights = att_weights.masked_fill(pad_mask, float('-inf'))

        att_weights = torch.softmax(att_weights, dim=1)        # [B, L, K]
        att_weights = att_weights.transpose(1, 2)              # [B, K, L]

        # Weighted sum of token representations per label
        m = torch.bmm(att_weights, H)                          # [B, K, H]

        # Per-label element-wise classification (faithful to Vu et al. / Huang et al.)
        logits = (m * self.third_linear.weight.unsqueeze(0)).sum(-1) + self.third_linear.bias
        # [B, K]

        return {"logits": logits}
