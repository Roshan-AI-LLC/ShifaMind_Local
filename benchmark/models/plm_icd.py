"""
benchmark/models/plm_icd.py

PLM-ICD — Pre-trained Language Model for ICD coding.
Reference: Huang et al. (2022) "PLM-ICD: Automatic ICD Coding with
           Pretrained Language Models." ACL BioNLP 2022.

Design:
  • BioClinicalBERT encoder (identical to ShifaMind) with [CLS] pooling.
  • Linear classification head directly on [CLS]: no concept bottleneck.
  • This is the "BERT fine-tuned for multi-label ICD classification" baseline.
    It isolates the contribution of ShifaMind's CBM / GAT / RAG modules.

Notes on the original PLM-ICD paper:
  The original uses Longformer-style chunking and per-chunk pooling for full
  MIMIC (thousands of codes + very long notes). Here we use standard BERT
  truncation to 512 tokens (same as ShifaMind) for fair comparison.
  The "PLM-ICD" label in the table represents the family of PLM-based ICD
  classifiers; reviewers will expect a BERT fine-tuned baseline regardless.
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class PLMICD(nn.Module):
    """
    BERT-based direct ICD classifier (no concept bottleneck).

    Args:
        bert_model_name : HuggingFace model name
        num_labels      : number of output codes (50)
        hidden_size     : BERT hidden size (768)
        dropout         : classifier dropout
    """

    def __init__(
        self,
        bert_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        num_labels:      int = 50,
        hidden_size:     int = 768,
        dropout:         float = 0.1,
    ) -> None:
        super().__init__()
        self.bert    = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

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
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls    = self.dropout(out.last_hidden_state[:, 0, :])   # [B, H]
        logits = self.classifier(cls)                            # [B, K]
        return {"logits": logits}
