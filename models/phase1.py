"""
models/phase1.py

Phase 1 architecture:
  ConceptBottleneckCrossAttention  — multiplicative concept gate per BERT layer
  ShifaMind2Phase1                 — BioClinicalBERT + concept bottleneck
                                     → Top-50 ICD-10 multilabel classification
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptBottleneckCrossAttention(nn.Module):
    """
    Multi-head cross-attention where text tokens attend to clinical concept
    embeddings, followed by a multiplicative sigmoid gate.

    Used at BERT layers 9 and 11 (configurable via fusion_layers).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        layer_idx: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = hidden_size // num_heads
        self.layer_idx   = layer_idx

        self.query    = nn.Linear(hidden_size, hidden_size)
        self.key      = nn.Linear(hidden_size, hidden_size)
        self.value    = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Multiplicative gate: combines pooled text + pooled context
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )

        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,        # [B, seq_len, H]
        concept_embeddings: torch.Tensor,   # [num_concepts, H]
        attention_mask=None,
    ):
        B, S, _ = hidden_states.shape
        C       = concept_embeddings.shape[0]
        H, Nh, Hd = self.hidden_size, self.num_heads, self.head_dim

        concepts = concept_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, C, H]

        Q = self.query(hidden_states).view(B, S, Nh, Hd).transpose(1, 2)
        K = self.key(concepts).view(B, C, Nh, Hd).transpose(1, 2)
        V = self.value(concepts).view(B, C, Nh, Hd).transpose(1, 2)

        scores       = torch.matmul(Q, K.transpose(-2, -1)) / (Hd ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)                    # [B, Nh, S, Hd]
        context = context.transpose(1, 2).contiguous().view(B, S, H)
        context = self.out_proj(context)

        # Multiplicative gate
        pooled_text    = hidden_states.mean(dim=1, keepdim=True).expand(-1, S, -1)
        pooled_context = context.mean(dim=1, keepdim=True).expand(-1, S, -1)
        gate           = self.gate_net(torch.cat([pooled_text, pooled_context], dim=-1))

        output = self.layer_norm(gate * context)

        return output, attn_weights.mean(dim=1), gate.mean()


class ShifaMind2Phase1(nn.Module):
    """
    Phase 1 model.

    Architecture:
      BioClinicalBERT (frozen / fine-tuned) →
      ConceptBottleneckCrossAttention at layers 9 & 11 →
      Concept head  (111 → sigmoid)
      Diagnosis head (50 → logits, BCEWithLogits externally)
    """

    def __init__(
        self,
        base_model,
        num_concepts: int,
        num_classes: int,
        fusion_layers: list = None,
    ) -> None:
        super().__init__()
        if fusion_layers is None:
            fusion_layers = [9, 11]

        self.base_model   = base_model
        self.hidden_size  = base_model.config.hidden_size
        self.num_concepts = num_concepts
        self.fusion_layers = fusion_layers

        # Learnable concept embedding matrix
        self.concept_embeddings = nn.Parameter(
            torch.randn(num_concepts, self.hidden_size) * 0.02
        )

        self.fusion_modules = nn.ModuleDict(
            {
                str(layer): ConceptBottleneckCrossAttention(
                    self.hidden_size, layer_idx=layer
                )
                for layer in fusion_layers
            }
        )

        self.concept_head   = nn.Linear(self.hidden_size, num_concepts)
        self.diagnosis_head = nn.Linear(self.hidden_size, num_classes)
        self.dropout        = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, return_attention: bool = False):
        bert_out      = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states  = bert_out.hidden_states   # tuple of [B, S, H] per layer
        current_hidden = bert_out.last_hidden_state
        attention_maps: dict = {}
        gate_values:    list = []

        for layer_idx in self.fusion_layers:
            key = str(layer_idx)
            if key in self.fusion_modules:
                fused, attn, gate = self.fusion_modules[key](
                    hidden_states[layer_idx],
                    self.concept_embeddings,
                    attention_mask,
                )
                current_hidden = fused
                gate_values.append(gate.item())
                if return_attention:
                    attention_maps[f"layer_{layer_idx}"] = attn

        cls_hidden       = self.dropout(current_hidden[:, 0, :])
        concept_scores   = torch.sigmoid(self.concept_head(cls_hidden))
        diagnosis_logits = self.diagnosis_head(cls_hidden)

        result = {
            "logits"         : diagnosis_logits,
            "concept_scores" : concept_scores,
            "hidden_states"  : current_hidden,
            "cls_hidden"     : cls_hidden,
            "avg_gate"       : float(np.mean(gate_values)) if gate_values else 0.0,
        }
        if return_attention:
            result["attention_maps"] = attention_maps

        return result
