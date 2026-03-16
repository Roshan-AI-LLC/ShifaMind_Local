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
      LAAT attention pooling  (per-label evidence selection over all 512 tokens)
      Concept head            (CLS → 111 concept scores via sigmoid)
      concept_to_label gate   (concept_scores [B,C] → additive bias [B,K])
      Diagnosis logits        (LAAT label reps + concept bias)

    The concept_to_label projection is the true bottleneck: concept scores
    causally influence every diagnosis logit via a learned affinity matrix,
    enabling per-prediction concept attribution at inference time.
    """

    def __init__(
        self,
        base_model,
        num_concepts: int,
        num_classes: int,
        fusion_layers: list = None,
        co_occ_matrix: "np.ndarray | None" = None,
    ) -> None:
        """
        Args:
            base_model    : pretrained HuggingFace BERT-like model
            num_concepts  : number of clinical concept dimensions (111)
            num_classes   : number of ICD-10 diagnosis labels (50)
            fusion_layers : encoder layer indices where concept cross-attention fires.
                            Default [17, 20] for BioClinical ModernBERT-base (22 layers).
                            Equivalent positions to [9, 11] in 12-layer BioClinicalBERT.
            co_occ_matrix : optional np.ndarray [num_concepts, num_classes] of
                            P(concept | label) co-occurrence values from training
                            data.  When supplied, concept_to_label is initialised
                            from this matrix instead of random weights.
        """
        super().__init__()
        if fusion_layers is None:
            fusion_layers = [17, 20]

        self.base_model    = base_model
        self.hidden_size   = base_model.config.hidden_size
        self.num_concepts  = num_concepts
        self.num_classes   = num_classes
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

        # ── Concept head (uses CLS — concept detection is global) ───────────────
        self.concept_head = nn.Linear(self.hidden_size, num_concepts)

        # ── LAAT — Label-Attention over all tokens ──────────────────────────────
        # laat_first  : token-level projection before attention scoring
        # laat_second : produces per-label attention score at each token position
        # laat_output : maps each per-label 768-d rep to a scalar logit
        self.laat_first  = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.laat_second = nn.Linear(self.hidden_size, num_classes,      bias=False)
        self.laat_output = nn.Linear(self.hidden_size, 1)

        # ── Concept → Label causal gate ─────────────────────────────────────────
        # Maps concept_scores [B, num_concepts] → additive bias [B, num_classes].
        # Weight shape (PyTorch Linear convention): [num_classes, num_concepts]
        # Initialised from co_occ.T if provided, so the model starts with
        # semantically meaningful concept→label affinity rather than random noise.
        self.concept_to_label = nn.Linear(num_concepts, num_classes, bias=False)
        if co_occ_matrix is not None:
            # co_occ_matrix: [num_concepts, num_classes]  →  need [num_classes, num_concepts]
            with torch.no_grad():
                self.concept_to_label.weight.data = torch.tensor(
                    co_occ_matrix.T, dtype=torch.float32
                )

        self.dropout = nn.Dropout(0.1)

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _laat(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Label-Attention mechanism over token sequence.

        Args:
            hidden         : [B, S, H]  concept-fused token representations
            attention_mask : [B, S]     1 for real tokens, 0 for padding

        Returns:
            label_reps : [B, num_classes, H]  per-label aggregated representations
        """
        H = self.dropout(hidden)                            # [B, S, H]
        U = torch.tanh(self.laat_first(H))                 # [B, S, H]
        att_scores = self.laat_second(U)                   # [B, S, num_classes]

        # Mask padding positions so they don't contribute to the softmax
        pad_mask = (attention_mask == 0).unsqueeze(-1)     # [B, S, 1]
        att_scores = att_scores.masked_fill(pad_mask, float("-inf"))

        att_weights = torch.softmax(att_scores, dim=1)     # [B, S, num_classes]
        att_weights = att_weights.transpose(1, 2)          # [B, num_classes, S]
        label_reps  = torch.bmm(att_weights, H)            # [B, num_classes, H]
        return label_reps

    # ── Forward ─────────────────────────────────────────────────────────────────

    def forward(self, input_ids, attention_mask, return_attention: bool = False):
        bert_out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states  = bert_out.hidden_states    # tuple of [B, S, H] per layer
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

        # Concept head — CLS representation (global concept detection)
        cls_hidden     = self.dropout(current_hidden[:, 0, :])   # [B, H]
        concept_logits = self.concept_head(cls_hidden)            # [B, num_concepts]
        concept_scores = torch.sigmoid(concept_logits)            # [B, num_concepts]

        # LAAT — per-label evidence pooling over all token positions
        label_reps       = self._laat(current_hidden, attention_mask)  # [B, K, H]
        diagnosis_logits = self.laat_output(label_reps).squeeze(-1)    # [B, K]

        # Concept → label causal gate: concept activations additively shift
        # each label's logit in proportion to learned concept-label affinity
        concept_bias     = self.concept_to_label(concept_scores)        # [B, K]
        diagnosis_logits = diagnosis_logits + concept_bias

        result = {
            "logits"         : diagnosis_logits,
            "concept_logits" : concept_logits,
            "concept_scores" : concept_scores,
            "hidden_states"  : current_hidden,
            "cls_hidden"     : cls_hidden,
            "label_reps"     : label_reps,
            "avg_gate"       : float(np.mean(gate_values)) if gate_values else 0.0,
        }
        if return_attention:
            result["attention_maps"] = attention_maps

        return result
