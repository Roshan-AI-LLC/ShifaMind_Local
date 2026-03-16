"""
models/phase2.py

Phase 2 architecture:
  GATEncoder        — multi-layer Graph Attention Network on the UMLS graph
  ShifaMindPhase2GAT — BioClinicalBERT + GAT + cross-attention concept bottleneck

This single class is used by:
  • Phase 2 training
  • Phase 2 threshold tuning
  • Phase 3 training (wrapped by ShifaMindPhase3RAG)
  • Phase 3 threshold tuning
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    """
    Multi-layer Graph Attention Network.

    Layer shapes:
      Input  : [N, in_channels]
      Hidden : [N, hidden_channels]   (attention-aggregated)
      Output : [N, hidden_channels]   (single head, averaged)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.convs      = nn.ModuleList()
        self.dropout    = nn.Dropout(dropout)

        # Layer 0: in_channels → hidden_channels  (multi-head, concat)
        self.convs.append(
            GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout, concat=True)
        )
        # Intermediate layers (if any)
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout, concat=True)
            )
        # Last layer: hidden_channels → hidden_channels  (single head, mean)
        if num_layers > 1:
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout, concat=False)
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = self.dropout(x)
        return x  # [N, hidden_channels]


class ShifaMindPhase2GAT(nn.Module):
    """
    Phase 2 model: BioClinicalBERT + GAT knowledge graph + LAAT concept bottleneck.

    Architecture:
      BERT text encoding →
      GAT-enriched concept embeddings (UMLS knowledge graph) →
      Cross-attention: text tokens attend to enhanced concepts →
      graph_scale residual: enriched_seq = layer_norm(hidden + sigmoid(s) * context) →
      LAAT attention: per-label evidence over all 512 tokens →
      concept_to_label causal gate: concept_scores → additive label bias →
      Diagnosis logits = LAAT logits + concept bias

    Key improvements over CLS-pooling Phase 2:
      • Each of the 50 labels independently selects evidence from all token positions
      • GAT knowledge enriches the cross-attention context that LAAT pools over
      • Concept → label affinity matrix (warm-started from training co-occurrence)
        provides causal interpretability: each prediction is attributable to concepts
      • graph_scale (sigmoid(-2) ≈ 0.12 at init) means GAT contributes from epoch 1
        rather than being effectively off for early training

    Forward inputs:
        input_ids               [B, seq_len]
        attention_mask          [B, seq_len]
        concept_embeddings_bert [num_concepts, 768]  — learned concept embeddings
    """

    def __init__(
        self,
        bert_model,
        gat_encoder: GATEncoder,
        graph_data,                               # torch_geometric.data.Data
        num_concepts: int,
        num_diagnoses: int,
        graph_hidden_dim: int = 256,
        concepts_list: Optional[List[str]] = None,
        co_occ_matrix: "np.ndarray | None" = None,
    ) -> None:
        """
        Args:
            bert_model       : pretrained HuggingFace BERT-like model
            gat_encoder      : GATEncoder instance
            graph_data       : torch_geometric.data.Data with UMLS graph
            num_concepts     : number of clinical concept dimensions (111)
            num_diagnoses    : number of ICD-10 diagnosis labels (50)
            graph_hidden_dim : GAT hidden dimension (projected to 768)
            concepts_list    : ordered list of concept strings (must match concept_embeddings)
            co_occ_matrix    : optional np.ndarray [num_concepts, num_diagnoses] of
                               P(concept | label) values.  When supplied, concept_to_label
                               is warm-started from this matrix.
        """
        super().__init__()
        import config
        self.bert          = bert_model
        self.gat           = gat_encoder
        self.hidden_size   = 768
        self.graph_hidden  = graph_hidden_dim
        self.num_concepts  = num_concepts
        self.num_diagnoses = num_diagnoses
        self._concepts     = concepts_list if concepts_list is not None else config.GLOBAL_CONCEPTS

        # Graph topology stored as persistent buffers (auto-moved with .to(device))
        self.register_buffer("graph_x",         graph_data.x)
        self.register_buffer("graph_edge_index", graph_data.edge_index)
        self.graph_node_to_idx = graph_data.node_to_idx
        self.graph_idx_to_node = graph_data.idx_to_node

        # ── GAT pipeline ────────────────────────────────────────────────────────
        # Project GAT output (graph_hidden) → BERT dimension (768)
        self.graph_proj = nn.Linear(self.graph_hidden, self.hidden_size)

        # Fuse BERT concept embeddings + GAT concept embeddings
        self.concept_fusion = nn.Sequential(
            nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Cross-attention: text tokens attend to GAT-enriched concept embeddings
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        # LayerNorm for the GAT-enriched token sequence residual
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        # ── Concept head (CLS — global concept detection) ───────────────────────
        self.concept_head = nn.Linear(self.hidden_size, num_concepts)

        # ── LAAT — Label-Attention over all tokens ──────────────────────────────
        # Same three-parameter design as Phase 1, but now applied over the
        # GAT-concept-enriched token sequence for richer label-specific evidence.
        self.laat_first  = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.laat_second = nn.Linear(self.hidden_size, num_diagnoses,    bias=False)
        self.laat_output = nn.Linear(self.hidden_size, 1)

        # ── Concept → Label causal gate ─────────────────────────────────────────
        # Maps concept_scores [B, num_concepts] → additive bias [B, num_diagnoses].
        # Warm-started from co_occ.T if provided.
        self.concept_to_label = nn.Linear(num_concepts, num_diagnoses, bias=False)
        if co_occ_matrix is not None:
            with torch.no_grad():
                self.concept_to_label.weight.data = torch.tensor(
                    co_occ_matrix.T, dtype=torch.float32
                )

        # ── Learnable GAT contribution scale ────────────────────────────────────
        # sigmoid(GRAPH_SCALE_INIT) ≈ 0.12 at init — GAT contributes from epoch 1.
        # Grows freely during training if GAT adds useful signal.
        self.graph_scale = nn.Parameter(torch.tensor(config.GRAPH_SCALE_INIT))

        self.dropout = nn.Dropout(0.1)

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def get_gat_concept_embeddings(self) -> torch.Tensor:
        """
        Run GAT on the full graph, extract the embedding for each concept node,
        project to 768-dim.

        Returns: [num_concepts, 768]
        """
        graph_embs = self.gat(self.graph_x, self.graph_edge_index)   # [N, graph_hidden]
        embeds = []
        for concept in self._concepts:
            if concept in self.graph_node_to_idx:
                idx = self.graph_node_to_idx[concept]
                embeds.append(graph_embs[idx])
            else:
                embeds.append(torch.zeros(self.graph_hidden, device=self.graph_x.device))
        embeds = torch.stack(embeds)      # [num_concepts, graph_hidden]
        return self.graph_proj(embeds)    # [num_concepts, 768]

    def _laat(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Label-Attention mechanism over token sequence.

        Args:
            hidden         : [B, S, H]  concept-enriched token representations
            attention_mask : [B, S]     1 for real tokens, 0 for padding

        Returns:
            label_reps : [B, num_diagnoses, H]  per-label aggregated representations
        """
        H = self.dropout(hidden)                              # [B, S, H]
        U = torch.tanh(self.laat_first(H))                   # [B, S, H]
        att_scores = self.laat_second(U)                     # [B, S, num_diagnoses]

        pad_mask   = (attention_mask == 0).unsqueeze(-1)     # [B, S, 1]
        att_scores = att_scores.masked_fill(pad_mask, float("-inf"))

        att_weights = torch.softmax(att_scores, dim=1)       # [B, S, num_diagnoses]
        att_weights = att_weights.transpose(1, 2)            # [B, num_diagnoses, S]
        label_reps  = torch.bmm(att_weights, H)              # [B, num_diagnoses, H]
        return label_reps

    # ── Forward ─────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        concept_embeddings_bert: torch.Tensor,   # [num_concepts, 768]
    ) -> dict:
        B = input_ids.shape[0]

        # 1. BERT text encoding
        bert_out      = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_out.last_hidden_state           # [B, S, 768]

        # 2. GAT-enhanced concept embeddings
        gat_concepts = self.get_gat_concept_embeddings()    # [num_concepts, 768]

        # 3. Fuse BERT concept embeddings with GAT concept embeddings
        bert_c            = concept_embeddings_bert.unsqueeze(0).expand(B, -1, -1)
        gat_c             = gat_concepts.unsqueeze(0).expand(B, -1, -1)
        enhanced_concepts = self.concept_fusion(
            torch.cat([bert_c, gat_c], dim=-1)              # [B, num_concepts, 1536]
        )                                                    # [B, num_concepts, 768]

        # 4. Cross-attention: text tokens attend to GAT-enhanced concepts
        context, attn_weights = self.cross_attention(
            query=hidden_states,
            key=enhanced_concepts,
            value=enhanced_concepts,
            need_weights=True,
        )                                                    # [B, S, 768]

        # 5. Concept head — CLS of BERT hidden (global concept detection)
        cls_hidden     = self.dropout(hidden_states[:, 0, :])
        concept_logits = self.concept_head(cls_hidden)       # [B, num_concepts]
        concept_scores = torch.sigmoid(concept_logits)

        # 6. Enrich token sequence with GAT concept context (token-level residual).
        #    graph_scale gates how much the cross-attended context contributes.
        #    sigmoid(-2) ≈ 0.12 at init — GAT signal is present from epoch 1.
        graph_strength = torch.sigmoid(self.graph_scale)
        enriched_seq   = self.layer_norm(
            hidden_states + graph_strength * context         # [B, S, 768]
        )

        # 7. LAAT — per-label evidence pooling over the enriched sequence.
        #    Each of the 50 labels independently selects supporting tokens from
        #    the GAT-concept-enriched representation.
        label_reps       = self._laat(enriched_seq, attention_mask)   # [B, K, H]
        diagnosis_logits = self.laat_output(label_reps).squeeze(-1)   # [B, K]

        # 8. Concept → label causal gate.
        #    Concept activations additively shift each label's logit via learned
        #    affinity (warm-started from training co-occurrence).
        concept_bias     = self.concept_to_label(concept_scores)       # [B, K]
        diagnosis_logits = diagnosis_logits + concept_bias

        return {
            "logits"            : diagnosis_logits,
            "concept_logits"    : concept_logits,
            "concept_scores"    : concept_scores,
            "label_reps"        : label_reps,
            "attention_weights" : attn_weights,
            "graph_strength"    : graph_strength.item(),
        }
