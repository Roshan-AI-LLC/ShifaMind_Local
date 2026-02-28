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
    Phase 2 model: BioClinicalBERT + GAT knowledge graph + cross-attention bottleneck.

    Forward inputs:
        input_ids               [B, seq_len]
        attention_mask          [B, seq_len]
        concept_embeddings_bert [num_concepts, 768]  — learned BERT concept embeddings
                                                        from Phase 1 (or freshly initialised)
    """

    def __init__(
        self,
        bert_model,
        gat_encoder: GATEncoder,
        graph_data,                         # torch_geometric.data.Data
        num_concepts: int,
        num_diagnoses: int,
        graph_hidden_dim: int = 256,
        concepts_list: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        import config
        self.bert           = bert_model
        self.gat            = gat_encoder
        self.hidden_size    = 768
        self.graph_hidden   = graph_hidden_dim
        self.num_concepts   = num_concepts
        self.num_diagnoses  = num_diagnoses
        # Use passed list or fall back to global concept list
        self._concepts      = concepts_list if concepts_list is not None else config.GLOBAL_CONCEPTS

        # Graph topology stored as persistent buffers (auto-moved with .to(device))
        self.register_buffer("graph_x",         graph_data.x)
        self.register_buffer("graph_edge_index", graph_data.edge_index)
        self.graph_node_to_idx = graph_data.node_to_idx
        self.graph_idx_to_node = graph_data.idx_to_node

        # Project GAT output (graph_hidden) → BERT dimension (768)
        self.graph_proj = nn.Linear(self.graph_hidden, self.hidden_size)

        # Fuse BERT concept embeddings + GAT concept embeddings
        self.concept_fusion = nn.Sequential(
            nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Cross-attention: text tokens attend to concept embeddings
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        # Multiplicative bottleneck gate
        self.gate_net = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid(),
        )

        self.layer_norm     = nn.LayerNorm(self.hidden_size)
        self.concept_head   = nn.Linear(self.hidden_size, num_concepts)
        self.diagnosis_head = nn.Linear(self.hidden_size, num_diagnoses)
        self.dropout        = nn.Dropout(0.1)

    # ------------------------------------------------------------------
    def get_gat_concept_embeddings(self) -> torch.Tensor:
        """
        Run GAT on the full graph, extract the embedding for each concept
        node, project to 768-dim.

        Returns: [num_concepts, 768]
        """
        graph_embs = self.gat(self.graph_x, self.graph_edge_index)  # [N, graph_hidden]

        embeds = []
        for concept in self._concepts:
            if concept in self.graph_node_to_idx:
                idx = self.graph_node_to_idx[concept]
                embeds.append(graph_embs[idx])
            else:
                # Concept not in UMLS graph → zero vector (same dim as GAT output)
                embeds.append(torch.zeros(self.graph_hidden, device=self.graph_x.device))

        embeds = torch.stack(embeds)          # [num_concepts, graph_hidden]
        return self.graph_proj(embeds)        # [num_concepts, 768]

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        concept_embeddings_bert: torch.Tensor,   # [num_concepts, 768]
    ) -> dict:
        B = input_ids.shape[0]

        # 1. BERT text encoding
        bert_out      = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_out.last_hidden_state   # [B, seq_len, 768]

        # 2. GAT-enhanced concept embeddings  [num_concepts, 768]
        gat_concepts = self.get_gat_concept_embeddings()

        # 3. Fuse BERT concept embeddings with GAT concept embeddings
        bert_c            = concept_embeddings_bert.unsqueeze(0).expand(B, -1, -1)
        gat_c             = gat_concepts.unsqueeze(0).expand(B, -1, -1)
        enhanced_concepts = self.concept_fusion(
            torch.cat([bert_c, gat_c], dim=-1)   # [B, num_concepts, 1536]
        )                                          # [B, num_concepts, 768]

        # 4. Cross-attention: text attends to enhanced concepts
        context, attn_weights = self.cross_attention(
            query=hidden_states,
            key=enhanced_concepts,
            value=enhanced_concepts,
            need_weights=True,
        )                                          # [B, seq_len, 768]

        # 5. Multiplicative bottleneck gate
        pooled_text    = hidden_states.mean(dim=1)  # [B, 768]
        pooled_context = context.mean(dim=1)         # [B, 768]
        gate           = self.gate_net(torch.cat([pooled_text, pooled_context], dim=-1))
        bottleneck     = self.layer_norm(gate * pooled_context)

        # 6. Output heads
        cls_hidden       = self.dropout(pooled_text)
        concept_logits   = self.concept_head(cls_hidden)
        concept_scores   = torch.sigmoid(concept_logits)
        diagnosis_logits = self.diagnosis_head(bottleneck)

        return {
            "logits"            : diagnosis_logits,
            "concept_logits"    : concept_logits,
            "concept_scores"    : concept_scores,
            "gate_values"       : gate,
            "attention_weights" : attn_weights,
            "bottleneck_output" : bottleneck,
        }
