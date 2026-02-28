"""
models/phase3.py

Phase 3 architecture:
  ShifaMindPhase3RAG — wraps ShifaMindPhase2GAT with a gated FAISS RAG layer.

RAG behaviour:
  • For each clinical note in the batch, retrieve the top-K most similar
    passages from the evidence store (clinical knowledge + MIMIC prototypes).
  • Encode the retrieved passage with the same sentence-transformer used
    to build the FAISS index.
  • Gate the RAG context with a learned sigmoid gate and ADD it to the
    concept embeddings (max influence capped at RAG_GATE_MAX = 40 %).
  • Pass the augmented concept embeddings to Phase 2's forward().

The concept_embeddings_bert tensor is frozen (not part of the Phase 3
optimiser) — only the RAG projection + gate and the Phase 2 model weights
are fine-tuned in Phase 3.
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

import config
from .phase2 import ShifaMindPhase2GAT


class ShifaMindPhase3RAG(nn.Module):
    """
    Phase 3: ShifaMindPhase2GAT + gated FAISS RAG.

    Args:
        phase2_model   : Pretrained / loaded ShifaMindPhase2GAT instance.
        rag_retriever  : A rag.retriever.SimpleRAG instance (index already built).
        hidden_size    : BERT hidden dimension (768).
    """

    def __init__(
        self,
        phase2_model: ShifaMindPhase2GAT,
        rag_retriever,
        hidden_size: int = 768,
    ) -> None:
        super().__init__()
        self.phase2_model = phase2_model
        self.rag          = rag_retriever
        self.hidden_size  = hidden_size

        # RAG encoder output dim: all-MiniLM-L6-v2 → 384
        rag_dim = 384

        # Project retrieved passage embedding into BERT space
        self.rag_projection = nn.Linear(rag_dim, hidden_size)

        # Gate: decides how much RAG context to mix in
        self.rag_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        concept_embeddings_bert: torch.Tensor,   # [num_concepts, 768]
        input_texts: Optional[List[str]] = None,
        use_rag: bool = True,
    ) -> dict:
        """
        Args:
            input_ids               [B, seq_len]
            attention_mask          [B, seq_len]
            concept_embeddings_bert [num_concepts, 768]  frozen Phase 2 embeddings
            input_texts             List[str] of raw clinical notes — needed for RAG
            use_rag                 Set False to skip retrieval (e.g. during export)
        """
        if use_rag and self.rag is not None and input_texts is not None:
            # --- Retrieve one passage per sample in the batch -----------------
            retrieved = [self.rag.retrieve(t) for t in input_texts]

            # Encode each retrieved passage with the RAG sentence-transformer
            # Shape: [B, 384]
            rag_np = np.stack(
                [
                    self.rag.encoder.encode([txt], convert_to_numpy=True)[0]
                    if txt
                    else np.zeros(384, dtype=np.float32)
                    for txt in retrieved
                ]
            )
            rag_tensor  = torch.tensor(rag_np, dtype=torch.float32, device=input_ids.device)
            rag_context = self.rag_projection(rag_tensor)   # [B, 768]

            # Get pooled BERT for gating (no grad — gating only)
            with torch.no_grad():
                bert_out    = self.phase2_model.bert(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                pooled_bert = bert_out.last_hidden_state.mean(dim=1)   # [B, 768]

            # Gate: how much RAG to inject, capped at RAG_GATE_MAX
            gate    = self.rag_gate(torch.cat([pooled_bert, rag_context], dim=-1))  # [B, 768]
            gate    = gate * config.RAG_GATE_MAX

            # Average the gated RAG context across the batch, broadcast to concepts
            rag_aug = (gate * rag_context).mean(dim=0, keepdim=True)     # [1, 768]
            concept_embeddings_aug = concept_embeddings_bert + rag_aug   # [num_concepts, 768]
        else:
            concept_embeddings_aug = concept_embeddings_bert

        return self.phase2_model(input_ids, attention_mask, concept_embeddings_aug)
