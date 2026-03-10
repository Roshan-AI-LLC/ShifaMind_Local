"""
models/phase3.py

Phase 3 architecture:
  ShifaMindPhase3RAG — wraps ShifaMindPhase2GAT with a gated FAISS RAG layer.

RAG behaviour (redesigned for per-sample signal):
  1. Phase 2 forward() runs normally → base logits + concept_scores.
  2. Top-K activated concept names (from concept_scores) are used as a
     focused RAG query — much better cosine similarity than full note text.
  3. All retrieved passages for the batch are encoded in ONE encode() call
     (no more one-by-one Python loop inside the forward pass).
  4. A learned linear head projects the 384-dim RAG embedding to num_dx
     logits, gated by a learnable scalar capped at RAG_GATE_MAX.
  5. Gated RAG boost is added PER SAMPLE to the base diagnosis logits.

Key architectural decisions:
  • Phase 2 BERT runs ONCE (no double-BERT bug from v1).
  • Concept embeddings stay frozen — RAG augments logits, not concept space.
  • RAG gate is a scalar parameter (interpretable, fast to converge).
  • Empty retrievals (gate * 0) → zero boost → RAG never hurts base model.
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from .phase2 import ShifaMindPhase2GAT


# How many top concept names to use as the RAG query per sample
_RAG_QUERY_TOP_CONCEPTS = 5


class ShifaMindPhase3RAG(nn.Module):
    """
    Phase 3: ShifaMindPhase2GAT + gated FAISS RAG (per-sample logit boost).

    Args:
        phase2_model   : Pretrained / loaded ShifaMindPhase2GAT instance.
        rag_retriever  : A rag.retriever.SimpleRAG instance (index already built).
        num_diagnoses  : Number of diagnosis labels (50 for top-50 ICD-10).
        hidden_size    : BERT hidden dimension (768).
        concepts_list  : Ordered list of concept name strings (len == num_concepts).
    """

    def __init__(
        self,
        phase2_model: ShifaMindPhase2GAT,
        rag_retriever,
        num_diagnoses: int,
        hidden_size: int = 768,
        concepts_list: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.phase2_model  = phase2_model
        self.rag           = rag_retriever
        self.hidden_size   = hidden_size
        self.num_diagnoses = num_diagnoses
        self._concepts     = concepts_list if concepts_list is not None else config.GLOBAL_CONCEPTS

        # RAG encoder output dim: all-MiniLM-L6-v2 → 384
        rag_dim = 384

        # Project RAG passage embedding → hidden space → diagnosis logits
        self.rag_projection = nn.Linear(rag_dim, hidden_size)
        self.rag_to_logits  = nn.Linear(hidden_size, num_diagnoses)

        # Per-diagnosis learnable gate magnitude: [num_diagnoses]
        # Scalar gate FAILS because positive/negative gradients across the 50 diagnoses
        # cancel each other → net gradient ≈ 0 → gate never moves (observed: stuck at 0.175).
        # Per-diagnosis gate allows each label to independently learn whether RAG helps.
        # Initialised to 0 → sigmoid(0) × RAG_GATE_MAX = 0.175 per diagnosis.
        # Diagnoses that benefit from RAG will have their gate grow; others stay low.
        self.rag_gate_logit = nn.Parameter(torch.zeros(num_diagnoses))

    # ------------------------------------------------------------------
    def _concept_query(
        self,
        concept_scores: torch.Tensor,   # [B, num_concepts]  (sigmoid probabilities)
        top_k: int = _RAG_QUERY_TOP_CONCEPTS,
    ) -> List[str]:
        """
        Build one focused RAG query string per sample from its top-K
        activated concept names.

        Using concept names (e.g. "sepsis pneumonia tachycardia") rather than
        the full clinical note greatly improves cosine similarity with the
        knowledge-base passages, which also use condition/symptom keywords.
        """
        queries = []
        topk_idxs = concept_scores.topk(min(top_k, concept_scores.shape[1]), dim=1).indices
        for row in topk_idxs:
            names = [self._concepts[i] for i in row.tolist() if i < len(self._concepts)]
            queries.append(" ".join(names))
        return queries

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        concept_embeddings_bert: torch.Tensor,   # [num_concepts, 768]  frozen
        input_texts: Optional[List[str]] = None,
        use_rag: bool = True,
    ) -> dict:
        """
        Args:
            input_ids               [B, seq_len]
            attention_mask          [B, seq_len]
            concept_embeddings_bert [num_concepts, 768]   frozen Phase 2 embeddings
            input_texts             List[str] of raw clinical notes (needed for RAG)
            use_rag                 False → skip retrieval (export / ablation)

        Returns:
            dict with all Phase 2 keys plus augmented 'logits'.
        """
        # --- 1. Phase 2 forward (single BERT call) -----------------------
        base_outputs = self.phase2_model(input_ids, attention_mask, concept_embeddings_bert)

        if not use_rag or self.rag is None or input_texts is None:
            return base_outputs

        # --- 2. Build focused RAG queries from top concept activations ---
        #  concept_scores: [B, num_concepts]  (already sigmoid probabilities)
        queries = self._concept_query(base_outputs["concept_scores"].detach())

        # --- 3. Retrieve one passage string per sample -------------------
        retrieved = [self.rag.retrieve(q) for q in queries]

        # --- 4. Batch-encode all retrieved passages in ONE call ----------
        #  Zeros for samples where nothing was retrieved (similarity < threshold)
        texts_for_encode = [txt if txt.strip() else "N/A" for txt in retrieved]
        rag_np = self.rag.encoder.encode(
            texts_for_encode,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=len(texts_for_encode),
        ).astype("float32")                             # [B, 384]

        # Zero out embeddings for samples that had no retrieval
        no_retrieval_mask = np.array([not txt.strip() for txt in retrieved], dtype=np.float32)
        rag_np *= (1.0 - no_retrieval_mask[:, None])

        rag_tensor = torch.tensor(rag_np, dtype=torch.float32, device=input_ids.device)  # [B, 384]

        # --- 5. Per-sample RAG boost to diagnosis logits -----------------
        rag_hidden  = F.relu(self.rag_projection(rag_tensor))   # [B, 768]
        rag_boost   = self.rag_to_logits(rag_hidden)             # [B, num_diagnoses]

        # Per-diagnosis gate: [num_diagnoses]  →  [1, num_diagnoses]  (broadcast over B)
        # sigmoid(0) × RAG_GATE_MAX = 0.175 per diagnosis at init.
        # Each of the 50 diagnoses independently controls its RAG influence,
        # so gradients do NOT cancel across labels and the gate can actually learn.
        gate = torch.sigmoid(self.rag_gate_logit) * config.RAG_GATE_MAX  # [num_dx]

        # --- 6. Augment logits (per sample, per diagnosis) ---------------
        outputs = dict(base_outputs)
        outputs["logits"]    = base_outputs["logits"] + gate.unsqueeze(0) * rag_boost
        outputs["rag_gate"]  = gate.mean().item()   # scalar summary for logging
        outputs["rag_boost"] = rag_boost.detach()
        return outputs
