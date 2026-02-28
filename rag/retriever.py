"""
rag/retriever.py

SimpleRAG — FAISS-backed retriever for Phase 3.

Features:
  • Index is serialised to disk (config.FAISS_INDEX) after first build.
  • On subsequent runs the index is loaded from cache — no re-encoding.
  • retrieve() returns a plain string (concatenated relevant passages)
    so it can be used directly inside ShifaMindPhase3RAG.forward().
  • Always uses a CPU FAISS index — MPS has no GPU FAISS support.
"""
import json
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

import config
from utils.logging_utils import get_logger

log = get_logger()


# ============================================================================
# EVIDENCE CORPUS BUILDER
# ============================================================================

_CLINICAL_KB = {
    # Respiratory
    "J"  : "Respiratory conditions: assess cough, dyspnea, chest imaging, oxygen saturation",
    "J18": "Pneumonia: fever, cough, chest infiltrates on imaging, leukocytosis",
    "J44": "COPD: chronic airflow limitation, emphysema, chronic bronchitis",
    "J96": "Respiratory failure: hypoxia, hypercapnia, requires oxygen support",
    # Cardiac
    "I"  : "Cardiovascular disease: chest pain, dyspnea, edema, cardiac markers",
    "I50": "Heart failure: dyspnea, edema, elevated BNP, reduced EF on echo",
    "I25": "Ischemic heart disease: angina, troponin, EKG changes",
    "I21": "MI: acute chest pain, troponin elevation, ST changes",
    "I48": "Atrial fibrillation: irregular rhythm, palpitations, stroke risk",
    "I10": "Hypertension: elevated BP, end-organ damage assessment",
    # Infection
    "A"  : "Infectious disease: fever, cultures, antibiotics",
    "A41": "Sepsis: organ dysfunction, hypotension, lactate >2, positive cultures",
    # Renal
    "N"  : "Renal disease: creatinine, BUN, urine output",
    "N17": "Acute kidney injury: rapid creatinine rise, oliguria",
    "N18": "Chronic kidney disease: GFR <60, proteinuria",
    "N39": "Urinary tract disorders: dysuria, frequency, positive culture",
    # Metabolic
    "E"  : "Endocrine/metabolic: glucose, electrolytes, hormone levels",
    "E11": "Type 2 diabetes: hyperglycemia, A1c >6.5%, insulin resistance",
    "E87": "Electrolyte disorders: sodium, potassium, calcium imbalance",
    "E86": "Volume depletion: dehydration, hypovolemia",
    # GI
    "K"  : "GI disease: abdominal pain, nausea, imaging",
    "K80": "Cholelithiasis: RUQ pain, ultrasound showing stones",
    "K21": "GERD: heartburn, acid reflux, esophagitis",
    # Mental health
    "F"  : "Mental health: psychiatric assessment, mood, cognition",
    "F32": "Depression: low mood, anhedonia, sleep disturbance",
    "F41": "Anxiety: excessive worry, panic, physical symptoms",
    # Injury / Neoplasm / Blood / Neuro
    "S"  : "Injury/trauma: mechanism, imaging, stabilisation",
    "T"  : "Poisoning/external causes: toxicology, supportive care",
    "C"  : "Malignancy: histology, staging, treatment planning",
    "D"  : "Benign neoplasm: imaging, biopsy if indicated",
    "D6" : "Anaemia: CBC, iron studies, transfusion if severe",
    "G"  : "Neurological: mental status, focal deficits, imaging",
    "G89": "Pain syndromes: assessment, multimodal analgesia",
}


def build_evidence_corpus(
    top50_codes: List[str],
    df_train: pd.DataFrame,
    prototypes_per_dx: int = config.PROTOTYPES_PER_DX,
    seed: int = config.SEED,
) -> List[dict]:
    """
    Build the evidence corpus for Phase 3 RAG.

    Two sources:
      1. Clinical knowledge base  — one passage per ICD chapter match
      2. MIMIC prototypes         — up to *prototypes_per_dx* real notes
                                    per diagnosis from the training set

    Returns a list of dicts:  {"text": str, "diagnosis": str, "source": str}
    """
    corpus: List[dict] = []

    # --- Clinical knowledge --------------------------------------------------
    for code in top50_codes:
        matched = False
        for prefix, knowledge in _CLINICAL_KB.items():
            if code.startswith(prefix):
                corpus.append({"text": f"{code}: {knowledge}", "diagnosis": code, "source": "clinical_knowledge"})
                matched = True
                break
        if not matched:
            corpus.append({"text": f"{code}: Diagnosis requiring clinical correlation", "diagnosis": code, "source": "clinical_knowledge"})

    log.info(f"Clinical knowledge passages: {len([c for c in corpus if c['source'] == 'clinical_knowledge'])}")

    # --- MIMIC prototypes ----------------------------------------------------
    code_idx_map = {code: i for i, code in enumerate(top50_codes)}
    for dx_code in tqdm(top50_codes, desc="Sampling MIMIC prototypes"):
        idx = code_idx_map[dx_code]

        # Support both column-based and list-of-labels formats
        if dx_code in df_train.columns:
            positives = df_train[df_train[dx_code] == 1]
        elif "labels" in df_train.columns:
            positives = df_train[df_train["labels"].apply(
                lambda x: isinstance(x, list) and len(x) > idx and x[idx] == 1
            )]
        else:
            positives = pd.DataFrame()

        n = min(len(positives), prototypes_per_dx)
        if n > 0:
            for _, row in positives.sample(n=n, random_state=seed).iterrows():
                corpus.append({
                    "text"     : str(row["text"])[:500],
                    "diagnosis": dx_code,
                    "source"   : "mimic_prototype",
                })

    n_kb   = len([c for c in corpus if c["source"] == "clinical_knowledge"])
    n_mimic = len([c for c in corpus if c["source"] == "mimic_prototype"])
    log.info(f"Evidence corpus built: {len(corpus)} passages  ({n_kb} clinical KB + {n_mimic} MIMIC prototypes)")
    return corpus


# ============================================================================
# SIMPLE RAG (FAISS)
# ============================================================================

class SimpleRAG:
    """
    FAISS-backed retriever.

    Usage:
        rag = SimpleRAG()
        rag.build_index(corpus, index_cache_path=config.FAISS_INDEX)
        text = rag.retrieve("patient presents with chest pain …")

    The index is CPU-only — MPS has no GPU FAISS support.
    """

    def __init__(
        self,
        model_name: str = config.RAG_MODEL_NAME,
        top_k: int = config.RAG_TOP_K,
        threshold: float = config.RAG_THRESHOLD,
    ) -> None:
        log.info(f"Loading RAG encoder: {model_name}")
        # Sentence-transformers works fine on CPU; MPS support is partial
        self.encoder   = SentenceTransformer(model_name, device="cpu")
        self.top_k     = top_k
        self.threshold = threshold
        self.index: Optional[faiss.Index] = None
        self.documents: List[dict] = []

    # ------------------------------------------------------------------
    def build_index(
        self,
        documents: List[dict],
        index_cache_path: Optional[Path] = None,
    ) -> None:
        """
        Build (or load from cache) the FAISS index.

        Args:
            documents        : list of {"text": str, …}
            index_cache_path : path to save / load the binary index
        """
        self.documents = documents

        if index_cache_path and Path(index_cache_path).exists():
            self.index = faiss.read_index(str(index_cache_path))
            log.info(f"FAISS index loaded from cache ({self.index.ntotal} vectors)  ← {Path(index_cache_path).name}")
            return

        log.info(f"Building FAISS index from {len(documents)} passages …")
        texts = [d["text"] for d in documents]

        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=config.RAG_ENCODE_BATCH_SIZE,
        ).astype("float32")

        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity
        self.index.add(embeddings)

        log.info(f"FAISS index built: {embeddings.shape[1]}-dim, {self.index.ntotal} vectors")

        if index_cache_path:
            faiss.write_index(self.index, str(index_cache_path))
            log.info(f"FAISS index saved → {Path(index_cache_path).name}")

    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> str:
        """
        Retrieve the top-K most relevant passages and return them as a
        single concatenated string (empty string if none exceed threshold).
        """
        if self.index is None or not query.strip():
            return ""

        q_emb = self.encoder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, self.top_k)

        texts = [
            self.documents[idx]["text"]
            for score, idx in zip(scores[0], indices[0])
            if score >= self.threshold
        ]
        return " ".join(texts)
