#!/usr/bin/env python3
"""
scripts/phase3_threshold.py

Phase 3 threshold tuning + final comprehensive evaluation.

Sweeps per-label probability thresholds on the validation set to
maximise per-label F1, then produces a full final report on the
held-out test set using both default (0.5) and optimal thresholds.

Run:
    cd ShifaMind_Local
    python scripts/phase3_threshold.py

Inputs (from Phase 3 training):
    shifamind_local/shared_data/val_split.pkl
    shifamind_local/shared_data/test_split.pkl
    shifamind_local/shared_data/val_concept_labels.npy
    shifamind_local/shared_data/test_concept_labels.npy
    shifamind_local/shared_data/top50_icd10_info.json
    shifamind_local/concept_store/phase2_concept_embeddings.pt
    shifamind_local/graph/phase2/graph_data.pt
    shifamind_local/checkpoints/phase3/phase3_best.pth
    shifamind_local/evidence_store/evidence_corpus.json
    shifamind_local/evidence_store/faiss.index

Outputs:
    shifamind_local/results/phase3/threshold_tuning.json
    shifamind_local/results/phase3/final_test_results.json
    shifamind_local/results/phase3/per_diagnosis_metrics.json
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

import config
from data import RAGDataset, make_loader
from models import GATEncoder, ShifaMindPhase2GAT, ShifaMindPhase3RAG
from rag.retriever import SimpleRAG
from utils import get_logger, load_checkpoint

# ============================================================================
# DEVICE
# ============================================================================

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = _get_device()
log    = get_logger()

log.info("=" * 72)
log.info("ShifaMind Phase 3 — Threshold Tuning & Final Evaluation")
log.info("=" * 72)
log.info(f"Device : {device}")

# ============================================================================
# LOAD SHARED DATA
# ============================================================================

log.info("Loading splits …")
for path, name in [
    (config.VAL_SPLIT,       "val_split.pkl"),
    (config.TEST_SPLIT,      "test_split.pkl"),
    (config.P3_BEST_CKPT,    "phase3_best.pth"),
    (config.P2_CONCEPT_EMBS, "phase2_concept_embeddings.pt"),
    (config.GRAPH_DATA_PT,   "graph_data.pt"),
]:
    assert path.exists(), f"{name} not found — run phase3_train.py first"

with open(config.VAL_SPLIT,  "rb") as f: df_val  = pickle.load(f)
with open(config.TEST_SPLIT, "rb") as f: df_test = pickle.load(f)

val_cl  = np.load(config.VAL_CONCEPT_LABELS)
test_cl = np.load(config.TEST_CONCEPT_LABELS)

with open(config.TOP50_INFO_OUT) as f:
    top50_info = json.load(f)
TOP_50_CODES = top50_info["top_50_codes"]
NUM_LABELS   = len(TOP_50_CODES)
NUM_CONCEPTS = len(config.GLOBAL_CONCEPTS)

log.info(f"Val: {len(df_val):,}  Test: {len(df_test):,}")

# ============================================================================
# REBUILD MODEL ARCHITECTURE
# ============================================================================

log.info("Loading BioClinicalBERT …")
tokenizer  = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
base_model = AutoModel.from_pretrained(config.BERT_MODEL_NAME).to(device)

log.info("Loading graph data …")
graph_data = torch.load(config.GRAPH_DATA_PT, map_location=device, weights_only=False)

gat_encoder = GATEncoder(
    in_channels     = 768,
    hidden_channels = config.GRAPH_HIDDEN_DIM,
    num_layers      = config.GAT_LAYERS,
    heads           = config.GAT_HEADS,
    dropout         = config.GAT_DROPOUT,
)

phase2_model = ShifaMindPhase2GAT(
    bert_model       = base_model,
    gat_encoder      = gat_encoder,
    graph_data       = graph_data,
    num_concepts     = NUM_CONCEPTS,
    num_diagnoses    = NUM_LABELS,
    graph_hidden_dim = config.GRAPH_HIDDEN_DIM,
    concepts_list    = config.GLOBAL_CONCEPTS,
).to(device)

# Load evidence corpus + RAG
log.info("Loading evidence corpus and FAISS index …")
assert config.EVIDENCE_CORPUS_JSON.exists(), "evidence_corpus.json not found — run phase3_train.py first"
with open(config.EVIDENCE_CORPUS_JSON) as f:
    evidence_corpus = json.load(f)

rag = SimpleRAG(
    model_name=config.RAG_MODEL_NAME,
    top_k=config.RAG_TOP_K,
    threshold=config.RAG_THRESHOLD,
)
rag.build_index(evidence_corpus, index_cache_path=config.FAISS_INDEX)

model = ShifaMindPhase3RAG(
    phase2_model=phase2_model,
    rag_retriever=rag,
    hidden_size=768,
).to(device)

# Load Phase 3 best checkpoint
best_ckpt = load_checkpoint(config.P3_BEST_CKPT, device)
model.load_state_dict(best_ckpt["model_state_dict"])
model.eval()
log.info(f"Phase 3 best model loaded (epoch {best_ckpt.get('epoch', '?') + 1})")

# Phase 2 concept embeddings (frozen)
p2_embs_ckpt      = torch.load(config.P2_CONCEPT_EMBS, map_location=device, weights_only=False)
concept_embs_bert = p2_embs_ckpt["concept_embeddings"].to(device).detach()
log.info(f"Phase 2 concept embeddings loaded (frozen): {tuple(concept_embs_bert.shape)}")

# ============================================================================
# DATALOADERS
# ============================================================================

val_ds  = RAGDataset(df_val["text"].tolist(),  df_val["labels"].tolist(),  val_cl,  tokenizer)
test_ds = RAGDataset(df_test["text"].tolist(), df_test["labels"].tolist(), test_cl, tokenizer)

val_loader  = make_loader(val_ds,  config.VAL_BATCH_SIZE)
test_loader = make_loader(test_ds, config.VAL_BATCH_SIZE)

# ============================================================================
# COLLECT VALIDATION PROBABILITIES
# ============================================================================

log.info("Running inference on validation set (with RAG) …")
val_probs_list, val_labels_list = [], []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Val inference"):
        inp   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        texts = batch["text"]
        out   = model(inp, mask, concept_embs_bert, input_texts=texts, use_rag=True)
        val_probs_list.append(torch.sigmoid(out["logits"]).cpu().numpy())
        val_labels_list.append(batch["labels"].numpy())

val_probs  = np.vstack(val_probs_list)
val_labels = np.vstack(val_labels_list)

# ============================================================================
# PER-LABEL THRESHOLD TUNING
# ============================================================================

log.info(f"Sweeping {len(config.THRESHOLD_CANDIDATES)} threshold candidates per label …")

optimal_thresholds = np.full(NUM_LABELS, 0.5, dtype=np.float32)
val_f1_per_label   = np.zeros(NUM_LABELS, dtype=np.float32)

for j in range(NUM_LABELS):
    best_t, best_f1 = 0.5, 0.0
    for t in config.THRESHOLD_CANDIDATES:
        preds = (val_probs[:, j] > t).astype(int)
        f1_j  = f1_score(val_labels[:, j], preds, zero_division=0)
        if f1_j > best_f1:
            best_f1, best_t = f1_j, t
    optimal_thresholds[j] = best_t
    val_f1_per_label[j]   = best_f1

log.info(
    f"Threshold tuning done — mean optimal threshold: {optimal_thresholds.mean():.3f}  "
    f"mean val F1: {val_f1_per_label.mean():.4f}"
)

threshold_tuning = {
    "thresholds"      : dict(zip(TOP_50_CODES, optimal_thresholds.tolist())),
    "val_f1_per_label": dict(zip(TOP_50_CODES, val_f1_per_label.tolist())),
    "mean_val_f1"     : float(val_f1_per_label.mean()),
    "mean_threshold"  : float(optimal_thresholds.mean()),
}
with open(config.P3_THRESH_JSON, "w") as f:
    json.dump(threshold_tuning, f, indent=2)
log.info(f"Threshold tuning saved → {config.P3_THRESH_JSON.name}")

# ============================================================================
# COLLECT TEST PROBABILITIES
# ============================================================================

log.info("Running inference on test set (with RAG) …")
test_probs_list, test_labels_list = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test inference"):
        inp   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        texts = batch["text"]
        out   = model(inp, mask, concept_embs_bert, input_texts=texts, use_rag=True)
        test_probs_list.append(torch.sigmoid(out["logits"]).cpu().numpy())
        test_labels_list.append(batch["labels"].numpy())

test_probs  = np.vstack(test_probs_list)
test_labels = np.vstack(test_labels_list)

# ============================================================================
# COMPREHENSIVE FINAL EVALUATION
# ============================================================================

def compute_metrics(labels, probs, thresholds, label_names):
    preds = (probs > thresholds).astype(int)

    # Per-class metrics
    per_class = {}
    for j, code in enumerate(label_names):
        n_pos = int(labels[:, j].sum())
        per_class[code] = {
            "f1"       : float(f1_score(labels[:, j], preds[:, j], zero_division=0)),
            "precision": float(precision_score(labels[:, j], preds[:, j], zero_division=0)),
            "recall"   : float(recall_score(labels[:, j], preds[:, j], zero_division=0)),
            "threshold": float(thresholds[j] if hasattr(thresholds, "__len__") else thresholds),
            "support"  : n_pos,
        }
        # AUC / AP only when both classes are present
        if n_pos > 0 and n_pos < len(labels):
            per_class[code]["auc_roc"] = float(roc_auc_score(labels[:, j], probs[:, j]))
            per_class[code]["avg_precision"] = float(
                average_precision_score(labels[:, j], probs[:, j])
            )

    return {
        "macro_f1"  : float(f1_score(labels, preds, average="macro",  zero_division=0)),
        "micro_f1"  : float(f1_score(labels, preds, average="micro",  zero_division=0)),
        "macro_p"   : float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_r"   : float(recall_score(labels, preds, average="macro",    zero_division=0)),
        "per_class" : per_class,
    }


default_metrics = compute_metrics(test_labels, test_probs, 0.5,                label_names=TOP_50_CODES)
tuned_metrics   = compute_metrics(test_labels, test_probs, optimal_thresholds, label_names=TOP_50_CODES)

log.info("=" * 60)
log.info("Phase 3 — Final Test Results")
log.info(f"  Default (0.5)  — Macro F1: {default_metrics['macro_f1']:.4f}  "
         f"Micro: {default_metrics['micro_f1']:.4f}")
log.info(f"  Tuned          — Macro F1: {tuned_metrics['macro_f1']:.4f}  "
         f"Micro: {tuned_metrics['micro_f1']:.4f}")
log.info(f"  Precision (tuned): {tuned_metrics['macro_p']:.4f}  "
         f"Recall (tuned): {tuned_metrics['macro_r']:.4f}")
log.info("=" * 60)

# Per-diagnosis detail table
per_dx_rows = []
for code in TOP_50_CODES:
    dm = tuned_metrics["per_class"][code]
    per_dx_rows.append({
        "icd_code"         : code,
        "threshold"        : dm["threshold"],
        "f1"               : dm["f1"],
        "precision"        : dm["precision"],
        "recall"           : dm["recall"],
        "support"          : dm["support"],
        "auc_roc"          : dm.get("auc_roc", float("nan")),
        "avg_precision"    : dm.get("avg_precision", float("nan")),
        "train_count"      : top50_info.get("top_50_counts", {}).get(code, 0),
    })
per_dx_df = pd.DataFrame(per_dx_rows).sort_values("f1", ascending=False)

# Top / bottom 5 by F1
log.info("Top 5 diagnoses by F1:")
for _, row in per_dx_df.head(5).iterrows():
    log.info(f"  {row['icd_code']:10s}  F1={row['f1']:.4f}  support={int(row['support'])}")
log.info("Bottom 5 diagnoses by F1:")
for _, row in per_dx_df.tail(5).iterrows():
    log.info(f"  {row['icd_code']:10s}  F1={row['f1']:.4f}  support={int(row['support'])}")

# ============================================================================
# SAVE ALL RESULTS
# ============================================================================

final_results = {
    "phase"        : "ShifaMind Phase 3 — Final Evaluation",
    "default_0.5"  : {
        "macro_f1": default_metrics["macro_f1"],
        "micro_f1": default_metrics["micro_f1"],
        "macro_p" : default_metrics["macro_p"],
        "macro_r" : default_metrics["macro_r"],
    },
    "optimal_tuned": {
        "macro_f1": tuned_metrics["macro_f1"],
        "micro_f1": tuned_metrics["micro_f1"],
        "macro_p" : tuned_metrics["macro_p"],
        "macro_r" : tuned_metrics["macro_r"],
    },
    "dataset_info" : {"val": len(df_val), "test": len(df_test)},
}

with open(config.P3_FINAL_RESULTS_JSON, "w") as f:
    json.dump(final_results, f, indent=2)

with open(config.P3_PER_DX_JSON, "w") as f:
    json.dump(tuned_metrics["per_class"], f, indent=2)

per_dx_df.to_csv(
    config.RESULTS_P3 / "per_diagnosis_metrics.csv", index=False
)

log.info(f"Final results saved  → {config.P3_FINAL_RESULTS_JSON.name}")
log.info(f"Per-diagnosis JSON   → {config.P3_PER_DX_JSON.name}")
log.info(f"Per-diagnosis CSV    → per_diagnosis_metrics.csv")
log.info("Phase 3 threshold tuning and final evaluation complete.")
