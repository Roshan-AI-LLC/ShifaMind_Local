#!/usr/bin/env python3
"""
scripts/phase3_threshold.py

Phase 3 threshold tuning + final comprehensive evaluation.

Sweeps per-label probability thresholds on the validation set to
maximise per-label F1, then produces a full final report on the
held-out test set using both default (0.5) and optimal thresholds.

Run:
    cd ShifaMind_Local
    python scripts/phase3_threshold.py              # Phase 2 base (default)
    python scripts/phase3_threshold.py --base-phase 1   # Phase 1 base

Inputs (from Phase 3 training):
    shifamind_local/shared_data/val_split.pkl
    shifamind_local/shared_data/test_split.pkl
    shifamind_local/shared_data/val_concept_labels.npy
    shifamind_local/shared_data/test_concept_labels.npy
    shifamind_local/shared_data/top50_icd10_info.json
    shifamind_local/concept_store/phase{N}_concept_embeddings.pt
    shifamind_local/graph/phase2/graph_data.pt
    shifamind_local/checkpoints/phase3_from_p{N}/<run_id>/phase3_best.pth
    shifamind_local/evidence_store/evidence_corpus.json
    shifamind_local/evidence_store/faiss.index

Outputs (in results/phase3_from_p{N}/):
    threshold_tuning.json
    final_test_results.json
    per_diagnosis_metrics.json
    per_diagnosis_metrics.csv
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

import config
from data import RAGDataset, make_loader
from models import GATEncoder, ShifaMindPhase2GAT, ShifaMindPhase3RAG
from rag.retriever import SimpleRAG
from utils import find_latest_checkpoint, get_logger, load_checkpoint

# ============================================================================
# ARGS
# ============================================================================

parser = argparse.ArgumentParser(description="ShifaMind Phase 3 Threshold Tuning")
parser.add_argument(
    "--base-phase", type=int, choices=[1, 2], default=2,
    help="Which base phase to evaluate (1 or 2). Must match the training run. Default: 2",
)
args = parser.parse_args()
BASE_PHASE = args.base_phase

# ============================================================================
# DEVICE + LOGGING
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
log.info(f"ShifaMind Phase 3 — Threshold Tuning & Final Evaluation (base: Phase {BASE_PHASE})")
log.info("=" * 72)
log.info(f"Device     : {device}")
log.info(f"Base phase : {BASE_PHASE}")

# Resolve phase-specific paths
CKPT_DIR, RESULTS_DIR = config.get_p3_paths(BASE_PHASE)
CONCEPT_EMBS_PATH = config.P2_CONCEPT_EMBS if BASE_PHASE == 2 else config.P1_CONCEPT_EMBS

# ============================================================================
# VALIDATE INPUTS
# ============================================================================

for path, name in [
    (config.VAL_SPLIT,         "val_split.pkl"),
    (config.TEST_SPLIT,        "test_split.pkl"),
    (CONCEPT_EMBS_PATH,        f"phase{BASE_PHASE}_concept_embeddings.pt"),
    (config.GRAPH_DATA_PT,     "graph_data.pt"),
    (config.EVIDENCE_CORPUS_JSON, "evidence_corpus.json"),
]:
    if not path.exists():
        log.error(f"Required file not found: {path}")
        log.error(f"  → Run phase3_train.py --base-phase {BASE_PHASE} first.")
        sys.exit(1)

# ============================================================================
# LOAD SHARED DATA
# ============================================================================

log.info("Loading splits …")
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
# REBUILD MODEL
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
with open(config.EVIDENCE_CORPUS_JSON) as f:
    evidence_corpus = json.load(f)

rag = SimpleRAG(
    model_name=config.RAG_MODEL_NAME,
    top_k=config.RAG_TOP_K,
    threshold=config.RAG_THRESHOLD,
)
rag.build_index(evidence_corpus, index_cache_path=config.FAISS_INDEX)

model = ShifaMindPhase3RAG(
    phase2_model  = phase2_model,
    rag_retriever = rag,
    num_diagnoses = NUM_LABELS,
    hidden_size   = 768,
    concepts_list = config.GLOBAL_CONCEPTS,
).to(device)

# Load Phase 3 best checkpoint
p3_best_path = find_latest_checkpoint(CKPT_DIR, "phase3_best.pth")
best_ckpt    = load_checkpoint(p3_best_path, device)

# Validate checkpoint base_phase matches CLI arg
ckpt_base = best_ckpt.get("base_phase", None)
if ckpt_base is not None and ckpt_base != BASE_PHASE:
    log.warning(
        f"Checkpoint was trained with base_phase={ckpt_base} "
        f"but --base-phase={BASE_PHASE} was requested. "
        f"Proceeding — double-check this is intentional."
    )

model.load_state_dict(best_ckpt["model_state_dict"])
model.eval()
log.info(f"Phase 3 best model loaded (epoch {best_ckpt.get('epoch', '?') + 1})")

# Phase concept embeddings (frozen)
embs_ckpt         = torch.load(CONCEPT_EMBS_PATH, map_location=device, weights_only=False)
concept_embs_bert = embs_ckpt["concept_embeddings"].to(device).detach()
log.info(
    f"Phase {BASE_PHASE} concept embeddings loaded (frozen): "
    f"{tuple(concept_embs_bert.shape)}"
)

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
    f"Threshold tuning done — "
    f"mean optimal threshold: {optimal_thresholds.mean():.3f}  "
    f"mean val F1: {val_f1_per_label.mean():.4f}"
)

threshold_tuning = {
    "thresholds"      : dict(zip(TOP_50_CODES, optimal_thresholds.tolist())),
    "val_f1_per_label": dict(zip(TOP_50_CODES, val_f1_per_label.tolist())),
    "mean_val_f1"     : float(val_f1_per_label.mean()),
    "mean_threshold"  : float(optimal_thresholds.mean()),
}
thresh_json = RESULTS_DIR / "threshold_tuning.json"
with open(thresh_json, "w") as f:
    json.dump(threshold_tuning, f, indent=2)
log.info(f"Threshold tuning saved → {thresh_json.name}")

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
# EVALUATE WITH OPTIMAL THRESHOLDS (tuned on val set)
# ============================================================================

def compute_metrics(labels, probs, thresholds, label_names):
    """Compute macro/micro aggregate + per-class metrics."""
    if hasattr(thresholds, "__len__"):
        preds = (probs > thresholds).astype(int)
    else:
        preds = (probs > thresholds).astype(int)

    per_class = {}
    for j, code in enumerate(label_names):
        n_pos = int(labels[:, j].sum())
        thr   = float(thresholds[j]) if hasattr(thresholds, "__len__") else float(thresholds)
        per_class[code] = {
            "f1"       : float(f1_score(labels[:, j], preds[:, j], zero_division=0)),
            "precision": float(precision_score(labels[:, j], preds[:, j], zero_division=0)),
            "recall"   : float(recall_score(labels[:, j], preds[:, j], zero_division=0)),
            "threshold": thr,
            "support"  : n_pos,
        }
        if n_pos > 0 and n_pos < len(labels):
            try:
                per_class[code]["auc_roc"]       = float(roc_auc_score(labels[:, j], probs[:, j]))
                per_class[code]["avg_precision"]  = float(
                    average_precision_score(labels[:, j], probs[:, j])
                )
            except ValueError:
                pass

    return {
        "macro_f1": float(f1_score(labels, preds, average="macro",  zero_division=0)),
        "micro_f1": float(f1_score(labels, preds, average="micro",  zero_division=0)),
        "macro_p" : float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_r" : float(recall_score(labels,    preds, average="macro", zero_division=0)),
        "per_class": per_class,
    }


tuned_metrics = compute_metrics(test_labels, test_probs, optimal_thresholds, label_names=TOP_50_CODES)

log.info("=" * 60)
log.info(f"Phase 3 (base: Phase {BASE_PHASE}) — Final Test Results")
log.info(
    f"  Tuned — Macro F1: {tuned_metrics['macro_f1']:.4f}  "
    f"Micro: {tuned_metrics['micro_f1']:.4f}"
)
log.info(
    f"  Precision (tuned): {tuned_metrics['macro_p']:.4f}  "
    f"Recall (tuned): {tuned_metrics['macro_r']:.4f}"
)
log.info("=" * 60)

# Per-diagnosis detail
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
    "phase"        : f"ShifaMind Phase 3 — Final Evaluation (base: Phase {BASE_PHASE})",
    "base_phase"   : BASE_PHASE,
    "checkpoint"   : str(p3_best_path),
    "optimal_tuned": {
        "macro_f1": tuned_metrics["macro_f1"],
        "micro_f1": tuned_metrics["micro_f1"],
        "macro_p" : tuned_metrics["macro_p"],
        "macro_r" : tuned_metrics["macro_r"],
    },
    "dataset_info" : {"val": len(df_val), "test": len(df_test)},
    "rag_config"   : {
        "top_k"    : config.RAG_TOP_K,
        "threshold": config.RAG_THRESHOLD,
        "gate_max" : config.RAG_GATE_MAX,
    },
}

final_json = RESULTS_DIR / "final_test_results.json"
per_dx_json = RESULTS_DIR / "per_diagnosis_metrics.json"
per_dx_csv  = RESULTS_DIR / "per_diagnosis_metrics.csv"

with open(final_json, "w") as f:
    json.dump(final_results, f, indent=2)

with open(per_dx_json, "w") as f:
    json.dump(tuned_metrics["per_class"], f, indent=2)

per_dx_df.to_csv(per_dx_csv, index=False)

log.info(f"Final results saved  → {final_json.name}")
log.info(f"Per-diagnosis JSON   → {per_dx_json.name}")
log.info(f"Per-diagnosis CSV    → {per_dx_csv.name}")
log.info(f"Results directory    → {RESULTS_DIR}")
log.info(f"Phase 3 threshold tuning and final evaluation complete  (base: Phase {BASE_PHASE})")
