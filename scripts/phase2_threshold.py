#!/usr/bin/env python3
"""
scripts/phase2_threshold.py

Phase 2 threshold tuning: sweep per-label probability thresholds on the
validation set, then evaluate on the held-out test set.

Run:
    cd ShifaMind_Local
    python scripts/phase2_threshold.py

Inputs (from Phase 2 training):
    shifamind_local/shared_data/val_split.pkl
    shifamind_local/shared_data/test_split.pkl
    shifamind_local/shared_data/val_concept_labels.npy
    shifamind_local/shared_data/test_concept_labels.npy
    shifamind_local/shared_data/top50_icd10_info.json
    shifamind_local/concept_store/phase2_concept_embeddings.pt
    shifamind_local/graph/phase2/graph_data.pt
    shifamind_local/checkpoints/phase2/phase2_best.pt

Outputs:
    shifamind_local/results/phase2/optimal_thresholds.json
    shifamind_local/results/phase2/threshold_tuning_results.json
    shifamind_local/results/phase2/threshold_comparison.csv
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

import config
from data import ConceptDataset, make_loader
from models import GATEncoder, ShifaMindPhase2GAT
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
log.info("ShifaMind Phase 2 — Threshold Tuning")
log.info("=" * 72)
log.info(f"Device : {device}")

# ============================================================================
# LOAD SHARED DATA
# ============================================================================

log.info("Loading splits …")
assert config.VAL_SPLIT.exists(),     "val_split.pkl not found — run phase2_train.py first"
assert config.TEST_SPLIT.exists(),    "test_split.pkl not found — run phase2_train.py first"
assert config.P2_BEST_CKPT.exists(),  f"Phase 2 checkpoint not found: {config.P2_BEST_CKPT}"
assert config.P2_CONCEPT_EMBS.exists(), f"Phase 2 concept embeddings not found: {config.P2_CONCEPT_EMBS}"
assert config.GRAPH_DATA_PT.exists(),   f"Graph data not found: {config.GRAPH_DATA_PT}"

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
# BUILD MODEL & LOAD BEST CHECKPOINT
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

model = ShifaMindPhase2GAT(
    bert_model       = base_model,
    gat_encoder      = gat_encoder,
    graph_data       = graph_data,
    num_concepts     = NUM_CONCEPTS,
    num_diagnoses    = NUM_LABELS,
    graph_hidden_dim = config.GRAPH_HIDDEN_DIM,
    concepts_list    = config.GLOBAL_CONCEPTS,
).to(device)

best_ckpt = load_checkpoint(config.P2_BEST_CKPT, device)
model.load_state_dict(best_ckpt["model_state_dict"])
model.eval()
log.info(f"Phase 2 best model loaded (epoch {best_ckpt.get('epoch', '?') + 1})")

# Load Phase 2 concept embeddings (frozen for inference)
p2_embs_ckpt    = torch.load(config.P2_CONCEPT_EMBS, map_location=device, weights_only=False)
concept_embs_p2 = p2_embs_ckpt["concept_embeddings"].to(device)
log.info(f"Phase 2 concept embeddings loaded: shape {tuple(concept_embs_p2.shape)}")

# ============================================================================
# DATALOADERS
# ============================================================================

val_ds  = ConceptDataset(df_val["text"].tolist(),  df_val["labels"].tolist(),  val_cl,  tokenizer)
test_ds = ConceptDataset(df_test["text"].tolist(), df_test["labels"].tolist(), test_cl, tokenizer)

val_loader  = make_loader(val_ds,  config.VAL_BATCH_SIZE)
test_loader = make_loader(test_ds, config.VAL_BATCH_SIZE)

# ============================================================================
# COLLECT VALIDATION PROBABILITIES
# ============================================================================

log.info("Running inference on validation set …")
val_probs_list, val_labels_list = [], []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Val inference"):
        inp  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out  = model(inp, mask, concept_embs_p2)
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

thresholds_dict = dict(zip(TOP_50_CODES, optimal_thresholds.tolist()))
with open(config.P2_THRESHOLDS_JSON, "w") as f:
    json.dump(thresholds_dict, f, indent=2)
log.info(f"Optimal thresholds saved → {config.P2_THRESHOLDS_JSON.name}")

# ============================================================================
# COLLECT TEST PROBABILITIES
# ============================================================================

log.info("Running inference on test set …")
test_probs_list, test_labels_list = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test inference"):
        inp  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out  = model(inp, mask, concept_embs_p2)
        test_probs_list.append(torch.sigmoid(out["logits"]).cpu().numpy())
        test_labels_list.append(batch["labels"].numpy())

test_probs  = np.vstack(test_probs_list)
test_labels = np.vstack(test_labels_list)

# ============================================================================
# EVALUATE DEFAULT vs OPTIMAL
# ============================================================================

def compute_metrics(labels, probs, thresholds):
    preds = (probs > thresholds).astype(int)
    return {
        "macro_f1"    : float(f1_score(labels, preds, average="macro",  zero_division=0)),
        "micro_f1"    : float(f1_score(labels, preds, average="micro",  zero_division=0)),
        "precision"   : float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall"      : float(recall_score(labels, preds, average="macro",    zero_division=0)),
        "per_class_f1": [
            float(f1_score(labels[:, j], preds[:, j], zero_division=0))
            for j in range(labels.shape[1])
        ],
    }


default_metrics = compute_metrics(test_labels, test_probs, 0.5)
tuned_metrics   = compute_metrics(test_labels, test_probs, optimal_thresholds)

log.info("=" * 60)
log.info("Phase 2 — Threshold Comparison (test set)")
log.info(f"  Default (0.5)  — Macro F1: {default_metrics['macro_f1']:.4f}  "
         f"Micro: {default_metrics['micro_f1']:.4f}")
log.info(f"  Tuned          — Macro F1: {tuned_metrics['macro_f1']:.4f}  "
         f"Micro: {tuned_metrics['micro_f1']:.4f}")
log.info("=" * 60)

# ============================================================================
# SAVE RESULTS
# ============================================================================

results = {
    "phase"       : "ShifaMind Phase 2 — Threshold Tuning",
    "default_0.5" : default_metrics,
    "optimal_tuned": tuned_metrics,
    "dataset_info" : {"val": len(df_val), "test": len(df_test)},
}

with open(config.P2_THRESH_RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=2)

comparison_df = pd.DataFrame({
    "icd_code"          : TOP_50_CODES,
    "optimal_threshold" : optimal_thresholds.tolist(),
    "val_f1_tuned"      : val_f1_per_label.tolist(),
    "test_f1_default"   : default_metrics["per_class_f1"],
    "test_f1_tuned"     : tuned_metrics["per_class_f1"],
    "train_count"       : [top50_info["top_50_counts"].get(c, 0) for c in TOP_50_CODES],
}).sort_values("test_f1_tuned", ascending=False)
comparison_df.to_csv(config.P2_THRESH_CSV, index=False)

log.info(f"Threshold results saved → {config.P2_THRESH_RESULTS_JSON.name}")
log.info(f"Threshold comparison   → {config.P2_THRESH_CSV.name}")
log.info("Phase 2 threshold tuning complete.")
