#!/usr/bin/env python3
"""
scripts/phase1_train.py

Phase 1 training: BioClinicalBERT + Concept Bottleneck → Top-50 ICD-10.

Run:
    cd ShifaMind_Local
    python scripts/phase1_train.py

Outputs (all under ./shifamind_local/):
    shared_data/train_split.pkl
    shared_data/val_split.pkl
    shared_data/test_split.pkl
    shared_data/train_concept_labels.npy
    shared_data/val_concept_labels.npy
    shared_data/test_concept_labels.npy
    shared_data/concept_list.json
    shared_data/top50_icd10_info.json
    concept_store/phase1_concept_embeddings.pt
    checkpoints/phase1/phase1_best.pt
    checkpoints/phase1/phase1_epoch_N.pt   (every epoch)
    results/phase1/results.json
    results/phase1/per_label_f1.csv
    logs/shifamind.log
    logs/metrics.jsonl
"""
import json
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# --- project root on sys.path -----------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

import config
from data import ConceptDataset, make_loader
from models import ShifaMind2Phase1
from training import MultiObjectiveLoss
from training.evaluate import evaluate_phase1
from utils import get_logger, log_metrics, save_best_checkpoint, save_epoch_checkpoint, log_memory_usage

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

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

log.info("=" * 72)
log.info("ShifaMind Phase 1 — Training")
log.info("=" * 72)
log.info(f"Device : {device}")

# ============================================================================
# LOAD SOURCE DATA
# ============================================================================

log.info("Loading source data …")
assert config.DATA_CSV.exists(), f"Data CSV not found: {config.DATA_CSV}"
assert config.TOP50_INFO_SRC.exists(), f"top50_icd10_info.json not found: {config.TOP50_INFO_SRC}"

with open(config.TOP50_INFO_SRC) as f:
    top50_info  = json.load(f)
TOP_50_CODES = top50_info["top_50_codes"]
NUM_LABELS   = len(TOP_50_CODES)
NUM_CONCEPTS = len(config.GLOBAL_CONCEPTS)

log.info(f"Top-50 ICD-10 codes loaded: {NUM_LABELS}")
log.info(f"Global concepts            : {NUM_CONCEPTS}")

df_all = pd.read_csv(config.DATA_CSV)
df_all["labels"] = df_all[TOP_50_CODES].values.tolist()
df     = df_all[["text", "labels"] + TOP_50_CODES].dropna(subset=["text"]).copy()

log.info(f"Dataset: {len(df):,} samples")

# ============================================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================================

train_idx, temp_idx = train_test_split(range(len(df)), test_size=0.30, random_state=config.SEED)
val_idx,  test_idx  = train_test_split(temp_idx,        test_size=0.50, random_state=config.SEED)

df_train = df.iloc[train_idx].reset_index(drop=True)
df_val   = df.iloc[val_idx].reset_index(drop=True)
df_test  = df.iloc[test_idx].reset_index(drop=True)

log.info(f"Split — train {len(df_train):,}  val {len(df_val):,}  test {len(df_test):,}")

# Save splits
with open(config.TRAIN_SPLIT, "wb") as f: pickle.dump(df_train, f)
with open(config.VAL_SPLIT,   "wb") as f: pickle.dump(df_val,   f)
with open(config.TEST_SPLIT,  "wb") as f: pickle.dump(df_test,  f)

# Copy top50_info to shared_data
shutil.copy(config.TOP50_INFO_SRC, config.TOP50_INFO_OUT)

# ============================================================================
# CONCEPT LABELS (keyword-based)
# ============================================================================

def generate_concept_labels(texts, concepts):
    labels = []
    for text in tqdm(texts, desc="Concept labels", leave=False):
        tl = str(text).lower()
        labels.append([1 if c in tl else 0 for c in concepts])
    return np.array(labels, dtype=np.float32)

log.info("Generating concept labels …")
train_cl = generate_concept_labels(df_train["text"], config.GLOBAL_CONCEPTS)
val_cl   = generate_concept_labels(df_val["text"],   config.GLOBAL_CONCEPTS)
test_cl  = generate_concept_labels(df_test["text"],  config.GLOBAL_CONCEPTS)

np.save(config.TRAIN_CONCEPT_LABELS, train_cl)
np.save(config.VAL_CONCEPT_LABELS,   val_cl)
np.save(config.TEST_CONCEPT_LABELS,  test_cl)

with open(config.CONCEPT_LIST, "w") as f:
    json.dump(config.GLOBAL_CONCEPTS, f, indent=2)

log.info(f"Concept labels: shape {train_cl.shape}, avg {train_cl.sum(1).mean():.2f} per sample")

# ============================================================================
# MODEL
# ============================================================================

log.info("Loading BioClinicalBERT …")
tokenizer  = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
base_model = AutoModel.from_pretrained(config.BERT_MODEL_NAME).to(device)

model = ShifaMind2Phase1(
    base_model, num_concepts=NUM_CONCEPTS, num_classes=NUM_LABELS, fusion_layers=[9, 11]
).to(device)

total_params = sum(p.numel() for p in model.parameters())
log.info(f"Phase 1 model: {total_params:,} parameters")

# ============================================================================
# DATASETS & LOADERS
# ============================================================================

train_ds = ConceptDataset(df_train["text"].tolist(), df_train["labels"].tolist(), train_cl, tokenizer)
val_ds   = ConceptDataset(df_val["text"].tolist(),   df_val["labels"].tolist(),   val_cl,   tokenizer)
test_ds  = ConceptDataset(df_test["text"].tolist(),  df_test["labels"].tolist(),  test_cl,  tokenizer)

train_loader = make_loader(train_ds, config.TRAIN_BATCH_SIZE, shuffle=True)
val_loader   = make_loader(val_ds,   config.VAL_BATCH_SIZE)
test_loader  = make_loader(test_ds,  config.VAL_BATCH_SIZE)

log.info(f"Loaders — train {len(train_loader)} batches  val {len(val_loader)}  test {len(test_loader)}")

# ============================================================================
# TRAINING SETUP
# ============================================================================

criterion = MultiObjectiveLoss(config.LAMBDA_DX, config.LAMBDA_ALIGN, config.LAMBDA_CONCEPT)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_loader) // 2,
    num_training_steps=len(train_loader) * config.NUM_EPOCHS_P1,
)

# ============================================================================
# TRAINING LOOP
# ============================================================================

best_f1 = 0.0
history = {"train_loss": [], "val_dx_f1": [], "val_concept_f1": []}

log.info("=" * 60)
log.info(f"Starting Phase 1 training ({config.NUM_EPOCHS_P1} epochs) …")
log.info("=" * 60)

for epoch in range(config.NUM_EPOCHS_P1):
    model.train()
    epoch_losses = defaultdict(list)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS_P1}")
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        dx_labels      = batch["labels"].to(device)
        concept_labels = batch["concept_labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss, comp = criterion(outputs, dx_labels, concept_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        for k, v in comp.items():
            epoch_losses[k].append(v)
        pbar.set_postfix(loss=f"{comp['total']:.4f}", dx=f"{comp['dx']:.4f}")

    avg_loss = float(np.mean(epoch_losses["total"]))
    log.info(f"Epoch {epoch+1} train loss: total={avg_loss:.4f}  dx={np.mean(epoch_losses['dx']):.4f}  "
             f"align={np.mean(epoch_losses['align']):.4f}  concept={np.mean(epoch_losses['concept']):.4f}")

    # --- Validation ---
    val_metrics = evaluate_phase1(model, val_loader, criterion, device)
    log.info(f"Epoch {epoch+1} val:   dx_f1={val_metrics['dx_f1']:.4f}  concept_f1={val_metrics['concept_f1']:.4f}")

    history["train_loss"].append(avg_loss)
    history["val_dx_f1"].append(val_metrics["dx_f1"])
    history["val_concept_f1"].append(val_metrics["concept_f1"])

    log_metrics("phase1", epoch + 1, {"train_loss": avg_loss, **val_metrics})
    log_memory_usage(device)

    # --- Checkpoint state ---
    ckpt_state = {
        "epoch"               : epoch,
        "model_state_dict"    : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "macro_f1"            : val_metrics["dx_f1"],
        "concept_f1"          : val_metrics["concept_f1"],
        # Phase 1 concept embeddings — saved explicitly for Phase 2 transfer
        "concept_embeddings"  : model.concept_embeddings.data.cpu(),
        "config": {
            "num_concepts" : NUM_CONCEPTS,
            "num_classes"  : NUM_LABELS,
            "fusion_layers": [9, 11],
            "top_50_codes" : TOP_50_CODES,
            "lambda_dx"    : config.LAMBDA_DX,
            "lambda_align" : config.LAMBDA_ALIGN,
            "lambda_concept": config.LAMBDA_CONCEPT,
        },
    }

    # Always save per-epoch snapshot
    save_epoch_checkpoint(ckpt_state, config.CKPT_P1, "phase1", epoch)

    # Save best
    if val_metrics["dx_f1"] > best_f1:
        best_f1 = val_metrics["dx_f1"]
        save_best_checkpoint(ckpt_state, config.P1_BEST_CKPT)
        log.info(f"  New best val dx_f1 = {best_f1:.4f}")

log.info(f"Training done. Best val dx_f1 = {best_f1:.4f}")

# ============================================================================
# SAVE CONCEPT EMBEDDINGS SEPARATELY
# ============================================================================

torch.save(
    {"concept_embeddings": model.concept_embeddings.data.cpu(),
     "num_concepts": NUM_CONCEPTS,
     "concepts": config.GLOBAL_CONCEPTS},
    config.P1_CONCEPT_EMBS,
)
log.info(f"Phase 1 concept embeddings → {config.P1_CONCEPT_EMBS.name}")

# ============================================================================
# FINAL TEST EVALUATION
# ============================================================================

log.info("Loading best model for test evaluation …")
ckpt = torch.load(config.P1_BEST_CKPT, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

all_dx_probs, all_dx_labels, all_c_probs, all_c_labels = [], [], [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test"):
        inp  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out  = model(inp, mask)
        all_dx_probs.append(torch.sigmoid(out["logits"]).cpu().numpy())
        all_dx_labels.append(batch["labels"].numpy())
        all_c_probs.append(out["concept_scores"].cpu().numpy())
        all_c_labels.append(batch["concept_labels"].numpy())

dx_probs  = np.vstack(all_dx_probs)
dx_labels = np.vstack(all_dx_labels)
c_probs   = np.vstack(all_c_probs)
c_labels  = np.vstack(all_c_labels)

dx_preds  = (dx_probs > 0.5).astype(int)

macro_f1  = float(f1_score(dx_labels, dx_preds, average="macro",  zero_division=0))
micro_f1  = float(f1_score(dx_labels, dx_preds, average="micro",  zero_division=0))
precision = float(precision_score(dx_labels, dx_preds, average="macro", zero_division=0))
recall    = float(recall_score(dx_labels, dx_preds, average="macro",    zero_division=0))
concept_f1 = float(f1_score(c_labels, (c_probs > 0.5).astype(int), average="macro", zero_division=0))

per_class_f1 = [float(f1_score(dx_labels[:, i], dx_preds[:, i], zero_division=0))
                for i in range(NUM_LABELS)]

log.info("=" * 60)
log.info("Phase 1 — Test Results (threshold = 0.5)")
log.info(f"  Macro F1   : {macro_f1:.4f}")
log.info(f"  Micro F1   : {micro_f1:.4f}")
log.info(f"  Precision  : {precision:.4f}")
log.info(f"  Recall     : {recall:.4f}")
log.info(f"  Concept F1 : {concept_f1:.4f}")
log.info("=" * 60)

# ============================================================================
# SAVE RESULTS
# ============================================================================

results = {
    "phase"             : "ShifaMind Phase 1",
    "diagnosis_metrics" : {"macro_f1": macro_f1, "micro_f1": micro_f1,
                           "precision": precision, "recall": recall,
                           "per_class_f1": dict(zip(TOP_50_CODES, per_class_f1))},
    "concept_metrics"   : {"concept_f1": concept_f1, "num_concepts": NUM_CONCEPTS},
    "training_history"  : history,
    "dataset_info"      : {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
}

with open(config.P1_RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=2)

per_label_df = pd.DataFrame({
    "icd_code"   : TOP_50_CODES,
    "f1_score"   : per_class_f1,
    "train_count": [top50_info.get("top_50_counts", {}).get(c, 0) for c in TOP_50_CODES],
}).sort_values("f1_score", ascending=False)
per_label_df.to_csv(config.P1_PER_LABEL_CSV, index=False)

log.info(f"Results saved → {config.P1_RESULTS_JSON.name}")
log.info(f"Per-label F1  → {config.P1_PER_LABEL_CSV.name}")
log.info("Phase 1 training complete.")
