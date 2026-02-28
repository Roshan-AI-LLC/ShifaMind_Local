#!/usr/bin/env python3
"""
scripts/phase3_train.py

Phase 3 training: Phase 2 GAT model + gated FAISS RAG augmentation
→ Top-50 ICD-10 multilabel classification.

Requires Phase 2 to have completed first.  The Phase 2 concept
embeddings are loaded once and FROZEN (not updated during Phase 3).
Only the RAG projection/gate layers and the Phase 2 model weights
are fine-tuned.

Run:
    cd ShifaMind_Local
    python scripts/phase3_train.py

Inputs (from Phase 2):
    shifamind_local/shared_data/train_split.pkl
    shifamind_local/shared_data/val_split.pkl
    shifamind_local/shared_data/test_split.pkl
    shifamind_local/shared_data/train_concept_labels.npy
    shifamind_local/shared_data/val_concept_labels.npy
    shifamind_local/shared_data/test_concept_labels.npy
    shifamind_local/shared_data/top50_icd10_info.json
    shifamind_local/concept_store/phase2_concept_embeddings.pt
    shifamind_local/graph/phase2/graph_data.pt
    shifamind_local/checkpoints/phase2/phase2_best.pt

Outputs:
    shifamind_local/evidence_store/evidence_corpus.json
    shifamind_local/evidence_store/faiss.index
    shifamind_local/checkpoints/phase3/phase3_best.pth
    shifamind_local/checkpoints/phase3/phase3_epoch_N.pth
    shifamind_local/results/phase3/results.json
    shifamind_local/results/phase3/test_predictions.npy
    shifamind_local/results/phase3/test_probabilities.npy
    shifamind_local/results/phase3/test_labels.npy
    logs/shifamind.log
    logs/metrics.jsonl
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

import config
from data import RAGDataset, make_loader
from models import GATEncoder, ShifaMindPhase2GAT, ShifaMindPhase3RAG
from rag.retriever import SimpleRAG, build_evidence_corpus
from training import MultiObjectiveLoss
from training.evaluate import evaluate_phase3
from utils import (
    get_logger, load_checkpoint, log_memory_usage, log_metrics,
    save_best_checkpoint, save_epoch_checkpoint,
)

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
log.info("ShifaMind Phase 3 — Training (GAT + FAISS RAG)")
log.info("=" * 72)
log.info(f"Device : {device}")

# ============================================================================
# LOAD SHARED DATA
# ============================================================================

log.info("Loading splits and concept labels from Phase 2 …")
for path, name in [
    (config.TRAIN_SPLIT,       "train_split.pkl"),
    (config.VAL_SPLIT,         "val_split.pkl"),
    (config.TEST_SPLIT,        "test_split.pkl"),
    (config.P2_BEST_CKPT,      "phase2_best.pt"),
    (config.P2_CONCEPT_EMBS,   "phase2_concept_embeddings.pt"),
    (config.GRAPH_DATA_PT,     "graph_data.pt"),
]:
    assert path.exists(), f"{name} not found at {path} — run phase2_train.py first"

with open(config.TRAIN_SPLIT, "rb") as f: df_train = pickle.load(f)
with open(config.VAL_SPLIT,   "rb") as f: df_val   = pickle.load(f)
with open(config.TEST_SPLIT,  "rb") as f: df_test  = pickle.load(f)

train_cl = np.load(config.TRAIN_CONCEPT_LABELS)
val_cl   = np.load(config.VAL_CONCEPT_LABELS)
test_cl  = np.load(config.TEST_CONCEPT_LABELS)

with open(config.TOP50_INFO_OUT) as f:
    top50_info = json.load(f)
TOP_50_CODES = top50_info["top_50_codes"]
NUM_LABELS   = len(TOP_50_CODES)
NUM_CONCEPTS = len(config.GLOBAL_CONCEPTS)

log.info(f"Dataset — train {len(df_train):,}  val {len(df_val):,}  test {len(df_test):,}")

# ============================================================================
# LOAD BIOCLINICALBERT + GRAPH
# ============================================================================

log.info("Loading BioClinicalBERT …")
tokenizer  = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
base_model = AutoModel.from_pretrained(config.BERT_MODEL_NAME).to(device)

log.info("Loading graph data …")
graph_data = torch.load(config.GRAPH_DATA_PT, map_location=device, weights_only=False)

# ============================================================================
# BUILD PHASE 2 MODEL & LOAD WEIGHTS
# ============================================================================

log.info("Reconstructing Phase 2 model and loading best checkpoint …")
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

p2_ckpt = load_checkpoint(config.P2_BEST_CKPT, device)
phase2_model.load_state_dict(p2_ckpt["model_state_dict"])
log.info(f"Phase 2 weights loaded (epoch {p2_ckpt.get('epoch', '?') + 1})")

# Load Phase 2 concept embeddings — FROZEN throughout Phase 3
p2_embs_ckpt       = torch.load(config.P2_CONCEPT_EMBS, map_location=device, weights_only=False)
concept_embs_bert  = p2_embs_ckpt["concept_embeddings"].to(device).detach()
# Freeze: no gradient updates on concept_embs_bert
concept_embs_bert.requires_grad_(False)
log.info(f"Phase 2 concept embeddings loaded (frozen): {tuple(concept_embs_bert.shape)}")

# ============================================================================
# BUILD / LOAD RAG EVIDENCE CORPUS & FAISS INDEX
# ============================================================================

if config.EVIDENCE_CORPUS_JSON.exists():
    log.info(f"Loading cached evidence corpus from {config.EVIDENCE_CORPUS_JSON.name} …")
    with open(config.EVIDENCE_CORPUS_JSON) as f:
        evidence_corpus = json.load(f)
else:
    log.info("Building evidence corpus …")
    evidence_corpus = build_evidence_corpus(
        top50_codes=TOP_50_CODES,
        df_train=df_train,
        prototypes_per_dx=config.PROTOTYPES_PER_DX,
        seed=config.SEED,
    )
    with open(config.EVIDENCE_CORPUS_JSON, "w") as f:
        json.dump(evidence_corpus, f, indent=2)
    log.info(f"Evidence corpus saved → {config.EVIDENCE_CORPUS_JSON.name}")

log.info(f"Evidence corpus: {len(evidence_corpus)} passages")

rag = SimpleRAG(
    model_name=config.RAG_MODEL_NAME,
    top_k=config.RAG_TOP_K,
    threshold=config.RAG_THRESHOLD,
)
rag.build_index(evidence_corpus, index_cache_path=config.FAISS_INDEX)

# ============================================================================
# PHASE 3 MODEL
# ============================================================================

log.info("Building Phase 3 RAG model …")
model = ShifaMindPhase3RAG(
    phase2_model=phase2_model,
    rag_retriever=rag,
    hidden_size=768,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
log.info(f"Phase 3 model: {total_params:,} parameters (concept_embs_bert frozen separately)")

# ============================================================================
# DATASETS & LOADERS
# ============================================================================

train_ds = RAGDataset(df_train["text"].tolist(), df_train["labels"].tolist(), train_cl, tokenizer)
val_ds   = RAGDataset(df_val["text"].tolist(),   df_val["labels"].tolist(),   val_cl,   tokenizer)
test_ds  = RAGDataset(df_test["text"].tolist(),  df_test["labels"].tolist(),  test_cl,  tokenizer)

train_loader = make_loader(train_ds, config.TRAIN_BATCH_SIZE, shuffle=True)
val_loader   = make_loader(val_ds,   config.VAL_BATCH_SIZE)
test_loader  = make_loader(test_ds,  config.VAL_BATCH_SIZE)

log.info(f"Loaders — train {len(train_loader)} batches  val {len(val_loader)}  test {len(test_loader)}")

# ============================================================================
# TRAINING SETUP
# ============================================================================

# LAMBDA_DX_P3 = 2.0 to emphasise diagnosis loss in Phase 3
criterion = MultiObjectiveLoss(config.LAMBDA_DX_P3, config.LAMBDA_ALIGN, config.LAMBDA_CONCEPT)

# Only model parameters — concept_embs_bert is frozen (not passed to optimizer)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_loader) // 2,
    num_training_steps=len(train_loader) * config.NUM_EPOCHS_P3,
)

# ============================================================================
# TRAINING LOOP
# ============================================================================

best_f1 = 0.0
history = {"train_loss": [], "val_macro_f1": [], "val_micro_f1": []}

log.info("=" * 60)
log.info(f"Starting Phase 3 training ({config.NUM_EPOCHS_P3} epochs) …")
log.info("=" * 60)

for epoch in range(config.NUM_EPOCHS_P3):
    model.train()
    epoch_losses = defaultdict(list)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS_P3}")
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        dx_labels      = batch["labels"].to(device)
        concept_labels = batch["concept_labels"].to(device)
        texts          = batch["text"]

        optimizer.zero_grad()
        outputs = model(
            input_ids, attention_mask, concept_embs_bert,
            input_texts=texts, use_rag=True,
        )
        loss, comp = criterion(outputs, dx_labels, concept_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        for k, v in comp.items():
            epoch_losses[k].append(v)
        pbar.set_postfix(loss=f"{comp['total']:.4f}", dx=f"{comp['dx']:.4f}")

    avg_loss = float(np.mean(epoch_losses["total"]))
    log.info(
        f"Epoch {epoch+1} train loss: total={avg_loss:.4f}  "
        f"dx={np.mean(epoch_losses['dx']):.4f}  "
        f"align={np.mean(epoch_losses['align']):.4f}  "
        f"concept={np.mean(epoch_losses['concept']):.4f}"
    )

    # --- Validation ---
    val_metrics = evaluate_phase3(
        model, val_loader, criterion, device,
        concept_embeddings=concept_embs_bert, use_rag=True,
    )
    log.info(
        f"Epoch {epoch+1} val:   macro_f1={val_metrics['macro_f1']:.4f}  "
        f"micro_f1={val_metrics['micro_f1']:.4f}  "
        f"loss={val_metrics['loss']:.4f}"
    )

    history["train_loss"].append(avg_loss)
    history["val_macro_f1"].append(val_metrics["macro_f1"])
    history["val_micro_f1"].append(val_metrics["micro_f1"])

    log_metrics("phase3", epoch + 1, {"train_loss": avg_loss, **val_metrics})
    log_memory_usage(device)

    # --- Checkpoint ---
    ckpt_state = {
        "epoch"                  : epoch,
        "model_state_dict"       : model.state_dict(),
        "concept_embeddings_bert": concept_embs_bert.cpu(),
        "optimizer_state_dict"   : optimizer.state_dict(),
        "scheduler_state_dict"   : scheduler.state_dict(),
        "macro_f1"               : val_metrics["macro_f1"],
        "config": {
            "num_concepts"     : NUM_CONCEPTS,
            "num_diagnoses"    : NUM_LABELS,
            "graph_hidden_dim" : config.GRAPH_HIDDEN_DIM,
            "top_50_codes"     : TOP_50_CODES,
            "lambda_dx"        : config.LAMBDA_DX_P3,
        },
    }

    save_epoch_checkpoint(ckpt_state, config.CKPT_P3, "phase3", epoch)

    if val_metrics["macro_f1"] > best_f1:
        best_f1 = val_metrics["macro_f1"]
        save_best_checkpoint(ckpt_state, config.P3_BEST_CKPT)
        log.info(f"  New best val macro_f1 = {best_f1:.4f}")

log.info(f"Training done. Best val macro_f1 = {best_f1:.4f}")

# ============================================================================
# FINAL TEST EVALUATION
# ============================================================================

log.info("Loading best Phase 3 model for test evaluation …")
best_ckpt = load_checkpoint(config.P3_BEST_CKPT, device)
model.load_state_dict(best_ckpt["model_state_dict"])
concept_embs_best = best_ckpt["concept_embeddings_bert"].to(device)
model.eval()

all_dx_probs, all_dx_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test"):
        inp   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        texts = batch["text"]
        out   = model(inp, mask, concept_embs_best, input_texts=texts, use_rag=True)
        all_dx_probs.append(torch.sigmoid(out["logits"]).cpu().numpy())
        all_dx_labels.append(batch["labels"].numpy())

dx_probs  = np.vstack(all_dx_probs)
dx_labels = np.vstack(all_dx_labels)
dx_preds  = (dx_probs > 0.5).astype(int)

macro_f1  = float(f1_score(dx_labels, dx_preds, average="macro",  zero_division=0))
micro_f1  = float(f1_score(dx_labels, dx_preds, average="micro",  zero_division=0))
precision = float(precision_score(dx_labels, dx_preds, average="macro", zero_division=0))
recall    = float(recall_score(dx_labels, dx_preds, average="macro",    zero_division=0))

per_class_f1 = [
    float(f1_score(dx_labels[:, i], dx_preds[:, i], zero_division=0))
    for i in range(NUM_LABELS)
]

log.info("=" * 60)
log.info("Phase 3 — Test Results (threshold = 0.5)")
log.info(f"  Macro F1   : {macro_f1:.4f}")
log.info(f"  Micro F1   : {micro_f1:.4f}")
log.info(f"  Precision  : {precision:.4f}")
log.info(f"  Recall     : {recall:.4f}")
log.info("=" * 60)

# ============================================================================
# SAVE RESULTS & ARRAYS
# ============================================================================

np.save(config.P3_TEST_PROBS_NPY,  dx_probs)
np.save(config.P3_TEST_PREDS_NPY,  dx_preds)
np.save(config.P3_TEST_LABELS_NPY, dx_labels)

results = {
    "phase"            : "ShifaMind Phase 3",
    "diagnosis_metrics": {
        "macro_f1" : macro_f1,  "micro_f1" : micro_f1,
        "precision": precision, "recall"   : recall,
        "per_class_f1": dict(zip(TOP_50_CODES, per_class_f1)),
    },
    "training_history" : history,
    "dataset_info"     : {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
}

with open(config.P3_RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=2)

log.info(f"Results saved         → {config.P3_RESULTS_JSON.name}")
log.info(f"Test probabilities    → {config.P3_TEST_PROBS_NPY.name}")
log.info(f"Test predictions      → {config.P3_TEST_PREDS_NPY.name}")
log.info(f"Test labels           → {config.P3_TEST_LABELS_NPY.name}")
log.info("Phase 3 training complete.")
