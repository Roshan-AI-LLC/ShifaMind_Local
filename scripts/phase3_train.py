#!/usr/bin/env python3
"""
scripts/phase3_train.py

Phase 3 training: Phase 2 (or Phase 1) model + gated FAISS RAG augmentation
→ Top-50 ICD-10 multilabel classification.

Supports two base-phase modes:
  --base-phase 2  (default) — starts from Phase 2 best checkpoint (BERT + GAT).
  --base-phase 1            — starts from Phase 1 best checkpoint (BERT only;
                              GAT layers are freshly initialised with graph_scale
                              fixed at −5 so GAT contribution ≈ 0 at start).

Run:
    cd ShifaMind_Local
    python scripts/phase3_train.py              # Phase 2 base (default)
    python scripts/phase3_train.py --base-phase 1   # Phase 1 base

Required inputs (Phase 2 base):
    shifamind_local/shared_data/train_split.pkl
    shifamind_local/shared_data/val_split.pkl
    shifamind_local/shared_data/test_split.pkl
    shifamind_local/shared_data/train_concept_labels.npy
    shifamind_local/shared_data/val_concept_labels.npy
    shifamind_local/shared_data/test_concept_labels.npy
    shifamind_local/shared_data/top50_icd10_info.json
    shifamind_local/concept_store/phase2_concept_embeddings.pt
    shifamind_local/graph/phase2/graph_data.pt
    shifamind_local/checkpoints/phase2/<run_id>/phase2_best.pt

Additional requirements for Phase 1 base (instead of Phase 2 checkpoint):
    shifamind_local/concept_store/phase1_concept_embeddings.pt
    shifamind_local/checkpoints/phase1/<run_id>/phase1_best.pt

Outputs (written to phase-specific subdirs):
    shifamind_local/evidence_store/evidence_corpus.json
    shifamind_local/evidence_store/faiss.index
    shifamind_local/checkpoints/phase3_from_p{N}/<run_id>/phase3_best.pth
    shifamind_local/checkpoints/phase3_from_p{N}/<run_id>/phase3_epoch_N.pth
    shifamind_local/results/phase3_from_p{N}/results.json
    shifamind_local/results/phase3_from_p{N}/test_predictions.npy
    shifamind_local/results/phase3_from_p{N}/test_probabilities.npy
    shifamind_local/results/phase3_from_p{N}/test_labels.npy
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    find_latest_checkpoint, get_logger, load_checkpoint,
    log_memory_usage, log_metrics, save_best_checkpoint,
)

# ============================================================================
# ARGS
# ============================================================================

parser = argparse.ArgumentParser(description="ShifaMind Phase 3 Training")
parser.add_argument(
    "--base-phase", type=int, choices=[1, 2], default=2,
    help="Which phase checkpoint to start from (1 = Phase 1 BERT, 2 = Phase 2 GAT+BERT). "
         "Default: 2",
)
parser.add_argument(
    "--rebuild-corpus", action="store_true",
    help="Force rebuild of the evidence corpus and FAISS index even if cached.",
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

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

log.info("=" * 72)
log.info(f"ShifaMind Phase 3 — Training (base: Phase {BASE_PHASE})")
log.info("=" * 72)
log.info(f"Device     : {device}")
log.info(f"Base phase : {BASE_PHASE}")
log.info(f"RAG encoder: {config.RAG_MODEL_NAME}")
log.info("NOTE: if RAG encoder changed since last run, pass --rebuild-corpus to rebuild FAISS index")

# Resolve phase-specific checkpoint and results directories
run_ckpt_root, RESULTS_DIR = config.get_p3_paths(BASE_PHASE)

# ============================================================================
# LOAD SHARED DATA
# ============================================================================

log.info("Loading splits and concept labels …")
for path, name in [
    (config.TRAIN_SPLIT,     "train_split.pkl"),
    (config.VAL_SPLIT,       "val_split.pkl"),
    (config.TEST_SPLIT,      "test_split.pkl"),
    (config.GRAPH_DATA_PT,   "graph_data.pt"),
    (config.TOP50_INFO_OUT,  "top50_icd10_info.json"),
]:
    if not path.exists():
        log.error(f"Required file not found: {path}")
        log.error(f"  → Run phase2_train.py first to generate shared data.")
        sys.exit(1)

# Concept embeddings depend on base phase
if BASE_PHASE == 2:
    CONCEPT_EMBS_PATH = config.P2_CONCEPT_EMBS
    CONCEPT_PHASE_NAME = "Phase 2"
else:
    CONCEPT_EMBS_PATH = config.P1_CONCEPT_EMBS
    CONCEPT_PHASE_NAME = "Phase 1"

if not CONCEPT_EMBS_PATH.exists():
    log.error(f"{CONCEPT_PHASE_NAME} concept embeddings not found: {CONCEPT_EMBS_PATH}")
    sys.exit(1)

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

log.info(
    f"Dataset — train {len(df_train):,}  val {len(df_val):,}  "
    f"test {len(df_test):,}  labels {NUM_LABELS}  concepts {NUM_CONCEPTS}"
)

# ============================================================================
# CONCEPT-LABEL CO-OCCURRENCE MATRIX
# ============================================================================
# Try Phase 2 checkpoint first → Phase 1 checkpoint → recompute from data.
# Used to:
#   1. Warm-start concept_to_label in ShifaMindPhase2GAT (model constructor)
#   2. Enable co-occurrence alignment loss (LAMBDA_ALIGN_P3)

def _compute_co_occ(cl_arr, df_split, n_labels):
    labels_np  = np.array(df_split["labels"].tolist(), dtype=np.float32)
    cnts       = labels_np.sum(axis=0) + 1e-8
    return (cl_arr.T @ labels_np / cnts).astype(np.float32)

_co_occ_loaded = False
for _ckpt_root, _fname in [
    (config.CKPT_P2, "phase2_best.pt"),
    (config.CKPT_P1, "phase1_best.pt"),
]:
    try:
        _path = find_latest_checkpoint(_ckpt_root, _fname)
        _ckpt = torch.load(_path, map_location="cpu", weights_only=False)
        if "co_occ_matrix" in _ckpt:
            co_occ = _ckpt["co_occ_matrix"].numpy().astype(np.float32)
            log.info(f"Co-occurrence matrix loaded from {_fname}: {co_occ.shape}")
            _co_occ_loaded = True
            break
    except FileNotFoundError:
        pass

if not _co_occ_loaded:
    co_occ = _compute_co_occ(train_cl, df_train, NUM_LABELS)
    log.info(f"Co-occurrence matrix computed from training data: {co_occ.shape}")

co_occ_tensor = torch.tensor(co_occ)

# ============================================================================
# LOAD BIOCLINICALBERT + GRAPH
# ============================================================================

log.info("Loading BioClinicalBERT …")
tokenizer  = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
base_model = AutoModel.from_pretrained(config.BERT_MODEL_NAME).to(device)

log.info("Loading graph data …")
graph_data = torch.load(config.GRAPH_DATA_PT, map_location=device, weights_only=False)

# ============================================================================
# RECONSTRUCT PHASE 2 MODEL ARCHITECTURE
# ============================================================================

log.info("Constructing Phase 2 model architecture …")
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
    co_occ_matrix    = co_occ,    # warm-starts concept_to_label from P(concept|label)
).to(device)

# ============================================================================
# LOAD BASE CHECKPOINT (Phase 1 or Phase 2)
# ============================================================================

if BASE_PHASE == 2:
    # ── Phase 2 base: full model state dict ──────────────────────────────────
    p2_ckpt_path = find_latest_checkpoint(config.CKPT_P2, "phase2_best.pt")
    p2_ckpt      = load_checkpoint(p2_ckpt_path, device)
    phase2_model.load_state_dict(p2_ckpt["model_state_dict"])
    log.info(f"Phase 2 weights loaded (epoch {p2_ckpt.get('epoch', '?') + 1})  ← {p2_ckpt_path.name}")

else:
    # ── Phase 1 base: BERT + concept_head + diagnosis_head from Phase 1 ──────
    #
    # Phase 1 model key layout                   Phase 2 target key
    # ─────────────────────────────────────────   ──────────────────────
    # base_model.embeddings.*                  →  bert.embeddings.*
    # base_model.encoder.layer.*              →  bert.encoder.layer.*
    # concept_head.weight / bias              →  concept_head.weight / bias  (same!)
    # diagnosis_head.weight / bias            →  diagnosis_head.weight / bias (same!)
    # concept_embeddings                       →  (Phase 2 receives externally, skip)
    # fusion_modules.*                         →  (Phase 2 has different arch, skip)
    #
    # Phase 2-specific modules NOT in Phase 1 (will be frozen to avoid
    # random initialisation corrupting Phase 1 quality during Phase 3 training):
    #   concept_fusion, cross_attention, gate_net, layer_norm, graph_proj, gat_encoder
    #
    p1_ckpt_path = find_latest_checkpoint(config.CKPT_P1, "phase1_best.pt")
    p1_ckpt      = load_checkpoint(p1_ckpt_path, device)
    p1_state     = p1_ckpt["model_state_dict"]

    # Build remapped state dict.
    # Phase 1 and Phase 2 now share the same LAAT + concept head keys:
    #   base_model.*           → bert.*
    #   laat_first.*           → laat_first.*    (same key)
    #   laat_second.*          → laat_second.*   (same key)
    #   laat_output.*          → laat_output.*   (same key)
    #   concept_head.*         → concept_head.*  (same key)
    #   concept_to_label.*     → concept_to_label.* (same key)
    # Skipped: concept_embeddings, fusion_modules.* (Phase 1-only)
    _P1_SHARED_PREFIXES = (
        "laat_first.", "laat_second.", "laat_output.",
        "concept_head.", "concept_to_label.",
    )
    p1_mapped = {}
    for k, v in p1_state.items():
        if k.startswith("base_model."):
            p1_mapped[k.replace("base_model.", "bert.", 1)] = v
        elif k.startswith(_P1_SHARED_PREFIXES):
            p1_mapped[k] = v

    missing, unexpected = phase2_model.load_state_dict(p1_mapped, strict=False)
    n_bert  = sum(1 for k in p1_mapped if k.startswith("bert."))
    n_heads = sum(1 for k in p1_mapped if not k.startswith("bert."))
    log.info(f"Phase 1 weights loaded into Phase 2 model  ← {p1_ckpt_path.name}")
    log.info(f"  BERT tensors : {n_bert}")
    log.info(f"  Head tensors : {n_heads}  (laat_first/second/output, concept_head, concept_to_label)")
    log.info(f"  Missing (P2-only GAT layers, will be frozen): {len(missing)}")

    # Keep graph_scale near-zero so freshly-initialised GAT layers don't
    # corrupt the Phase 1 predictions at epoch 0.  sigmoid(-5) ≈ 0.007.
    with torch.no_grad():
        phase2_model.graph_scale.fill_(-5.0)
    log.info("  graph_scale = -5.0  (GAT near-disabled; grows as GAT layers train)")

# ============================================================================
# LOAD CONCEPT EMBEDDINGS  (frozen throughout Phase 3)
# ============================================================================

embs_ckpt         = torch.load(CONCEPT_EMBS_PATH, map_location=device, weights_only=False)
concept_embs_bert = embs_ckpt["concept_embeddings"].to(device).detach()
concept_embs_bert.requires_grad_(False)
log.info(
    f"{CONCEPT_PHASE_NAME} concept embeddings loaded (frozen): "
    f"{tuple(concept_embs_bert.shape)}  ← {CONCEPT_EMBS_PATH.name}"
)

# ============================================================================
# BUILD / LOAD RAG EVIDENCE CORPUS & FAISS INDEX
# ============================================================================

if args.rebuild_corpus and config.EVIDENCE_CORPUS_JSON.exists():
    log.info("--rebuild-corpus: removing cached corpus and index.")
    config.EVIDENCE_CORPUS_JSON.unlink(missing_ok=True)
    config.FAISS_INDEX.unlink(missing_ok=True)

if config.EVIDENCE_CORPUS_JSON.exists():
    log.info(f"Loading cached evidence corpus ← {config.EVIDENCE_CORPUS_JSON.name} …")
    with open(config.EVIDENCE_CORPUS_JSON) as f:
        evidence_corpus = json.load(f)
    # Auto-rebuild if corpus size suggests stale config (e.g. PROTOTYPES_PER_DX changed)
    expected_min = len(TOP_50_CODES) * (1 + 1)  # at least 1 KB + 1 prototype per code
    if len(evidence_corpus) < expected_min:
        log.warning(
            f"Cached corpus has only {len(evidence_corpus)} passages "
            f"(expected ≥ {expected_min}). Rebuilding …"
        )
        evidence_corpus = None
else:
    evidence_corpus = None

if evidence_corpus is None:
    log.info("Building evidence corpus …")
    evidence_corpus = build_evidence_corpus(
        top50_codes=TOP_50_CODES,
        df_train=df_train,
        prototypes_per_dx=config.PROTOTYPES_PER_DX,
        seed=config.SEED,
    )
    config.EVIDENCE_P3.mkdir(parents=True, exist_ok=True)
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
    phase2_model  = phase2_model,
    rag_retriever = rag,
    num_diagnoses = NUM_LABELS,
    hidden_size   = 768,
    concepts_list = config.GLOBAL_CONCEPTS,
).to(device)

total_params    = sum(p.numel() for p in model.parameters())
trainable_all   = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"Phase 3 model: {total_params:,} parameters total (concept_embs frozen separately)")

# ============================================================================
# WARM-START rag_to_logits FROM CORPUS EMBEDDINGS
# ============================================================================
# Problem solved: with zero-mean random init of rag_to_logits, rag_boost
# averages to ~0 across any batch.  Then gate_logit gradient =
#   ∂L/∂logit * rag_boost * sigmoid'(gate_logit) * RAG_GATE_MAX ≈ 0
# so the gate never moves regardless of LR or initialisation.
#
# Fix: set rag_to_logits.weight[i] to the unit direction produced by running
# the mean corpus embedding for diagnosis i through rag_projection.  Now
# rag_boost[i] > 0 for relevant-diagnosis passages from epoch 1, giving the
# gate non-zero gradient so it can actually learn which diagnoses benefit from
# RAG augmentation.

def _warmstart_rag_head() -> None:
    """
    Initialise rag_to_logits from the evidence corpus so that
    rag_boost[i] > 0 for diagnosis i's own passages from step 1.

    Steps
    ─────
    1. Group corpus passages by diagnosis code (TOP_50_CODES order).
    2. Encode up to _WS_PASSAGES passages per code → mean embed [384].
    3. Forward through rag_projection (current random weights) → [768].
    4. L2-normalise → unit direction per diagnosis.
    5. Set rag_to_logits.weight[i]  = unit_hidden[i] * _WS_SCALE.
       Set rag_to_logits.bias[:] = 0.
    """
    _WS_PASSAGES = 10   # corpus passages to average per diagnosis
    _WS_SCALE    = 0.5  # weight norm after init; large enough for real gradient

    log.info("Warm-starting rag_to_logits from corpus mean embeddings …")

    # 1. Group passages
    dx_to_texts: dict = defaultdict(list)
    for doc in evidence_corpus:
        diag = doc.get("diagnosis", "")
        for code in TOP_50_CODES:
            if diag == code or diag.startswith(code) or code.startswith(diag):
                dx_to_texts[code].append(doc["text"])
                break

    # 2. Encode mean embedding per diagnosis (CPU encoder, numpy output)
    mean_embeds = []
    for code in TOP_50_CODES:
        texts = dx_to_texts.get(code, [])[:_WS_PASSAGES]
        if not texts:
            texts = [f"{code} clinical diagnosis treatment management findings"]
        embs = rag.encoder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=len(texts),
        ).astype("float32")
        mean_embeds.append(embs.mean(axis=0))   # [384]

    mean_tensor = torch.tensor(
        np.array(mean_embeds), dtype=torch.float32, device=device
    )  # [num_dx, 384]

    # 3–5. Forward → normalise → set weights
    with torch.no_grad():
        hidden      = torch.nn.functional.relu(model.rag_projection(mean_tensor))  # [num_dx, 768]
        hidden_norm = torch.nn.functional.normalize(hidden, dim=-1)                # unit vectors
        model.rag_to_logits.weight.copy_(hidden_norm * _WS_SCALE)
        model.rag_to_logits.bias.zero_()

    w_norms = model.rag_to_logits.weight.norm(dim=-1)
    log.info(
        f"  rag_to_logits warm-started  "
        f"weight_norm: mean={w_norms.mean():.4f}  min={w_norms.min():.4f}  max={w_norms.max():.4f}"
    )


_warmstart_rag_head()

# ============================================================================
# DATASETS & LOADERS
# ============================================================================

train_ds = RAGDataset(df_train["text"].tolist(), df_train["labels"].tolist(), train_cl, tokenizer)
val_ds   = RAGDataset(df_val["text"].tolist(),   df_val["labels"].tolist(),   val_cl,   tokenizer)
test_ds  = RAGDataset(df_test["text"].tolist(),  df_test["labels"].tolist(),  test_cl,  tokenizer)

train_loader = make_loader(train_ds, config.TRAIN_BATCH_SIZE, shuffle=True)
val_loader   = make_loader(val_ds,   config.VAL_BATCH_SIZE)
test_loader  = make_loader(test_ds,  config.VAL_BATCH_SIZE)

log.info(
    f"Loaders — train {len(train_loader)} batches  "
    f"val {len(val_loader)}  test {len(test_loader)}"
)

# ============================================================================
# TRAINING SETUP
# ============================================================================

# Phase 3 re-enables the co-occurrence alignment loss (LAMBDA_ALIGN_P3=0.03).
# Now that concept embeddings are frozen and the model is stable, this loss
# re-enforces interpretability without fighting active gradient updates.
criterion = MultiObjectiveLoss(
    config.LAMBDA_DX_P3, config.LAMBDA_ALIGN_P3, config.LAMBDA_CONCEPT,
    focal_gamma=config.FOCAL_GAMMA, focal_alpha=config.FOCAL_ALPHA,
    co_occ_matrix=co_occ_tensor,
)

# ── Freeze strategy ───────────────────────────────────────────────────────────
#
# Freeze all Phase 2 layers EXCEPT the final classification heads:
#   laat_output      — maps per-label reps to scalar logits
#   concept_to_label — concept → label affinity gate
#
# These are the "head" layers (analogous to old diagnosis_head) that benefit
# from adapting to the RAG augmented training distribution while BERT, GAT,
# concept_fusion, and cross_attention remain frozen and stable.
#
_P3_TRAINABLE_P2 = ("laat_output.", "concept_to_label.")
for name, param in model.phase2_model.named_parameters():
    if not name.startswith(_P3_TRAINABLE_P2):
        param.requires_grad_(False)

freeze_label = "all Phase 2 except laat_output + concept_to_label"

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"Phase 3 trainable parameters: {trainable:,}  (frozen: {freeze_label})")

# Two-group optimizer: RAG head layers need a much higher LR than BERT.
# BERT is already well-trained (Phase 2); rag_projection / rag_to_logits / rag_gate_logit
# train from random initialisation. Using BERT LR (2e-5) for the RAG head is why
# the gate never moved — the AdamW step for a freshly-initialised linear layer
# needs to be ~50-100× larger than for a converged BERT weight.
rag_param_ids = {
    id(p) for p in [
        model.rag_projection.weight, model.rag_projection.bias,
        model.rag_to_logits.weight,  model.rag_to_logits.bias,
        model.rag_gate_logit,
    ]
}
rag_params   = [p for p in model.parameters() if p.requires_grad and id(p) in rag_param_ids]
other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in rag_param_ids]

log.info(
    f"Optimizer: P2 heads (laat_output, concept_to_label) LR={config.LEARNING_RATE:.0e}  "
    f"RAG head LR={config.RAG_HEAD_LR:.0e}"
)

optimizer = torch.optim.AdamW(
    [
        {"params": other_params, "lr": config.LEARNING_RATE,   "weight_decay": config.WEIGHT_DECAY},
        {"params": rag_params,   "lr": config.RAG_HEAD_LR,     "weight_decay": 0.0},
    ]
)
num_optimizer_steps = (len(train_loader) // config.GRAD_ACCUM_STEPS) * config.NUM_EPOCHS_P3
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(1, num_optimizer_steps // 10),
    num_training_steps=num_optimizer_steps,
)

# ============================================================================
# TRAINING LOOP
# ============================================================================

RUN_ID       = datetime.now().strftime("%Y%m%d_%H%M%S")
run_ckpt_dir = run_ckpt_root / RUN_ID
run_ckpt_dir.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
_BEST_CKPT   = run_ckpt_dir / "phase3_best.pth"

log.info(f"Run ID : {RUN_ID}")
log.info(f"Checkpoints → {run_ckpt_dir}")
log.info(f"Results     → {RESULTS_DIR}")

best_f1 = 0.0
history = {"train_loss": [], "val_macro_f1": [], "val_micro_f1": []}

log.info("=" * 60)
log.info(f"Starting Phase 3 training ({config.NUM_EPOCHS_P3} epochs) …")
log.info("=" * 60)

for epoch in range(config.NUM_EPOCHS_P3):
    model.train()
    epoch_losses = defaultdict(list)

    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS_P3}")
    for step, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        dx_labels      = batch["labels"].to(device)
        concept_labels = batch["concept_labels"].to(device)
        texts          = batch["text"]

        outputs = model(
            input_ids, attention_mask, concept_embs_bert,
            input_texts=texts, use_rag=True,
        )
        loss, comp = criterion(outputs, dx_labels, concept_labels)
        (loss / config.GRAD_ACCUM_STEPS).backward()

        if (step + 1) % config.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        for k, v in comp.items():
            epoch_losses[k].append(v)
        pbar.set_postfix(
            loss=f"{comp['total']:.4f}",
            dx=f"{comp['dx']:.4f}",
            gate=f"{outputs.get('rag_gate', 0.0):.3f}",
        )

    avg_loss = float(np.mean(epoch_losses["total"]))
    log.info(
        f"Epoch {epoch+1} train: total={avg_loss:.4f}  "
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
        f"precision={val_metrics['precision']:.4f}  "
        f"recall={val_metrics['recall']:.4f}  "
        f"loss={val_metrics['loss']:.4f}"
    )

    # Per-diagnosis gate diagnostics — shows whether gate is learning per-label
    with torch.no_grad():
        gate_vals = (torch.sigmoid(model.rag_gate_logit) * config.RAG_GATE_MAX).cpu()
    log.info(
        f"Epoch {epoch+1} gate:  mean={gate_vals.mean():.4f}  "
        f"std={gate_vals.std():.4f}  "
        f"min={gate_vals.min():.4f}  "
        f"max={gate_vals.max():.4f}  "
        f"(target: std>0 means per-dx differentiation is happening)"
    )

    history["train_loss"].append(avg_loss)
    history["val_macro_f1"].append(val_metrics["macro_f1"])
    history["val_micro_f1"].append(val_metrics["micro_f1"])

    log_metrics("phase3", epoch + 1, {"train_loss": avg_loss, **val_metrics})
    log_memory_usage(device)

    # --- Checkpoint (every epoch + track best) ---
    ckpt_state = {
        "epoch"                  : epoch,
        "model_state_dict"       : model.state_dict(),
        "concept_embeddings_bert": concept_embs_bert.cpu(),
        "optimizer_state_dict"   : optimizer.state_dict(),
        "scheduler_state_dict"   : scheduler.state_dict(),
        "macro_f1"               : val_metrics["macro_f1"],
        "base_phase"             : BASE_PHASE,
        "config": {
            "num_concepts"     : NUM_CONCEPTS,
            "num_diagnoses"    : NUM_LABELS,
            "graph_hidden_dim" : config.GRAPH_HIDDEN_DIM,
            "top_50_codes"     : TOP_50_CODES,
            "lambda_dx"        : config.LAMBDA_DX_P3,
            "lambda_align"     : config.LAMBDA_ALIGN_P3,
            "rag_model"        : config.RAG_MODEL_NAME,
            "rag_gate_max"     : config.RAG_GATE_MAX,
            "base_phase"       : BASE_PHASE,
            "architecture"     : "laat+concept_bottleneck+gat+rag",
        },
    }

    # Save every-epoch checkpoint
    epoch_ckpt = run_ckpt_dir / f"phase3_epoch_{epoch+1}.pth"
    torch.save(ckpt_state, epoch_ckpt)

    if val_metrics["macro_f1"] > best_f1:
        best_f1 = val_metrics["macro_f1"]
        save_best_checkpoint(ckpt_state, _BEST_CKPT)
        log.info(f"  *** New best val macro_f1 = {best_f1:.4f} (epoch {epoch+1})")

log.info(f"Training done. Best val macro_f1 = {best_f1:.4f}")

# ============================================================================
# FINAL TEST EVALUATION  (best checkpoint)
# ============================================================================

log.info("Loading best Phase 3 model for test evaluation …")
best_ckpt = load_checkpoint(_BEST_CKPT, device)
model.load_state_dict(best_ckpt["model_state_dict"])
concept_embs_best = best_ckpt["concept_embeddings_bert"].to(device)
model.eval()

all_dx_probs, all_dx_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test inference"):
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
recall    = float(recall_score(dx_labels, dx_preds,    average="macro", zero_division=0))

per_class_f1 = [
    float(f1_score(dx_labels[:, i], dx_preds[:, i], zero_division=0))
    for i in range(NUM_LABELS)
]

log.info("=" * 60)
log.info(f"Phase 3 (base: Phase {BASE_PHASE}) — Test Results (threshold = 0.5)")
log.info(f"  Macro F1   : {macro_f1:.4f}")
log.info(f"  Micro F1   : {micro_f1:.4f}")
log.info(f"  Precision  : {precision:.4f}")
log.info(f"  Recall     : {recall:.4f}")
log.info("=" * 60)

# ============================================================================
# SAVE RESULTS & ARRAYS
# ============================================================================

np.save(RESULTS_DIR / "test_probabilities.npy", dx_probs)
np.save(RESULTS_DIR / "test_predictions.npy",   dx_preds)
np.save(RESULTS_DIR / "test_labels.npy",         dx_labels)

results = {
    "phase"            : f"ShifaMind Phase 3 (base: Phase {BASE_PHASE})",
    "base_phase"       : BASE_PHASE,
    "run_id"           : RUN_ID,
    "best_val_macro_f1": best_f1,
    "diagnosis_metrics": {
        "macro_f1" : macro_f1,  "micro_f1" : micro_f1,
        "precision": precision, "recall"   : recall,
        "per_class_f1": dict(zip(TOP_50_CODES, per_class_f1)),
    },
    "training_history" : history,
    "dataset_info"     : {
        "train": len(df_train), "val": len(df_val), "test": len(df_test)
    },
    "rag_config"       : {
        "top_k"          : config.RAG_TOP_K,
        "threshold"      : config.RAG_THRESHOLD,
        "gate_max"       : config.RAG_GATE_MAX,
        "prototypes_per_dx": config.PROTOTYPES_PER_DX,
        "corpus_size"    : len(evidence_corpus),
    },
}

with open(RESULTS_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

log.info(f"Results saved         → {RESULTS_DIR / 'results.json'}")
log.info(f"Test probabilities    → {RESULTS_DIR / 'test_probabilities.npy'}")
log.info(f"Test predictions      → {RESULTS_DIR / 'test_predictions.npy'}")
log.info(f"Test labels           → {RESULTS_DIR / 'test_labels.npy'}")
log.info(f"Best checkpoint       → {_BEST_CKPT}")
log.info(f"Phase 3 training complete  (base: Phase {BASE_PHASE})")
