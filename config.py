"""
config.py — Single source of truth for ALL paths and hyperparameters.

Every path in the project is defined here and only here.
No other file should contain hardcoded paths.
"""
import os
from pathlib import Path

# ============================================================================
# HUGGING FACE TOKEN  (optional — removes rate-limit warning on public models)
# Set HF_TOKEN in your shell or in a .env file next to this file.
# ============================================================================
def _load_hf_token() -> None:
    """Read HF_TOKEN from .env if present, then log in to the Hub."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        env_file = Path(__file__).resolve().parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN=") and not line.startswith("#"):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass  # hub login is optional; don't crash if offline

_load_hf_token()

# ============================================================================
# SOURCE DATA  (Google Drive — read-only, never modified by this project)
# ============================================================================

GDRIVE_BASE = Path(
    "/Users/mohammedsameersyed/Library/CloudStorage/"
    "GoogleDrive-sdmohammedsameer@gmail.com/My Drive/ShifaMind"
)

# Original MIMIC run that holds the pre-processed CSV and top-50 info
SOURCE_RUN = GDRIVE_BASE / "10_ShifaMind" / "run_20260102_203225"

# UMLS 2025AA full metathesaurus
UMLS_DIR = (
    GDRIVE_BASE
    / "01_Raw_Datasets"
    / "Extracted"
    / "umls-2025AA-metathesaurus-full"
    / "2025AA"
    / "META"
)

# ── Derived source paths ───────────────────────────────────────────────────────
DATA_CSV        = SOURCE_RUN / "mimic_dx_data_top50.csv"
TOP50_INFO_SRC  = SOURCE_RUN / "shared_data" / "top50_icd10_info.json"
UMLS_MRCONSO    = UMLS_DIR / "MRCONSO.RRF"
UMLS_MRREL      = UMLS_DIR / "MRREL.RRF"

# ============================================================================
# LOCAL OUTPUT ROOT  (all generated files live here)
# ============================================================================

LOCAL = Path(__file__).resolve().parent / "shifamind_local"

# ── Shared data (produced by Phase 1, consumed by all later phases) ────────────
SHARED_DATA            = LOCAL / "shared_data"
TRAIN_SPLIT            = SHARED_DATA / "train_split.pkl"
VAL_SPLIT              = SHARED_DATA / "val_split.pkl"
TEST_SPLIT             = SHARED_DATA / "test_split.pkl"
TRAIN_CONCEPT_LABELS   = SHARED_DATA / "train_concept_labels.npy"
VAL_CONCEPT_LABELS     = SHARED_DATA / "val_concept_labels.npy"
TEST_CONCEPT_LABELS    = SHARED_DATA / "test_concept_labels.npy"
CONCEPT_LIST           = SHARED_DATA / "concept_list.json"
TOP50_INFO_OUT         = SHARED_DATA / "top50_icd10_info.json"

# ── Concept embeddings (saved per phase for reproducibility) ───────────────────
CONCEPT_STORE          = LOCAL / "concept_store"
P1_CONCEPT_EMBS        = CONCEPT_STORE / "phase1_concept_embeddings.pt"
P2_CONCEPT_EMBS        = CONCEPT_STORE / "phase2_concept_embeddings.pt"

# ── Phase 1 ────────────────────────────────────────────────────────────────────
CKPT_P1                = LOCAL / "checkpoints" / "phase1"
P1_BEST_CKPT           = CKPT_P1 / "phase1_best.pt"
# per-epoch: phase1_epoch_{N}.pt  (written by utils/checkpoints.py)

RESULTS_P1             = LOCAL / "results" / "phase1"
P1_RESULTS_JSON        = RESULTS_P1 / "results.json"
P1_PER_LABEL_CSV       = RESULTS_P1 / "per_label_f1.csv"
P1_THRESHOLDS_JSON     = RESULTS_P1 / "optimal_thresholds.json"
P1_THRESH_RESULTS_JSON = RESULTS_P1 / "threshold_tuning_results.json"
P1_THRESH_CSV          = RESULTS_P1 / "threshold_comparison.csv"

# ── Phase 2 ────────────────────────────────────────────────────────────────────
GRAPH_P2               = LOCAL / "graph" / "phase2"
UMLS_GRAPH_GPICKLE     = GRAPH_P2 / "umls_knowledge_graph.gpickle"
GRAPH_DATA_PT          = GRAPH_P2 / "graph_data.pt"

CKPT_P2                = LOCAL / "checkpoints" / "phase2"
P2_BEST_CKPT           = CKPT_P2 / "phase2_best.pt"

RESULTS_P2             = LOCAL / "results" / "phase2"
P2_TRAIN_HISTORY_JSON  = RESULTS_P2 / "training_history.json"
P2_RESULTS_JSON        = RESULTS_P2 / "results.json"
P2_PER_LABEL_CSV       = RESULTS_P2 / "per_label_f1.csv"
P2_THRESHOLDS_JSON     = RESULTS_P2 / "optimal_thresholds.json"
P2_THRESH_RESULTS_JSON = RESULTS_P2 / "threshold_tuning_results.json"
P2_THRESH_CSV          = RESULTS_P2 / "threshold_comparison.csv"

# ── Phase 3 ────────────────────────────────────────────────────────────────────
EVIDENCE_P3            = LOCAL / "evidence_store"
EVIDENCE_CORPUS_JSON   = EVIDENCE_P3 / "evidence_corpus.json"
FAISS_INDEX            = EVIDENCE_P3 / "faiss.index"

CKPT_P3                = LOCAL / "checkpoints" / "phase3"
P3_BEST_CKPT           = CKPT_P3 / "phase3_best.pth"

RESULTS_P3             = LOCAL / "results" / "phase3"
P3_RESULTS_JSON        = RESULTS_P3 / "results.json"
P3_TEST_PREDS_NPY      = RESULTS_P3 / "test_predictions.npy"
P3_TEST_PROBS_NPY      = RESULTS_P3 / "test_probabilities.npy"
P3_TEST_LABELS_NPY     = RESULTS_P3 / "test_labels.npy"
P3_THRESH_JSON         = RESULTS_P3 / "threshold_tuning.json"
P3_FINAL_RESULTS_JSON  = RESULTS_P3 / "final_test_results.json"
P3_PER_DX_JSON         = RESULTS_P3 / "per_diagnosis_metrics.json"

# ── Logs ───────────────────────────────────────────────────────────────────────
LOGS                   = LOCAL / "logs"
LOG_FILE               = LOGS / "shifamind.log"
METRICS_JSONL          = LOGS / "metrics.jsonl"

# ============================================================================
# DIRECTORY BOOTSTRAP
# Every output directory is created here so any script can safely import
# config without worrying about missing dirs.
# ============================================================================

_ALL_OUTPUT_DIRS = [
    SHARED_DATA,
    CONCEPT_STORE,
    CKPT_P1, CKPT_P2, CKPT_P3,
    RESULTS_P1, RESULTS_P2, RESULTS_P3,
    GRAPH_P2,
    EVIDENCE_P3,
    LOGS,
]

for _d in _ALL_OUTPUT_DIRS:
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

SEED = 42
BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# ── Batch sizes  (MPS / Apple M4 Max 32 GB unified memory) ────────────────────
TRAIN_BATCH_SIZE  = 8     # effective batch = 8 × GRAD_ACCUM_STEPS
VAL_BATCH_SIZE    = 16
INFER_BATCH_SIZE  = 32    # inference only (no gradients kept)
NUM_WORKERS       = 0     # MPS requires 0 — spawn start method re-imports the script in workers
PREFETCH_FACTOR   = 2     # only applied when NUM_WORKERS > 0

# ── Optimiser ──────────────────────────────────────────────────────────────────
LEARNING_RATE     = 2e-5
WEIGHT_DECAY      = 0.01
GRAD_ACCUM_STEPS  = 4     # effective batch = 8 × 4 = 32
MAX_GRAD_NORM     = 1.0

# ── Sequence length ────────────────────────────────────────────────────────────
MAX_LENGTH        = 512   # increased from 384 — captures more of long clinical notes

# ── Epochs ─────────────────────────────────────────────────────────────────────
NUM_EPOCHS_P1     = 12    # increased from 7 — val dx_f1 was still rising at epoch 7
NUM_EPOCHS_P2     = 7
NUM_EPOCHS_P3     = 5

# ── Loss weights ───────────────────────────────────────────────────────────────
LAMBDA_DX         = 1.0
LAMBDA_ALIGN      = 0.1   # reduced from 0.5 — 50×111 all-pair alignment adds noise
LAMBDA_CONCEPT    = 0.05  # reduced from 0.3 — keyword concept F1 (~0.09) is too noisy
LAMBDA_DX_P3      = 2.0   # Phase 3 emphasises diagnosis loss

# ── Focal loss (diagnosis head) ────────────────────────────────────────────────
FOCAL_GAMMA       = 2.0   # focusing exponent; 0 = weighted BCE, 2 = standard focal
FOCAL_ALPHA       = 0.75  # positive-class weight; addresses severe label imbalance

# ── Graph (Phase 2) ────────────────────────────────────────────────────────────
GRAPH_HIDDEN_DIM  = 256
GAT_HEADS         = 4
GAT_LAYERS        = 2
GAT_DROPOUT       = 0.3

# ── RAG (Phase 3) ──────────────────────────────────────────────────────────────
RAG_MODEL_NAME        = "sentence-transformers/all-MiniLM-L6-v2"
RAG_TOP_K             = 3
RAG_THRESHOLD         = 0.7
RAG_GATE_MAX          = 0.4    # cap RAG influence at 40 %
PROTOTYPES_PER_DX     = 20
RAG_ENCODE_BATCH_SIZE = 32     # sentence-transformers encode batch (MPS-safe)

# ── Threshold tuning ───────────────────────────────────────────────────────────
THRESHOLD_CANDIDATES = [round(t, 2) for t in [x / 100 for x in range(5, 96, 5)]]

# ============================================================================
# GLOBAL CONCEPT SPACE  (111 unique clinical concepts)
# Duplicates ('fever', 'edema') that existed in the original were removed.
# ============================================================================

GLOBAL_CONCEPTS = [
    # Symptoms
    "fever", "cough", "dyspnea", "pain", "nausea", "vomiting", "diarrhea", "fatigue",
    "headache", "dizziness", "weakness", "confusion", "syncope", "chest", "abdominal",
    "dysphagia", "hemoptysis", "hematuria", "hematemesis", "melena", "jaundice",
    "edema", "rash", "pruritus", "weight", "anorexia", "malaise",
    # Vital signs / physical findings
    "hypotension", "hypertension", "tachycardia", "bradycardia", "tachypnea", "hypoxia",
    "hypothermia", "shock", "altered", "lethargic", "obtunded",
    # Organ systems
    "cardiac", "pulmonary", "renal", "hepatic", "neurologic", "gastrointestinal",
    "respiratory", "cardiovascular", "genitourinary", "musculoskeletal", "endocrine",
    "hematologic", "dermatologic", "psychiatric",
    # Common conditions
    "infection", "sepsis", "pneumonia", "uti", "cellulitis", "meningitis",
    "failure", "infarction", "ischemia", "hemorrhage", "thrombosis", "embolism",
    "obstruction", "perforation", "rupture", "stenosis", "regurgitation",
    "hypertrophy", "atrophy", "neoplasm", "malignancy", "metastasis",
    # Lab / diagnostic
    "elevated", "decreased", "anemia", "leukocytosis", "thrombocytopenia",
    "hyperglycemia", "hypoglycemia", "acidosis", "alkalosis", "hypoxemia",
    "creatinine", "bilirubin", "troponin", "bnp", "lactate", "wbc", "cultures",
    # Imaging / procedures
    "infiltrate", "consolidation", "effusion", "cardiomegaly",
    "ultrasound", "ct", "mri", "xray", "echo", "ekg",
    # Treatments
    "antibiotics", "diuretics", "vasopressors", "insulin", "anticoagulation",
    "oxygen", "ventilation", "dialysis", "transfusion", "surgery",
]

assert len(GLOBAL_CONCEPTS) == 111, (
    f"Expected 111 concepts, got {len(GLOBAL_CONCEPTS)}. "
    "Check for accidental duplicates."
)
