# ShifaMind Local

Multi-phase clinical NLP pipeline for ICD-10 diagnosis coding (Top-50 codes).
Runs locally on Apple Silicon (MPS), CUDA, or CPU.

Architecture: BioClinicalBERT → Concept Bottleneck (Phase 1) → UMLS GAT (Phase 2) → RAG (Phase 3).

---

## Prerequisites

- Python 3.11
- Google Drive synced locally (macOS default path assumed in `config.py`)
- Source data from run `10_ShifaMind/run_20260102_203225` available on Google Drive
- UMLS 2025AA full metathesaurus downloaded to `01_Raw_Datasets/Extracted/`

---

## 1. Configure paths

Open `config.py` and update `GDRIVE_BASE` to match your local Google Drive mount path:

```python
GDRIVE_BASE = Path(
    "/Users/YOUR_NAME/Library/CloudStorage/"
    "GoogleDrive-YOUR_EMAIL/My Drive/ShifaMind"
)
```

Everything else is derived from that one constant — no other file needs editing.

---

## 2. Install dependencies

```bash
# PyTorch — pick the right backend for your machine
pip install torch torchvision torchaudio                                    # macOS MPS
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # Linux CUDA

# torch-geometric (must come after torch)
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

# everything else
pip install -r requirements.txt
```

---

## 3. Run ShifaMind

All scripts run from the repo root. Phases must be run **in order** — each phase reads outputs from the previous one.

```bash
cd ShifaMind_Local

# ── Phase 1: Concept Bottleneck ────────────────────────────────────────────
# Trains BioClinicalBERT + concept head + diagnosis head.
# Writes splits, concept labels, and checkpoints on first run.
python scripts/phase1_train.py

# Tune per-label thresholds on val set → evaluate on test set.
python scripts/phase1_threshold.py

# ── Phase 2: UMLS Knowledge Graph (GAT) ───────────────────────────────────
# Builds PyG graph from UMLS on first run (~minutes); cached to shifamind_local/graph/.
# Trains Phase 1 model + GAT encoder end-to-end.
python scripts/phase2_train.py

# Tune thresholds for Phase 2.
python scripts/phase2_threshold.py

# ── Phase 3: RAG-Augmented Diagnosis ──────────────────────────────────────
# Trains Phase 2 model + FAISS-based evidence retrieval + per-diagnosis gate.
# Builds FAISS index on first run; cached to shifamind_local/evidence_store/.
python scripts/phase3_train.py

# Tune thresholds for Phase 3 + full final evaluation.
python scripts/phase3_threshold.py
```

All results are saved under `shifamind_local/results/`.

---

## 4. Run the benchmark

The benchmark trains 5 baselines (CAML, LAAT, PLM-ICD, MSMN, Vanilla CBM) on the **same splits** as ShifaMind, then evaluates frontier LLMs zero-shot and generates a comparison table.

### Full benchmark run order

```bash
cd ShifaMind_Local

# ── Step 1: train all Group A baselines ───────────────────────────────────
# Saves best-val-loss checkpoints to benchmark/checkpoints/
python benchmark/train_all.py

# To train a specific model only:
python benchmark/train_all.py --models caml laat
python benchmark/train_all.py --models plm_icd
python benchmark/train_all.py --models msmn
python benchmark/train_all.py --models vanilla_cbm

# To skip already-trained models:
python benchmark/train_all.py --skip-trained

# ── Step 2: run ShifaMind threshold scripts (if not already done) ──────────
python scripts/phase1_threshold.py
python scripts/phase2_threshold.py
python scripts/phase3_threshold.py

# ── Step 3: evaluate all baselines + ShifaMind (per-label threshold tuning)
# Loads checkpoints, runs inference, tunes thresholds, writes all_results.json
python benchmark/evaluate_all.py

# To re-evaluate specific models (e.g. after retraining):
python benchmark/evaluate_all.py --models caml plm_icd shifamind

# To force re-run inference (ignore cached .npy probs):
python benchmark/evaluate_all.py --rerun-inference

# ── Step 4: frontier LLMs zero-shot (optional) ────────────────────────────
# Requires OPENROUTER_API_KEY in .env  (see .env.example)
python benchmark/llm_eval.py --dry-run   # print cost estimate only
python benchmark/llm_eval.py             # run GPT-5.4, Claude Sonnet 4.6, Gemini 2.5 Pro

# ── Step 5: generate comparison table ────────────────────────────────────
# Reads benchmark/results/all_results.json → writes .tex and .csv
python benchmark/generate_table.py
```

### Frontier LLM setup

All models are called through a single [OpenRouter](https://openrouter.ai) key — no separate API accounts required.

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY
```

---

## Output layout

```
shifamind_local/
├── shared_data/              # train/val/test splits, concept labels, ICD-10 info
│   ├── train_split.pkl
│   ├── val_split.pkl
│   ├── test_split.pkl
│   ├── train/val/test_concept_labels.npy
│   ├── concept_list.json
│   └── top50_icd10_info.json
├── concept_store/            # BERT concept embeddings (phase1 + phase2)
│   ├── phase1_concept_embeddings.pt
│   └── phase2_concept_embeddings.pt
├── graph/phase2/             # compiled PyG graph
│   └── graph_data.pt
├── evidence_store/           # RAG corpus + FAISS index (phase3)
│   ├── evidence_corpus.json
│   └── faiss.index
├── checkpoints/
│   ├── phase1/<run_id>/phase1_best.pt
│   ├── phase2/<run_id>/phase2_best.pt
│   └── phase3_from_p2/<run_id>/phase3_best.pth
├── results/
│   ├── phase1/               # threshold_tuning_results.json, threshold_comparison.csv
│   ├── phase2/               # threshold_tuning_results.json, threshold_comparison.csv
│   └── phase3_from_p2/       # final_test_results.json, per_diagnosis_metrics.*
└── logs/
    ├── shifamind.log
    └── metrics.jsonl

benchmark/
├── checkpoints/              # baseline checkpoints (gitignored)
│   ├── caml_best.pt
│   ├── laat_best.pt
│   ├── plm_icd_best.pt
│   ├── msmn_best.pt
│   └── vanilla_cbm_best.pt
└── results/
    ├── all_results.json      # merged results for all models
    ├── inference_cache/      # cached val/test probs (.npy)
    ├── comparison_table.tex
    └── comparison_table.csv
```

---

## Notes

- **Evaluation protocol**: All models use per-label threshold tuning on the validation set (19 candidates, 0.05–0.95). Only tuned results are reported — no default 0.5 baseline.
- **Speed (MPS)**: All training scripts use `num_workers=0`, `zero_grad(set_to_none=True)`, and `torch.mps.empty_cache()` after every epoch to avoid MPS memory accumulation and progressive slowdown.
- **UMLS graph build**: Slow on first run (~minutes depending on drive speed); subsequent runs skip it if `graph_data.pt` exists.
- **FAISS index**: Rebuilt on first Phase 3 run and cached for subsequent runs.
- **Baselines architecture**: Each baseline matches its original paper exactly — CAML (Mullenbach et al. 2018), LAAT (Vu et al. 2020), PLM-ICD (Huang et al. 2022), MSMN (Yuan et al. 2022), Vanilla CBM (Koh et al. 2020).
- **Checkpoint naming**: ShifaMind uses `phase1_best.pt` / `phase2_best.pt` / `phase3_best.pth`; baselines use `{model_name}_best.pt`.
