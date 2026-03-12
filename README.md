# ShifaMind Local

Multi-phase clinical NLP pipeline ported from Google Colab to run locally on Apple Silicon (MPS) or CUDA.

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
# Install PyTorch first (picks the right backend for your machine)
pip install torch torchvision torchaudio          # macOS MPS
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # Linux CUDA

# Install torch-geometric (must come after torch)
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

# Install everything else
pip install -r requirements.txt
```

---

## 3. Run the pipeline

All scripts are run from the repo root. Outputs go to `./shifamind_local/`.

```bash
cd ShifaMind_Local

# Phase 1 — train concept bottleneck model
python scripts/phase1_train.py

# Phase 1 — tune classification thresholds
python scripts/phase1_threshold.py

# Phase 2 — train GNN over UMLS knowledge graph
#   (builds the graph on first run; cached to shifamind_local/graph/)
python scripts/phase2_train.py

# Phase 2 — tune thresholds
python scripts/phase2_threshold.py

# Phase 3 — train RAG-augmented diagnosis model
python scripts/phase3_train.py

# Phase 3 — tune thresholds + final evaluation
python scripts/phase3_threshold.py
```

Each script prints its input and output paths at startup so you can verify everything is wired correctly before training begins.

---

## Output layout

```
shifamind_local/
├── shared_data/          # splits, labels, concept list (written by phase1_train)
├── concept_store/        # phase1 + phase2 concept embeddings
├── graph/phase2/         # compiled PyG graph (graph_data.pt)
├── evidence_store/       # RAG corpus (phase3)
├── checkpoints/
│   ├── phase1/
│   ├── phase2/
│   └── phase3/
├── results/
│   ├── phase1/
│   ├── phase2/
│   └── phase3/
├── faiss/                # FAISS index cache
└── logs/                 # metrics.jsonl append log
```

---

---

## 4. Benchmark (Group A baselines + frontier LLMs)

Trains CAML, LAAT, PLM-ICD, MSMN, and Vanilla CBM on the same splits as ShifaMind, then evaluates frontier LLMs zero-shot.

```bash
# Train all Group A baselines (saves checkpoints to benchmark/checkpoints/)
python benchmark/train_all.py

# Evaluate frontier LLMs via OpenRouter (requires OPENROUTER_API_KEY in .env)
python benchmark/llm_eval.py --dry-run   # cost estimate only
python benchmark/llm_eval.py             # run all three LLMs

# Generate the comparison table (LaTeX + CSV)
python benchmark/evaluate_all.py
python benchmark/generate_table.py
```

Outputs go to `benchmark/results/`. Checkpoints are gitignored.

### Frontier LLM setup

All models (GPT-4o, Claude Sonnet 3.7, Gemini 2.5 Pro) are called through a single [OpenRouter](https://openrouter.ai) key — no separate OpenAI / Anthropic / Google accounts required.

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY
```

---

## Notes

- Phases must be run in order (each phase reads outputs from the previous one).
- The UMLS graph build is slow on first run (~minutes depending on drive speed); subsequent runs skip it if `graph_data.pt` already exists.
- Phase 1 threshold tuning reads `shared_data/` and the Phase 1 checkpoint — run `phase1_train.py` first.
- The FAISS index is rebuilt if the cache file doesn't exist and cached for subsequent runs.
