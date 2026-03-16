# long_context/ — BioClinical ModernBERT experiment

This folder contains Phase 1/2/3 scripts adapted for **long-context base models**,
starting with **BioClinical ModernBERT-base**.

The original working code (best Phase 1 result: 0.4727 macro-F1) lives in the
project root `scripts/` and `models/`. This folder is for the next attempt.

## Why BioClinical ModernBERT?

| | BioClinicalBERT (current) | BioClinical ModernBERT (this folder) |
|---|---|---|
| Context | 512 tokens | **4096–8192 tokens** |
| Architecture | BERT | RoPE + Flash Attention + GeGLU |
| Pretraining data | MIMIC-III | MIMIC-IV + 20 institutions (53.5B tokens) |
| ICD codes in pretraining | No | **Yes (ICD-9→12, CPT, meds)** |
| Layers | 12 | **22** |

PLM-ICD scores 0.5083 macro-F1. The 512-token context cap is the main reason
we fall short — we read only ~35% of the average MIMIC discharge summary.

Paper: https://arxiv.org/abs/2506.10896
HuggingFace: https://huggingface.co/thomas-sounack/BioClinical-ModernBERT-base

## Key differences vs root scripts

- `config.py` → `BERT_MODEL_NAME`, `MAX_LENGTH=4096`, batch sizes reduced
- `FUSION_LAYERS_P1 = [17, 20]` (equivalent to [9,11] in 12-layer BERT)
- Output dir: `../shifamind_local_lc/` (separate from main runs)
- `models/phase1.py` default fusion layers updated to [17, 20]

## How to run

```bash
cd ShifaMind_Local
python long_context/scripts/phase1_train.py
python long_context/scripts/phase2_train.py
python long_context/scripts/phase3_train.py
```

## Memory notes (Mac MPS)

- `TRAIN_BATCH_SIZE=2`, `GRAD_ACCUM_STEPS=16` → effective batch 32
- 4096 tokens uses ~4–6× memory vs 512 tokens
- If OOM: reduce `MAX_LENGTH` to 2048 first, then batch size
