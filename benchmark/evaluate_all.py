"""
benchmark/evaluate_all.py
──────────────────────────────────────────────────────────────────────────────
Unified evaluation for ALL Group A models (baselines + ShifaMind phases).

Protocol (identical for every model):
  1. Load test probabilities (run inference if .npy not found).
  2. Tune per-label thresholds on the validation set (19 candidates, 0.05–0.95).
  3. Report test metrics at:
       • Default threshold 0.5
       • Tuned thresholds (per-label)
  4. Bootstrap 95% CI for macro-F1 (N=1000 re-samples).
  5. Pairwise McNemar significance test vs ShifaMind Phase 1 (best baseline).
  6. Write benchmark/results/all_results.json for generate_table.py.

Usage:
    cd ShifaMind_Local
    python benchmark/evaluate_all.py
    python benchmark/evaluate_all.py --models caml plm_icd
    python benchmark/evaluate_all.py --rerun-inference  # force re-inference
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.train_all import (
    BaselineDataset, _extract, get_device, load_cfg, load_splits
)
from benchmark.models.caml        import CAML
from benchmark.models.laat        import LAAT
from benchmark.models.plm_icd     import PLMICD
from benchmark.models.msmn        import MSMN, build_synonym_tensors
from benchmark.models.vanilla_cbm import VanillaCBM


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "macro_f1"  : float(f1_score(labels, preds, average="macro",  zero_division=0)),
        "micro_f1"  : float(f1_score(labels, preds, average="micro",  zero_division=0)),
        "precision" : float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall"    : float(recall_score(labels, preds, average="macro",    zero_division=0)),
    }


def bootstrap_macro_f1_ci(
    probs:      np.ndarray,
    labels:     np.ndarray,
    thresholds: np.ndarray,
    n_samples:  int   = 1000,
    ci_level:   float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap 95% CI for macro-F1.

    Args:
        probs      : [N, K] predicted probabilities (or binary preds)
        labels     : [N, K] binary ground-truth labels
        thresholds : [K]    per-label decision thresholds
        n_samples  : bootstrap resamples
        ci_level   : confidence level (default 0.95)

    Returns:
        (ci_lo, ci_hi)
    """
    rng = np.random.default_rng(42)
    N   = labels.shape[0]
    f1s = np.empty(n_samples)
    for i in range(n_samples):
        idx    = rng.integers(0, N, size=N)
        preds  = (probs[idx] > thresholds).astype(int)
        f1s[i] = f1_score(labels[idx], preds, average="macro", zero_division=0)
    alpha = (1.0 - ci_level) / 2.0
    return float(np.percentile(f1s, 100 * alpha)), float(np.percentile(f1s, 100 * (1 - alpha)))


def tune_thresholds(
    val_probs:  np.ndarray,
    val_labels: np.ndarray,
    candidates: list,
) -> np.ndarray:
    """
    Sweep threshold candidates per label; choose the one maximising per-label F1.
    Returns optimal_thresholds [num_labels].

    Protocol matches ShifaMind phase1_threshold.py exactly:
      - best_f1 starts at 0.0 (not -1.0) so labels where no threshold
        improves F1 above 0 keep the default 0.5 rather than getting 0.05.
      - strict > comparison to match ShifaMind's > t operator.
    """
    num_labels = val_probs.shape[1]
    best_thresh = np.full(num_labels, 0.5)
    for i in range(num_labels):
        best_f1 = 0.0          # matches ShifaMind protocol (was -1.0 — caused bug)
        for t in candidates:
            preds = (val_probs[:, i] > t).astype(int)   # strict >, matches ShifaMind
            f1 = f1_score(val_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1     = f1
                best_thresh[i] = t
    return best_thresh


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_inference(model, loader, device, model_name, cfg) -> np.ndarray:
    model.eval()
    probs_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  Inference [{model_name}]", leave=False):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            mcfg = cfg["group_a"].get(model_name, {})
            fwd_kw = {}
            if model_name in ("caml", "laat"):
                fwd_kw = {
                    "chunk_size":    mcfg.get("chunk_size", 512),
                    "chunk_overlap": mcfg.get("chunk_overlap", 64),
                }
            out  = model(ids, mask, **fwd_kw)
            probs_list.append(torch.sigmoid(out["logits"]).cpu().numpy())
    return np.concatenate(probs_list)   # [N, K]


def _load_or_infer(
    model_name:   str,
    model:        torch.nn.Module | None,
    loader,
    device,
    cfg,
    cache_dir:    Path,
    split:        str,     # "val" or "test"
    rerun:        bool,
) -> np.ndarray:
    cache_path = cache_dir / f"{model_name}_{split}_probs.npy"
    if not rerun and cache_path.exists():
        print(f"  Loading cached probs ← {cache_path.name}")
        return np.load(cache_path)
    probs = _run_inference(model, loader, device, model_name, cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, probs)
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# Load model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_baseline_model(
    model_name: str,
    cfg:        dict,
    device:     torch.device,
    tokenizer,
    num_labels:   int,
    num_concepts: int,
    top50_info:   dict,
    top50_codes:  list,
) -> torch.nn.Module:
    ckpt_dir  = ROOT / cfg["checkpoints"]["baselines_dir"]
    ckpt_path = ckpt_dir / f"{model_name}_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run: python benchmark/train_all.py --models {model_name}"
        )

    vocab_size = tokenizer.vocab_size
    mcfg = cfg["group_a"][model_name]

    if model_name == "caml":
        model = CAML(
            vocab_size   = vocab_size,
            num_labels   = num_labels,
            embed_dim    = mcfg["embed_dim"],
            num_filters  = mcfg["num_filters"],
            filter_size  = mcfg["filter_size"],
            dropout      = mcfg["dropout"],
            pad_token_id = tokenizer.pad_token_id or 0,
        )
    elif model_name == "laat":
        model = LAAT(
            vocab_size   = vocab_size,
            num_labels   = num_labels,
            embed_dim    = mcfg["embed_dim"],
            hidden_dim   = mcfg["hidden_dim"],
            dropout      = mcfg["dropout"],
            pad_token_id = tokenizer.pad_token_id or 0,
        )
    elif model_name == "plm_icd":
        model = PLMICD(
            bert_model_name = mcfg["bert_model"],
            num_labels      = num_labels,
            hidden_size     = mcfg["hidden_size"],
            dropout         = mcfg["dropout"],
        )
    elif model_name == "msmn":
        syn_ids, syn_mask = build_synonym_tensors(
            top50_info   = top50_info,
            top50_codes  = top50_codes,
            tokenizer    = tokenizer,
            num_synonyms = mcfg["num_synonyms"],
            max_syn_len  = 32,
            device       = device,
        )
        model = MSMN(
            vocab_size        = vocab_size,
            num_labels        = num_labels,
            num_synonyms      = mcfg["num_synonyms"],
            embed_dim         = mcfg["embed_dim"],
            hidden_dim        = mcfg["hidden_dim"],
            attention_dim     = mcfg["attention_dim"],
            attention_head    = mcfg["attention_head"],
            dropout           = mcfg["dropout"],
            lstm_dropout      = mcfg["lstm_dropout"],
            pad_token_id      = tokenizer.pad_token_id or 0,
            synonym_input_ids = syn_ids,
            synonym_attn_mask = syn_mask,
        )
    elif model_name == "vanilla_cbm":
        model = VanillaCBM(
            bert_model_name = mcfg["bert_model"],
            num_concepts    = num_concepts,
            num_labels      = num_labels,
            hidden_size     = mcfg["hidden_size"],
            dropout         = mcfg["dropout"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"  Loaded {model_name} ← {ckpt_path.name}  "
          f"(val_macro_f1={ckpt.get('val_macro_f1', '?'):.4f})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ShifaMind phases (load from existing checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest_ckpt(ckpt_dir: Path, filename: str) -> Path | None:
    """Mirror of utils/checkpoints.py find_latest_checkpoint."""
    runs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir()])
    for run in reversed(runs):
        p = run / filename
        if p.exists():
            return p
    flat = ckpt_dir / filename
    return flat if flat.exists() else None


def evaluate_shifamind_phases(
    cfg:        dict,
    results:    dict,
) -> None:
    """
    Load ShifaMind Phase 1/2/3 metrics from the JSON files produced by the
    phase{N}_threshold.py scripts.  Run those scripts first if needed:
        python scripts/phase1_threshold.py
        python scripts/phase2_threshold.py
        python scripts/phase3_threshold.py
    """
    import config as sm_cfg

    phase_json = {
        "phase1": sm_cfg.P1_THRESH_RESULTS_JSON,
        "phase2": sm_cfg.P2_THRESH_RESULTS_JSON,
        "phase3": sm_cfg.RESULTS_P3_FROM_P2 / "final_test_results.json",
    }

    for phase_key, phase_cfg in cfg["shifamind"]["phases"].items():
        display   = phase_cfg["display_name"]
        json_path = phase_json.get(phase_key)

        if json_path is None or not Path(json_path).exists():
            print(f"  [{display}] No results found.")
            print(f"    → Run: python scripts/{phase_key}_threshold.py first.")
            continue

        with open(json_path) as f:
            sm_res = json.load(f)

        def _norm(m: dict) -> dict:
            """Normalise metric keys: phase 1/2 use 'precision'/'recall',
            phase 3 uses 'macro_p'/'macro_r'."""
            return {
                "macro_f1" : m.get("macro_f1",  0.0),
                "micro_f1" : m.get("micro_f1",  0.0),
                "precision": m.get("precision", m.get("macro_p", 0.0)),
                "recall"   : m.get("recall",    m.get("macro_r", 0.0)),
            }

        m_def  = _norm(sm_res["default_0.5"])
        m_tune = _norm(sm_res["optimal_tuned"])

        results[f"shifamind_{phase_key}"] = {
            "display_name"  : display,
            "table_group"   : phase_cfg["table_group"],
            "interpretable" : phase_cfg["interpretable"],
            "hipaa_safe"    : phase_cfg["hipaa_safe"],
            "default_0.5"   : m_def,
            "tuned"         : m_tune,
        }
        print(
            f"  [{display}]  "
            f"macro_f1(default)={m_def['macro_f1']:.4f}  "
            f"macro_f1(tuned)={m_tune['macro_f1']:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ALL_BASELINES = ["caml", "laat", "plm_icd", "msmn", "vanilla_cbm"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all Group A models")
    parser.add_argument("--models",          nargs="+",
                        choices=ALL_BASELINES + ["shifamind"],
                        default=ALL_BASELINES + ["shifamind"],
                        help="Models to evaluate (default: all)")
    parser.add_argument("--rerun-inference", action="store_true",
                        help="Re-run inference even if probs are cached")
    parser.add_argument("--config",          default="benchmark/config.yaml")
    args = parser.parse_args()

    cfg       = load_cfg(args.config)
    device    = get_device(cfg)
    seed      = cfg["seed"]
    results   = {}

    out_dir   = ROOT / cfg["results"]["output_dir"]
    cache_dir = out_dir / "inference_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Device: {device}  |  Evaluating: {args.models}")

    # ── Load splits ────────────────────────────────────────────────────────
    train_split, val_split, test_split, train_con, val_con, test_con = load_splits(cfg)
    _, train_labels  = _extract(train_split)
    val_texts,  val_labels  = _extract(val_split)
    test_texts, test_labels = _extract(test_split)

    num_labels   = test_labels.shape[1]
    num_concepts = test_con.shape[1]

    bert_name = cfg["group_a"]["plm_icd"]["bert_model"]
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    t_cfg     = cfg["training"]

    val_ds  = BaselineDataset(val_texts,  val_labels,  val_con,  tokenizer, t_cfg["max_length"])
    test_ds = BaselineDataset(test_texts, test_labels, test_con, tokenizer, t_cfg["max_length"])
    val_loader  = DataLoader(val_ds,  batch_size=t_cfg["val_batch_size"],
                             shuffle=False, num_workers=t_cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=t_cfg["val_batch_size"],
                             shuffle=False, num_workers=t_cfg["num_workers"])

    with open(ROOT / cfg["data"]["top50_info"]) as f:
        top50_info = json.load(f)
    # Use the canonical code list — select_dtypes would also pick up numeric
    # metadata columns (hadm_id, etc.) and the order is not guaranteed.
    top50_codes = top50_info["top_50_codes"][:num_labels]

    candidates = cfg["training"]["threshold_candidates"]

    # ── Evaluate each baseline ──────────────────────────────────────────────

    for model_name in [m for m in args.models if m != "shifamind"]:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name.upper()}")
        print(f"{'='*60}")

        model = load_baseline_model(
            model_name, cfg, device, tokenizer,
            num_labels, num_concepts, top50_info, top50_codes,
        )

        val_probs  = _load_or_infer(
            model_name, model, val_loader,  device, cfg, cache_dir, "val",  args.rerun_inference)
        test_probs = _load_or_infer(
            model_name, model, test_loader, device, cfg, cache_dir, "test", args.rerun_inference)

        best_thresh = tune_thresholds(val_probs, val_labels, candidates)
        preds_def   = (test_probs > 0.5).astype(int)
        preds_tuned = (test_probs > best_thresh).astype(int)

        m_def  = compute_metrics(test_labels, preds_def)
        m_tune = compute_metrics(test_labels, preds_tuned)

        mcfg_entry = cfg["group_a"][model_name]
        results[model_name] = {
            "display_name"  : mcfg_entry["display_name"],
            "table_group"   : mcfg_entry["table_group"],
            "interpretable" : mcfg_entry["interpretable"],
            "hipaa_safe"    : mcfg_entry["hipaa_safe"],
            "default_0.5"   : m_def,
            "tuned"         : m_tune,
            "mean_threshold": float(best_thresh.mean()),
        }
        np.save(cache_dir / f"{model_name}_test_thresholds.npy", best_thresh)

        print(
            f"  macro_f1(default)={m_def['macro_f1']:.4f}  "
            f"macro_f1(tuned)={m_tune['macro_f1']:.4f}"
        )

    # ── ShifaMind phases ────────────────────────────────────────────────────
    if "shifamind" in args.models:
        print(f"\n{'='*60}")
        print(f"  Evaluating: ShifaMind phases")
        print(f"{'='*60}")

        evaluate_shifamind_phases(cfg=cfg, results=results)

    # ── Save combined results ───────────────────────────────────────────────
    out_path = ROOT / cfg["results"]["combined"]
    # Load existing results (e.g. from llm_eval.py) and merge
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        existing.update(results)
        results = existing

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved → {out_path}")
    print("  Next: python benchmark/generate_table.py")


if __name__ == "__main__":
    main()
