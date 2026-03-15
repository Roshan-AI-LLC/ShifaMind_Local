"""
benchmark/llm_eval.py
──────────────────────────────────────────────────────────────────────────────
Zero-shot evaluation of frontier LLMs on 500 stratified test samples.

Protocol:
  • 500 samples stratified by most-frequent codes, seed=42 (reproducible).
  • System prompt: expert ICD-10 coder, predict from fixed Top-50 list only.
  • User prompt: discharge note truncated to max_input_chars characters.
  • Expected output: JSON {"predicted_codes": ["I5032", ...]}
  • Parse failures → retry up to 3×, then log failure → treat as all-zeros.
  • Evaluation: fixed threshold 0.5 (no tuning — clearly noted in table).
  • Cost estimate printed before any API call.

API key (read from .env in project root or environment variable):
  OPENROUTER_API_KEY   — single key for all models via openrouter.ai

All three models (GPT, Claude, Gemini) are called through OpenRouter's
OpenAI-compatible endpoint so only one API key is needed.

Usage:
    cd ShifaMind_Local
    python benchmark/llm_eval.py --dry-run      # show cost estimate only
    python benchmark/llm_eval.py                # run all 3 LLMs
    python benchmark/llm_eval.py --models claude # run one LLM
    python benchmark/llm_eval.py --resume       # skip already-cached samples

Outputs:
  benchmark/results/llm_cache.json   — per-sample predictions (resumable)
  benchmark/results/all_results.json — merged with evaluate_all.py results
"""
import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.train_all import _extract, load_cfg, load_splits
from benchmark.evaluate_all import compute_metrics, bootstrap_macro_f1_ci


# ─────────────────────────────────────────────────────────────────────────────
# .env loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_env() -> None:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()


# ─────────────────────────────────────────────────────────────────────────────
# Stratified sampling
# ─────────────────────────────────────────────────────────────────────────────

def stratified_sample(
    texts:      list,
    labels:     np.ndarray,
    n:          int,
    seed:       int = 42,
) -> tuple[list, np.ndarray, np.ndarray]:
    """
    Return n indices stratified to cover all labels as evenly as possible.
    Uses iterative label-balanced selection (greedy coverage).

    Returns: (sampled_texts, sampled_labels, original_indices)
    """
    rng       = np.random.default_rng(seed)
    N         = len(texts)
    num_labels = labels.shape[1]

    # Shuffle first for randomness within strata
    perm = rng.permutation(N)
    texts_p  = [texts[i] for i in perm]
    labels_p = labels[perm]

    selected   = []
    label_count = np.zeros(num_labels, dtype=int)

    # Priority: samples that add the most uncovered labels
    for i in range(N):
        if len(selected) >= n:
            break
        row_labels = labels_p[i].astype(bool)
        gain = (row_labels & (label_count < (n // num_labels + 1))).sum()
        if gain > 0 or len(selected) < n:
            selected.append(i)
            label_count += row_labels.astype(int)

    # Fill to n if not enough high-gain samples
    remaining_pool = [i for i in range(N) if i not in selected]
    while len(selected) < n and remaining_pool:
        selected.append(remaining_pool.pop(rng.integers(len(remaining_pool))))

    selected = selected[:n]
    orig_indices = perm[selected]
    return [texts_p[i] for i in selected], labels_p[selected], orig_indices


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert medical coder specialising in ICD-10-CM coding for hospital \
discharge summaries. Your task is to predict which ICD-10 diagnosis codes from \
a fixed list of 50 codes apply to the clinical note below.

Rules:
1. Only output codes from the provided list. Do NOT invent codes.
2. Predict all applicable codes (multi-label — a note can have many codes).
3. Output ONLY valid JSON in this exact format, no explanation:
   {"predicted_codes": ["I5032", "E1165", ...]}
4. If no code applies, output: {"predicted_codes": []}"""


def make_user_prompt(note: str, top50_codes: list, max_chars: int) -> str:
    code_list_str = ", ".join(top50_codes)
    note_trunc    = note[:max_chars]
    return (
        f"Available ICD-10 codes (predict only from this list):\n{code_list_str}\n\n"
        f"Clinical note:\n{note_trunc}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_llm_response(response_text: str, top50_codes: list) -> list[int]:
    """
    Parse LLM JSON response → binary vector [num_labels].
    Returns list of 0/1 integers.

    Handles:
      - Clean JSON: {"predicted_codes": ["I5032", ...]}
      - JSON embedded in markdown: ```json ... ```
      - Partial match: strips whitespace, normalises code case
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", response_text).strip()

    try:
        parsed = json.loads(text)
        codes  = parsed.get("predicted_codes", [])
    except json.JSONDecodeError:
        # Try to extract a list anywhere in the response
        m = re.search(r'\[([^\]]*)\]', text)
        if m:
            raw = m.group(0)
            try:
                codes = json.loads(raw)
            except Exception:
                codes = re.findall(r'[A-Z][0-9]{2,4}[\w]*', text)
        else:
            codes = re.findall(r'[A-Z][0-9]{2,4}[\w]*', text)

    # Normalise: uppercase, strip spaces AND dots.
    # Gemini returns ICD-10 codes in clinical dot-notation ("I50.32", "E11.65")
    # but our top50_codes list uses the raw format ("I5032", "E1165").
    codes_norm = [str(c).strip().upper().replace(".", "") for c in codes if isinstance(c, str)]
    top50_upper = [c.upper() for c in top50_codes]

    vec = [int(c in codes_norm) for c in top50_upper]
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# API caller — all models via OpenRouter (single key, OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def call_openrouter(
    model_id:     str,
    system:       str,
    user:         str,
    max_tokens:   int,
    max_retries:  int,
    retry_delay:  float,
) -> str:
    """
    Call any model through OpenRouter's OpenAI-compatible endpoint.

    Set OPENROUTER_API_KEY in .env.  Model IDs use OpenRouter's format:
      openai/gpt-4o, anthropic/claude-3-7-sonnet-20250219,
      google/gemini-2.5-pro-preview, etc.
    Full model list: https://openrouter.ai/models
    """
    import openai
    client = openai.OpenAI(
        base_url   = "https://openrouter.ai/api/v1",
        api_key    = os.environ.get("OPENROUTER_API_KEY", ""),
        default_headers = {
            "HTTP-Referer": "https://github.com/Roshan-AI-LLC/ShifaMind_Local",
            "X-Title": "ShifaMind Benchmark",
        },
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model          = model_id,
                messages       = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens     = max_tokens,
                temperature    = 0.0,
                # Forces valid JSON output — OpenRouter passes this to Gemini/GPT/Claude.
                # Prevents partial/malformed responses that cause parse failures.
                # Models that don't support it ignore it silently (OpenRouter behaviour).
                response_format = {"type": "json_object"},
            )
            # Guard: resp.choices can be None or [] when Gemini's safety filter
            # triggers or OpenRouter returns a non-standard error response.
            if not resp or not resp.choices:
                raise ValueError(
                    f"Empty choices in response "
                    f"(finish_reason={getattr(resp, 'choices', None)})"
                )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"    OpenRouter error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
    return ""


_CALLERS = {
    "openrouter": call_openrouter,
    # Legacy aliases so existing config.yaml provider values still work
    "openai"    : call_openrouter,
    "anthropic" : call_openrouter,
    "google"    : call_openrouter,
}


# ─────────────────────────────────────────────────────────────────────────────
# Cost estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_cost(
    n_samples:        int,
    max_input_chars:  int,
    max_tokens_out:   int,
    model_cfgs:       dict,
) -> dict:
    """
    Returns {model_key: estimated_usd}.
    Rough approximation: 1 token ≈ 4 characters.
    """
    approx_in_tokens  = max_input_chars // 4 + 200   # +200 for system + code list
    approx_out_tokens = max_tokens_out
    costs = {}
    for key, mc in model_cfgs.items():
        cost_in  = (approx_in_tokens  / 1_000_000) * mc["cost_per_1m_input"]  * n_samples
        cost_out = (approx_out_tokens / 1_000_000) * mc["cost_per_1m_output"] * n_samples
        costs[key] = cost_in + cost_out
    return costs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frontier LLMs zero-shot")
    parser.add_argument("--models",   nargs="+",
                        choices=["gpt", "claude", "gemini"],
                        default=["gpt", "claude", "gemini"])
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print cost estimate and exit without calling APIs")
    parser.add_argument("--resume",   action="store_true",
                        help="Resume from cache — skip already-evaluated samples")
    parser.add_argument("--clear-model", nargs="+",
                        choices=["gpt", "claude", "gemini"],
                        help="Wipe cached results for these models before running "
                             "(use when a previous run cached bad/zero results). "
                             "GPT/Claude entries in all_results.json are preserved.")
    parser.add_argument("--config",   default="benchmark/config.yaml")
    args = parser.parse_args()

    cfg      = load_cfg(args.config)
    llm_cfg  = cfg["llm_eval"]
    n        = llm_cfg["n_samples"]
    seed     = llm_cfg["seed"]
    max_chars = llm_cfg["max_input_chars"]
    max_tok  = llm_cfg["max_tokens"]
    retries  = llm_cfg["max_retries"]
    delay    = llm_cfg["retry_delay_s"]

    out_dir  = ROOT / cfg["results"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path  = ROOT / cfg["results"]["llm_cache"]
    combined_path = ROOT / cfg["results"]["combined"]

    # ── Cost estimate ──────────────────────────────────────────────────────
    model_cfgs = {k: v for k, v in llm_cfg["models"].items() if k in args.models}
    costs = estimate_cost(n, max_chars, max_tok, model_cfgs)
    total = sum(costs.values())
    print("\n  ── Estimated API cost ───────────────────────────────────")
    for k, c in costs.items():
        print(f"    {llm_cfg['models'][k]['display_name']:<25} ${c:.2f}")
    print(f"    {'TOTAL':<25} ${total:.2f}")
    print(f"  {'(500 samples, ~2000 token notes, JSON output)'}")
    print()

    if args.dry_run:
        print("  --dry-run: exiting without API calls.")
        return

    if total > 25.0:
        print(f"  WARNING: estimated cost ${total:.2f} exceeds $25. Proceed? [y/N] ", end="")
        if input().strip().lower() != "y":
            print("  Aborted.")
            return

    # ── Load test data ────────────────────────────────────────────────────
    print("Loading test split …")
    _, _, test_split, _, _, _ = load_splits(cfg)
    test_texts, test_labels   = _extract(test_split)

    with open(ROOT / cfg["data"]["top50_info"]) as f:
        top50_info = json.load(f)
    # Use the canonical code list from top50_info — do NOT derive from DataFrame
    # columns, which also contain "labels" (list column), metadata columns, etc.
    top50_codes = top50_info["top_50_codes"]

    # ── Stratified 500-sample subset ──────────────────────────────────────
    sample_texts, sample_labels, sample_idx = stratified_sample(
        test_texts, test_labels, n=n, seed=seed
    )
    print(f"Sampled {n} test examples  "
          f"(label coverage: {(sample_labels.sum(0) > 0).sum()}/{test_labels.shape[1]} codes)")

    # ── Load / init cache ─────────────────────────────────────────────────
    cache: dict = {}
    if args.resume and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Resuming from cache ({sum(len(v) for v in cache.values())} existing predictions)")

    # ── Clear stale model cache if requested ──────────────────────────────────
    if args.clear_model:
        for mk in args.clear_model:
            if mk in cache:
                del cache[mk]
                print(f"  Cache cleared for {mk} (will re-run all 500 samples)")
        with open(cache_path, "w") as f:
            json.dump(cache, f)

    # ── Run each LLM ──────────────────────────────────────────────────────
    all_results = {}
    if combined_path.exists():
        with open(combined_path) as f:
            all_results = json.load(f)

    for model_key in args.models:
        mc      = llm_cfg["models"][model_key]
        mid     = mc["model_id"]
        prov    = mc["provider"]
        display = mc["display_name"]
        caller  = _CALLERS[prov]

        print(f"\n{'='*60}")
        print(f"  Evaluating: {display}  ({mid})")
        print(f"{'='*60}")

        model_cache = cache.get(model_key, {})   # {str(idx): [0/1 vector]}
        failures    = 0

        for i, (text, label_row) in enumerate(zip(sample_texts, sample_labels)):
            cache_key = str(sample_idx[i])
            if args.resume and cache_key in model_cache:
                continue

            user_prompt = make_user_prompt(text, top50_codes, max_chars)
            response    = caller(mid, _SYSTEM_PROMPT, user_prompt, max_tok, retries, delay)

            if response:
                vec = parse_llm_response(response, top50_codes)
            else:
                vec = [0] * len(top50_codes)
                failures += 1
                print(f"    [!] Sample {i} failed (all-zeros fallback)")

            model_cache[cache_key] = vec

            if (i + 1) % 50 == 0:
                cache[model_key] = model_cache
                with open(cache_path, "w") as f:
                    json.dump(cache, f)
                print(f"  [{display}] {i+1}/{n}  failures={failures}")

        cache[model_key] = model_cache
        with open(cache_path, "w") as f:
            json.dump(cache, f)

        # ── Metrics ────────────────────────────────────────────────────────
        preds_list = []
        for idx in sample_idx:
            vec = model_cache.get(str(idx), [0] * len(top50_codes))
            preds_list.append(vec)
        preds  = np.array(preds_list, dtype=int)
        labels = sample_labels.astype(int)

        m = compute_metrics(labels, preds)
        ci_lo, ci_hi = bootstrap_macro_f1_ci(
            preds.astype(float), labels.astype(float),
            np.full(len(top50_codes), 0.5),
            n_samples=cfg["bootstrap"]["n_samples"],
            ci_level=cfg["bootstrap"]["ci_level"],
        )

        all_results[model_key] = {
            "display_name"  : display,
            "table_group"   : mc["table_group"],
            "interpretable" : mc["interpretable"],
            "hipaa_safe"    : mc["hipaa_safe"],
            "default_0.5"   : m,
            "tuned"         : m,   # no tuning for LLMs — same as default
            "ci_95"         : {"lo": ci_lo, "hi": ci_hi},
            "mean_threshold": 0.5,
            "llm_note"      : f"500-sample stratified subset, fixed threshold=0.5",
            "parse_failures": failures,
            "n_samples"     : n,
        }

        print(
            f"  [{display}]  "
            f"macro_f1={m['macro_f1']:.4f}  "
            f"micro_f1={m['micro_f1']:.4f}  "
            f"95%CI=[{ci_lo:.4f},{ci_hi:.4f}]  "
            f"failures={failures}/{n}"
        )

    # ── Save combined results ──────────────────────────────────────────────
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  LLM results merged → {combined_path}")
    print("  Next: python benchmark/generate_table.py")


if __name__ == "__main__":
    main()
