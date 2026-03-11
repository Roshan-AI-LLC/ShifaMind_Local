"""
benchmark/generate_table.py
──────────────────────────────────────────────────────────────────────────────
Read benchmark/results/all_results.json and produce:
  • benchmark/results/comparison_table.tex   — publication-ready LaTeX table
  • benchmark/results/comparison_table.csv   — spreadsheet version

LaTeX table structure:
  Columns: Model | Macro-F1 (tuned) | Micro-F1 | Precision | Recall |
           Interpretable | HIPAA-Safe
  Row groups (midrule-separated):
    1. CNN/Attention Baselines
    2. Transformer Baselines
    3. Ontology/Concept Baselines   ← includes ShifaMind phases
    4. Frontier LLMs†
  Bold the overall best Macro-F1 (Group A only — LLMs are a different subset).
  95% CI shown in parentheses below the F1 value.

Usage:
    cd ShifaMind_Local
    python benchmark/generate_table.py
    python benchmark/generate_table.py --metric macro_f1  # default
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val: float, bold: bool = False, ci: tuple | None = None) -> str:
    """Format a metric value; optionally bold and append CI."""
    s = f"{val:.4f}"
    if bold:
        s = r"\textbf{" + s + "}"
    if ci:
        s += r" \\ \scriptsize{[" + f"{ci[0]:.4f}–{ci[1]:.4f}" + "]}"
    return s


_INTERPRETABLE_MAP = {
    "✓Full"        : r"\checkmark Full",
    "✓Partial"     : r"\checkmark Partial",
    "Attention only": "Attn.",
    "×"            : r"$\times$",
}

_HIPAA_MAP = {
    True : r"\checkmark On-prem",
    False: r"$\times$ PHI\,out",
}

# Row order within each group (controls display order in the table)
_GROUP_ORDER = [
    "CNN/Attention Baselines",
    "Transformer Baselines",
    "Ontology/Concept Baselines",
    "Frontier LLMs",
]

# Key-to-priority within a group (lower = first)
_KEY_PRIORITY = {
    "caml"             : 0,
    "laat"             : 1,
    "plm_icd"          : 0,
    "msmn"             : 1,
    "vanilla_cbm"      : 0,
    "shifamind_phase1" : 1,
    "shifamind_phase2" : 2,
    "shifamind_phase3" : 3,
    "gpt"              : 0,
    "claude"           : 1,
    "gemini"           : 2,
}


def _is_shifamind(key: str) -> bool:
    return key.startswith("shifamind_")


# ─────────────────────────────────────────────────────────────────────────────
# Table builder
# ─────────────────────────────────────────────────────────────────────────────

def build_table(
    results:      dict,
    metric:       str = "macro_f1",
    show_ci:      bool = True,
    show_mcnemar: bool = True,
) -> tuple[str, list[list]]:
    """
    Build LaTeX and CSV representations.

    Returns:
        (latex_str, csv_rows)
    """
    # Organise by group
    groups: dict[str, list[tuple[str, dict]]] = {g: [] for g in _GROUP_ORDER}
    for key, res in results.items():
        group = res.get("table_group", "Ontology/Concept Baselines")
        # Normalise "Frontier LLMs†" → "Frontier LLMs"
        group = group.rstrip("†").strip()
        if group not in groups:
            groups[group] = []
        groups[group].append((key, res))

    # Sort within each group
    for g in groups:
        groups[g].sort(key=lambda x: _KEY_PRIORITY.get(x[0], 99))

    # Find best Group A macro_f1 (for bolding)
    group_a_keys  = [k for k in results if not results[k].get("llm_note")]
    best_f1       = max(
        (results[k]["tuned"].get(metric, 0) for k in group_a_keys),
        default=0.0,
    )

    # ── LaTeX ─────────────────────────────────────────────────────────────
    col_spec = r"l r r r r c c"
    header_cols = [
        "Model",
        r"Macro-F1$\uparrow$",
        r"Micro-F1$\uparrow$",
        r"Precision$\uparrow$",
        r"Recall$\uparrow$",
        "Interpretable",
        "HIPAA-Safe",
    ]

    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{ShifaMind benchmark on MIMIC-III Top-50 ICD-10 diagnosis coding. "
        r"Group A models evaluated on full test set (N=17,266) with per-label "
        r"threshold tuning; 95\% bootstrap CI in brackets. "
        r"Group B LLMs evaluated on a stratified 500-sample subset at fixed "
        r"threshold=0.5 (no tuning). "
        r"\textbf{Bold}: best Group A macro-F1. "
        r"$^{\dagger}$: 500-sample subset; not directly comparable to Group A.}"
    )
    lines.append(r"\label{tab:benchmark}")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header_cols) + r" \\")
    lines.append(r"\midrule")

    csv_rows = [
        ["Model", "Group", "Macro-F1 (tuned)", "CI-lo", "CI-hi",
         "Micro-F1", "Precision", "Recall", "Interpretable", "HIPAA-Safe",
         "McNemar-p-vs-P1"],
    ]

    first_group = True
    for group_name in _GROUP_ORDER:
        rows = groups.get(group_name, [])
        if not rows:
            continue

        if not first_group:
            lines.append(r"\midrule")
        first_group = False

        # Group label row
        is_llm = (group_name == "Frontier LLMs")
        group_display = group_name + (r"$^{\dagger}$" if is_llm else "")
        lines.append(
            r"\multicolumn{7}{l}{\textit{" + group_display + r"}} \\"
        )

        for key, res in rows:
            m_tuned = res.get("tuned", {})
            ci      = res.get("ci_95", {})
            ci_lo   = ci.get("lo", float("nan"))
            ci_hi   = ci.get("hi", float("nan"))

            macro_f1  = m_tuned.get("macro_f1",  float("nan"))
            micro_f1  = m_tuned.get("micro_f1",  float("nan"))
            precision = m_tuned.get("precision", float("nan"))
            recall    = m_tuned.get("recall",    float("nan"))

            is_bold = (not is_llm) and abs(macro_f1 - best_f1) < 1e-6

            display_name = res.get("display_name", key)
            # ShifaMind rows get "(Ours)" annotation
            if _is_shifamind(key):
                display_name += r" \textit{(Ours)}"

            # Significance marker
            p_val = res.get("mcnemar_p_vs_p1")
            sig   = ""
            if show_mcnemar and p_val is not None and not _is_shifamind(key) and key != "shifamind_phase1":
                if p_val < 0.001:
                    sig = r"$^{***}$"
                elif p_val < 0.01:
                    sig = r"$^{**}$"
                elif p_val < 0.05:
                    sig = r"$^{*}$"

            ci_arg = (ci_lo, ci_hi) if show_ci and not is_llm else None
            f1_str = _fmt(macro_f1, bold=is_bold, ci=ci_arg)

            interp = _INTERPRETABLE_MAP.get(
                res.get("interpretable", "×"), res.get("interpretable", "×")
            )
            hipaa = _HIPAA_MAP.get(
                res.get("hipaa_safe", False), r"$\times$"
            )

            row_parts = [
                display_name + sig,
                f1_str,
                f"{micro_f1:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                interp,
                hipaa,
            ]
            lines.append(" & ".join(row_parts) + r" \\")

            csv_rows.append([
                res.get("display_name", key),
                group_name,
                f"{macro_f1:.4f}",
                f"{ci_lo:.4f}",
                f"{ci_hi:.4f}",
                f"{micro_f1:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                res.get("interpretable", "×"),
                str(res.get("hipaa_safe", False)),
                f"{p_val:.4f}" if p_val is not None else "N/A",
            ])

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines), csv_rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison LaTeX table")
    parser.add_argument("--metric",    default="macro_f1",
                        help="Primary ranking metric (default: macro_f1)")
    parser.add_argument("--no-ci",     action="store_true",
                        help="Omit bootstrap CIs from LaTeX table")
    parser.add_argument("--config",    default="benchmark/config.yaml")
    args = parser.parse_args()

    import yaml
    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    combined_path = ROOT / cfg["results"]["combined"]
    if not combined_path.exists():
        print(f"ERROR: {combined_path} not found.")
        print("Run evaluate_all.py (and optionally llm_eval.py) first.")
        sys.exit(1)

    with open(combined_path) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} model results from {combined_path.name}")

    latex_str, csv_rows = build_table(
        results,
        metric   = args.metric,
        show_ci  = not args.no_ci,
    )

    tex_path = ROOT / cfg["results"]["table_tex"]
    csv_path = ROOT / cfg["results"]["table_csv"]

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(latex_str)
    print(f"LaTeX table → {tex_path}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"CSV table   → {csv_path}")

    # ── Console summary ─────────────────────────────────────────────────────
    print("\n  ── Results summary ──────────────────────────────────────────")
    print(f"  {'Model':<35} {'Macro-F1':>10}  {'95% CI':>18}  {'Micro-F1':>10}")
    print(f"  {'-'*35} {'-'*10}  {'-'*18}  {'-'*10}")
    for key, res in sorted(results.items(),
                            key=lambda x: x[1].get("tuned", {}).get("macro_f1", 0),
                            reverse=True):
        name  = res.get("display_name", key)
        f1    = res.get("tuned", {}).get("macro_f1", float("nan"))
        mf1   = res.get("tuned", {}).get("micro_f1", float("nan"))
        ci    = res.get("ci_95", {})
        ci_lo = ci.get("lo", float("nan"))
        ci_hi = ci.get("hi", float("nan"))
        llm_flag = " *" if res.get("llm_note") else ""
        print(f"  {name:<35} {f1:>10.4f}  [{ci_lo:.4f}–{ci_hi:.4f}]  {mf1:>10.4f}{llm_flag}")
    print()
    print("  * LLM results on 500-sample subset — not directly comparable.")


if __name__ == "__main__":
    main()
