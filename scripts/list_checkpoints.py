#!/usr/bin/env python3
"""
scripts/list_checkpoints.py

Utility: scan all checkpoint directories and print a table of every
saved checkpoint with its key metrics.  Run this before Phase 3 to
quickly identify the best Phase 1 / Phase 2 checkpoint to build on.

Run:
    cd ShifaMind_Local
    python scripts/list_checkpoints.py
    python scripts/list_checkpoints.py --phase 2       # Phase 2 only
    python scripts/list_checkpoints.py --phase 3 --base 1   # P3-from-P1 only
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import config
from utils.logging_utils import get_logger

log = get_logger()


def _safe_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"_load_error": str(e)}


def _fmt(v) -> str:
    if v is None:
        return "  —  "
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def _scan_phase_dir(phase_label: str, ckpt_dir: Path, filename: str) -> list:
    """
    Scan ckpt_dir for run subdirs containing *filename*.
    Returns list of row dicts sorted by run_id (newest last).
    """
    rows = []
    candidates = sorted(ckpt_dir.glob(f"*/{filename}"))

    # Also check legacy flat path
    flat = ckpt_dir / filename
    if flat.exists() and flat not in [c for c in candidates]:
        candidates.append(flat)

    for ckpt_path in candidates:
        ckpt = _safe_load(ckpt_path)
        if "_load_error" in ckpt:
            row = {
                "phase"    : phase_label,
                "run_id"   : ckpt_path.parent.name,
                "path"     : str(ckpt_path),
                "epoch"    : "ERR",
                "macro_f1" : None,
                "micro_f1" : None,
                "note"     : ckpt["_load_error"][:60],
            }
        else:
            epoch    = ckpt.get("epoch", None)
            macro_f1 = (
                ckpt.get("macro_f1")
                or ckpt.get("val_macro_f1")
                or (ckpt.get("val_metrics") or {}).get("macro_f1")
            )
            # For phase1/2 checkpoints the metric is stored differently
            if macro_f1 is None:
                macro_f1 = ckpt.get("best_val_f1") or ckpt.get("dx_f1")

            # Check if there's a results.json next to it
            results_json = ckpt_path.parent.parent.parent / "results" / phase_label.lower().replace(" ", "") / "results.json"
            micro_f1 = None
            if results_json.exists():
                try:
                    with open(results_json) as f:
                        res = json.load(f)
                    diag = res.get("diagnosis_metrics", {})
                    if micro_f1 is None:
                        micro_f1 = diag.get("micro_f1")
                    if macro_f1 is None:
                        macro_f1 = diag.get("macro_f1")
                except Exception:
                    pass

            base_phase = ckpt.get("base_phase", ckpt.get("config", {}).get("base_phase", "—"))
            row = {
                "phase"     : phase_label,
                "run_id"    : ckpt_path.parent.name if ckpt_path.parent != ckpt_dir else "legacy",
                "path"      : str(ckpt_path.relative_to(config.LOCAL)),
                "epoch"     : (epoch + 1) if epoch is not None else "?",
                "macro_f1"  : macro_f1,
                "base_phase": base_phase,
            }
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description="List ShifaMind checkpoints with metrics")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None,
                        help="Filter to a specific phase (default: all)")
    parser.add_argument("--base", type=int, choices=[1, 2], default=None,
                        help="For Phase 3: filter by base phase")
    args = parser.parse_args()

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    phase1_rows = _scan_phase_dir("phase1", config.CKPT_P1, "phase1_best.pt")
    # ── Phase 2 ───────────────────────────────────────────────────────────────
    phase2_rows = _scan_phase_dir("phase2", config.CKPT_P2, "phase2_best.pt")
    # ── Phase 3 (from P2) ─────────────────────────────────────────────────────
    phase3_p2_rows = _scan_phase_dir("phase3/p2", config.CKPT_P3_FROM_P2, "phase3_best.pth")
    # ── Phase 3 (from P1) ─────────────────────────────────────────────────────
    phase3_p1_rows = _scan_phase_dir("phase3/p1", config.CKPT_P3_FROM_P1, "phase3_best.pth")
    # ── Legacy Phase 3 ────────────────────────────────────────────────────────
    phase3_leg_rows = _scan_phase_dir("phase3/legacy", config.CKPT_P3, "phase3_best.pth")

    all_rows = phase1_rows + phase2_rows + phase3_p2_rows + phase3_p1_rows + phase3_leg_rows

    if args.phase == 1:
        all_rows = phase1_rows
    elif args.phase == 2:
        all_rows = phase2_rows
    elif args.phase == 3:
        all_rows = phase3_p2_rows + phase3_p1_rows + phase3_leg_rows
        if args.base == 1:
            all_rows = phase3_p1_rows
        elif args.base == 2:
            all_rows = phase3_p2_rows

    if not all_rows:
        print("\nNo checkpoints found.  Run training scripts first.\n")
        return

    # ── Print table ───────────────────────────────────────────────────────────
    col_w = {"phase": 14, "run_id": 18, "epoch": 6, "macro_f1": 10, "base_phase": 10, "path": 60}
    header = (
        f"{'Phase':{col_w['phase']}}  "
        f"{'Run ID':{col_w['run_id']}}  "
        f"{'Ep':>{col_w['epoch']}}  "
        f"{'MacroF1':>{col_w['macro_f1']}}  "
        f"{'BasePh':>{col_w['base_phase']}}  "
        f"{'Path'}"
    )
    sep = "─" * (sum(col_w.values()) + 12)

    print()
    print(sep)
    print(header)
    print(sep)
    for row in all_rows:
        bp = str(row.get("base_phase", "—"))
        print(
            f"{row['phase']:{col_w['phase']}}  "
            f"{str(row['run_id']):{col_w['run_id']}}  "
            f"{str(row['epoch']):>{col_w['epoch']}}  "
            f"{_fmt(row.get('macro_f1')):>{col_w['macro_f1']}}  "
            f"{bp:>{col_w['base_phase']}}  "
            f"{row['path']}"
        )
    print(sep)
    print(f"\nTotal: {len(all_rows)} checkpoint(s)\n")

    # ── Best checkpoint per phase ─────────────────────────────────────────────
    def _best(rows, label):
        valid = [r for r in rows if r.get("macro_f1") is not None]
        if not valid:
            return
        best = max(valid, key=lambda r: float(r["macro_f1"]))
        print(
            f"  Best {label:18s} → epoch {best['epoch']:>3}  "
            f"MacroF1={_fmt(best['macro_f1'])}  "
            f"run={best['run_id']}"
        )

    print("Best checkpoints:")
    _best(phase1_rows,     "Phase 1")
    _best(phase2_rows,     "Phase 2")
    _best(phase3_p2_rows,  "Phase 3 (P2 base)")
    _best(phase3_p1_rows,  "Phase 3 (P1 base)")
    if phase3_leg_rows:
        _best(phase3_leg_rows, "Phase 3 (legacy)")
    print()


if __name__ == "__main__":
    main()
