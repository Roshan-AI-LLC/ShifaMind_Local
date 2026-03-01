"""
utils/checkpoints.py

Checkpoint helpers:
  • save_best_checkpoint  — overwrite best.pt when metric improves
  • load_checkpoint       — torch.load with map_location + weights_only=False
"""
from pathlib import Path
from typing import Any, Dict

import torch

from .logging_utils import get_logger

log = get_logger()


def save_best_checkpoint(state: Dict[str, Any], path: Path) -> None:
    """Save state dict to *path*; overwrites the previous best."""
    torch.save(state, path)
    log.info(f"Best checkpoint saved  → {path.name}")



def find_latest_checkpoint(ckpt_dir: Path, filename: str) -> Path:
    """
    Return the path to *filename* inside the most recent timestamped run
    subdirectory under *ckpt_dir*.

    Directory layout expected:
        ckpt_dir/
            20260301_143022/phase2_best.pt   ← latest (returned)
            20260228_091500/phase2_best.pt   ← older

    Subdirectory names must be sortable by time (YYYYMMDD_HHMMSS works).
    Falls back to ``ckpt_dir/filename`` for backward compatibility with
    runs created before this scheme was introduced.

    Raises:
        FileNotFoundError — if no checkpoint is found anywhere.
    """
    candidates = sorted(ckpt_dir.glob(f"*/{filename}"))
    if candidates:
        found = candidates[-1]  # lexicographic order = chronological order
        log.info(f"Found checkpoint ← {found.relative_to(ckpt_dir.parent)}")
        return found
    fallback = ckpt_dir / filename
    if fallback.exists():
        log.warning(f"No run subdirs found; using legacy path {fallback.name}")
        return fallback
    raise FileNotFoundError(
        f"No '{filename}' found under {ckpt_dir}. "
        "Run the previous phase first."
    )


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """
    Load a checkpoint, always using weights_only=False for custom objects
    (PyG Data, concept lists, etc.) and mapping tensors to *device*.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    log.info(f"Loaded checkpoint ← {path.name}")
    return ckpt
