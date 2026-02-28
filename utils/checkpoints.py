"""
utils/checkpoints.py

Checkpoint helpers:
  • save_best_checkpoint   — overwrite best.pt when metric improves
  • save_epoch_checkpoint  — write phase_epoch_N.pt after every epoch
  • load_checkpoint        — torch.load with map_location + weights_only=False
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


def save_epoch_checkpoint(
    state: Dict[str, Any],
    ckpt_dir: Path,
    stem: str,
    epoch: int,
) -> None:
    """
    Save a per-epoch snapshot alongside the best checkpoint.

    Naming convention:  <stem>_epoch_<N>.pt   (1-indexed)

    Example:
        save_epoch_checkpoint(state, CKPT_P1, "phase1", epoch=2)
        # writes  checkpoints/phase1/phase1_epoch_3.pt
    """
    suffix = ".pth" if stem.startswith("phase3") else ".pt"
    ep_path = ckpt_dir / f"{stem}_epoch_{epoch + 1}{suffix}"
    torch.save(state, ep_path)
    log.info(f"Epoch checkpoint saved → {ep_path.name}")


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """
    Load a checkpoint, always using weights_only=False for custom objects
    (PyG Data, concept lists, etc.) and mapping tensors to *device*.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    log.info(f"Loaded checkpoint ← {path.name}")
    return ckpt
