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



def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """
    Load a checkpoint, always using weights_only=False for custom objects
    (PyG Data, concept lists, etc.) and mapping tensors to *device*.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    log.info(f"Loaded checkpoint ← {path.name}")
    return ckpt
