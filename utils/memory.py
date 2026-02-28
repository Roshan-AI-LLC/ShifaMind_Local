"""
utils/memory.py

Unified memory-usage reporter for MPS / CUDA / CPU.
Call log_memory_usage() anywhere to print current accelerator memory.
"""
import torch

from .logging_utils import get_logger

log = get_logger()


def log_memory_usage(device: torch.device) -> None:
    """Log current accelerator memory to the shared logger."""
    if device.type == "mps":
        mb = torch.mps.current_allocated_memory() / 1e6
        log.info(f"MPS memory in use : {mb:,.0f} MB")
    elif device.type == "cuda":
        mb = torch.cuda.memory_allocated() / 1e6
        log.info(f"GPU memory in use : {mb:,.0f} MB")
    else:
        log.info("Running on CPU — no accelerator memory stats available")


def clear_accelerator_cache(device: torch.device) -> None:
    """Free unused memory from the accelerator cache."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
