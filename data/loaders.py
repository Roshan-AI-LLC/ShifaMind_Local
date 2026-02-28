"""
data/loaders.py

Factory that builds DataLoaders with MPS-safe settings:
  • pin_memory=False   (pin_memory is a CUDA-only optimisation)
  • prefetch_factor    only supplied when num_workers > 0
"""
from torch.utils.data import DataLoader, Dataset

import config


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = None,
) -> DataLoader:
    """
    Returns a DataLoader configured for Apple MPS / local execution.

    Args:
        dataset     : any torch Dataset
        batch_size  : samples per batch
        shuffle     : whether to shuffle (True for train, False for val/test)
        num_workers : override config.NUM_WORKERS if needed
    """
    nw = num_workers if num_workers is not None else config.NUM_WORKERS
    kwargs: dict = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=False,   # CUDA-only; no benefit on MPS
    )
    if nw > 0:
        kwargs["prefetch_factor"] = config.PREFETCH_FACTOR
    return DataLoader(dataset, **kwargs)
