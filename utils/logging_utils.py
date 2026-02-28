"""
utils/logging_utils.py

Logging setup:
  • Human-readable lines to stdout AND to logs/shifamind.log
  • Structured epoch metrics appended as JSON lines to logs/metrics.jsonl
    (easy to parse later with pandas or jq)
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import config

_FMT     = "%(asctime)s  [%(levelname)-8s]  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_logger: logging.Logger | None = None


def setup_logging() -> logging.Logger:
    """
    Initialise the 'shifamind' logger (idempotent — safe to call multiple times).
    Returns the logger instance.
    """
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger("shifamind")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

        # stdout handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # file handler (append mode — survives restarts)
        fh = logging.FileHandler(config.LOG_FILE, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Return the global logger, initialising it if necessary."""
    return setup_logging()


def log_metrics(phase: str, epoch: int, metrics: Dict[str, Any]) -> None:
    """
    Append one JSON line to logs/metrics.jsonl.

    Format:
        {"phase": "phase1", "epoch": 3, "ts": "...", "macro_f1": 0.42, ...}

    This file can be read later with:
        import pandas as pd
        df = pd.read_json("shifamind_local/logs/metrics.jsonl", lines=True)
    """
    entry = {
        "phase": phase,
        "epoch": epoch,
        "ts": datetime.now().isoformat(timespec="seconds"),
        **metrics,
    }
    with open(config.METRICS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
