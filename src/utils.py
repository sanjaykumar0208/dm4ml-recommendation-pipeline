"""Shared utilities used across every pipeline stage.

Keeping config loading, logging and small IO helpers in one place keeps the
stage modules focused on their actual logic and makes the whole pipeline
reproducible (single source of truth for paths and the random seed).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file (src/utils.py -> project root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the YAML config into a plain dict."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel_path: str) -> Path:
    """Resolve a config-relative path against the project root."""
    return (PROJECT_ROOT / rel_path).resolve()


def ensure_parent(path: str | os.PathLike) -> Path:
    """Make sure the parent directory of `path` exists; return the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to console and logs/pipeline.log."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
