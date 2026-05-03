"""
src/utils/logger.py
-------------------
Singleton loguru logger for the IFS-MCDM-AutoML-XAI framework.

Usage
-----
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started")
    logger.debug("Shape: {shape}", shape=df.shape)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger as _loguru_logger

# ---------------------------------------------------------------------------
# Internal state — tracks whether the logger has been configured
# ---------------------------------------------------------------------------
_configured: bool = False
_LOG_DIR = Path("logs")


def setup_logger(
    config_path: str | Path = "config/logging.yaml",
    log_level_override: Optional[str] = None,
) -> None:
    """
    Configure the global loguru logger from ``config/logging.yaml``.

    This function is idempotent — calling it multiple times has no effect
    after the first successful configuration.

    Parameters
    ----------
    config_path : str | Path
        Path to logging.yaml. Relative paths are resolved from the current
        working directory (i.e., the project root).
    log_level_override : str | None
        If provided, overrides the console log level (e.g. ``"DEBUG"``).
    """
    global _configured
    if _configured:
        return

    config_path = Path(config_path)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Remove loguru's default stderr handler
    _loguru_logger.remove()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    cfg: dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    loguru_cfg = cfg.get("loguru", {})

    # ------------------------------------------------------------------
    # Console handler
    # ------------------------------------------------------------------
    console_cfg = loguru_cfg.get("console", {})
    console_level = log_level_override or console_cfg.get("level", "INFO")
    _loguru_logger.add(
        sink=sys.stderr,
        level=console_level,
        colorize=console_cfg.get("colorize", True),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        backtrace=console_cfg.get("backtrace", True),
        diagnose=console_cfg.get("diagnose", False),
    )

    # ------------------------------------------------------------------
    # File handler
    # ------------------------------------------------------------------
    file_cfg = loguru_cfg.get("file", {})
    _loguru_logger.add(
        sink=str(_LOG_DIR / "run_{time:YYYYMMDD_HHmmss}.log"),
        level=file_cfg.get("level", "DEBUG"),
        colorize=False,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
        rotation=file_cfg.get("rotation", "50 MB"),
        retention=file_cfg.get("retention", "30 days"),
        compression=file_cfg.get("compression", "zip"),
        backtrace=file_cfg.get("backtrace", True),
        diagnose=file_cfg.get("diagnose", False),
        enqueue=file_cfg.get("enqueue", True),
    )

    _configured = True
    _loguru_logger.debug("Logger initialised from '{}'", config_path)


def get_logger(name: str = "ifs_mcdm"):
    """
    Return a loguru logger bound to *name*.

    The first call implicitly runs :func:`setup_logger` with defaults so that
    modules can call ``get_logger(__name__)`` without needing to call
    ``setup_logger`` first.

    Parameters
    ----------
    name : str
        Logical name attached to every log record from this logger.
        Convention: pass ``__name__`` from the calling module.

    Returns
    -------
    loguru.Logger
        A context-bound loguru logger instance.
    """
    if not _configured:
        setup_logger()
    return _loguru_logger.bind(name=name)


def reset_logger() -> None:
    """
    Reset logger configuration (useful in tests that need fresh state).
    """
    global _configured
    _loguru_logger.remove()
    _configured = False
