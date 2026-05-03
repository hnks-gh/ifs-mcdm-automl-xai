"""
main.py — IFS-MCDM-AutoML-XAI Framework Entry Point
-----------------------------------------------------

Simple "Run" Button Approach
----------------------------
Simply execute: python main.py

The framework will:
1. Load config from config/config.yaml
2. Detect which pipelines are enabled in config.pipeline.*
3. Execute enabled pipelines (MCDM and/or ML)
4. Save all outputs to output/

All behavior is controlled via config.yaml — no CLI arguments needed!

Usage
-----
    python main.py                    # Run with defaults (all enabled)

All configuration is read from:
    config/config.yaml

Outputs go to:
    output/mcdm/      # MCDM weights, rankings, analysis
    output/ml/        # Imputed data, forecasts, SHAP, rankings on 2025
    output/figures/   # All visualizations
    logs/             # Pipeline execution logs
"""

from __future__ import annotations

import sys

from src.pipeline.runner import run_pipeline
from src.utils.logger import setup_logger


def main() -> int:
    """
    Entry point: Run the complete IFS-MCDM-AutoML-XAI framework.

    Returns
    -------
    int
        Exit code (0 = success, 1 = failure).
    """
    # Setup logging with default level (can be overridden in config.yaml)
    setup_logger(log_level_override="INFO")

    # Run the unified pipeline
    return run_pipeline(config_path="config/config.yaml")


if __name__ == "__main__":
    sys.exit(main())
