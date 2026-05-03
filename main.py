"""
main.py — IFS-MCDM-AutoML-XAI Framework Entry Point
-----------------------------------------------------
Usage
-----
    python main.py --pipeline mcdm --config config/config.yaml
    python main.py --pipeline ml   --config config/config.yaml
    python main.py --pipeline all  --config config/config.yaml --log-level DEBUG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils.logger import get_logger, setup_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IFS-MCDM-AutoML-XAI Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pipeline",
        choices=["mcdm", "ml", "all"],
        default="all",
        help="Which pipeline to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Initialise logger first so all subsequent imports can use it
    setup_logger(log_level_override=args.log_level)
    logger = get_logger(__name__)

    logger.info("=== IFS-MCDM-AutoML-XAI Framework ===")
    logger.info("Pipeline: {}  |  Config: {}", args.pipeline, args.config)

    try:
        from src.core.data_loader import load_config
        cfg = load_config(args.config)

        if args.pipeline in ("mcdm", "all"):
            # Phase 3-5 will populate this
            from src.pipeline.mcdm_pipeline import MCDMPipeline
            MCDMPipeline(cfg).run()

        if args.pipeline in ("ml", "all"):
            # Phase 6-8 will populate this
            from src.pipeline.ml_pipeline import MLPipeline
            MLPipeline(cfg).run()

    except Exception as exc:
        logger.exception("Pipeline failed: {}", exc)
        return 1

    logger.info("=== Pipeline completed successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
