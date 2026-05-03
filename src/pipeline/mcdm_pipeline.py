"""
src/pipeline/mcdm_pipeline.py
------------------------------
MCDM pipeline orchestrator — populated in Phase 3-5.
"""
from __future__ import annotations
from src.core.schema import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MCDMPipeline:
    """Orchestrates the full IFS-MCDM pipeline (weighting + ranking + analysis)."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(self) -> None:
        logger.warning("MCDMPipeline.run() is a stub — implement in Phase 3-5")
