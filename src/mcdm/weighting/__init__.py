"""
src/mcdm/weighting/__init__.py
-------------------------------
Public API for the IF-CRITIC two-level weighting subsystem.
"""

from src.mcdm.weighting.if_critic import (
    compute_critic_weights,
    compute_stage1_weights,
    compute_stage2_weights,
    handle_missing_subcriteria,
)
from src.mcdm.weighting.two_level_aggregator import (
    aggregate_regime_weights,
    compute_final_subcriteria_weights,
    compute_weights_for_all_years,
    compute_weights_for_year,
)

__all__ = [
    # if_critic.py
    "compute_critic_weights",
    "compute_stage1_weights",
    "compute_stage2_weights",
    "handle_missing_subcriteria",
    # two_level_aggregator.py
    "aggregate_regime_weights",
    "compute_final_subcriteria_weights",
    "compute_weights_for_all_years",
    "compute_weights_for_year",
]
