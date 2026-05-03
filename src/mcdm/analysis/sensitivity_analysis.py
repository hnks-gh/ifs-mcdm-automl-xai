"""
src/mcdm/analysis/sensitivity_analysis.py
------------------------------------------
Monte Carlo Sensitivity Analysis for IF-CRITIC Weights & Rankings.

This module quantifies ranking robustness by perturbing IF-CRITIC weights via
Dirichlet distribution sampling and re-running all ranking methods.

Methodology
===========

**Step 1: Baseline ranking**
Compute IF-WASPAS, IF-TOPSIS, IF-PROMETHEE II on the original decision matrix
using the original (unperturbed) weights.

**Step 2: Weight perturbation**
For each of n_simulations iterations:
1. Sample perturbed weights from Dirichlet(α) where α_j = base_weight_j × α_scale
2. Normalize perturbed weights to sum to 1.0
3. Re-rank using all 3 ranking methods

**Step 3: Ranking stability**
Compute Kendall's tau-b (weighted variant) between baseline rankings and each
perturbed ranking. This measures rank reordering sensitivity.

**Step 4: Statistical summary**
Report mean τ_b, std τ_b, 95th percentile CI per method.

Theoretical Foundation
======================

**Dirichlet perturbation**: Provides a principled way to sample weight vectors
that respect the simplex constraint (all weights >= 0, sum = 1). The concentration
parameter α directly controls variance:
- Higher α_j → lower variance for weight j
- α_j = 0 → weight not included (extreme)
- Scaling α by a factor λ controls overall "noise level"

**Kendall tau-b (weighted)**: A rank correlation coefficient that down-weights
disagreements involving less important sub-criteria. For sub-criteria j with
weight w_j, discordant pairs involving j contribute (1 - w_j) to the penalty.

**95% CI**: Bootstrap-style confidence interval from the empirical distribution
of τ_b values; computed as [percentile(τ_b, 2.5), percentile(τ_b, 97.5)].

References
----------
Dirichlet distribution: Continuous multivariate analogue of Beta distribution.
    Commonly used for random simplex sampling (compositional data).

Kendall tau-b (weighted variant): Derived from weighted distance metrics.
    - Standard: τ = (C - D) / sqrt(T0 × T1)
    - Weighted: penalize disagreements by importance weights

Data Leakage Prevention
=======================
This analysis uses ONLY the decision matrix and weights already computed from
the MCDM pipeline. NO additional fitting or data contamination occurs; we are
sampling from a prior distribution over the weight space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.exceptions import FrameworkError
from src.core.ifs_arithmetic import IFSMatrix
from src.core.schema import RankingMethod, WeightVector
from src.mcdm.ranking import if_waspas, if_topsis, if_promethee2

logger = logging.getLogger(__name__)

_TOL: float = 1e-12


@dataclass
class SensitivityResult:
    """
    Monte Carlo sensitivity analysis output.

    Attributes
    ----------
    baseline_ranks : dict[RankingMethod, list[int]]
        Original ranking (1..n) for each method.
    n_simulations : int
        Number of Monte Carlo samples.
    perturbation_scale : float
        Dirichlet concentration scaling factor α_scale.
    random_state : int
        Random seed used.
    kendall_tau_b_distributions : dict[RankingMethod, list[float]]
        τ_b values from all simulations for each method.
    summary_stats : dict[RankingMethod, dict[str, float]]
        Mean, std, 2.5% and 97.5% percentiles of τ_b per method.
    """
    baseline_ranks: Dict[RankingMethod, List[int]]
    n_simulations: int
    perturbation_scale: float
    random_state: int
    kendall_tau_b_distributions: Dict[RankingMethod, List[float]]
    summary_stats: Dict[RankingMethod, Dict[str, float]]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "baseline_ranks": {k.value: v for k, v in self.baseline_ranks.items()},
            "n_simulations": self.n_simulations,
            "perturbation_scale": self.perturbation_scale,
            "random_state": self.random_state,
            "kendall_tau_b_distributions": {
                k.value: v for k, v in self.kendall_tau_b_distributions.items()
            },
            "summary_stats": {
                k.value: v for k, v in self.summary_stats.items()
            },
        }

    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert to pandas DataFrames for export.

        Returns
        -------
        tau_b_df : pd.DataFrame
            Shape (n_simulations, n_methods) - τ_b values.
        stats_df : pd.DataFrame
            Shape (n_methods, 5) - summary statistics per method.
        """
        # τ_b distribution dataframe
        tau_b_df = pd.DataFrame(self.kendall_tau_b_distributions)

        # Summary statistics dataframe
        stats_rows = []
        for method, stats in self.summary_stats.items():
            stats_rows.append({
                "Method": method.value,
                "Mean_τb": stats["mean"],
                "Std_τb": stats["std"],
                "CI_Lower_2.5%": stats["ci_lower"],
                "CI_Upper_97.5%": stats["ci_upper"],
            })
        stats_df = pd.DataFrame(stats_rows)

        return tau_b_df, stats_df


# =============================================================================
# Kendall's tau-b (weighted variant)
# =============================================================================

def kendall_tau_b_weighted(
    rank1: List[int],
    rank2: List[int],
    weights: np.ndarray,
) -> float:
    """
    Compute weighted Kendall's tau-b between two rankings.

    The standard tau-b ignores ties; the weighted variant down-weights
    discordances involving less important criteria (lower weights).

    Parameters
    ----------
    rank1, rank2 : list[int]
        Two rankings (permutations of 1..n).
    weights : ndarray, shape (n,)
        Weights for each item (sub-criterion importance).
        Larger weight → disagreement on this item is more significant.

    Returns
    -------
    tau_b : float
        Weighted Kendall's tau-b ∈ [-1, 1].
        +1 = identical rankings
        -1 = reverse rankings
        0 = random agreement

    Raises
    ------
    ValueError
        If ranks and weights have mismatched lengths or invalid rank values.
    """
    rank1 = np.array(rank1)
    rank2 = np.array(rank2)
    weights = np.array(weights)

    n = len(rank1)
    if len(rank2) != n or len(weights) != n:
        raise ValueError(
            f"rank1, rank2, and weights must have same length: "
            f"got {len(rank1)}, {len(rank2)}, {len(weights)}"
        )

    # Check validity: ranks should be permutations of 1..n
    if set(rank1) != set(range(1, n+1)) or set(rank2) != set(range(1, n+1)):
        raise ValueError("Ranks must be permutations of 1..n (no duplicates, no gaps)")

    # Normalize weights to sum to 1
    weights_norm = weights / np.sum(weights)

    # Compute concordant/discordant pairs
    concordant_weight = 0.0
    discordant_weight = 0.0

    for i in range(n):
        for j in range(i+1, n):
            # Pair (i, j)
            pair_weight = weights_norm[i] * weights_norm[j]

            # Check order consistency
            rank1_order = rank1[i] < rank1[j]
            rank2_order = rank2[i] < rank2[j]

            if rank1_order == rank2_order:
                concordant_weight += pair_weight
            else:
                discordant_weight += pair_weight

    # tau-b = (C - D) / (C + D) where C = concordant, D = discordant
    total = concordant_weight + discordant_weight
    if abs(total) < _TOL:
        return 0.0
    
    tau_b = (concordant_weight - discordant_weight) / total
    return float(np.clip(tau_b, -1.0, 1.0))


# =============================================================================
# Weight perturbation
# =============================================================================

def sample_dirichlet_weights(
    base_weights: np.ndarray,
    alpha_scale: float = 10.0,
    n_samples: int = 1000,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Sample weight vectors from Dirichlet distribution.

    Parameters
    ----------
    base_weights : ndarray, shape (n,)
        Base weight vector (must sum to 1.0).
    alpha_scale : float
        Scaling factor for concentration parameters: α_j = base_weight_j × α_scale.
        Higher α_scale → lower variance (less perturbation).
        Default 10.0 provides moderate noise.
    n_samples : int
        Number of samples to draw.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    samples : ndarray, shape (n_samples, n)
        Sampled weight vectors (each row sums to 1.0).

    Raises
    ------
    ValueError
        If base_weights do not sum to 1 or contain negative values.
    """
    base_weights = np.array(base_weights, dtype=float)
    n = len(base_weights)

    if not np.isclose(np.sum(base_weights), 1.0, atol=_TOL):
        raise ValueError(
            f"base_weights must sum to 1.0, got sum={np.sum(base_weights):.6f}"
        )
    if np.any(base_weights < -_TOL):
        raise ValueError(f"base_weights must be non-negative, got min={np.min(base_weights):.6f}")

    # Compute concentration parameters
    alpha = base_weights * alpha_scale

    # Sample from Dirichlet
    rng = np.random.RandomState(random_state)
    samples = rng.dirichlet(alpha, size=n_samples)

    return samples


# =============================================================================
# Main sensitivity analysis function
# =============================================================================

def run_montecarlo_sensitivity(
    ifs_matrix: IFSMatrix,
    base_weights: WeightVector,
    ranking_methods: List[RankingMethod],
    ranking_configs: Dict,  # Dict[RankingMethod, config]
    n_simulations: int = 10000,
    dirichlet_alpha_scale: float = 10.0,
    random_state: int = 42,
) -> SensitivityResult:
    """
    Execute Monte Carlo sensitivity analysis via weight perturbation.

    Parameters
    ----------
    ifs_matrix : IFSMatrix
        Decision matrix (provinces × sub-criteria).
    base_weights : WeightVector
        Original IF-CRITIC weights.
    ranking_methods : list[RankingMethod]
        Methods to evaluate (IF-WASPAS, IF-TOPSIS, IF-PROMETHEE II).
    ranking_configs : dict[RankingMethod, config]
        Configuration for each ranking method (lambda, p parameter, etc.).
    n_simulations : int
        Number of Monte Carlo samples. Default 10000.
    dirichlet_alpha_scale : float
        Concentration parameter scaling. Default 10.0.
    random_state : int
        Random seed. Default 42.

    Returns
    -------
    result : SensitivityResult
        Complete sensitivity analysis output.

    Raises
    ------
    FrameworkError
        If ranking computation fails or data is invalid.
    """
    logger.info(
        f"Running Monte Carlo sensitivity analysis: n_simulations={n_simulations}, "
        f"alpha_scale={dirichlet_alpha_scale}, random_state={random_state}"
    )

    # Compute baseline rankings
    logger.info("Computing baseline rankings with original weights...")
    baseline_ranks = {}

    try:
        if RankingMethod.IF_WASPAS in ranking_methods:
            result_waspas = if_waspas.rank(
                ifs_matrix,
                np.array(base_weights.values),
                lambda_param=ranking_configs.get(RankingMethod.IF_WASPAS, {}).get("lambda_param", 0.5),
            )
            baseline_ranks[RankingMethod.IF_WASPAS] = result_waspas.ranks

        if RankingMethod.IF_TOPSIS in ranking_methods:
            result_topsis = if_topsis.rank(
                ifs_matrix,
                np.array(base_weights.values),
            )
            baseline_ranks[RankingMethod.IF_TOPSIS] = result_topsis.ranks

        if RankingMethod.IF_PROMETHEE2 in ranking_methods:
            result_promethee2 = if_promethee2.rank(
                ifs_matrix,
                np.array(base_weights.values),
                p_parameter=ranking_configs.get(RankingMethod.IF_PROMETHEE2, {}).get("p_parameter", 0.1),
            )
            baseline_ranks[RankingMethod.IF_PROMETHEE2] = result_promethee2.ranks
    except Exception as e:
        raise FrameworkError(
            f"Failed to compute baseline rankings: {str(e)}",
            context={"error": str(e)},
        ) from e

    # Sample perturbed weights
    logger.info(f"Sampling {n_simulations} perturbed weight vectors from Dirichlet...")
    perturbed_weight_samples = sample_dirichlet_weights(
        np.array(base_weights.values),
        alpha_scale=dirichlet_alpha_scale,
        n_samples=n_simulations,
        random_state=random_state,
    )

    # Run simulations
    logger.info("Running ranking simulations with perturbed weights...")
    tau_b_distributions = {method: [] for method in ranking_methods}

    for sim in range(n_simulations):
        if (sim + 1) % max(1, n_simulations // 10) == 0:
            logger.debug(f"  Simulation {sim+1}/{n_simulations}")

        perturbed_weights = perturbed_weight_samples[sim]

        try:
            # Rank with perturbed weights
            if RankingMethod.IF_WASPAS in ranking_methods:
                result_waspas_pert = if_waspas.rank(
                    ifs_matrix,
                    perturbed_weights,
                    lambda_param=ranking_configs.get(RankingMethod.IF_WASPAS, {}).get("lambda_param", 0.5),
                )
                tau_b_w = kendall_tau_b_weighted(
                    baseline_ranks[RankingMethod.IF_WASPAS],
                    result_waspas_pert.ranks,
                    np.array(base_weights.values),
                )
                tau_b_distributions[RankingMethod.IF_WASPAS].append(tau_b_w)

            if RankingMethod.IF_TOPSIS in ranking_methods:
                result_topsis_pert = if_topsis.rank(
                    ifs_matrix,
                    perturbed_weights,
                )
                tau_b_t = kendall_tau_b_weighted(
                    baseline_ranks[RankingMethod.IF_TOPSIS],
                    result_topsis_pert.ranks,
                    np.array(base_weights.values),
                )
                tau_b_distributions[RankingMethod.IF_TOPSIS].append(tau_b_t)

            if RankingMethod.IF_PROMETHEE2 in ranking_methods:
                result_promethee2_pert = if_promethee2.rank(
                    ifs_matrix,
                    perturbed_weights,
                    p_parameter=ranking_configs.get(RankingMethod.IF_PROMETHEE2, {}).get("p_parameter", 0.1),
                )
                tau_b_p = kendall_tau_b_weighted(
                    baseline_ranks[RankingMethod.IF_PROMETHEE2],
                    result_promethee2_pert.ranks,
                    np.array(base_weights.values),
                )
                tau_b_distributions[RankingMethod.IF_PROMETHEE2].append(tau_b_p)

        except Exception as e:
            logger.warning(f"Simulation {sim+1} failed: {str(e)}; skipping")
            continue

    # Compute summary statistics
    logger.info("Computing summary statistics...")
    summary_stats = {}

    for method in ranking_methods:
        tau_b_vals = np.array(tau_b_distributions[method])
        summary_stats[method] = {
            "mean": float(np.mean(tau_b_vals)),
            "std": float(np.std(tau_b_vals, ddof=1)),
            "ci_lower": float(np.percentile(tau_b_vals, 2.5)),
            "ci_upper": float(np.percentile(tau_b_vals, 97.5)),
            "n_samples": len(tau_b_vals),
        }
        logger.info(
            f"{method.value}: τ_b = {summary_stats[method]['mean']:.4f} "
            f"± {summary_stats[method]['std']:.4f}, CI=[{summary_stats[method]['ci_lower']:.4f}, "
            f"{summary_stats[method]['ci_upper']:.4f}]"
        )

    result = SensitivityResult(
        baseline_ranks=baseline_ranks,
        n_simulations=n_simulations,
        perturbation_scale=dirichlet_alpha_scale,
        random_state=random_state,
        kendall_tau_b_distributions=tau_b_distributions,
        summary_stats=summary_stats,
    )

    logger.info("Monte Carlo sensitivity analysis complete")
    return result


def save_sensitivity_analysis(
    result: SensitivityResult,
    output_dir: str,
) -> None:
    """
    Save sensitivity analysis results to disk.

    Parameters
    ----------
    result : SensitivityResult
        Analysis result.
    output_dir : str
        Directory to save outputs.
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save τ_b distributions
    tau_b_df, stats_df = result.to_dataframe()
    tau_b_df.to_csv(output_dir / "sensitivity_kendall_tau_b_simulations.csv", index=False)
    logger.info(f"Saved τ_b distributions to {output_dir / 'sensitivity_kendall_tau_b_simulations.csv'}")

    # Save summary statistics
    stats_df.to_csv(output_dir / "sensitivity_summary_statistics.csv", index=False)
    logger.info(f"Saved summary statistics to {output_dir / 'sensitivity_summary_statistics.csv'}")

    # Save detailed JSON for reproducibility
    import json
    with open(output_dir / "sensitivity_result.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Saved detailed result to {output_dir / 'sensitivity_result.json'}")
