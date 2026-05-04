"""
src/core/schema.py
------------------
Pydantic v2 configuration models, enumerations, and typed dataclasses for
the IFS-MCDM-AutoML-XAI framework.

Design principles
-----------------
* All configuration parameters are validated at startup, not at use-time.
* Dataclasses are typed but lightweight — they do NOT contain heavy numpy
  arrays (those live in pipeline-internal variables).  They carry metadata
  such as labels, shapes, and provenance.
* Enums provide exhaustive, type-safe choices that prevent silent misspellings
  in config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Enumerations
# =============================================================================

class RankingMethod(str, Enum):
    """Supported IF-MCDM ranking methods."""
    IF_WASPAS = "if_waspas"
    IF_TOPSIS = "if_topsis"
    IF_PROMETHEE2 = "if_promethee2"


class HesitancyMethod(str, Enum):
    """IFS hesitancy assignment strategies."""
    FIXED_PI = "fixed_pi"          # pi = constant; mu = x/max*(1-pi); nu = 1-mu-pi
    COMPLEMENT = "complement"       # nu = 1 - mu; pi = 0  (crisp membership)


class WeightBlendMethod(str, Enum):
    """How per-regime weights are blended into cross-regime weights."""
    PROPORTIONAL_YEARS = "proportional_years"   # weight ∝ number of years in regime
    SIMPLE_AVERAGE = "simple_average"            # equal blend across regimes


class CorrelationMethod(str, Enum):
    """Correlation method used inside CRITIC."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"


class PerturbationDistribution(str, Enum):
    """Distribution used in Monte Carlo weight perturbation."""
    DIRICHLET = "dirichlet"


class ExplainerType(str, Enum):
    """SHAP explainer type."""
    AUTO = "auto"
    TREE = "tree"
    KERNEL = "kernel"
    DEEP = "deep"


class TabularFormat(str, Enum):
    """Serialisation format for tabular outputs."""
    CSV = "csv"
    PARQUET = "parquet"


# =============================================================================
# Pydantic configuration models
# =============================================================================

class RegimeConfig(BaseModel):
    """Configuration for a single year-regime."""
    years: List[int] = Field(..., description="Calendar years belonging to this regime")
    absent_subcriteria: List[str] = Field(
        default_factory=list,
        description="Sub-criteria columns that are structurally absent (all-NaN) in these years",
    )
    active_subcriteria: List[str] = Field(
        ..., description="Sub-criteria columns present and usable in these years"
    )
    n_active: int = Field(..., ge=1, description="Count of active sub-criteria")

    @model_validator(mode="after")
    def validate_n_active(self) -> "RegimeConfig":
        if self.n_active != len(self.active_subcriteria):
            raise ValueError(
                f"n_active ({self.n_active}) does not match "
                f"len(active_subcriteria) ({len(self.active_subcriteria)})"
            )
        return self


class BlankProvinceYear(BaseModel):
    """A province-year combination where all sub-criteria are blank (Type 2 missing)."""
    year: int
    province: str  # e.g. "P15"


class DataConfig(BaseModel):
    """Data layer configuration."""
    csv_dir: str = "data/csv"
    codebook_dir: str = "data/codebook"
    years: List[int] = Field(..., min_length=1)
    province_col: str = "Province"
    n_provinces: int = Field(63, ge=1)
    n_subcriteria: int = Field(29, ge=1)
    n_criteria: int = Field(8, ge=1)
    all_subcriteria: List[str] = Field(..., min_length=1)
    criteria_subcriteria_map: Dict[str, List[str]] = Field(
        ..., description="Mapping criterion code -> list of sub-criteria codes"
    )
    cost_criteria: List[str] = Field(
        default_factory=list,
        description="Sub-criteria where lower values are preferred (none for PAPI)",
    )
    regimes: Dict[str, RegimeConfig] = Field(
        ..., description="Regime definitions keyed by regime id (R1, R2, ...)"
    )
    blank_province_years: List[BlankProvinceYear] = Field(
        default_factory=list,
        description="Province-year combinations that are entirely blank (Type 2 missing)",
    )

    @field_validator("years")
    @classmethod
    def years_are_sorted(cls, v: List[int]) -> List[int]:
        if v != sorted(v):
            raise ValueError("years must be in ascending order")
        return v

    @field_validator("all_subcriteria")
    @classmethod
    def subcriteria_unique(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            raise ValueError("all_subcriteria contains duplicates")
        return v

    @model_validator(mode="after")
    def validate_criteria_map(self) -> "DataConfig":
        all_sc_in_map: List[str] = []
        for crit, scs in self.criteria_subcriteria_map.items():
            all_sc_in_map.extend(scs)

        if sorted(all_sc_in_map) != sorted(self.all_subcriteria):
            raise ValueError(
                "criteria_subcriteria_map must cover exactly all_subcriteria; "
                f"map covers {sorted(all_sc_in_map)}, "
                f"all_subcriteria = {sorted(self.all_subcriteria)}"
            )

        # Validate n_criteria
        if len(self.criteria_subcriteria_map) != self.n_criteria:
            raise ValueError(
                f"n_criteria ({self.n_criteria}) does not match "
                f"criteria_subcriteria_map keys ({len(self.criteria_subcriteria_map)})"
            )
        return self


class IFSConfig(BaseModel):
    """IFS conversion configuration."""
    score_max: float = Field(3.33, gt=0.0, description="Maximum observable raw score")
    hesitancy_method: HesitancyMethod = HesitancyMethod.FIXED_PI
    fixed_pi_value: float = Field(
        0.05, ge=0.0, lt=1.0,
        description="Fixed hesitancy π applied to all IFS conversions"
    )

    @model_validator(mode="after")
    def validate_pi_leaves_room_for_mu_nu(self) -> "IFSConfig":
        if self.hesitancy_method == HesitancyMethod.FIXED_PI:
            if self.fixed_pi_value >= 1.0:
                raise ValueError("fixed_pi_value must be < 1.0 so mu and nu can be positive")
        return self


class WASPASConfig(BaseModel):
    """IF-WASPAS specific parameters."""
    lambda_param: float = Field(0.5, ge=0.0, le=1.0)


class TOPSISConfig(BaseModel):
    """IF-TOPSIS specific parameters."""
    distance_metric: str = "normalized_euclidean"


class PROMETHEE2Config(BaseModel):
    """IF-PROMETHEE II specific parameters."""
    preference_function: str = "gaussian"
    p_parameter: float = Field(0.1, gt=0.0)


class RankingConfig(BaseModel):
    """Ranking methods configuration."""
    methods: List[RankingMethod] = Field(
        default_factory=lambda: list(RankingMethod)
    )
    if_waspas: WASPASConfig = Field(default_factory=WASPASConfig)
    if_topsis: TOPSISConfig = Field(default_factory=TOPSISConfig)
    if_promethee2: PROMETHEE2Config = Field(default_factory=PROMETHEE2Config)


class TemporalStabilityConfig(BaseModel):
    """Temporal stability analysis configuration."""
    window_size: int = Field(5, ge=2)
    n_windows: int = Field(10, ge=1)
    metrics: List[str] = Field(default_factory=lambda: ["rmsd", "cv"])


class SensitivityConfig(BaseModel):
    """Monte Carlo sensitivity analysis configuration."""
    n_simulations: int = Field(10000, ge=100)
    perturbation_distribution: PerturbationDistribution = PerturbationDistribution.DIRICHLET
    dirichlet_concentration_scale: float = Field(10.0, gt=0.0)
    correlation_metric: str = "kendall_tau_b_weighted"
    random_state: int = 42


class WeightingAnalysisConfig(BaseModel):
    """Weight analysis configuration."""
    temporal_stability: TemporalStabilityConfig = Field(
        default_factory=TemporalStabilityConfig
    )
    sensitivity: SensitivityConfig = Field(default_factory=SensitivityConfig)


class WeightingConfig(BaseModel):
    """IF-CRITIC weighting configuration."""
    method: str = "two_level_if_critic"
    correlation_method: CorrelationMethod = CorrelationMethod.PEARSON
    min_variance_threshold: float = Field(1e-9, ge=0.0)
    weight_blend_method: WeightBlendMethod = WeightBlendMethod.PROPORTIONAL_YEARS


class AnalysisConfig(BaseModel):
    """All analysis configurations."""
    weighting: WeightingAnalysisConfig = Field(default_factory=WeightingAnalysisConfig)
    ranking_inter_method_correlation: str = "spearman"
    ranking_discriminatory_power_metric: str = "iqr"
    ranking_temporal_persistence_metric: str = "spearman_yoy"


class MCDMConfig(BaseModel):
    """Full MCDM configuration."""
    weighting: WeightingConfig = Field(default_factory=WeightingConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)


class ImputationConfig(BaseModel):
    """MICE imputation configuration."""
    method: str = "mice"
    max_iter: int = Field(10, ge=1)
    n_nearest_features: Optional[int] = None
    initial_strategy: str = "mean"
    random_state: int = 42


class ForecastingConfig(BaseModel):
    """AutoGluon forecasting configuration."""
    target_year: int = 2025
    prediction_length: int = Field(1, ge=1)
    presets: str = "best_quality"
    refit_full: bool = True
    known_covariates_names: Optional[List[str]] = None
    eval_metric: str = "MASE"
    freq: str = "Y"
    random_state: int = 42
    model_save_dir: str = "output/ml/ag_models"


class SHAPConfig(BaseModel):
    """SHAP explainability configuration."""
    explainer_type: ExplainerType = ExplainerType.AUTO
    n_background_samples: int = Field(100, ge=10)
    random_state: int = 42


class MLConfig(BaseModel):
    """Full ML pipeline configuration."""
    imputation: ImputationConfig = Field(default_factory=ImputationConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    shap: SHAPConfig = Field(default_factory=SHAPConfig)


class OutputConfig(BaseModel):
    """Output path and format configuration."""
    mcdm_dir: str = "output/mcdm"
    ml_dir: str = "output/ml"
    figures_dir: str = "output/figures"
    reports_dir: str = "output/reports"
    tabular_format: TabularFormat = TabularFormat.CSV
    figure_dpi: int = Field(300, ge=72)
    figure_format: str = "png"


class PipelineConfig(BaseModel):
    """
    Pipeline orchestration configuration — controls which components execute.
    
    Attributes
    ----------
    mcdm_enabled : bool
        Enable the entire MCDM pipeline (weighting + ranking + analysis).
    ml_enabled : bool
        Enable the entire ML pipeline (imputation + forecasting + SHAP).
    mcdm_weighting_enabled : bool
        Enable MCDM weighting computation (only if mcdm_enabled=True).
    mcdm_ranking_enabled : bool
        Enable MCDM ranking methods (only if mcdm_enabled=True).
    mcdm_analysis_enabled : bool
        Enable MCDM analysis (temporal stability, sensitivity) (only if mcdm_enabled=True).
    ml_imputation_enabled : bool
        Enable MICE imputation (only if ml_enabled=True).
    ml_forecasting_enabled : bool
        Enable AutoGluon forecasting (only if ml_enabled=True).
    ml_shap_enabled : bool
        Enable SHAP explainability (only if ml_enabled=True).
    ml_forecast_ranking_enabled : bool
        Apply MCDM ranking to 2025 forecasted values (only if ml_enabled=True).
    log_level : str
        Logging level: DEBUG, INFO, WARNING, ERROR.
    """
    mcdm_enabled: bool = True
    ml_enabled: bool = True
    mcdm_weighting_enabled: bool = True
    mcdm_ranking_enabled: bool = True
    mcdm_analysis_enabled: bool = True
    ml_imputation_enabled: bool = True
    ml_forecasting_enabled: bool = True
    ml_shap_enabled: bool = True
    ml_forecast_ranking_enabled: bool = True
    log_level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")


class AppConfig(BaseModel):
    """
    Root application configuration.

    Loaded from ``config/config.yaml`` via :func:`src.core.data_loader.load_config`.
    """
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    data: DataConfig
    ifs: IFSConfig = Field(default_factory=IFSConfig)
    mcdm: MCDMConfig = Field(default_factory=MCDMConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# =============================================================================
# Lightweight dataclasses (metadata carriers, not heavy arrays)
# =============================================================================

@dataclass
class Regime:
    """
    Metadata for a single year-regime.

    Attributes
    ----------
    regime_id : str
        Identifier (e.g. "R1", "R2").
    years : list[int]
        Calendar years belonging to this regime.
    active_subcriteria : list[str]
        Sub-criteria usable in this regime.
    absent_subcriteria : list[str]
        Structurally absent sub-criteria in this regime.
    """
    regime_id: str
    years: List[int]
    active_subcriteria: List[str]
    absent_subcriteria: List[str]

    @property
    def n_active(self) -> int:
        return len(self.active_subcriteria)

    @property
    def n_years(self) -> int:
        return len(self.years)


class DataConfigExt(DataConfig):
    """Extended DataConfig with computed properties."""
    
    @property
    def n_years(self) -> int:
        """Number of years in the dataset."""
        return len(self.years)


@dataclass
class WeightVector:
    """
    A named weight vector over sub-criteria or criteria.

    Attributes
    ----------
    labels : list[str]
        Ordered names corresponding to each weight.
    values : list[float]
        Non-negative weights summing to 1.0 (over active entries).
    year : int | None
        Year this weight vector was computed for (None = cross-year blend).
    regime_id : str | None
        Regime used during computation.
    stage : int | None
        1 = intra-criterion (sub-criteria), 2 = inter-criterion (criteria).
    """
    labels: List[str]
    values: List[float]
    year: Optional[int] = None
    regime_id: Optional[str] = None
    stage: Optional[int] = None

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.values):
            raise ValueError(
                f"labels length ({len(self.labels)}) != "
                f"values length ({len(self.values)})"
            )

    def as_dict(self) -> Dict[str, float]:
        return dict(zip(self.labels, self.values))


@dataclass
class RankingResult:
    """
    Output of a single MCDM ranking run.

    Attributes
    ----------
    method : RankingMethod
        The ranking method that produced this result.
    year : int
        Year of the ranking.
    provinces : list[str]
        Province codes in the same order as ``scores`` and ``ranks``.
    scores : list[float]
        Composite score for each province (method-specific semantics).
    ranks : list[int]
        Rank assigned to each province (1 = best).
    """
    method: RankingMethod
    year: int
    provinces: List[str]
    scores: List[float]
    ranks: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.provinces)
        if len(self.scores) != n:
            raise ValueError("provinces and scores must have the same length")
        if self.ranks and len(self.ranks) != n:
            raise ValueError("provinces and ranks must have the same length")


@dataclass
class PAPIPanel:
    """
    The full PAPI raw-data panel.

    Attributes
    ----------
    data : dict[int, pd.DataFrame]
        Mapping year → DataFrame (provinces × sub-criteria), read from CSV.
        This is the **read-only** source for the MCDM pipeline.
    regimes : dict[str, Regime]
        Regime metadata derived from column presence analysis.
    codebook : dict
        Parsed codebook: keys ``"provinces"``, ``"criteria"``, ``"subcriteria"``.
    """
    # Using Any to avoid importing pandas at module load time
    data: Dict[int, "pd.DataFrame"]  # type: ignore[name-defined]
    regimes: Dict[str, Regime]
    codebook: Dict[str, "pd.DataFrame"]  # type: ignore[name-defined]

    @property
    def years(self) -> List[int]:
        return sorted(self.data.keys())

    def get_year(self, year: int) -> "pd.DataFrame":  # type: ignore[name-defined]
        if year not in self.data:
            raise KeyError(f"Year {year} not in panel. Available: {self.years}")
        return self.data[year]


@dataclass
class SHAPResult:
    """
    SHAP explainability result for a single target sub-criterion.

    Represents SHAP values computed for one AutoGluon forecast target.
    Contains all necessary information to reproduce plots and analysis.

    Attributes
    ----------
    target_name : str
        Target sub-criterion code (e.g., 'SC11').
    shap_values : np.ndarray
        SHAP values array: shape (n_provinces, n_features).
        Each row corresponds to a province; each column to a covariate.
    base_values : float
        Base value (expected model output) from SHAP computation.
    feature_names : list[str]
        Ordered list of feature names (covariates).
    province_codes : list[str]
        Ordered list of province codes (P01, ..., P63).
    explainer_type : str
        Type of explainer used: "tree", "kernel", or "deep".
    n_background : int
        Number of background samples used (for KernelExplainer).
    """
    target_name: str
    shap_values: "np.ndarray"  # type: ignore[name-defined]
    base_values: float
    feature_names: List[str]
    province_codes: List[str]
    explainer_type: str
    n_background: int = 100

    def __post_init__(self) -> None:
        """Validate dimensions and consistency."""
        if self.shap_values.ndim != 2:
            raise ValueError(
                f"shap_values must be 2D array; got shape {self.shap_values.shape}"
            )
        n_provinces, n_features = self.shap_values.shape
        if n_provinces != len(self.province_codes):
            raise ValueError(
                f"shap_values first dimension ({n_provinces}) != "
                f"len(province_codes) ({len(self.province_codes)})"
            )
        if n_features != len(self.feature_names):
            raise ValueError(
                f"shap_values second dimension ({n_features}) != "
                f"len(feature_names) ({len(self.feature_names)})"
            )

    @property
    def n_provinces(self) -> int:
        return self.shap_values.shape[0]

    @property
    def n_features(self) -> int:
        return self.shap_values.shape[1]

    def global_importance(self) -> "np.ndarray":  # type: ignore[name-defined]
        """
        Compute global feature importance as mean absolute SHAP value.

        Returns
        -------
        np.ndarray
            1D array of shape (n_features,) with mean |SHAP| per feature.
        """
        import numpy as np
        return np.mean(np.abs(self.shap_values), axis=0)


@dataclass
class SHAPAggregation:
    """
    Aggregated SHAP importance across all targets.

    Summarizes feature importance across the 28 forecasting targets
    for global interpretation.

    Attributes
    ----------
    feature_names : list[str]
        All feature (covariate) names.
    target_names : list[str]
        All target sub-criterion names.
    mean_absolute_shap : np.ndarray
        Shape (n_features,): mean absolute SHAP per feature across all targets.
    feature_ranks : list[int]
        Rank of each feature by importance (1 = most important).
    """
    feature_names: List[str]
    target_names: List[str]
    mean_absolute_shap: "np.ndarray"  # type: ignore[name-defined]
    feature_ranks: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate dimensions."""
        if len(self.mean_absolute_shap) != len(self.feature_names):
            raise ValueError(
                f"mean_absolute_shap length ({len(self.mean_absolute_shap)}) != "
                f"len(feature_names) ({len(self.feature_names)})"
            )
        # Auto-compute feature ranks if not provided
        if not self.feature_ranks:
            import numpy as np
            # Rank descending (higher SHAP → lower rank number)
            self.feature_ranks = list(
                np.argsort(-self.mean_absolute_shap) + 1
            )

    def top_features(self, n: int = 10) -> List[tuple]:
        """
        Get top-n most important features by global SHAP magnitude.

        Parameters
        ----------
        n : int
            Number of top features to return. Default: 10.

        Returns
        -------
        list[tuple]
            List of (feature_name, mean_abs_shap) tuples, sorted by
            importance descending.
        """
        import numpy as np
        indices = np.argsort(-self.mean_absolute_shap)[:n]
        return [
            (self.feature_names[i], float(self.mean_absolute_shap[i]))
            for i in indices
        ]
