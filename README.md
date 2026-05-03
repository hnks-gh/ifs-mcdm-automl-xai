# IFS-MCDM-AutoML-XAI Framework: Integrated Analytical System for Indicator Sets

## Project Overview

This repository implements an integrated analytical framework combining Intuitionistic Fuzzy Set (IFS) Multi-Criteria Decision Making (MCDM), Automated Machine Learning, and Explainable Artificial Intelligence (XAI). The framework is designed for comprehensive analysis of multi-dimensional indicator datasets, with empirical evaluation on the Vietnam Provincial Governance and Public Administration Performance Index (PAPI) covering the period 2011–2024 across 63 provinces and 29 sub-criteria.

### Core Objectives

The framework addresses four fundamental analytical dimensions: hierarchical weighting of decision criteria using advanced fuzzy mathematical methods, multi-method ranking comparison to assess ranking consistency and discriminatory power, multivariate time series forecasting of future performance indicators, and model-agnostic explainability to understand feature contributions to forecasts.

### Framework Architecture

The system operates through four integrated analytical pipelines: the MCDM Pipeline performs two-level IF-CRITIC weighting and applies three ranking methods (IF-WASPAS, IF-TOPSIS, IF-PROMETHEE II) to generate consensus rankings; the Weighting Analysis Pipeline conducts temporal stability assessment and Monte Carlo sensitivity analysis to evaluate weight robustness; the Ranking Analysis Pipeline computes inter-method agreement, discriminatory power metrics, and temporal persistence to validate ranking quality; the ML Pipeline executes MICE imputation on the historical panel, trains AutoGluon multivariate time series forecasting models for future predictions, and applies SHAP explainability to interpret model decisions.

## Dataset Characteristics

The PAPI dataset comprises 882 observations (63 provinces × 14 years, 2011–2024) structured as a balanced panel. The 29 sub-criteria are organized hierarchically into 8 parent criteria and aggregate into a composite governance index. Missingness patterns exhibit structural (column-level absence in certain years) and occasional (individual cell blanks) characteristics, motivating the separation of analytical paths: MCDM uses raw data with regime-specific handling, while ML employs MICE imputation for forecasting.

## Installation and Environment

### Prerequisites

The framework requires Python 3.11 or later. All dependencies are specified in `requirements.txt` with pinned versions to ensure reproducibility. Key dependencies include NumPy and Pandas for data manipulation, scikit-learn for imputation, AutoGluon for time series forecasting, SHAP for explainability, and Matplotlib/Seaborn for visualization.

### Setup Instructions

Clone the repository and install dependencies via pip in a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Configuration parameters are centralized in `config/config.yaml` and may be customized per analysis requirements. Logging configuration resides in `config/logging.yaml`.

## Project Structure

The source code is organized into modular packages under `src/`:

The `core` package provides foundational utilities: `data_loader.py` handles CSV ingestion and codebook parsing, `preprocessor.py` normalizes scores and converts to IFS representation, `ifs_arithmetic.py` implements all fuzzy mathematical operations, `schema.py` defines Pydantic data models and configuration structures, and `exceptions.py` establishes custom exception hierarchy.

The `mcdm` package contains domain-specific MCDM implementations: the `weighting` subpackage implements IF-CRITIC algorithm across two hierarchical levels; the `ranking` subpackage implements IF-WASPAS, IF-TOPSIS, and IF-PROMETHEE II methods; the `analysis` subpackage provides temporal stability assessment, Monte Carlo sensitivity analysis, and ranking validation metrics.

The `ml` package orchestrates machine learning operations: the `imputation` subpackage performs MICE-based panel imputation; the `forecasting` subpackage manages AutoGluon time series prediction loop for all sub-criteria targets; the `explainability` subpackage computes and visualizes SHAP values with automatic explainer type detection.

The `pipeline` package contains orchestration logic: `mcdm_pipeline.py` sequences weighting and ranking operations, `ml_pipeline.py` chains imputation through forecasting to explainability, `runner.py` provides command-line interface, and `visualization_orchestration.py` manages figure generation.

The `utils` package supplies cross-cutting concerns: `logger.py` implements structured logging, `io_utils.py` manages file operations, `plot_utils.py` provides matplotlib wrappers, `stats_utils.py` offers statistical helpers, `math_utils.py` contains general numerical utilities, and `validators.py` implements input verification.

Output artifacts are organized by analytical type in `output/`: MCDM results (weights, rankings, analyses) in `mcdm/`, ML results (imputed panel, forecasts, SHAP values) in `ml/`, publication-ready figures in `figures/`, and aggregated HTML reports in `reports/`.

## Core Methodology

### MCDM Approach: Intuitionistic Fuzzy Two-Level CRITIC Weighting

The weighting methodology implements IF-CRITIC algorithm across two hierarchical levels. Stage 1 computes sub-criteria weights within each of the 8 parent criteria by analyzing intrinsic variability (standard deviation of IFS score functions) and relative importance (through adapted Pearson correlation of score functions), producing within-criterion weights. Stage 2 aggregates criteria-level IFS composites using Stage 1 weights, then applies CRITIC analysis on these aggregates to determine criteria-level weights. Final sub-criteria weights result from Stage 1 and Stage 2 weight products. The two-level architecture respects the hierarchical structure of the governance index while automatically adapting to structural missingness patterns through regime-specific weight computation.

### Ranking Methods: Multi-Method Consensus Approach

Three distinct ranking methods provide complementary perspectives on provincial performance. IF-WASPAS blends arithmetic (WSM) and geometric (WPM) aggregation of weighted criteria with balancing parameter lambda (default 0.5), accommodating both compensatory and conjunctive decision preferences. IF-TOPSIS applies Euclidean distance metrics to IFS-valued criteria relative to ideal and anti-ideal solutions, emphasizing minimization of distance to positive ideal and maximization from negative ideal. IF-PROMETHEE II constructs pairwise preference matrices using preference functions (Gaussian in this implementation) and derives net outranking flows, enabling partial ranking and non-compensatory comparison. Multi-method comparison quantifies inter-method agreement through Spearman correlation, assesses discriminatory power via inter-quartile range of score distributions, and measures temporal persistence through year-to-year rank correlation.

### Time Series Forecasting: AutoGluon Multivariate Architecture

The forecasting component treats prediction as a univariate time series problem replicated for each of the 29 sub-criteria, with the remaining 28 sub-criteria serving as static covariates within the AutoGluon TimeSeriesPredictor framework. This architecture acknowledges temporal dependencies within each sub-criterion while capturing cross-sectional relationships through covariate structure. The system trains on 14 historical years (2011–2024) and predicts 2025 values across all 63 provinces. Preset configuration "best_quality" activates an ensemble of multiple base models (DeepAR, PatchTST, TemporalFusionTransformer, AutoETS, AutoARIMA, among others) with hyperparameter optimization, selecting the best performer and refitting on complete historical data before final prediction.

### Explainability: SHAP Model-Agnostic Interpretation

SHAP explainability provides instance-level and global feature importance estimates. The framework automatically detects the underlying model architecture from AutoGluon's best model and selects appropriate explainer type (TreeExplainer for tree-based models, KernelExplainer for neural architectures). SHAP values represent each feature's marginal contribution to prediction, enabling comparison of influence across provinces and identification of consistently important sub-criteria. Visualizations include global feature importance rankings (mean absolute SHAP), beeswarm plots showing per-observation SHAP distributions, and waterfall plots decomposing predictions for selected provinces.

## Running the Framework

### MCDM Pipeline Execution

Launch the MCDM pipeline via command line:

```
python main.py --pipeline mcdm --config config/config.yaml --output output/
```

This executes the complete MCDM workflow: loads raw PAPI data, detects temporal regimes based on sub-criteria availability, converts numerical scores to IFS representation, computes two-level IF-CRITIC weights, applies three ranking methods, and generates analysis outputs including temporal stability, sensitivity analysis, and ranking validation metrics.

### ML Pipeline Execution

Execute the complete machine learning pipeline:

```
python main.py --pipeline ml --config config/config.yaml --output output/
```

This performs sequential operations: loads raw panel data, runs MICE imputation producing fully-observed panel dataset, trains AutoGluon forecasting models for each of 29 sub-criteria targets, generates 2025 point predictions for all provinces, and computes SHAP explainability values.

### Full Pipeline Execution

Execute all analytical components:

```
python main.py --pipeline all --config config/config.yaml --output output/
```

## Documentation

### Technical Specifications

Detailed mathematical formulations and algorithmic specifications are documented in the following technical reports:

**docs/weighting_methodology.md** provides comprehensive mathematical exposition of the two-level IF-CRITIC weighting algorithm, including IFS conversion, standard deviation computation on score functions, correlation matrix derivation, CRITIC information measure, stage-specific weight calculation, regime-based aggregation, and numerical validation properties.

**docs/ranking_methodology.md** (extended from existing docs/ranking_methods.md) presents complete mathematical derivations of IF-WASPAS (WSM and WPM component formulas, IFS addition operation), IF-TOPSIS (ideal/anti-ideal solution construction, distance computation, closeness coefficient), and IF-PROMETHEE II (preference functions, flow calculation, outranking relations), with interpretation guidance and parameter selection rationale.

**docs/ml_forecasting_methodology.md** describes the AutoGluon multivariate time series architecture, including TimeSeriesDataFrame construction, covariate specification, base model composition, hyperparameter optimization strategy, ensemble aggregation, and refit-full procedure to ensure 2025 predictions leverage complete historical information.

**docs/explainability_methodology.md** explains SHAP theoretical foundations under the Shapley value framework, automatic explainer type selection heuristics, background data sampling strategy, interpretation of SHAP values for instance-level and global explanations, and visualization types employed in this framework.

**docs/analysis_validation_methodology.md** covers four critical components: temporal stability analysis using rolling windows and root-mean-square-deviation metrics, Monte Carlo sensitivity analysis via Dirichlet perturbation with weighted Kendall's tau-b correlation, ranking validation including inter-method Spearman correlations and discriminatory power metrics, and comprehensive testing strategy spanning unit tests, integration tests, and mathematical property validation.

**docs/data.md** documents dataset structure, missingness patterns, regime definitions, sub-criteria to criteria hierarchy, and data integrity constraints.

### Demonstration Scripts

The `scripts/` directory contains demonstration scripts illustrating each framework component on representative data:

`scripts/01_data_exploration.py` loads complete PAPI dataset, produces summary statistics, visualizes missingness patterns, and reports regime boundaries.

`scripts/02_mcdm_weighting_demo.py` applies two-level IF-CRITIC weighting to a single complete year (2019), displaying Stage 1 and Stage 2 weight matrices with interpretation.

`scripts/03_mcdm_ranking_demo.py` generates province rankings using all three ranking methods for a single year, comparing rank assignments and computing inter-method agreement metrics.

`scripts/04_ml_forecasting_demo.py` demonstrates MICE imputation, AutoGluon training on a single sub-criterion target, and 2025 prediction generation for selected provinces.

`scripts/05_shap_explainability_demo.py` computes SHAP values for forecasted sub-criterion, generates global importance ranking and per-province waterfall explanations.

`scripts/06_phase10_visualization_demo.py` produces all publication-ready figures organized by analytical type (weighting, ranking, forecasting, explainability).

## Testing and Quality Assurance

Comprehensive testing ensures mathematical correctness and operational reliability across all framework components. Unit tests in `tests/unit/` verify individual components: `test_ifs_arithmetic.py` validates all fuzzy operations against analytical solutions, `test_if_critic.py` confirms weight computation and normalization properties, `test_ranking_*` files verify each ranking method's correctness on small analytical examples, `test_mice_imputer.py` ensures zero post-imputation missingness and value range preservation, `test_autogluon_forecaster.py` validates data construction and output shapes, `test_shap_explainer.py` verifies SHAP value computation and aggregation.

Integration tests in `tests/integration/` validate end-to-end workflows: `test_mcdm_pipeline.py` confirms complete weighting and ranking pipeline on synthetic mini-dataset, `test_ml_pipeline.py` validates imputation through forecasting to explainability on representative data, `test_full_pipeline.py` executes smoke test covering both MCDM and ML pipelines sequentially.

Mathematical validation confirms: all weight vectors sum to 1.0 within floating-point tolerance, all IFS values satisfy constraint $\mu + \nu + \pi = 1$ with $\mu, \nu, \pi \geq 0$, all rankings are permutations of 1 to 63 with no gaps or duplicates, imputed panel contains zero missing values with all imputed values bounded in valid range, forecast output dimensions match specification (63 provinces × 29 sub-criteria), SHAP values have expected dimensionality (63 provinces × 28 covariates per target).

## Key Design Principles

### Data Integrity and Separation

The framework maintains strict separation between MCDM and ML analytical paths. MCDM operations use exclusively raw PAPI CSV data with regime-specific handling of structural missingness, ensuring weight and ranking computation remain grounded in actual observed values. ML operations utilize the MICE-imputed panel stored in separate output directory, preventing any contamination of MCDM path. This design enforces data authenticity for MCDM while enabling complete-panel forecasting for ML.

### Configuration-Driven Implementation

All algorithmic parameters, file paths, and system hyperparameters are externalized to `config/config.yaml`, enabling reproducible studies and sensitivity analysis without code modification. Configuration is loaded via Pydantic v2 schema, providing type checking and validation at runtime.

### Regime-Aware Analysis

Temporal regime concept acknowledges structural changes in data availability across the study period. Four regimes (R1: 2011–2017, R2: 2018, R3: 2019–2020, R4: 2021–2024) reflect sub-criteria availability patterns. MCDM analysis computes regime-specific weights and rankings, then blends results proportionally by regime duration, maintaining methodological integrity despite changing sub-criteria sets.

### Reproducibility and Logging

All random number generation employs fixed seeds specified in configuration (`random_state: 42`). Comprehensive structured logging via Loguru records pipeline execution timestamps, input/output shapes, intermediate metrics, and anomalies to `logs/` directory, enabling complete audit trail and troubleshooting.

## Publications and References

The framework implements methodologies from peer-reviewed literature in fuzzy MCDM, time series forecasting, and explainable AI. Specific mathematical formulations follow established conventions in intuitionistic fuzzy set theory and multi-criteria decision-making domains. TimeSeriesPredictor architecture leverages AutoGluon research demonstrating superior empirical performance on diverse forecasting tasks. SHAP explainability builds on Shapley value game-theoretic foundations providing consistent and locally accurate explanations.

## Authors and Contributions

This framework was developed as an integrated research platform for empirical evaluation of governance indicators using advanced computational methods combining classical decision science with modern machine learning. The implementation reflects years of specialized expertise in fuzzy set mathematics, MCDM methodology, machine learning architecture, and explainable AI.

## License and Usage

The framework is provided for research and educational purposes. All code implements standard algorithms from published literature and established software libraries. Users should cite this work appropriately when publishing research utilizing the framework.

## Contact and Support

For technical questions, bug reports, or methodology clarifications, please refer to technical documentation in the `docs/` directory or examine relevant source code files with inline mathematical annotations.
