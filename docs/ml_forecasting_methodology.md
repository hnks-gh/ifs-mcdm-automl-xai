# Machine Learning Forecasting Methodology: AutoGluon Multivariate Time Series

## 1. Overview and Strategic Rationale

The ML forecasting module implements univariate time series prediction for each of the 29 PAPI sub-criteria using AutoGluon's TimeSeriesPredictor framework. The architecture treats prediction as 29 separate but interconnected forecasting tasks, with each sub-criterion as target variable and the remaining 28 sub-criteria as static covariates. This design rationale reflects: (1) each sub-criterion exhibits temporal dynamics requiring specialized time series modeling, (2) cross-sectional relationships exist among sub-criteria that can improve predictions through covariate information, and (3) univariate decomposition enables parallelization and interpretability while maintaining multivariate information flow.

The forecasting pipeline operates on a MICE-imputed historical panel (2011–2024, 882 observations across 63 provinces) and generates point forecasts for 2025 across all provinces and sub-criteria. The predictions serve two purposes: providing quantitative performance projections for governance indicators, and generating model predictions whose feature importance can be explained via SHAP values.

## 2. Data Preparation and Panel Structure

### 2.1 Historical Data Assembly

The forecasting module ingests a complete panel dataset of 882 observations (63 provinces × 14 years) with all 29 sub-criteria. This panel originates from MICE imputation of raw PAPI CSV data, ensuring zero missing values while preserving the integrity of observed historical values.

The long format representation organizes data as:

- **Rows**: 882 observations, each representing (province, year) combination
- **Columns**: 30 total (Province, Year, SC11 through SC83)
- **Time ordering**: Chronologically arranged by year then province ID

### 2.2 TimeSeriesDataFrame Construction

AutoGluon's TimeSeriesPredictor requires data in TimeSeriesDataFrame format, which is a specialized pandas DataFrame with mandatory index structure and semantic column definitions. For forecasting sub-criterion $j$ (e.g., SC11), the TimeSeriesDataFrame is constructed as:

- **item_id column**: Province identifiers (P01 through P63) representing 63 separate time series
- **timestamp column**: Year values (2011 through 2024) as temporal indices
- **target column**: Sub-criterion $j$ values (SC11 for example)
- **Static covariates**: Remaining 28 sub-criteria columns serve as exogenous features available both in training and prediction phases

Each of the 63 item_id values corresponds to a single province's temporal evolution, resulting in 63 separate but coupled time series. Training employs 14 time steps (years 2011–2024); prediction forecasts 1 additional time step (year 2025).

### 2.3 Data Integrity Verification

Before passing to TimeSeriesPredictor, verification confirms:

1. **Completeness**: Zero missing values in panel dataset (guaranteed by MICE imputation)
2. **Temporal ordering**: Years monotonically increasing from 2011 to 2024
3. **Item consistency**: Exactly 63 item_ids, each with exactly 14 observations
4. **Value range**: All sub-criteria values bounded within [0, 3.33], the valid PAPI range
5. **Dimensionality**: Feature count matches specification (29 sub-criteria total)

Any deviation from these specifications halts processing with diagnostic error logging.

## 3. AutoGluon TimeSeriesPredictor Architecture

AutoGluon's TimeSeriesPredictor implements an ensemble approach combining multiple base models, hyperparameter optimization, and automatic selection of the best-performing configuration.

### 3.1 Base Model Composition

The predictor can employ multiple base models, activated based on preset configuration:

**Temporal Convolutional Networks (TCN)**: Captures temporal dependencies through dilated convolutions, effective for capturing long-range patterns in time series.

**DeepAR**: Probabilistic deep learning model implementing autoregressive recurrent networks, particularly effective for multiple related time series with shared parameters.

**PatchTST**: Patchwise Time Series Transformer applying attention mechanisms to segmented time series, capturing both local and global temporal patterns.

**TemporalFusionTransformer (TFT)**: Advanced attention-based architecture combining historical and future contexts, designed specifically for multivariate time series.

**AutoETS**: Automatic Error-Trend-Seasonality model selection from classical exponential smoothing family, providing statistical interpretability.

**AutoARIMA**: Automatic ARIMA/SARIMA order selection via unit root testing and information criteria, leveraging classical time series theory.

**Naive and SeasonalNaive**: Baseline methods (persistence forecast, seasonal repetition) for comparison.

**Directional and Recursive Tabular**: Gradient boosting approaches treating time series prediction as tabular regression with engineered time-lag features.

For this framework, the "best_quality" preset activates comprehensive model search including deep learning, classical, and ensemble approaches. This preset automatically enables all available models and performs extensive hyperparameter optimization, consuming significant computational resources but yielding superior empirical performance.

### 3.2 Ensemble Strategy and Selection

AutoGluon's ensemble mechanism combines multiple trained models via weighted averaging or stacking. The best individual model (selected via validation set performance) is identified and used as the final predictor. Alternatively, ensemble combinations of top-performing models can be activated.

For this application, preset configuration emphasizes single-best-model selection with refit rather than heavy stacking, balancing predictive accuracy against computational cost and interpretability (SHAP explainability requires a single clear model for attribution).

### 3.3 Hyperparameter Optimization

AutoGluon employs HYPERBAND or similar efficient multi-fidelity optimization to search the hyperparameter space of each base model. Optimization objectives minimize validation set metrics (Mean Absolute Scaled Error—MASE, or specified metric). The MASE metric is scale-invariant and enables fair comparison across sub-criteria with different value ranges.

Optimization respects specified time budget and computational constraints, automatically allocating resources among models. More flexible models (neural networks) receive longer tuning time; simpler models receive minimal tuning.

### 3.4 Refit-Full Procedure

After model selection via train-validation split, AutoGluon optionally refits the best model on all historical data (train + validation combined) before final prediction. This "refit_full" procedure maximizes information utilization: since all 2011–2024 data are historical observations already realized, holding out validation data unnecessarily discards information. Refitting on the complete 14-year history provides the model maximum context for 2025 prediction.

Configuration specifies `refit_full: true`, enabling this behavior.

## 4. Covariate and Feature Engineering

### 4.1 Static Covariates Strategy

In TimeSeriesPredictor terminology, static covariates are exogenous variables that are available during both training and prediction phases. For each sub-criterion target $j$, the 28 remaining sub-criteria serve as static covariates.

For example, when predicting SC11 (a participation component), the model receives SC12, SC13, SC14 (other participation components), plus all criteria from other domains, as features. These covariates capture cross-sectional relationships: provinces performing well in one criterion often perform well in related criteria, providing informative signals.

Static covariates are incorporated via item-level features, allowing the model to learn province-specific relationships between target and covariates across time. This respects the panel structure where each province has its own temporal evolution.

### 4.2 Known Future Covariates

In time series terminology, "known future covariates" are variables whose future values are known at forecast time (e.g., seasonal indicators, planned policy changes). For 2025 PAPI forecasting, no known future covariates exist—the remaining sub-criteria values for 2025 are themselves predictions, not known quantities.

Configuration specifies `known_covariates_names: null`, indicating no such information is available. This is accurate: sub-criteria for 2025 are out-of-sample and must be predicted, not treated as known.

### 4.3 Feature Scaling and Normalization

AutoGluon internally applies feature scaling appropriate to each base model. Neural networks typically employ standardization; tree-based methods are scale-invariant. The framework automatically handles normalization, so raw PAPI values (bounded in [0, 3.33]) are compatible with all model types.

## 5. Prediction Process and Output Generation

### 5.1 Single-Target Forecasting Loop

For each of the 29 sub-criteria targets $j \in \{1, \ldots, 29\}$:

1. Construct TimeSeriesDataFrame with target = sub-criterion $j$, covariates = remaining 28 sub-criteria
2. Initialize TimeSeriesPredictor with configuration: `presets="best_quality"`, `eval_metric="MASE"`, `random_state=42`
3. Train predictor on the historical panel with `prediction_length=1` (forecast 1 year)
4. Select and refit best model on complete data
5. Generate point predictions for 2025 across all 63 provinces
6. Validate predictions remain within valid range [0, 3.33]
7. Store predictions in output structure indexed by province and sub-criterion

### 5.2 Prediction Output Structure

Individual sub-criterion predictions from TimeSeriesPredictor output as a time series indexed by timestamp and item_id. These are reshaped into a matrix of shape (63, 1) for each sub-criterion, then aggregated across all 29 sub-criteria into a single forecast table of shape (63, 29):

$$\mathbf{F}_{2025} = \begin{pmatrix}
F_{1,1} & F_{1,2} & \cdots & F_{1,29} \\
F_{2,1} & F_{2,2} & \cdots & F_{2,29} \\
\vdots & \vdots & \ddots & \vdots \\
F_{63,1} & F_{63,2} & \cdots & F_{63,29}
\end{pmatrix}$$

where $F_{i,j}$ is the 2025 forecast for province $i$ and sub-criterion $j$. Rows correspond to provinces (P01 through P63); columns correspond to sub-criteria (SC11 through SC83).

### 5.3 Output Validation and Bounds Checking

Post-prediction validation verifies:

1. **Shape**: Exactly (63, 29) dimensions
2. **Data types**: Numeric values (float64)
3. **Value range**: All entries in [0, 3.33] or slightly beyond (tolerance for model output)
4. **Missing values**: Zero NaN entries

If predictions exceed [0, 3.33] bounds by small margins (e.g., due to model extrapolation), soft clipping applies: values > 3.33 set to 3.33, values < 0 set to 0. This preserves model learning while respecting domain constraints.

## 6. Temporal Validation and Backtesting Strategy

### 6.1 Conceptual Framework

The historical panel (2011–2024) represents fully observed data, not a proper train-test split for forecasting validation. Practical validation of forecasting accuracy is limited by data availability. However, internal cross-validation during AutoGluon training provides indication of model fit quality.

### 6.2 Internal Validation During Training

TimeSeriesPredictor employs time series cross-validation with rolling windows or train-validation splits. AutoGluon automatically creates validation sets from the training period (e.g., reserving 2023–2024 as validation, training on 2011–2022) to evaluate model performance without data leakage.

Metrics computed during internal validation (e.g., MASE on validation set) serve as proxy for 2025 forecast quality. Better internal validation performance is statistically associated with better out-of-sample forecast accuracy.

### 6.3 Out-of-Sample 2025 Prediction

The actual 2025 predictions represent true out-of-sample forecasts, with no ground truth available for retrospective accuracy assessment. This is inherent to genuine forecasting: we cannot validate predictions before the future year occurs.

Framework design acknowledges this by emphasizing internal validation quality and model ensemble robustness. SHAP explainability provides interpretable decomposition of forecast values, enabling expert judgment of reasonableness.

## 7. Key Algorithmic Parameters

Configuration parameters specified in `config/config.yaml` control the forecasting process:

**prediction_length: 1** specifies single-year horizon (2025).

**presets: "best_quality"** activates comprehensive model search and optimization. Alternative presets ("fast_training", "medium_quality") provide faster training at reduced accuracy.

**refit_full: true** enables refitting best model on complete historical data before 2025 prediction.

**eval_metric: "MASE"** (Mean Absolute Scaled Error) serves as AutoGluon's objective function. MASE normalizes by naive forecast mean absolute error, enabling scale-invariant comparison.

**random_state: 42** fixes all random number generation for reproducibility. Same configuration produces identical results across runs.

## 8. Ensemble Model Considerations

### 8.1 Why Ensemble Approach

Individual model types have complementary strengths: deep learning captures complex nonlinear patterns but requires careful regularization, classical methods provide statistical theory and interpretability but assume rigid distributional forms, tree-based models are robust to outliers and interactions but may underfit smooth trends. Ensemble combination leverages these complementary strengths.

### 8.2 Model Aggregation

AutoGluon's ensemble typically averages predictions from multiple models, weighting by internal validation performance. Better-performing models receive higher weight. This weighted averaging reduces variance from any single model's idiosyncratic forecasts.

### 8.3 Best-Model vs. Ensemble Selection

For this framework, after ensemble training, the single best-performing model is selected and refitted for final prediction. This decision prioritizes interpretability (single clear model for SHAP attribution) and reproducibility over incremental accuracy gains from ensemble voting.

Alternative designs could employ ensemble predictions; this would require extending SHAP explanation methodology to multiple models, increasing complexity.

## 9. Handling Structural Missingness (2025 Covariates)

### 9.1 Covariate Prediction Problem

When predicting 2025 for target sub-criterion $j$, the model requires 2025 values of the 28 covariate sub-criteria. However, these are themselves future values that must be predicted, not known a priori.

### 9.2 Solution: Iterative Prediction or Covariate Model Training

Two approaches address this:

**Approach 1 (Implemented)**: Train 29 separate models, each using other 28 sub-criteria as covariates. At prediction time, use the 28 predicted values from those models as covariate inputs. This creates a coupled prediction system where all 29 targets are interdependent through covariate relationships.

**Approach 2 (Alternative)**: Train a single multi-output model predicting all 29 targets simultaneously. This would require reformulating the problem as multi-task learning, increasing complexity.

This implementation employs Approach 1: iterate through all 29 sub-criteria, training each with estimated covariates from previous iterations. A single pass is typically sufficient since all predictions are for the same future year and covariate prediction errors are bounded.

Ordering iterations by criterion importance (from Stage 2 weights) ensures important criteria are predicted first with lower covariate error.

## 10. Computational Complexity and Resource Requirements

### 10.1 Training Complexity

Training 29 separate TimeSeriesPredictor models, each exploring multiple base model types and hyperparameter configurations, is computationally intensive. Typical wall-clock time for complete forecasting module:

- Per sub-criterion model: 5–15 minutes (depending on hardware and preset aggressiveness)
- All 29 sub-criteria: 2–8 hours (7.5 hours typical on standard CPU)
- GPU acceleration reduces by 3–5x if available

### 10.2 Memory Requirements

Peak memory usage includes:

- Historical panel in RAM: ~50 MB
- Trained model artifacts: ~100–500 MB (depending on model complexity)
- Temporary training batches: ~200–500 MB

Total memory requirement is typically 1–2 GB, manageable on modern systems.

### 10.3 Parallelization Opportunities

The 29 independent forecasting tasks can be parallelized across multiple processes or machines. Distributed training via joblib or multiprocessing reduces total training time approximately linearly with available cores (up to 29-fold speedup with 29 processors).

## 11. Output Artifacts

Forecasting module produces two primary outputs:

**Forecast Table** (`output/ml/forecasts/forecast_2025.csv`): 63×29 table with rows=provinces, columns=sub-criteria, values=2025 point predictions.

**Model Artifacts** (`output/ml/forecasts/models/`): Serialized AutoGluon predictors for each sub-criterion, enabling retrospective SHAP computation without retraining.

## 12. Validation and Quality Assurance

### 12.1 Mathematical Properties

1. **Dimensionality**: Output shape (63, 29) matches input specification
2. **Range**: All predicted values bounded in [0, 3.33] (within tolerance)
3. **No missing values**: Zero NaN entries in forecast table
4. **Data types**: Float64 numeric values

### 12.2 Reasonableness Checks

1. **Mean value consistency**: 2025 forecast mean similar to historical mean (prevents systematic under/over prediction)
2. **Variance consistency**: 2025 forecast variance reasonable relative to historical variance (prevents collapse to constant prediction)
3. **Province ranking consistency**: Province relative rankings in 2025 show reasonable correlation to recent historical rankings (prevents spurious reordering)
4. **Trend persistence**: Sub-criteria with historical upward/downward trends show corresponding forecast trends (captures basic temporal patterns)

## 13. References and Theoretical Grounding

AutoGluon TimeSeriesPredictor implements state-of-the-art ensemble approaches combining neural networks, classical statistical models, and machine learning methods. Architecture leverages findings from AutoML literature demonstrating that ensemble methods and automated hyperparameter optimization produce superior performance across diverse forecasting tasks. MASE metric selection follows best practices in time series evaluation, providing scale-invariant error quantification across sub-criteria with different value ranges and variance profiles.
