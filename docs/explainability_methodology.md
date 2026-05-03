# Explainability Methodology: SHAP Model-Agnostic Feature Attribution

## 1. Theoretical Foundation and Motivation

SHAP (SHapley Additive exPlanations) provides theoretically grounded feature attribution explaining machine learning model predictions through Shapley values from cooperative game theory. SHAP values quantify each feature's marginal contribution to moving a prediction from the base (average) value toward the actual predicted value.

For forecasting models predicting 2025 PAPI sub-criteria, SHAP explainability answers critical questions: Which historical sub-criteria most strongly influence 2025 predictions? Do important features differ across provinces or sub-criteria targets? Can we identify systematic biases in model predictions?

The SHAP framework operates model-agnostically: the same explanation methodology applies whether the underlying predictor is tree-based (DeepAR ensemble), neural network (PatchTST), or classical (AutoETS). This enables consistent interpretation across AutoGluon's diverse base models.

## 2. Shapley Value Theory and SHAP Framework

### 2.1 Shapley Value Definition

Shapley values originated in cooperative game theory to fairly distribute payouts among coalition members. In machine learning context, features are "players" and the model prediction is the "payoff."

Given a model prediction $f(\mathbf{x}) = \hat{y}$ for input features $\mathbf{x} = (x_1, \ldots, x_p)$, the Shapley value for feature $j$ is:

$$\phi_j = \sum_{S \subseteq \{1,\ldots,p\} \setminus \{j\}} \frac{|S|!(p-|S|-1)!}{p!} \left[ f(S \cup \{j\}) - f(S) \right]$$

where the sum is over all possible coalitions $S$ (subsets of features excluding $j$), and $f(S)$ denotes model prediction using only features in coalition $S$ (with others marginalised via expectation).

This formula embodies fairness axioms:

**Efficiency**: $\sum_{j=1}^{p} \phi_j = f(\mathbf{x}) - E[f(\mathbf{X})]$. Shapley values sum to the total deviation from baseline prediction.

**Symmetry**: If two features contribute identically across all coalitions, they receive equal Shapley values.

**Dummy**: Features that never change prediction receive zero Shapley value.

**Additivity**: For models expressible as sums of subfunctions, Shapley values decompose additively.

### 2.2 SHAP Approximation Strategies

Exact Shapley value computation requires evaluating $2^p$ coalitions (exponential in feature count), becoming intractable for $p > 20$. SHAP implements several approximation strategies.

**TreeExplainer**: For tree-based models, exploits tree structure to compute Shapley values in polynomial time. Efficient for random forests, gradient boosting, and ensemble tree models. Primary advantage: speed (typically milliseconds per instance).

**KernelExplainer**: Model-agnostic approach using weighted local linear regression to approximate Shapley values. Works with any model but computationally expensive (microsecond to second per instance depending on sample size).

**DeepExplainer**: Specialized for deep neural networks, using backpropagation to compute gradient-based attribution. Not applicable here since AutoGluon's neural models are treated as black-boxes.

For this framework, automatic explainer selection chooses TreeExplainer if AutoGluon's best model is tree-based, otherwise defaults to KernelExplainer.

## 3. Explainer Type Detection and Selection

### 3.1 Model Type Classification

After AutoGluon training completes, the best model is identified and its type determined. Classification logic:

**Tree-Based Models**: Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost), GBDT variants. These expose tree structure enabling TreeExplainer.

**Neural Network Models**: DeepAR, PatchTST, TemporalFusionTransformer, other deep learning architectures. No direct tree structure; use KernelExplainer.

**Classical Statistical Models**: AutoETS, AutoARIMA. Treated as generic models, use KernelExplainer.

**Ensemble Models**: If final predictor is weighted ensemble of multiple models, heuristic selects explainer based on dominant model type. If ensemble contains mix of tree and neural, defaults to KernelExplainer for consistency.

### 3.2 TreeExplainer Strategy

When tree-based model identified, instantiate TreeExplainer via SHAP library passing the trained model object. TreeExplainer automatically extracts tree structure and computes exact Shapley values efficiently.

For ensemble trees (random forests, boosted ensembles), TreeExplainer operates on aggregated tree structure, computing Shapley values with respect to final ensemble output.

TreeExplainer is preferred due to computational efficiency and theoretical exactness within tree model's decision boundaries.

### 3.3 KernelExplainer Strategy

For neural and classical models, instantiate KernelExplainer with:

**Model wrapper**: Function accepting feature matrix and returning predictions (single target)

**Background data**: Reference dataset representing "typical" feature distributions (detailed in Section 4)

**Feature names**: Explicit labels for interpretability (sub-criteria names SC11, SC12, etc.)

KernelExplainer generates synthetic perturbations of input features, evaluates model predictions on perturbed instances, and performs weighted linear regression to approximate Shapley values. Accuracy improves with larger background datasets and more perturbation samples.

## 4. Background Data Sampling

### 4.1 Role of Background Data

Background data $\mathbf{B} = \{\mathbf{b}_1, \ldots, \mathbf{b}_m\}$ represents the reference distribution used for marginalizing features. When computing Shapley value contribution of feature $j$, the explainer marginalizes other features by averaging over background data:

$$f(S) \approx E_{\mathbf{b} \in \mathbf{B}} [f(\mathbf{x}_{S} \cup \mathbf{b}_{S^c})]$$

where $\mathbf{x}_S$ denotes features in coalition $S$ from instance $\mathbf{x}$, and $\mathbf{b}_{S^c}$ denotes features outside $S$ from background instance $\mathbf{b}$.

Background data quality significantly impacts SHAP value accuracy and interpretability. Poor background data (unrepresentative or skewed) produces unreliable Shapley approximations.

### 4.2 Background Sampling Procedure

For time series forecasting context, background data should represent typical provincial time series patterns:

1. **Temporal subset**: Sample background data from historical years (2011–2024) rather than artificially constructed values.

2. **Stratified by province**: Ensure background data includes representation from all 63 provinces, capturing heterogeneous provincial characteristics.

3. **Sample size**: Configuration specifies `n_samples: 100`. This draws 100 random (province-year) observations from historical panel, balancing statistical representativeness against computational cost.

4. **Value distribution**: Background sample should respect empirical distribution of sub-criteria values. Random sampling automatically preserves marginal distributions.

Sampling procedure:

```
For each of 100 samples:
  - Randomly select province i uniformly from {1,...,63}
  - Randomly select year t uniformly from {2011,...,2024}
  - Append (province i, year t) row from historical panel to background set
```

Result is 100×28 matrix (100 samples, 28 covariate sub-criteria) representing typical conditions the model encountered during training.

### 4.3 Background Data Validation

Verify background data satisfies:

1. **Completeness**: Zero missing values
2. **Value range**: All values in [0, 3.33]
3. **Representativeness**: Mean and variance of background data reasonably match historical panel
4. **Independence**: Samples statistically representative of empirical distribution (avoid clustering or bias)

## 5. SHAP Value Computation

### 5.1 Single-Instance Explanation

For each (province, sub-criterion target) combination, compute SHAP values with respect to historical feature values:

Given prediction $\hat{y}_{i,j}^{2025}$ for province $i$ forecasted sub-criterion $j$, and using historical features (other 28 sub-criteria) as $\mathbf{x}_{i,t}$:

Compute $\phi_k^{(i,j)} = $ Shapley value for feature (sub-criterion) $k$ in explaining prediction for province $i$, target $j$.

These values satisfy:

$$\sum_{k=1}^{28} \phi_k^{(i,j)} = \hat{y}_{i,j}^{2025} - E[\hat{y}_j^{2025}]$$

where $E[\hat{y}_j^{2025}]$ is baseline prediction (mean forecast across all provinces for target $j$).

### 5.2 Batch Computation

Compute SHAP values for all 63 provinces simultaneously using vectorized batch evaluation. SHAP library enables batch processing, computing Shapley values for multiple instances efficiently:

Input: 63×28 matrix of features (63 provinces, 28 covariates)
Output: 63×28 matrix of Shapley values

This batch computation reduces overall runtime by avoiding repeated model initialization and enabling vectorized operations.

### 5.3 Per-Target Computation

Repeat SHAP computation for each of 29 sub-criteria targets:

For each target $j \in \{1, \ldots, 29\}$:
  1. Load trained predictor for target $j$
  2. Extract 2025 predictions and covariate features
  3. Instantiate explainer (TreeExplainer or KernelExplainer)
  4. Compute batch Shapley values for 63 provinces
  5. Store SHAP values and explanatory metadata

Total computation: 29 targets × 63 provinces = 1,827 individual instances explained.

## 6. Global and Local Feature Importance

### 6.1 Global Feature Importance

Aggregate SHAP values across all provinces to determine overall feature importance for target $j$:

$$I_k^{(j)} = \frac{1}{63} \sum_{i=1}^{63} |\phi_k^{(i,j)}|$$

Mean absolute SHAP value represents average magnitude of feature $k$'s contribution across provinces. Features with high $I_k^{(j)}$ consistently influence predictions for target $j$, regardless of direction (positive or negative).

**Global Feature Importance Ranking**: Sort features by $I_k^{(j)}$ in descending order to identify most influential sub-criteria for each target.

**Cross-Target Importance**: Compute $\bar{I}_k = \frac{1}{29} \sum_{j=1}^{29} I_k^{(j)}$ representing feature $k$'s average importance across all targets. This identifies sub-criteria that universally influence forecasting quality.

### 6.2 Local Feature Importance (Per-Province)

For individual province $i$ and target $j$, SHAP values $\phi_k^{(i,j)}$ represent feature importance specific to that instance. Province-specific importance may differ from global average if province exhibits atypical feature relationships.

**Instance-level interpretation**: Positive SHAP value ($\phi_k > 0$) indicates feature $k$ increases prediction toward higher 2025 value; negative ($\phi_k < 0$) indicates feature reduces prediction.

**Province comparison**: Compare SHAP values for same feature across different provinces to identify heterogeneous relationships. If sub-criterion SC11 has high positive SHAP for province 1 but negative for province 2, forecasting relationships differ by province.

## 7. SHAP Visualizations

### 7.1 Global Importance Bar Plot

**Purpose**: Display feature importance ranking for each sub-criterion target.

**Construction**: For target $j$, compute $I_k^{(j)}$ for all 28 covariates, sort descending, and create horizontal bar chart with feature names as labels and mean |SHAP| as bar lengths.

**Interpretation**: Longer bars indicate features with stronger average influence on predictions. Top-ranked features are most critical for model decisions.

**Output**: 29 separate plots (one per target) or single aggregated plot if cross-target importance desired.

### 7.2 SHAP Beeswarm Plot

**Purpose**: Display distribution of SHAP values across provinces for a specific target and feature subset.

**Construction**: For target $j$ and top-K features (e.g., K=10), create scatter plot with:
- X-axis: SHAP value (magnitude and sign)
- Y-axis: Feature name
- Color: Encoded original feature value (red=high, blue=low) or outcome direction

Each point represents one province's SHAP value for that feature-target pair. Multiple points vertically stacked indicate heterogeneous importance across provinces.

**Interpretation**: Concentration near zero indicates inconsistent influence; spread left/right indicates mixed positive/negative relationships; color gradient reveals association between feature value and prediction direction.

**Output**: 1 beeswarm plot per target (or subset for readability).

### 7.3 SHAP Waterfall Plot

**Purpose**: Decompose prediction for individual province into feature contributions.

**Construction**: For province $i$ forecasted target $j$:
1. Baseline value: $E[\hat{y}_j^{2025}]$ (mean forecast)
2. Feature contributions: Display each feature's SHAP value as horizontal bar from baseline
3. Sequence: Order features by absolute SHAP value (largest first)
4. Final prediction: $E[\hat{y}_j^{2025}] + \sum_k \phi_k^{(i,j)}$

Bars extending right increase prediction; bars extending left decrease prediction.

**Interpretation**: Immediate visual decomposition showing which features most substantially push 2025 prediction higher or lower for specific province. Useful for understanding atypical predictions or validating reasonableness.

**Output**: 1 waterfall plot per selected province (typically top 5 by prediction magnitude or heterogeneity).

## 8. Mathematical Properties and Validation

### 8.1 Shapley Value Verification

Verify computed SHAP values satisfy theoretical properties:

**Efficiency**: $\sum_{k=1}^{28} \phi_k^{(i,j)} \approx \hat{y}_{i,j}^{2025} - E[\hat{y}_j^{2025}]$ within numerical tolerance.

**Consistency**: If two features $k_1, k_2$ contribute identically across all model behaviors, $\phi_{k_1}^{(i,j)} \approx \phi_{k_2}^{(i,j)}$ for all provinces $i$.

**Dummy**: Features with zero contribution (constant or unused by model) have $\phi_k \approx 0$.

### 8.2 Implementation Artifacts

**TreeExplainer** guarantees exact Shapley values within the tree structure. Numerical errors are minimal (machine epsilon level).

**KernelExplainer** produces approximations whose accuracy depends on:
- Background data size: Larger background improves accuracy
- Perturbation samples: More synthetic perturbed instances improve accuracy
- Model complexity: Linear models are exactly explained; nonlinear models approximate

Configuration `n_samples: 100` provides reasonable approximation accuracy for governance forecasting context where interpretability suffices (not high-precision financial applications).

## 9. Output Artifacts

Explainability module produces multiple outputs:

**SHAP Values Matrix** (`output/ml/shap/shap_values_target_j.parquet`): 63×28 matrix of Shapley values for target $j$, rows=provinces, columns=covariates.

**Baseline Values** (`output/ml/shap/baseline_values.csv`): Expected value $E[\hat{y}_j^{2025}]$ for each target $j$, representing baseline prediction before feature contributions.

**Feature Names** (`output/ml/shap/feature_names.json`): Explicit mapping of covariate indices to sub-criteria names for interpretation.

**Visualizations**:
- Global importance bar plots: `output/figures/ml/shap_global_importance_target_j.png`
- Beeswarm plots: `output/figures/ml/shap_beeswarm_target_j.png`
- Waterfall plots for top provinces: `output/figures/ml/shap_waterfall_province_i_target_j.png`

## 10. Computational Complexity

### 10.1 TreeExplainer Performance

For tree-based AutoGluon models, TreeExplainer computation is very fast: approximately 0.001–0.01 seconds per instance. Computing SHAP values for 63 provinces × 29 targets requires approximately 1–3 minutes total.

### 10.2 KernelExplainer Performance

For neural and classical models using KernelExplainer, computation is slower: approximately 0.1–1.0 seconds per instance with 100 background samples. Computing 1,827 instances requires approximately 3–30 minutes.

### 10.3 Parallelization

SHAP computation can be parallelized across instances (provinces) and targets. Using joblib multiprocessing with N workers reduces wall-clock time approximately linearly (up to 29-fold with 29 workers, limited by per-process overhead).

## 11. Interpretation Guidelines for Practitioners

### 11.1 Feature Importance Interpretation

High global importance ($I_k^{(j)}$ large) indicates feature $k$ is essential for target $j$ predictions. If SC11 (participation component) has highest importance for participation target, this validates that the model captures expected relationships. Unexpected importance patterns may indicate data quality issues or spurious correlations.

### 11.2 Sign Interpretation

Positive average SHAP across provinces indicates feature tends to increase predictions. For governance forecasts, this should align with domain knowledge: better historical participation should increase predicted future participation.

### 11.3 Heterogeneity and Provincial Differences

Diverse SHAP values across provinces (beeswarm spread) indicates model learned complex province-specific relationships. Homogeneous values indicate simple, consistent relationships. Heterogeneity is not problematic but should be interpreted with domain context.

### 11.4 Outlier Detection

Provinces with unusual SHAP patterns (e.g., strong negative SHAP for a feature typically positive) may warrant inspection: could indicate data entry errors, provincial exceptionalism, or model overfitting.

## 12. Limitations and Caveats

### 12.1 Model Dependence

SHAP values explain model predictions, not ground truth relationships. If AutoGluon model exhibits biases or spurious correlations, SHAP explanations faithfully reflect these biases.

### 12.2 Covariate Collinearity

When covariates are highly correlated, SHAP values may distribute importance unevenly among correlated features depending on tree structure or linear regression coefficients. Individual feature importance may be unstable; global patterns remain stable.

### 12.3 Causality Fallacy

High positive SHAP value for feature $k$ does not imply $k$ causally influences predictions. SHAP explains model decision process (associations), not causal mechanisms.

## 13. References

SHAP methodology builds on Shapley values from cooperative game theory (Shapley, 1953) and their adaptation to machine learning (Lundberg & Lee, 2017). Implementation leverages the SHAP Python library providing efficient approximations and visualization. Interpretation follows best practices in explainable AI for governance and policy applications.
