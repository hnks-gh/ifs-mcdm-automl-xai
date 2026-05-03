# Analysis, Validation, and Testing Methodology

## 1. Temporal Stability Analysis of Weights

### 1.1 Motivation and Objective

Temporal stability analysis evaluates whether the two-level IF-CRITIC weighting algorithm produces consistent importance rankings across different time periods. Governance indicators typically evolve gradually; weight structures should also exhibit temporal persistence rather than erratic fluctuation. Instability could indicate: (1) data quality issues, (2) genuine structural changes in indicator relationships, or (3) algorithmic sensitivity to year-to-year variations.

This analysis provides empirical assessment of weight robustness, enabling confidence in long-term MCDM analyses and forecasting applications.

### 1.2 Rolling Window Construction

From 14 years of data (2011–2024), construct 10 overlapping windows of 5 consecutive years:

Window $t$ spans years $2010 + t$ to $2014 + t$ for $t \in \{1, 2, \ldots, 10\}$:

- Window 1: 2011–2015
- Window 2: 2012–2016
- ...
- Window 10: 2020–2024

Each window contains sufficient data (5 years × 63 provinces = 315 observations) for stable CRITIC analysis while enabling detection of temporal variation. Window overlap enables continuous assessment rather than gapped segments.

### 1.3 Window-Specific Weight Computation

For each window $t$, execute the complete two-level IF-CRITIC algorithm independently:

1. Load raw PAPI data for years in window $t$
2. Detect applicable regimes (R1, R2, R3, R4) within window
3. Apply regime-specific IF-CRITIC weighting
4. Blend regime weights proportionally by representation in window
5. Produce 29-element final weight vector $\mathbf{w}^{(t)}$ for window $t$

Result: 10 weight vectors $\mathbf{w}^{(1)}, \ldots, \mathbf{w}^{(10)}$ representing governance structure across different temporal windows.

### 1.4 Root Mean Square Deviation (RMSD) Between Consecutive Windows

Quantify temporal continuity via RMSD between consecutive window weight vectors:

$$\text{RMSD}_t = \sqrt{\frac{1}{29} \sum_{j=1}^{29} (w_j^{(t)} - w_j^{(t+1)})^2}$$

This metric measures average squared weight change per sub-criterion between windows $t$ and $t+1$, normalized by feature count.

**Interpretation**:
- RMSD near 0.01–0.02: Excellent stability; weights change minimally from year to year
- RMSD near 0.05–0.10: Moderate stability; meaningful but not drastic changes
- RMSD > 0.15: Concerning instability; suggests either data issues or genuine structural breaks

Compute sequence $\text{RMSD}_1, \text{RMSD}_2, \ldots, \text{RMSD}_9$ (9 values for 10 windows).

**Aggregated stability metric**: $\overline{\text{RMSD}} = \frac{1}{9} \sum_{t=1}^{9} \text{RMSD}_t$ provides single scalar characterizing overall weight stability.

### 1.5 Coefficient of Variation (CV) of Sub-Criterion Weights

Assess relative variability of each sub-criterion's importance across the 10 windows:

For sub-criterion $j$, compute coefficient of variation:

$$\text{CV}_j = \frac{\sigma(w_j^{(1)}, \ldots, w_j^{(10)})}{\bar{w}_j}$$

where $\sigma$ is sample standard deviation and $\bar{w}_j$ is mean weight across windows.

**Interpretation**:
- CV < 0.1: Very stable sub-criterion importance; ranking robust to window choice
- CV ∈ [0.1, 0.3]: Moderately stable; meaningful variation but consistent relative importance
- CV > 0.5: Highly variable; sub-criterion importance fluctuates substantially

Sub-criteria with high CV may indicate: (1) sensitive metrics responding to short-term governance changes, (2) measurement unreliability, or (3) lack of persistent underlying structure.

**Distribution analysis**: Report CV for all 29 sub-criteria, including min, max, mean, and percentiles. Identify which sub-criteria are most/least stable.

### 1.6 Output Format

Generate `output/mcdm/analysis/temporal_stability.csv` containing:

- Columns: Window 1 through Window 10 (10 columns)
- Rows: Each of 29 sub-criteria (29 rows)
- Cells: Weight values for that sub-criterion in that window

Generate `output/mcdm/analysis/stability_metrics.json` containing:

```json
{
  "rmsd_consecutive": [0.023, 0.019, ..., 0.031],
  "mean_rmsd": 0.025,
  "cv_per_subcriteria": [0.052, 0.078, ..., 0.141],
  "mean_cv": 0.087,
  "assessment": "Excellent stability"
}
```

## 2. Monte Carlo Sensitivity Analysis

### 2.1 Objective and Rationale

Sensitivity analysis assesses ranking robustness under perturbation of weights. In real governance assessment, weights are estimated from data; estimation uncertainty or parameter ambiguity could affect rankings. Sensitivity analysis quantifies this impact: if small weight changes produce drastically different rankings, decision-makers should treat rankings with caution. If rankings prove robust, confidence in rankings is warranted.

This analysis employs Monte Carlo simulation: repeatedly perturb weight vector via random sampling, recompute rankings under perturbed weights, and measure ranking correlation with baseline.

### 2.2 Dirichlet Perturbation Strategy

The weight vector $\mathbf{w} \in \mathbb{R}^{29}$ lies in the 28-dimensional simplex (sum-to-1, all non-negative). Natural distribution over this space is the Dirichlet distribution.

Generate perturbed weights via Dirichlet sampling with concentration parameters proportional to baseline weights:

$$\tilde{\mathbf{w}}^{(s)} \sim \text{Dirichlet}(\alpha \mathbf{w})$$

where $\alpha$ is concentration parameter controlling perturbation magnitude and $s \in \{1, \ldots, n_{sims}\}$ indexes simulation iterations.

**Concentration parameter selection**: $\alpha \in [10, 100]$ balances exploration (small $\alpha$ produces wide variation) and consistency (large $\alpha$ concentrates near baseline). Configuration specifies $\alpha = 50$ by default, producing moderate perturbations.

**Interpretation**: Concentration $\alpha \mathbf{w}$ ensures perturbed weights remain near baseline; if baseline weight is large, perturbed weight distribution concentrates nearby. If baseline weight is small, perturbation allows larger relative variation.

### 2.3 Ranking Recomputation Under Perturbation

For each simulation iteration $s$:

1. Sample perturbed weight vector $\tilde{\mathbf{w}}^{(s)}$ from Dirichlet
2. Normalize to ensure $\sum w_j = 1.0$ (automatically satisfied by Dirichlet)
3. For each ranking method (IF-WASPAS, IF-TOPSIS, IF-PROMETHEE II):
   a. Apply ranking algorithm with perturbed weights
   b. Obtain province ranking $\tilde{\mathbf{r}}_m^{(s)}$ for method $m$
4. Compare perturbed ranking $\tilde{\mathbf{r}}_m^{(s)}$ with baseline ranking $\mathbf{r}_m$ via Kendall tau-b correlation

Total: $n_{sims} \times 3$ ranking comparisons (typically $n_{sims} = 10,000$).

### 2.4 Weighted Kendall Tau-B Correlation

Kendall's tau-b rank correlation measures agreement between two orderings:

$$\tau_b = \frac{\text{# concordant pairs} - \text{# discordant pairs}}{n(n-1)/2}$$

where $n = 63$ provinces. Tau-b ranges from -1 (perfect reverse) through 0 (independence) to +1 (perfect agreement).

For sensitivity analysis, weight correlation metric by sub-criterion importance to emphasize changes in highly weighted criteria:

$$\tau_{b,w} = \frac{\sum_{i < j} w_{ij} \cdot \text{sgn}(\mathbf{r}_i - \mathbf{r}_j) \cdot \text{sgn}(\tilde{\mathbf{r}}_i - \tilde{\mathbf{r}}_j)}{\sum_{i < j} w_{ij}}$$

where $w_{ij} = (w_{score_i} + w_{score_j})/2$ weights province pairs by their average sub-criterion score importance, and $\text{sgn}(\mathbf{r}_i - \mathbf{r}_j)$ indicates relative ordering direction.

This weighting emphasizes preservation of relative positions among high-importance provinces (high average scores), which are typically focus of governance analysis.

### 2.5 Sensitivity Result Aggregation

For each ranking method $m$, collect correlation sequence $\tau_{b,w}^{(1)}, \ldots, \tau_{b,w}^{(n_{sims})}$ and report:

**Mean correlation**: $\bar{\tau}_m = \frac{1}{n_{sims}} \sum_s \tau_{b,w}^{(s)}$

**Standard deviation**: $\sigma_m = \sqrt{\frac{1}{n_{sims}} \sum_s (\tau_{b,w}^{(s)} - \bar{\tau}_m)^2}$

**95th percentile interval**: $[\tau_m^{0.025}, \tau_m^{0.975}]$ from sorted correlation values

**Assessment**: $\bar{\tau}_m > 0.95$ indicates excellent robustness (very similar rankings under perturbation); $\bar{\tau}_m > 0.85$ indicates good robustness; $\bar{\tau}_m < 0.75$ indicates concerning sensitivity (small weight changes substantially alter rankings).

### 2.6 Output Format

Generate `output/mcdm/analysis/sensitivity_analysis.json` containing:

```json
{
  "perturbation_alpha": 50,
  "n_simulations": 10000,
  "methods": {
    "if_waspas": {
      "mean_tau_b": 0.943,
      "std_tau_b": 0.018,
      "percentile_2_5": 0.902,
      "percentile_97_5": 0.968,
      "assessment": "Excellent robustness"
    },
    "if_topsis": {...},
    "if_promethee2": {...}
  }
}
```

## 3. Ranking Validation and Inter-Method Agreement

### 3.1 Inter-Method Spearman Correlation

Measure agreement between three ranking methods via Spearman's rank correlation coefficient:

$$\rho_{m_1,m_2} = 1 - \frac{6 \sum_{i=1}^{63} (r_{m_1,i} - r_{m_2,i})^2}{63 \times 62 \times 64}$$

where $r_{m,i}$ is province $i$'s rank under method $m$.

Compute correlations for all $\binom{3}{2} = 3$ method pairs:

- $\rho_{WASPAS,TOPSIS}$
- $\rho_{WASPAS,PROMETHEE2}$
- $\rho_{TOPSIS,PROMETHEE2}$

For each year $t \in \{2011, \ldots, 2024\}$, compute method correlations independently, then average across years.

**Interpretation**:
- $\rho > 0.9$: Strong agreement; methods produce nearly identical rankings
- $\rho \in [0.7, 0.9]$: Moderate agreement; methods rank provinces similarly with some differences
- $\rho < 0.7$: Weak agreement; methods produce substantially different rankings

High agreement indicates robustness of conclusions; all methods converge on similar provincial importance rankings. Disagreement suggests different methods emphasize different aspects of governance.

### 3.2 Discriminatory Power Analysis

Assess each ranking method's ability to differentiate provinces via inter-quartile range (IQR) of score distributions.

For each method $m$ and year $t$, compute IFS score values $S(\mathbf{a}_{i,t,m})$ for all 63 provinces, where $\mathbf{a}_{i,t,m}$ is province $i$'s aggregated IFS score for year $t$ under method $m$.

Compute quartiles:

$$Q_1^{(m,t)} = \text{25th percentile of scores}$$
$$Q_3^{(m,t)} = \text{75th percentile of scores}$$

$$\text{IQR}_{m,t} = Q_3^{(m,t)} - Q_1^{(m,t)}$$

Larger IQR indicates greater spread of scores, enabling finer differentiation among provinces. Smaller IQR indicates scores cluster, making ranking less discriminative.

Average IQR across years: $\overline{\text{IQR}}_m = \frac{1}{14} \sum_{t=2011}^{2024} \text{IQR}_{m,t}$

**Interpretation**: Higher discriminatory power (larger IQR) is desirable for governance assessment, as it enables clearer identification of high-performing and low-performing provinces.

### 3.3 Temporal Persistence: Year-to-Year Rank Correlation

Measure whether province rankings remain stable across consecutive years, indicating persistent performance patterns:

For each method $m$ and consecutive years $t, t+1$:

$$\rho_m^{(t)} = \text{Spearman}(\mathbf{r}_{m,t}, \mathbf{r}_{m,t+1})$$

Compute sequence $\rho_m^{(2011)}, \ldots, \rho_m^{(2023)}$ (13 correlations for 14 years).

Average year-to-year correlation: $\bar{\rho}_m = \frac{1}{13} \sum_{t=2011}^{2023} \rho_m^{(t)}$

**Interpretation**: $\bar{\rho}_m > 0.85$ indicates excellent persistence; provinces maintain similar relative positions year-to-year, reflecting stable governance structures. $\bar{\rho}_m < 0.70$ indicates ranks fluctuate substantially, suggesting either rapid governance changes or ranking instability.

### 3.4 Output Format

Generate `output/mcdm/analysis/ranking_validation.json` containing:

```json
{
  "inter_method_spearman": {
    "waspas_topsis": 0.874,
    "waspas_promethee2": 0.821,
    "topsis_promethee2": 0.803,
    "mean": 0.833,
    "assessment": "Moderate agreement"
  },
  "discriminatory_power_iqr": {
    "if_waspas": 0.487,
    "if_topsis": 0.521,
    "if_promethee2": 0.463,
    "best_method": "if_topsis"
  },
  "temporal_persistence": {
    "if_waspas": 0.887,
    "if_topsis": 0.901,
    "if_promethee2": 0.843,
    "mean": 0.877,
    "assessment": "Excellent persistence"
  }
}
```

## 4. Comprehensive Testing Strategy

### 4.1 Unit Testing: IFS Arithmetic

Unit tests verify correctness of all IFS mathematical operations against analytical solutions.

**Test cases for IFS addition**:

Given $A = (\mu_1, \nu_1, \pi_1)$ and $B = (\mu_2, \nu_2, \pi_2)$ with known values, verify IFS addition produces correct result:

$$A \oplus B = (1-(1-\mu_1)(1-\mu_2), \nu_1 \nu_2, \pi)$$

Tests cover: (1) general case with all components non-zero, (2) boundary cases (one operand zero, one certainty), (3) commutativity $A \oplus B = B \oplus A$, (4) constraint verification $\mu + \nu + \pi = 1$.

**Test cases for score and accuracy functions**: Verify $S(A) = \mu - \nu$ and $H(A) = \mu + \nu$ with boundary inputs.

**Test cases for distances**: Verify Hamming and Euclidean distance calculations against manual computation on small examples.

Target coverage: 100% of ifs_arithmetic.py functions with pass/fail results.

### 4.2 Unit Testing: IF-CRITIC Weighting

Test IF-CRITIC correctness on small synthetic datasets with known analytical solutions.

**Synthetic test case**: 3 criteria, 2 sub-criteria per criterion, 5 provinces with manually constructed IFS values where importance is analytically determinable.

Verify: (1) Stage 1 weights sum to 1 per criterion, (2) Stage 2 weights sum to 1, (3) combined weights sum to 1, (4) weight orderings match analytical expectations, (5) regime handling produces correct regime-specific weights.

Tests include: (1) missing sub-criteria handling (regime-specific masking), (2) correlation matrix computation against numpy corrcoef, (3) CRITIC information measure formula verification.

Target coverage: All code paths in if_critic.py and two_level_aggregator.py.

### 4.3 Unit Testing: Ranking Methods

Test each ranking method (IF-WASPAS, IF-TOPSIS, IF-PROMETHEE II) on analytical examples.

**Test case for IF-WASPAS**: 3 provinces, 3 criteria with IFS values and weights. Compute WSM and WPM components separately, verify correct IFS arithmetic in combination.

**Test case for IF-TOPSIS**: Verify PIS construction (max membership, min non-membership per criterion), NIS construction (min membership, max non-membership), distance computations, closeness coefficient formula.

**Test case for IF-PROMETHEE II**: Verify preference matrix construction using Gaussian preference function, flow calculations (phi+, phi-, phi_net), and ranking correctness.

Verify all rankings are valid permutations (no gaps, no duplicates, all values in 1..n).

### 4.4 Unit Testing: ML Components

**MICE Imputation tests**: Verify zero missing values post-imputation, value range [0, 3.33] respected, original observed values unchanged.

**AutoGluon wrapper tests**: Mock TimeSeriesPredictor to test data construction (TimeSeriesDataFrame format, covariate specification), prediction shape verification, bounds checking.

**SHAP computation tests**: Mock SHAP explainer to test batch processing, output dimensionality, baseline value consistency.

### 4.5 Integration Testing: MCDM Pipeline

End-to-end test of complete MCDM pipeline on mini-dataset (7 provinces, 3 years, 8 sub-criteria).

Execute: raw data loading → regime detection → IFS conversion → two-level weighting → three ranking methods → temporal stability → sensitivity analysis → ranking validation.

Verify: (1) each pipeline stage produces expected output shape, (2) no numerical errors or NaN propagation, (3) weight and ranking properties satisfied, (4) output files created in correct directory structure.

### 4.6 Integration Testing: ML Pipeline

End-to-end test of complete ML pipeline on representative subset (10 provinces, all years, all sub-criteria).

Execute: raw data loading → MICE imputation → AutoGluon training (limited to 1–2 sub-criteria for speed) → SHAP computation.

Verify: (1) imputed panel has expected shape and zero NaNs, (2) forecasts produced with correct dimensionality, (3) SHAP values computed and stored correctly.

### 4.7 Mathematical Property Validation

Systematic verification of algorithmic properties:

1. **Weight normalization**: All weight vectors $\sum w_j = 1.0$ (tolerance $10^{-10}$)
2. **IFS constraints**: All IFS values $\mu + \nu + \pi = 1.0$ with $\mu, \nu, \pi \in [0,1]$
3. **Ranking validity**: Rankings are permutations (no gaps, no duplicates)
4. **Covariate completeness**: Imputed panel has zero missing values
5. **Forecast dimensionality**: (63, 29) for all sub-criteria
6. **SHAP efficiency**: $\sum_{k=1}^{28} \phi_k^{(i,j)} \approx f(i,j) - E[f]$

### 4.8 Test Coverage Metrics

Employ coverage tools (pytest-cov) to measure test coverage:

Target: ≥ 85% line coverage for all src/ modules; ≥ 95% for core mathematical functions.

Identify untested branches via coverage reports; add targeted tests for edge cases and error conditions.

### 4.9 Continuous Integration Strategy

Tests executed automatically on code commits:

1. Unit tests first (fast feedback, ~30 seconds)
2. Integration tests on successful unit tests (~5 minutes)
3. Coverage reports generated and tracked

Fail build on test failures or coverage regressions.

## 5. Quality Assurance Checklist

Pre-release quality assurance confirms:

- [ ] All unit tests pass with 100% success rate
- [ ] All integration tests pass on representative data
- [ ] Code coverage ≥ 85% (≥ 95% for mathematical core)
- [ ] No linting violations (PEP8 compliance via flake8)
- [ ] Type hints complete and verified via mypy
- [ ] Documentation strings present and accurate
- [ ] Mathematical formulas verified against literature
- [ ] Output files verified for correctness and completeness
- [ ] Performance profiling shows acceptable runtime
- [ ] Numerical stability confirmed (no inf/nan propagation)
- [ ] Data integrity preserved throughout pipeline
- [ ] Reproducibility verified (identical results with fixed seeds)

## 6. Error Handling and Diagnostics

Framework implements comprehensive error handling:

**DataLoadError**: Raised when raw CSV files missing or malformed.

**RegimeDetectionError**: Raised when regime boundaries cannot be determined from column presence.

**IFSValueError**: Raised when IFS conversion produces invalid values violating constraints.

**MCDMError**: Raised when weighting or ranking computation fails (e.g., singular correlation matrix).

**ImputationError**: Raised when MICE imputation fails (non-convergence, memory issues).

**ForecastingError**: Raised when AutoGluon training fails or produces invalid forecasts.

All errors logged with full context (input shapes, parameter values, partial results) to enable debugging. Exception tracebacks preserved for diagnosis.

## 7. Mathematical References

Temporal stability metrics follow time series analysis conventions. Sensitivity analysis employs Dirichlet perturbation following Bayesian uncertainty quantification practices. Rank correlation metrics follow standard statistical definitions. Testing strategy aligns with software engineering best practices for mathematical libraries.
