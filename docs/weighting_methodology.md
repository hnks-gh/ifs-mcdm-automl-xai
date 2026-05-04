# Weighting Methodology: Two-Level Intuitionistic Fuzzy CRITIC Algorithm

## 1. Overview and Theoretical Foundation

The weighting module implements the two-level Intuitionistic Fuzzy CRITIC (Criteria Importance Through Intercriteria Correlation) algorithm, a hierarchical approach that respects the multi-level structure of governance indicator systems. The PAPI dataset comprises 29 sub-criteria organized into 8 parent criteria, establishing a natural two-level hierarchy. The algorithm computes sub-criteria weights within each criterion group (Stage 1), then aggregates these to determine inter-criterion importance weights (Stage 2), finally combining them to produce final normalized weights for each sub-criterion.

The theoretical foundation extends the classical CRITIC method into the intuitionistic fuzzy domain. Classical CRITIC assumes crisp numerical scores; this framework accommodates fuzzy membership and non-membership degrees, capturing imprecision and hesitation inherent in governance assessment. The hierarchical structure reflects domain knowledge that sub-criteria importance varies both within criterion groups and relative to other criteria.

## 2. Intuitionistic Fuzzy Set Representation

Before applying CRITIC analysis, numerical PAPI scores are converted to Intuitionistic Fuzzy Set (IFS) representation. This conversion acknowledges measurement uncertainty while maintaining mathematical structure for fuzzy operations.

### 2.1 IFS Conversion Formula

Each raw score $x \in [0, x_{max}]$ is converted to an IFS number $A = (\mu, \nu, \pi)$ where:

$$\mu = \frac{x}{x_{max}}$$

$$\nu = \alpha \left(1 - \frac{x}{x_{max}}\right)$$

$$\pi = 1 - \mu - \nu$$

Here $x_{max} = 3.33$ represents the theoretical maximum of the PAPI scale. The non-membership coefficient $\nu$ is scaled by parameter $\alpha \in [0, 1]$ to control hesitancy level. The hesitancy degree $\pi$ ensures membership constraint $\mu + \nu + \pi = 1$.

For this implementation, $\alpha$ is typically set to $(1 - \mu)$ following the complement method, yielding:

$$\mu = \frac{x}{x_{max}}, \quad \nu = \left(1 - \frac{x}{x_{max}}\right)^2, \quad \pi = \frac{x}{x_{max}} \left(2 - \frac{x}{x_{max}}\right)$$

This formulation ensures: (1) high scores produce high membership and low non-membership, (2) zero scores produce zero membership and high non-membership, (3) hesitancy is maximized at intermediate scores reflecting genuine uncertainty, and (4) the constraint is always satisfied.

### 2.2 IFS Ordering and Score Function

The IFS score function $S(A) = \mu - \nu$ provides a total ordering on IFS numbers, enabling comparison and ranking. For decision analysis, this function summarizes both membership commitment and disbelief, with higher scores indicating stronger positive assessment.

## 3. Stage 1: Sub-Criteria Weighting Within Criteria

Stage 1 analysis operates within each of the 8 criterion groups independently. For each criterion, the algorithm analyzes the IFS matrix of that criterion's sub-criteria across all 63 provinces.

### 3.1 Data Structure

For criterion $C_k$ with $p_k$ active sub-criteria in a given year/regime, the Stage 1 input is an IFS matrix $X^{(k)}$ of shape $(n \times p_k)$ where $n = 63$ provinces and rows correspond to provinces, columns to sub-criteria.

### 3.2 Standard Deviation of Score Functions

The first CRITIC component captures intrinsic variability of each sub-criterion. For each sub-criterion $j$ within criterion $k$, extract the score function vector:

$$S_j^{(k)} = \{S(x_{1j}^{(k)}), S(x_{2j}^{(k)}), \ldots, S(x_{nj}^{(k)})\}$$

Compute sample standard deviation:

$$\sigma_j^{(k)} = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} \left(S(x_{ij}^{(k)}) - \bar{S}_j^{(k)}\right)^2}$$

where $\bar{S}_j^{(k)} = \frac{1}{n} \sum_{i=1}^{n} S(x_{ij}^{(k)})$ is the mean score.

Higher standard deviation indicates greater discrimination among provinces, suggesting the sub-criterion is more important for differentiating provincial performance.

### 3.3 Correlation Matrix on Score Functions

The second CRITIC component captures inter-criterion relationships. Construct the correlation matrix $R^{(k)} \in \mathbb{R}^{p_k \times p_k}$ where entry $r_{jl}$ is the Pearson correlation coefficient between score function vectors of sub-criteria $j$ and $l$:

$$r_{jl}^{(k)} = \frac{\text{Cov}(S_j^{(k)}, S_l^{(k)})}{\sigma_j^{(k)} \sigma_l^{(k)}}$$

Correlation is computed on score functions rather than raw IFS tuples, providing a scalar-valued comparison that respects the fuzzy structure via the score function definition.

High correlation between sub-criteria indicates redundancy: they carry similar information about provinces. Low correlation indicates complementarity: they provide distinct perspectives. CRITIC methodology penalizes redundancy and rewards complementarity.

### 3.4 CRITIC Information Measure

The CRITIC information measure for sub-criterion $j$ within criterion $k$ is defined as:

$$C_j^{(k)} = \sigma_j^{(k)} \sum_{l=1}^{p_k} (1 - r_{jl}^{(k)})$$

This formula combines: (1) standard deviation $\sigma_j^{(k)}$ as base importance (higher variability = more discriminative), and (2) sum of complement correlation values $\sum_l(1 - r_{jl}^{(k)})$ as independence bonus. Sub-criteria that are less redundant with their peers receive higher $C_j$ values, reflecting the principle that independent information sources deserve higher weight.

The term $(1 - r_{jj}^{(k)}) = 0$ (correlation with self is 1), ensuring only true inter-criterion relationships contribute to independence bonus.

### 3.5 Stage 1 Weight Normalization

Stage 1 weights for criterion $k$ are obtained by normalizing CRITIC measures:

$$w_j^{(1,k)} = \frac{C_j^{(k)}}{\sum_{l=1}^{p_k} C_l^{(k)}}$$

These weights sum to 1.0 and are strictly positive (assuming no degenerate sub-criteria with zero variance). Weights represent the relative importance of each sub-criterion in explaining province performance differences within criterion $k$.

## 4. Stage 2: Criteria Weighting

Stage 2 determines the relative importance of the 8 criteria themselves, using similar CRITIC logic applied at the criterion level.

### 4.1 Criterion-Level IFS Aggregation

Using Stage 1 weights from criterion $k$, compute criterion-level IFS composite for each province $i$:

$$A_{i}^{(k)} = \text{IFS-WAM}\left(\{x_{i1}^{(k)}, x_{i2}^{(k)}, \ldots, x_{ip_k}^{(k)}\}, \{w_1^{(1,k)}, w_2^{(1,k)}, \ldots, w_{p_k}^{(1,k)}\}\right)$$

The Intuitionistic Fuzzy Weighted Arithmetic Mean is computed as:

$$\mu_{agg} = 1 - \prod_{j=1}^{p_k} (1 - \mu_j)^{w_j^{(1,k)}}$$

$$\nu_{agg} = \prod_{j=1}^{p_k} \nu_j^{w_j^{(1,k)}}$$

$$\pi_{agg} = 1 - \mu_{agg} - \nu_{agg}$$

This aggregation weights membership upward and non-membership downward according to Stage 1 weights, producing a single IFS composite per criterion per province.

### 4.2 Stage 2 Score and Correlation

Extract criterion-level score functions:

$$S_k^{(2)} = \{S(A_1^{(k)}), S(A_2^{(k)}), \ldots, S(A_n^{(k)})\}$$

Compute standard deviation and Pearson correlation matrix across the 8 criteria-level scores, obtaining $\sigma_k^{(2)}$ and correlation matrix $R^{(2)} \in \mathbb{R}^{8 \times 8}$.

### 4.3 Stage 2 CRITIC Measure and Weights

Apply identical CRITIC formula at criterion level:

$$C_k^{(2)} = \sigma_k^{(2)} \sum_{m=1}^{8} (1 - r_{km}^{(2)})$$

Normalize to obtain Stage 2 weights:

$$w_k^{(2)} = \frac{C_k^{(2)}}{\sum_{m=1}^{8} C_m^{(2)}}$$

These weights represent the relative importance of the 8 criteria in explaining overall governance performance variation.

## 5. Combined Sub-Criteria Weights

The final weight for sub-criterion $j$ belonging to criterion $k$ is computed as the product of its Stage 1 and Stage 2 weights:

$$w_j^{\text{final}} = w_j^{(1,k)} \times w_k^{(2)}$$

This multiplicative combination reflects hierarchical importance: a sub-criterion's final importance depends both on its relative importance within its criterion (Stage 1) and its criterion's relative importance overall (Stage 2).

The 29-element final weight vector $\mathbf{w} = [w_1^{\text{final}}, \ldots, w_{29}^{\text{final}}]$ sums to 1.0 and is used in all downstream ranking calculations.

Verification: $\sum_{j=1}^{29} w_j^{\text{final}} = \sum_{k=1}^{8} \sum_{j \in C_k} w_j^{(1,k)} w_k^{(2)} = \sum_{k=1}^{8} w_k^{(2)} \sum_{j \in C_k} w_j^{(1,k)} = \sum_{k=1}^{8} w_k^{(2)} = 1.0$.

## 6. Regime-Based Weighting and Aggregation

The PAPI dataset exhibits temporal structural breaks in sub-criteria availability, necessitating regime-specific weighting.

### 6.1 Regime Definition

Four temporal regimes are defined based on active sub-criteria:

**Regime R1 (2011–2017, 7 years)**: 22 active sub-criteria. Sub-criteria SC24 (transparency sub-domain) and SC71–SC83 (environmental governance and e-governance) are structurally absent. Weighting uses only 22 active sub-criteria; absent sub-criteria receive zero weight.

**Regime R2 (2018, 1 year)**: 28 active sub-criteria. Only SC83 (e-governance sub-criterion) is absent; SC24 transitions to active. This single-year regime represents a data collection transition.

**Regime R3 (2019–2020, 2 years)**: 29 active sub-criteria. Complete data with all criteria present. Represents the ideal case with full information.

**Regime R4 (2021–2024, 4 years)**: 28 active sub-criteria. Sub-criterion SC52 (public administration procedures) becomes absent. All other criteria remain active.

### 6.2 Missing Sub-Criteria Handling

For a sub-criterion absent in a given regime, two approaches are employed:

1. **Exclusion from Stage 1**: When computing Stage 1 weights within criterion $k$, only active sub-criteria are included. The normalization sums only over active sub-criteria, producing weights that sum to 1.0 within each criterion. Absent sub-criteria receive weight 0.

2. **Propagation of Zero Weight**: When constructing the full 29-element weight vector, absent sub-criteria are assigned 0.0 weight explicitly, while active sub-criteria maintain their computed values.

This approach ensures: (1) mathematical consistency (weights always sum appropriately), (2) transparent handling of missingness, and (3) interpretability (zero weight clearly indicates unavailable data).

### 6.3 Multi-Regime Aggregation

When computing weights for a full year or across a year group spanning multiple regimes, weights are computed for each applicable regime separately, then blended proportionally:

$$w_j^{\text{blended}} = \frac{\sum_{r} n_r \times w_j^{(r)}}{\sum_{r} n_r}$$

where $n_r$ is the number of years in regime $r$ contributing to the analysis, and $w_j^{(r)}$ is the weight for sub-criterion $j$ computed specifically for regime $r$.

For example, if a 5-year window includes 3 years from R1 and 2 years from R3, the blended weight is $\frac{3 w_j^{(R1)} + 2 w_j^{(R3)}}{5}$. This proportional blending ensures temporal consistency while respecting changing data availability.

ponding to changing conditions.

### 7.4 Coefficient of Variation (CV)

For each sub-criterion, compute variation of weights across all 10 windows:

$$\text{CV}_j = \frac{\sigma(\mathbf{w}_j)}{\bar{\mathbf{w}}_j}$$

where $\sigma(\mathbf{w}_j) = \sqrt{\frac{1}{10} \sum_{t=1}^{10} (w_j^{(t)} - \bar{w}_j)^2}$ and $\bar{\mathbf{w}}_j = \frac{1}{10} \sum_{t=1}^{10} w_j^{(t)}$.

CV near zero indicates stable sub-criterion importance; CV > 1.0 indicates high variability suggesting sensitive dependence on window composition.

## 8. Mathematical Properties and Validation
## 7. Temporal Stability Analysis

Temporal stability assessment evaluates weight robustness across overlapping time windows.

### 7.1 Rolling Window Construction

From 14 years of data, construct 10 overlapping windows of 5 consecutive years each:

- Window 1: 2011–2015
- Window 2: 2012–2016
- ...
- Window 10: 2020–2024

Each window represents a 5-year snapshot with sufficient temporal coverage for CRITIC analysis.

### 7.2 Window-Specific Weight Computation

For each window, apply complete two-level IF-CRITIC algorithm on that window's years, producing window-specific weight vectors $\mathbf{w}^{(t)}$ for $t = 1, \ldots, 10$.

### 7.3 Root Mean Square Deviation (RMSD)

Measure weight stability through consecutive window RMSD:

$$\text{RMSD}_t = \sqrt{\frac{1}{29} \sum_{j=1}^{29} (w_j^{(t)} - w_j^{(t+1)})^2}$$

Compute sequence $\text{RMSD}_1, \ldots, \text{RMSD}_9$ representing discontinuity between consecutive windows. Lower RMSD indicates more stable weights; higher RMSD indicates weights res
### 8.1 Weight Normalization Verification

All weight vectors must satisfy:

$$\sum_{j=1}^{29} w_j^{\text{final}} = 1.0 \quad (\text{within numerical tolerance } 10^{-10})$$

This property is verified post-computation to ensure numerical stability.

### 8.2 Non-Negativity

All weights satisfy $w_j^{\text{final}} \geq 0$ by construction. For sub-criteria with zero variance (degenerate cases), weights may be exactly zero rather than positive, which is acceptable.

### 8.3 Rank Stability

Weights are semantically meaningful: if criterion $A$ has CRITIC information measure double that of criterion $B$, then $A$ should receive approximately double the weight, reflecting genuine importance differences.

## 9. Numerical Implementation Notes

The implementation in `src/mcdm/weighting/if_critic.py` employs:

1. **IFS Matrix Representation**: Stored as structured arrays of shape $(n, m, 3)$ where the third dimension contains $(\mu, \nu, \pi)$ tuples, or as numpy arrays of IFSNumber objects for readability.

2. **Score Function Extraction**: Applied to matrices in vectorized form to produce $(n, m)$ matrices of score values for correlation and std dev computation.

3. **Correlation via NumPy**: Pearson correlation matrix computed via `numpy.corrcoef()` on score function arrays, handling degenerate cases where variance is zero.

4. **Numerical Stability**: When correlation is undefined (zero variance in one or both variables), correlation is set to 0 rather than NaN, providing conservative independence estimate.

5. **Regime Indexing**: Active sub-criteria tracked via boolean mask or index list, applied consistently in normalization to avoid division-by-zero errors.

6. **Output Format**: Final weights exported as pandas Series with sub-criteria names as index, enabling downstream ranking methods to access weights by name.

## 10. References and Theoretical Grounding

The two-level CRITIC methodology extends classical CRITIC (Diakoulaki et al., 1995) into intuitionistic fuzzy domain following frameworks in Atanassov's IFS theory. The hierarchical approach respects multi-level indicator system structure common in governance and sustainability assessment. Mathematical formulations maintain consistency with established practices in fuzzy MCDM literature while ensuring practical applicability to real governance datasets with structural missingness.
