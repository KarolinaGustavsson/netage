# Biological Age Modelling in the AMORIS Cohort

## Background and rationale

Biological age aims to capture the physiological state of an individual in a way that is not fully determined by chronological time. A useful biological age estimator assigns two individuals of the same chronological age different values when their underlying health trajectories differ, and the resulting *age gap* — the difference between biological and chronological age — carries prognostic information beyond what chronological age alone provides.

The AMORIS cohort offers an unusually strong setting for this question. Baseline clinical and biochemical measurements on approximately 260,000 individuals, combined with long-term survival follow-up on a large fraction of the cohort, permit both the estimation of a biological age model and its validation against hard mortality outcomes.

A naive approach — training a regression model to predict chronological age from biomarkers and interpreting the residual as biological age — is known to be problematic. If the model predicts chronological age perfectly, the residual carries no information about aging. If it predicts poorly, the residual is dominated by noise. Moreover, ordinary residuals of regression on age exhibit systematic regression-to-the-mean bias, assigning positive age gaps to young individuals and negative gaps to old individuals as a geometric artifact rather than a biological signal. Any downstream analysis of such residuals partly explains this artifact rather than biology.

This project adopts a formulation that sidesteps these problems by construction: biological age is defined through a mortality-calibrated mapping, and the neural network is trained on a survival objective rather than on chronological age prediction.

## Aims

1. Construct a biological age estimator from 15 baseline variables in the AMORIS cohort, calibrated against observed mortality.
2. Quantify the age gap between biological and chronological age and characterize its relationship to downstream health outcomes.
3. Identify the baseline variables driving biological age and the age gap, and contrast the two to distinguish markers of aging from drivers of accelerated aging.

## Methods

### Biological age definition

We replace the linear predictor in the Levine PhenoAge framework with a neural network. A feedforward network η_θ(x, t) maps the 15 baseline variables x and chronological age at baseline t to an individual log-hazard. The network is trained under the Cox partial likelihood, using attained age as the time scale with left-truncation at age of baseline measurement. The baseline hazard is estimated nonparametrically via Breslow's estimator on a reference subset of the cohort, stratified by sex.

Biological age g(x, t) is defined as the chronological age at which the reference-population 10-year mortality equals the individual's predicted 10-year mortality derived from η_θ(x, t). The age gap is Δ(x, t) = g(x, t) − t.

This construction has three properties relevant to the scientific claims:

- The training objective is mortality prediction, not chronological age prediction. The degenerate solution in which the network reproduces t without biological content does not exist in this formulation.
- Nonlinear interactions among biomarkers are captured directly. This is the principal reason for adopting a neural formulation over linear Cox.
- Biological age is expressed in units of years, calibrated such that an individual's biological age corresponds to the chronological age at which a reference individual carries equivalent mortality risk.

### Model capacity and regularization

Given the low input dimensionality, a modest-capacity network is appropriate: two to three hidden layers with 64–128 units, ReLU activations, dropout, and weight decay. Hyperparameters will be selected on cross-validated Cox concordance. A held-out Cox C-index improvement over linear Cox is required to justify the neural formulation; marginal improvements will prompt a switch to the linear baseline for the sake of interpretability.

### Comparators

Two classical biological age estimators will serve as reference points:

- **Linear Cox PhenoAge**, following Levine et al., using the same 15 variables with linear effects.
- **Klemera–Doubal biological age**, using precision-weighted linear regressions of each biomarker on age.

All three estimators will be compared on downstream mortality and incident disease outcomes.

### Feature importance

Feature importance is a primary scientific objective of this project. Attributions will be computed on two distinct targets:

- **Biological age g(x, t)**: decomposes the absolute level of biological age. Answers the prognostic question of which variables drive mortality risk overall.
- **Age gap Δ(x, t)**: decomposes the deviation of biological age from chronological age. Answers the age-acceleration question of which variables flag individuals aging faster or slower than their cohort.

These are different quantities. A variable can contribute substantially to biological age while contributing little to the age gap if its effect is largely explained by age covariation — a *marker of aging* rather than a *driver of accelerated aging*. The explicit contrast between the two attributions is the central analytic contribution of the project.

SHAP will be the primary attribution method, computed with a shared age-stratified cohort-mean baseline so that the two decompositions are placed on comparable scales. Integrated gradients will serve as a consistency check. Given the 15-dimensional input, exact or near-exact SHAP is computationally feasible. SHAP interaction values will be computed to quantify non-additive contributions.

The analysis will characterize features by their position in the (|φ^{BA}|, |φ^{Δ}|) plane:

- High BA importance, high Δ importance: drivers of accelerated aging.
- High BA importance, low Δ importance: markers of aging (large age-expected effect, small deviation effect).
- Low BA importance, high Δ importance: anomaly-driven features whose level matters less than individual deviation.
- Low both: candidates for exclusion from reduced models.

Attributions will be reported both at the individual-feature level and aggregated over pre-specified biomarker groups (lipid, inflammation, renal, hepatic, glycemic, hematologic). Group-level summaries are more stable to within-group correlations and typically carry clearer biological interpretation.

### Robustness and sanity checks

- **Age-stratified attribution**, computed within age deciles, to characterize how importance varies across the lifespan.
- **Chronological age as negative control**: attribution of t to Δ should be near zero on average; systematic deviations diagnose residual miscalibration in the biological age construction.
- **Bootstrap over individuals** to produce uncertainty intervals on importance rankings.
- **Linear Cox SHAP comparison** as a cross-check on top-ranked features.
- **Permutation importance** against held-out Cox concordance as a predictive-utility cross-check.
- **Seed stability**: retraining under multiple random seeds to confirm top features are robust to initialization.
- **Pre-specified effect-size threshold** for reporting meaningful drivers of Δ, to avoid over-interpreting statistically significant but biologically negligible contributions in a cohort of this size.

### Validation

The primary figure of merit is not age-prediction accuracy but downstream utility of Δ:

- Predictive value of Δ for all-cause mortality beyond chronological age alone (incremental C-index, likelihood ratio tests).
- Predictive value for cause-specific incidence (cardiovascular, cancer, dementia) in directions consistent with biological expectation.
- Calibration of 10-year mortality risk predictions on held-out data.

## Ethical and data considerations

Work is conducted on de-identified AMORIS cohort data under the existing ethical approvals governing AMORIS research. All modelling and analysis is performed in compliance with local data-access protocols; no individual-level data leaves the permitted computational environment.

## Expected outcomes

A neural-network-based biological age estimator for the AMORIS cohort, calibrated to observed mortality, together with a characterization of the baseline variables driving biological age and the age gap. The principal methodological contribution is the explicit contrast between importance on g and on Δ, distinguishing markers of aging from drivers of accelerated aging. The principal epidemiological contribution is the validation of Δ as a prognostic quantity beyond chronological age in a large Swedish cohort.
