# CLAUDE.md

Guidance for Claude Code working on the AMORIS biological age project.

## Project overview

Train a neural Cox model on 15 baseline variables from the AMORIS cohort (~260k individuals, with survival follow-up on a large fraction), define biological age via Levine-style inversion of the population mortality curve, and analyze feature importance with explicit contrast between attributions on biological age g(x, t) and the age gap Δ(x, t) = g(x, t) − t.

Read `PROJECT_DESCRIPTION.md` for scientific framing before making non-trivial methodological decisions.

## Methodological commitments

These are settled decisions. Do not silently deviate. If you believe a deviation is warranted, raise it explicitly rather than substituting your own judgment.

1. **Training target is mortality, not chronological age.** The network η_θ(x, t) is trained under the Cox partial likelihood. Biological age is obtained by inverting the population mortality curve, not by regressing on t.
2. **Time scale is attained age with left-truncation at age of baseline measurement.** Do not use follow-up time as the time scale.
3. **Baseline hazard is nonparametric (Breslow), stratified by sex.** Do not fit a parametric Gompertz unless specifically asked for a comparator.
4. **Biological age is defined via 10-year mortality equivalence**, not instantaneous hazard. Compute the individual's predicted 10-year mortality from the Cox model and find the chronological age in the reference population at which this mortality level is reached.
5. **Attributions are computed on two targets, g and Δ, with a common age-stratified cohort-mean baseline.** Never compare attributions computed against different baselines.
6. **Model capacity is modest by design.** Two to three hidden layers, 64–128 units, dropout, weight decay. A network that substantially outperforms linear Cox should be scrutinized for leakage before being accepted.
7. **The age-prediction paradox is not a concern in this formulation** because the network does not predict age. Do not add regularization targeting age-MAE.

## Repository structure

```
amoris-bioage/
├── PROJECT_DESCRIPTION.md       # Scientific framing — read first
├── CLAUDE.md                    # This file
├── README.md
├── pyproject.toml               # Use uv or poetry; pin dependencies
├── src/amoris_bioage/
│   ├── data/                    # Loading, QC, splits
│   ├── models/                  # Network definitions, Cox loss
│   ├── training/                # Training loops, hyperparameter search
│   ├── bioage/                  # Breslow estimator, inversion, Δ computation
│   ├── attribution/             # SHAP and IG wrappers for g and Δ
│   ├── validation/              # C-index, calibration, downstream outcomes
│   └── comparators/             # Linear Cox PhenoAge, Klemera–Doubal
├── scripts/                     # Runnable entry points
├── tests/                       # Unit tests; see testing policy below
├── configs/                     # YAML hyperparameter configs
└── notebooks/                   # Exploratory analysis only; not the primary artifact
```

## Dependencies

- **PyTorch** for the neural network.
- **lifelines** for Cox baseline hazard, C-index, and classical survival analysis.
- **scikit-survival** as an alternative Cox implementation; useful for comparators.
- **shap** for Shapley attributions. Use the `Explainer` interface with an explicit masker; do not rely on library defaults for the background distribution.
- **captum** for integrated gradients, used as a consistency check against SHAP.
- **pandas, numpy, scipy** for data handling.
- **pytest** for tests.

Avoid: autograd-incompatible Cox loss implementations, hand-rolled SHAP, and any library that does not handle left-truncated survival data correctly.

## Data handling

- Input data lives outside the repo. Paths are supplied via config, never hardcoded.
- Raw data is read-only. All preprocessing writes to a separate derived location.
- Individual-level data never appears in logs, error messages, commit diffs, or notebook outputs checked into git. Summary statistics over groups of sufficient size are acceptable.
- Splits: train / validation / test at the individual level, stratified by sex and age decile. Fix the random seed for splits and document it. Do not tune anything on the test set.
- Missing data: the AMORIS variables have known patterns of missingness; handle them explicitly (imputation strategy documented in the data module) rather than dropping silently.

## Training

- Cox partial likelihood with Efron tie handling (not Breslow ties) in the training loss.
- Left-truncation is implemented in the risk set construction. Individuals enter the risk set at their age of baseline measurement and exit at event or censoring. Verify this explicitly in unit tests.
- Sex stratification in the baseline hazard is applied at the inversion step, not in the training loss. The network sees sex as a covariate.
- Optimizer: AdamW, learning rate in the 1e-4 to 1e-3 range, cosine schedule. Early stopping on validation C-index.
- Hyperparameter selection is on validation C-index, not training loss.

## Biological age computation

Implemented in `src/amoris_bioage/bioage/`:

1. Fit Breslow cumulative baseline hazard on the training set, stratified by sex.
2. For each individual, compute predicted 10-year mortality probability from η_θ and the baseline hazard.
3. Construct the reference mapping from chronological age to expected 10-year mortality in the reference population (sex-stratified), via a monotone smoother.
4. Invert the mapping for each individual to obtain g(x, t) in years. Δ(x, t) = g(x, t) − t.

Sanity checks that must pass before any downstream analysis:

- Mean Δ across the cohort is close to zero (by construction of the reference).
- Mean attribution of t to Δ is close to zero. A systematic nonzero value indicates miscalibration of the inversion.
- Δ distribution is approximately symmetric around zero within each age decile.
- Δ correlates positively with 10-year mortality risk in held-out data.

If any of these fails, stop and diagnose before proceeding. Do not paper over calibration issues with post-hoc corrections unless the correction is principled and documented.

## Attribution

Implemented in `src/amoris_bioage/attribution/`:

- Primary method: SHAP with a common age-stratified cohort-mean background. Compute attributions on both g and Δ as scalar-valued functions of (x, t).
- Secondary method: integrated gradients with the same baseline, for consistency.
- SHAP interaction values for pairwise non-additive contributions.
- Attributions computed on the held-out test set, not on training data.
- Report both individual-feature and pre-specified group-level aggregations. Groups: lipid, inflammation, renal, hepatic, glycemic, hematologic. Exact group membership lives in `configs/variable_groups.yaml` and is defined before attribution runs.

Do not compute SHAP by monkey-patching library internals. If the library interface does not support what is needed, raise it rather than working around it.

## Validation

- Primary: incremental C-index and likelihood-ratio test of Δ against a Cox model containing only t, on held-out data.
- Secondary: cause-specific incidence (CVD, cancer, dementia) against Δ, stratified by sex.
- Calibration: plot predicted vs observed 10-year mortality by decile of predicted risk.

## Comparators

- **Linear Cox PhenoAge**: same variables, same Cox formulation, linear predictor instead of neural network. Same inversion procedure.
- **Klemera–Doubal biological age**: per-variable linear regressions on age, precision-weighted aggregation.

Both comparators must share the data split, preprocessing, and inversion machinery with the neural model so that differences in downstream metrics reflect modelling choices, not data handling.

## Testing

Tests live in `tests/` and are run with `pytest`.

Required coverage:

- Cox partial likelihood with and without left-truncation, tested against lifelines on synthetic data with known ground truth.
- Risk set construction under left-truncation: at least one test with a constructed cohort where the correct risk set membership is computed by hand.
- Breslow baseline hazard: reproduce lifelines output on a shared dataset.
- Biological age inversion: on synthetic data where the ground-truth mapping is known, recover it within tolerance.
- Attribution consistency: SHAP attributions on a linear Cox model reduce to β_j × (x_j − mean) within numerical tolerance.
- Δ attribution identity: for the cohort-mean reference individual, per-feature SHAP on Δ sums (with the bias term) to Δ at the individual, to within numerical tolerance.

Do not skip tests to make a commit pass. If a test fails, either the code or the test is wrong; resolve which before proceeding.

## Reporting

All figures and tables produced for the paper are generated by scripts in `scripts/reporting/`, reading from saved model checkpoints and attribution outputs. No manual edits to figure files. A single `make report` (or equivalent) regenerates every figure from scratch.

Key figures:

- Distribution of Δ by age decile and sex.
- Kaplan–Meier survival stratified by Δ decile.
- Importance contrast plot: mean |φ^{BA}| vs mean |φ^{Δ}| per feature.
- Group-level importance contrast.
- Age-stratified importance for top features.
- Top pairwise SHAP interactions.
- Comparator benchmark: incremental C-index of Δ over t, for the neural model, linear Cox PhenoAge, and Klemera–Doubal.

## Style

- Python: Black formatting, Ruff linting, full type hints, Google-style docstrings.
- Configs in YAML, loaded into typed dataclasses (pydantic or attrs). No magic strings scattered across the codebase.
- Logging via `logging`, not print. INFO level for scripts, DEBUG for internal diagnostics.
- Randomness: every random draw goes through a seeded generator. No bare `np.random` or `torch.randn` without an explicit seed or generator.
- Prefer explicit over clever. The readership is methodological; the code is part of the argument.

## Working style

- When uncertain about a methodological choice, ask rather than guess. The scientific commitments in this file exist because the alternatives were considered and rejected for specific reasons.
- When tests fail or sanity checks fail, stop and diagnose. Do not mask failures with try/except or by loosening tolerances.
- Small, well-scoped commits with descriptive messages. One conceptual change per commit.
- New methodological code is accompanied by tests in the same commit.
- No individual-level data in any artifact committed to the repository.
