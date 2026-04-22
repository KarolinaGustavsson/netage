# netage — Neural Biological Age from the AMORIS Cohort

Train a neural Cox proportional hazards model on 15 baseline biomarkers from the AMORIS cohort (~260 000 individuals), define biological age via inversion of the population mortality curve, and analyse feature importance with explicit contrast between attributions on biological age *g(x, t)* and the age gap *Δ(x, t) = g − t*.

## Scientific background

Biological age is defined by the 10-year mortality equivalence principle: *g(x, t)* is the chronological age at which a reference individual has the same predicted 10-year mortality as the measured individual. This formulation avoids the age-prediction paradox and produces attributions that are directly interpretable in terms of mortality risk.

The central analytic contribution is the contrast between

- **φ^{BA}**: feature attributions on *g* — what drives an individual to look biologically older/younger overall, and
- **φ^{Δ}**: feature attributions on *Δ* — what drives the deviation from the expected age trajectory.

See `PROJECT_DESCRIPTION.md` for the full scientific framing.

## Installation

```bash
conda env create -f environment.yml
conda activate netage
```

Requires Python 3.11, PyTorch ≥ 2.2, lifelines ≥ 0.29, shap 0.51, captum 0.9. On macOS the OpenMP library conflict is resolved automatically by the conda activation script.

## Running the pipeline

Edit `configs/default.yaml` to set `data.raw_path` to your AMORIS CSV, then:

```bash
# Full pipeline — all figures
snakemake --cores 4 --configfile configs/default.yaml

# Individual stages
python scripts/preprocess.py   --config configs/default.yaml
python scripts/train.py        --config configs/default.yaml
python scripts/compute_bioage.py --config configs/default.yaml
python scripts/evaluate.py     --config configs/default.yaml
python scripts/attribute.py    --config configs/default.yaml [--n-explain 500]

# Individual figures
python scripts/reporting/plot_delta_distribution.py
python scripts/reporting/plot_importance_contrast.py
# … etc.
```

Outputs are written to `outputs/` (derived data, model checkpoints, results, figures). Nothing in `outputs/` is committed to git.

## Repository structure

```
src/amoris_bioage/
  data/          load, QC, split, preprocess
  models/        CoxMLP network; Efron Cox loss
  training/      training loop (AdamW, cosine LR, early stopping)
  bioage/        Breslow estimator; biological age inversion
  validation/    C-index, calibration, incremental LRT
  attribution/   SHAP and integrated-gradients wrappers
scripts/
  preprocess.py  train/val/test split + standardisation
  train.py       CoxMLP training + Breslow fitting
  compute_bioage.py  compute g and Δ for all splits
  evaluate.py    metrics (C-index, calibration, LRT)
  attribute.py   SHAP values for g and Δ; IG for η
  reporting/     one script per paper figure
tests/           pytest suite (151+ tests)
configs/
  default.yaml   pipeline configuration
  variable_groups.yaml  pre-specified feature groups for attribution
Snakefile        full pipeline DAG
```

## Testing

```bash
conda run -n netage pytest -q
```

All 151+ tests should pass. Do not skip tests or loosen tolerances to make a commit pass; resolve the underlying issue instead.

## Key methodological commitments

1. Training target is **mortality**, not chronological age.
2. Time scale is **attained age** with left-truncation at baseline measurement.
3. Baseline hazard is **nonparametric (Breslow)**, stratified by sex.
4. Biological age is defined via **10-year mortality equivalence**.
5. Attributions are computed on **two targets** (*g* and *Δ*) with a **common age-stratified cohort-mean baseline**.
6. Model capacity is **modest by design** (2–3 hidden layers, 64–128 units).

See `CLAUDE.md` for the full specification.
