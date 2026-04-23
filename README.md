# netage — Neural Biological Age from the AMORIS Cohort

Train a neural Cox proportional hazards model on 17 baseline biomarkers from the AMORIS cohort (~260 000 individuals), define biological age via inversion of the population mortality curve, and analyse feature importance with explicit contrast between attributions on biological age *g(x, t)* and the age gap *Δ(x, t) = g − t*.

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

## Running locally vs on HPC

### Local Machine

Setup:
```bash
source scripts/setup_env.sh local
```

Then run:
```bash
snakemake --cores 4 --configfile configs/default.yaml
```

**Estimated time:** 6–20 hours (depending on hardware and GPU availability)

### HPC (tensor cluster)

Setup:
```bash
source scripts/setup_env.sh hpc
```

Submit job:
```bash
sbatch scripts/submit_hpc.sh configs/hpc.yaml all
```

Monitor:
```bash
squeue -u kargus
```

**Cluster:** tensor (CPU-only: 7 nodes × 96 CPUs/node, 748 GB RAM/node)

**Estimated time:** 16–24 hours on 12 CPUs

See `LOCAL_vs_HPC.md` for full configuration and troubleshooting.

## Running the pipeline

1. **Set up environment** (choose one):
   ```bash
   source scripts/setup_env.sh local    # for local machine
   source scripts/setup_env.sh hpc      # for HPC cluster
   ```

2. **Update data path** in appropriate config:
   - `configs/default.yaml` — for local
   - `configs/hpc.yaml` — for HPC

3. **Run pipeline:**

   **Local machine:**
   ```bash
   snakemake --cores 4 --configfile configs/default.yaml
   
   # Or individual stages
   python scripts/preprocess.py --config configs/default.yaml
   python scripts/train.py --config configs/default.yaml
   ```

   **HPC cluster:**
   ```bash
   sbatch scripts/submit_hpc.sh configs/hpc.yaml all
   
   # Or single stages for testing
   sbatch scripts/submit_hpc.sh configs/hpc.yaml preprocess
   sbatch scripts/submit_hpc.sh configs/hpc.yaml train
   ```

4. **View results:**
   ```bash
   python scripts/reporting/plot_delta_distribution.py
   python scripts/reporting/plot_importance_contrast.py
   # … see scripts/reporting/ for all figures
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
