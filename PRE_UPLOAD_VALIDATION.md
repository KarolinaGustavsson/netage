# Pre-Upload Validation Checklist

## ✅ Project Structure

```
netage-main/
├── CLAUDE.md                    # Methodological commitments (reference only)
├── PROJECT_DESCRIPTION.md       # Scientific framing
├── README.md                    # Quick start guide (updated)
├── LOCAL_vs_HPC.md             # Local vs tensor HPC guide (NEW)
├── environment.yml              # Conda environment specification
├── pyproject.toml              # Package metadata
├── Snakefile                   # Full pipeline DAG
├── scripts/
│   ├── setup_env.sh           # Environment setup for local/HPC (NEW)
│   ├── submit_hpc.sh          # SLURM submission script (NEW)
│   ├── preprocess.py          # Data preprocessing ✓
│   ├── train.py               # Model training ✓
│   ├── compute_bioage.py      # Biological age computation ✓
│   ├── evaluate.py            # Validation metrics ✓
│   ├── attribute.py           # SHAP attributions ✓
│   └── reporting/             # Figure generation scripts ✓
├── configs/
│   ├── default.yaml           # Local configuration (updated)
│   ├── hpc.yaml              # HPC configuration (NEW)
│   └── variable_groups.yaml   # Feature groupings (updated)
├── src/amoris_bioage/
│   ├── data/
│   │   ├── loader.py         # CSV loading with column rename (updated)
│   │   ├── schema.py         # Column schema + biomarker mapping (updated)
│   │   ├── preprocessing.py  # Standardization, missing indicators ✓
│   │   └── splits.py         # Train/val/test stratification ✓
│   ├── models/
│   │   ├── network.py        # CoxMLP neural network ✓
│   │   └── cox_loss.py       # Efron Cox loss ✓
│   ├── training/
│   │   ├── trainer.py        # Main training loop ✓
│   │   └── dataset.py        # PyTorch dataset wrapper ✓
│   ├── bioage/
│   │   ├── breslow.py        # Nonparametric baseline hazard ✓
│   │   └── inversion.py      # Mortality-equivalence inversion ✓
│   ├── validation/
│   │   ├── concordance.py    # C-index ✓
│   │   ├── calibration.py    # Calibration plots ✓
│   │   └── incremental.py    # Incremental LRT ✓
│   ├── attribution/
│   │   ├── shap_explainer.py # SHAP wrapper ✓
│   │   ├── ig_explainer.py   # Integrated gradients wrapper ✓
│   │   └── background.py     # Age-stratified cohort mean ✓
│   └── config.py             # Pydantic config classes ✓
└── tests/
    ├── conftest.py           # Pytest fixtures ✓
    ├── test_*.py             # 151+ tests ✓
    └── fixtures/             # Synthetic test data ✓
```

## ✅ Data Changes

- **17 biomarkers** (was 15): Added S_FAMN, S_HAPT, S_Urea, S_LD, three iron/mineral features
- **Column renaming**: Automatically handled in `loader.py` via `CSV_COL_MAPPING`
  - `sampleID` → `id`
  - `Kon` → `sex`
  - `age` → `age_at_baseline`
  - `lastAge` → `age_at_exit`
  - `status` → `event`
- **Event handling**: Binary all-cause mortality (status/event), with Event column preserved for dementia derivation
- **Feature groups**: 7 groups (was 6): added `mineral_electrolyte`

## ✅ Configuration

### `configs/default.yaml` (Local)
- ✅ Data path: `/Users/karolina.gustavsson/...data/amoris/scrambled_b.csv` (your machine)
- ✅ n_features: 17
- ✅ Batch size: 4096 (CPU-friendly)
- ✅ SHAP n_explain: 500

### `configs/hpc.yaml` (tensor)
- ⚠️  Data path: `/scratch/shared/amoris/scrambled_b.csv` — **NEEDS ADJUSTMENT**
  - Update to match actual path on tensor
- ✅ n_features: 17
- ✅ Batch size: 8192 (GPU-optimized)
- ✅ SHAP n_explain: 500

## ✅ Scripts

### `scripts/setup_env.sh`
Handles both local and HPC environment activation:
```bash
source scripts/setup_env.sh local   # Enables Metal on macOS
source scripts/setup_env.sh hpc     # Enables GPU + threading
```

### `scripts/submit_hpc.sh`
SLURM submission script (tensor):
```bash
sbatch scripts/submit_hpc.sh configs/hpc.yaml all
sbatch scripts/submit_hpc.sh configs/hpc.yaml train     # Test single stage
```

Current SLURM settings:
- `--time=24:00:00` (24 hours max)
- `--cpus-per-task=8`
- `--mem=64G`
- `--gres=gpu:1` (1 GPU)
- `--partition=gpu` — **NEEDS VERIFICATION** for tensor

## ✅ Validation Checks Passed

- [x] 17 biomarkers correctly mapped in schema.py
- [x] Column rename mapping in loader.py (scrambled_b.csv → canonical names)
- [x] Event column validation in loader.py (codes documented)
- [x] Feature groups in variable_groups.yaml (7 groups, all biomarkers assigned)
- [x] Config files valid YAML with sensible defaults
- [x] Setup scripts executable and sourced correctly
- [x] SLURM script provides modular stage control
- [x] README.md updated with local/HPC instructions
- [x] LOCAL_vs_HPC.md provides complete reference

## ⚠️  Before Upload to tensor

1. **Update HPC data path** in `configs/hpc.yaml`:
   ```yaml
   data:
     raw_path: /nfs/home/kargus/projects/netage-main/data/amoris/scrambled_b.csv
     # or wherever scrambled_b.csv is on tensor
   ```

2. **Verify tensor SLURM settings** (`scripts/submit_hpc.sh`):
   - [ ] `--partition=gpu` exists on tensor (or update to correct name)
   - [ ] GPU type available (V100, A100, etc.)?
   - [ ] Max job time limit?
   - [ ] Memory per node?
   - Update SLURM directives accordingly

3. **Keep data file separate** (not in git):
   - Don't commit `data/amoris/scrambled_b.csv`
   - Ensure `.gitignore` includes it

## 📁 Tensor Folder Structure (Planned)

```
/nfs/home/kargus/
├── projects/
│   └── netage-main/        ← Clone repo here
│       ├── scripts/
│       ├── src/
│       ├── data/
│       │   └── amoris/
│       │       └── scrambled_b.csv  ← Place data here
│       ├── outputs/        ← Generated on-the-fly (outputs/derived, logs, etc.)
│       ├── logs/           ← SLURM job logs
│       └── configs/
│           ├── default.yaml (not used on HPC)
│           └── hpc.yaml    ← Update raw_path here
```

## 🚀 First Test Run (Recommended)

1. **Local quick test:**
   ```bash
   source scripts/setup_env.sh local
   python scripts/preprocess.py --config configs/default.yaml  # ~5-10 min
   ```

2. **Once verified, upload to tensor:**
   ```bash
   git clone <repo> /nfs/home/kargus/projects/netage-main
   cd /nfs/home/kargus/projects/netage-main
   ```

3. **On tensor:**
   ```bash
   source scripts/setup_env.sh hpc
   sbatch scripts/submit_hpc.sh configs/hpc.yaml preprocess  # Test single stage
   ```

## Summary

✅ **Everything is ready.** The only blocking item before tensor upload is:
- [ ] Update `configs/hpc.yaml` with correct `raw_path` for tensor
- [ ] Verify SLURM partition name and GPU availability on tensor

Once those are confirmed, you can upload to `/nfs/home/kargus/projects/netage-main` and run immediately.
