# Running NETAGE: Local vs HPC

This guide explains how to run the AMORIS biological age pipeline on your local machine or on the tensor HPC cluster.

## Quick Start

### Local Machine

1. **Set up environment:**
   ```bash
   source scripts/setup_env.sh local
   ```

2. **Update data path in config** (edit `configs/default.yaml`):
   ```yaml
   data:
     raw_path: /Users/karolina.gustavsson/Library/CloudStorage/OneDrive-KarolinskaInstitutet/Coding_PhD_cloud/Project2/Scripts/NETAGE/netage-main/data/amoris/scrambled_b.csv
   ```

3. **Run pipeline:**
   ```bash
   # Full pipeline
   python scripts/preprocess.py --config configs/default.yaml
   python scripts/train.py --config configs/default.yaml
   python scripts/compute_bioage.py --config configs/default.yaml
   python scripts/evaluate.py --config configs/default.yaml
   python scripts/attribute.py --config configs/default.yaml
   
   # Or use Snakemake for all at once:
   snakemake --cores 4 --configfile configs/default.yaml
   ```

### HPC (tensor cluster)

1. **Place data on tensor:**
   ```bash
   # Copy scrambled_b.csv to tensor (from local machine)
   scp data/amoris/scrambled_b.csv kargus@tensor:/nfs/home/kargus/projects/netage-main/data/amoris/
   ```

2. **HPC config** (already configured in `configs/hpc.yaml`):
   ```yaml
   data:
     raw_path: /nfs/home/kargus/projects/netage-main/data/amoris/scrambled_b.csv
   ```

3. **Set up environment on tensor:**
   ```bash
   ssh tensor
   cd /nfs/home/kargus/projects/netage-main
   source scripts/setup_env.sh hpc
   ```

4. **Submit job to SLURM:**
   ```bash
   # Full pipeline (96 hours max)
   sbatch scripts/submit_hpc.sh configs/hpc.yaml all
   
   # Single stage (faster, for testing)
   sbatch scripts/submit_hpc.sh configs/hpc.yaml preprocess
   sbatch scripts/submit_hpc.sh configs/hpc.yaml train
   sbatch scripts/submit_hpc.sh configs/hpc.yaml attribute
   ```

5. **Monitor job:**
   ```bash
   squeue -u kargus           # Check job status
   tail -f logs/netage_*.log  # Watch logs (if still running)
   cat logs/netage_*.log      # View completed logs after finished
   ```

**Cluster specs:** 7 nodes, 96 CPUs/node, 748 GB RAM/node, CPU-only (no GPU)  
**Requested resources:** 12 CPUs, 96 GB RAM, up to 96 hours  
**Estimated runtime:** 16–24 hours for full pipeline

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Local machine (default) |
| `configs/hpc.yaml` | HPC tensor cluster |

Each config specifies:
- **Data paths** — where to find/write data
- **Batch size** — optimized per environment
- **SHAP explanation count** — full pipeline on HPC, reduced for local testing

## Customization

### For Testing (Faster Iteration)

Create `configs/test.yaml`:
```yaml
data:
  train_frac: 0.10     # Use only 10% of data
  raw_path: ./data/amoris/scrambled_b.csv

training:
  max_epochs: 20       # Stop early
  batch_size: 4096

attribution:
  n_explain: 10        # Explain only 10 individuals
```

Then run:
```bash
# Local
python scripts/train.py --config configs/test.yaml

# HPC
sbatch scripts/submit_hpc.sh configs/test.yaml train
```

### For Full Production Run

Use `configs/hpc.yaml` as-is. Full pipeline takes ~16–24 hours on tensor (12 CPUs, 96 GB RAM).

## Troubleshooting

### Local: Out of Memory
- Reduce `batch_size` in config (e.g., 2048)
- Reduce `n_explain` for SHAP (e.g., 100)

### HPC: Job Fails
Check logs:
```bash
cat logs/netage_<jobid>.err
```

Common issues:
- **Path not found:** Update `raw_path` in `configs/hpc.yaml`
- **Out of memory:** Increase `--mem` in `scripts/submit_hpc.sh`
- **Timeout:** Increase `--time` in `scripts/submit_hpc.sh` (max is 25 days on tensor)
- **CPU throttling:** Reduce `batch_size` (currently 4096) or request fewer CPUs to avoid resource contention

## Output Locations

All outputs go to `./outputs/`:
```
outputs/
├── derived/              # Preprocessed data splits
├── models/               # Trained model checkpoints
├── results/              # Validation metrics, attributions
└── figures/              # Paper figures
```

Outputs sync back from HPC — include `outputs/` in your file transfer.

## Performance Tips

| Environment | Optimization |
|-------------|-------------|
| **Local (Mac)** | Enable Metal (PyTorch MPS) for 2-5× speedup on CPU |
| **Local (Linux)** | Use CUDA if GPU available; otherwise optimize threading |
| **tensor (CPU)** | Uses 12 CPUs with automatic thread management via OMP_NUM_THREADS |

## Questions?

Refer to `CLAUDE.md` for methodological commitments and `README.md` for pipeline overview.
