#!/bin/bash
#SBATCH --job-name=netage-amoris
#SBATCH --output=logs/netage_%j.log
#SBATCH --error=logs/netage_%j.err
#SBATCH --time=96:00:00               # 4 days (max on tensor)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12            # 12 CPUs for efficient PyTorch parallelization
#SBATCH --mem=96G                     # 96 GB RAM (plenty headroom for 260k individuals)
#SBATCH --partition=core              # Shared partition (CPU-only on tensor)

# NETAGE HPC submission script for tensor cluster (CPU-only, 7 nodes × 96 CPUs each).
# Optimized for distributed CPU computation across 12 cores with threading.
# 
# Cluster specs: 7 nodes, 96 CPUs/node, 748 GB RAM/node, max 25-day jobs
# Usage: sbatch scripts/submit_hpc.sh [config_file] [pipeline_stage]
#
# Example:
#   sbatch scripts/submit_hpc.sh configs/hpc.yaml
#   sbatch scripts/submit_hpc.sh configs/hpc.yaml preprocess
#   sbatch scripts/submit_hpc.sh configs/hpc.yaml train

set -euo pipefail

CONFIG_FILE="${1:-configs/hpc.yaml}"
PIPELINE_STAGE="${2:-all}"  # 'preprocess', 'train', 'bioage', 'evaluate', 'attribute', or 'all'

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "NETAGE HPC Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG_FILE"
echo "Stage: $PIPELINE_STAGE"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'None detected')"
echo "=========================================="
echo ""

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate netage
echo "Environment: $(which python)"
echo ""

# Run pipeline stages
case "$PIPELINE_STAGE" in
  preprocess)
    echo "Running preprocessing..."
    python scripts/preprocess.py --config "$CONFIG_FILE"
    ;;
  train)
    echo "Running training..."
    python scripts/train.py --config "$CONFIG_FILE"
    ;;
  bioage)
    echo "Computing biological age..."
    python scripts/compute_bioage.py --config "$CONFIG_FILE"
    ;;
  evaluate)
    echo "Evaluating model..."
    python scripts/evaluate.py --config "$CONFIG_FILE"
    ;;
  attribute)
    echo "Computing SHAP attributions..."
    python scripts/attribute.py --config "$CONFIG_FILE"
    ;;
  all)
    echo "Running full pipeline..."
    python scripts/preprocess.py --config "$CONFIG_FILE"
    python scripts/train.py --config "$CONFIG_FILE"
    python scripts/compute_bioage.py --config "$CONFIG_FILE"
    python scripts/evaluate.py --config "$CONFIG_FILE"
    python scripts/attribute.py --config "$CONFIG_FILE"
    ;;
  *)
    echo "Unknown stage: $PIPELINE_STAGE"
    echo "Valid stages: preprocess, train, bioage, evaluate, attribute, all"
    exit 1
    ;;
esac

echo ""
echo "=========================================="
echo "Job completed successfully"
echo "=========================================="
