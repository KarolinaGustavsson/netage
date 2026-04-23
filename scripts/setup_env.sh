#!/bin/bash
# Environment setup script for local and HPC execution.
# 
# Usage (local):
#   source scripts/setup_env.sh local
#
# Usage (HPC):
#   source scripts/setup_env.sh hpc

set -euo pipefail

ENV_TYPE="${1:-local}"  # 'local' or 'hpc'

echo "Setting up environment for: $ENV_TYPE"

# Activate conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install conda or activate it manually."
    exit 1
fi

eval "$(conda shell.bash hook)"

# Check if environment exists
if ! conda env list | grep -q "^netage "; then
    echo "Creating netage conda environment..."
    conda env create -f environment.yml
fi

# Activate environment
conda activate netage

# Set environment variables
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)/src"

if [ "$ENV_TYPE" = "hpc" ]; then
    # HPC-specific settings (CPU-only on tensor cluster)
    echo "Configuring for HPC (CPU-only)..."
    
    # CPU threading optimization (tensor has 96 CPUs per node)
    # SLURM sets SLURM_CPUS_PER_TASK automatically
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
    export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
    export PYTORCH_NUM_THREADS="${OMP_NUM_THREADS}"
    echo "  CPU threads: ${OMP_NUM_THREADS}"
    echo "  No GPU available (tensor is CPU-only)"
    
elif [ "$ENV_TYPE" = "local" ]; then
    # Local-specific settings
    echo "Configuring for local machine..."
    
    # Try to enable Metal Performance Shaders on macOS
    if [[ $(uname) == "Darwin" ]]; then
        export PYTORCH_ENABLE_MPS_FALLBACK=1
        echo "  macOS detected, MPS acceleration enabled (if available)"
    fi
fi

# Verify installation
echo ""
echo "Verification:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import lifelines; print(f'  lifelines: {lifelines.__version__}')"
python -c "import shap; print(f'  shap: {shap.__version__}')"
python -c "import amoris_bioage; print(f'  amoris_bioage: installed')"

echo ""
echo "Environment ready. To deactivate later, run: conda deactivate"
