"""Snakemake workflow for the AMORIS biological age pipeline.

Stages
------
1. preprocess  — QC, split, and standardise raw data
2. train       — train CoxMLP; fit Breslow baseline hazard
3. compute_bioage — compute g and Δ for all splits
4. evaluate    — C-index, calibration, incremental LRT
5. attribute   — SHAP values for g and Δ; integrated gradients for η
6. report      — all paper figures

Quick start::

    snakemake --cores 4 --configfile configs/default.yaml

Cluster (SLURM)::

    snakemake --profile profiles/slurm --configfile configs/default.yaml
"""

configfile: "configs/default.yaml"

# Expose the config file path to Python scripts via snakemake.config.
config["_config_path"] = "configs/default.yaml"

OUTDIR   = "outputs"
DERIVED  = f"{OUTDIR}/derived"
MODELS   = f"{OUTDIR}/models"
RESULTS  = f"{OUTDIR}/results"
FIGURES  = f"{OUTDIR}/figures"


# ---------------------------------------------------------------------------
# Target: generate all paper figures
# ---------------------------------------------------------------------------

rule all:
    input:
        f"{FIGURES}/delta_distribution.pdf",
        f"{FIGURES}/km_by_delta.pdf",
        f"{FIGURES}/importance_contrast.pdf",
        f"{FIGURES}/group_importance.pdf",
        f"{FIGURES}/age_stratified_importance.pdf",
        f"{FIGURES}/shap_interactions.pdf",
        f"{FIGURES}/comparator_benchmark.pdf",


# ---------------------------------------------------------------------------
# Stage 1: preprocess
# ---------------------------------------------------------------------------

rule preprocess:
    input:
        config["data"]["raw_path"],
    output:
        train=f"{DERIVED}/train.csv",
        val=f"{DERIVED}/val.csv",
        test=f"{DERIVED}/test.csv",
        preprocessor=f"{DERIVED}/preprocessor.pkl",
    log:
        f"{OUTDIR}/logs/preprocess.log",
    script:
        "scripts/preprocess.py"


# ---------------------------------------------------------------------------
# Stage 2: train
# ---------------------------------------------------------------------------

rule train:
    input:
        train=f"{DERIVED}/train.csv",
        val=f"{DERIVED}/val.csv",
    output:
        checkpoint=f"{MODELS}/best_model.pt",
        breslow=f"{MODELS}/breslow.pkl",
        history=f"{MODELS}/training_history.json",
    log:
        f"{OUTDIR}/logs/train.log",
    resources:
        # Request a GPU if available; falls back to CPU.
        gpu=1,
    script:
        "scripts/train.py"


# ---------------------------------------------------------------------------
# Stage 3: compute biological ages
# ---------------------------------------------------------------------------

rule compute_bioage:
    input:
        checkpoint=f"{MODELS}/best_model.pt",
        breslow=f"{MODELS}/breslow.pkl",
    output:
        test=f"{RESULTS}/bioage_test.csv",
        # train and val are side-effects written unconditionally by the script.
    log:
        f"{OUTDIR}/logs/compute_bioage.log",
    script:
        "scripts/compute_bioage.py"


# ---------------------------------------------------------------------------
# Stage 4: evaluate
# ---------------------------------------------------------------------------

rule evaluate:
    input:
        test=f"{DERIVED}/test.csv",
        bioage=f"{RESULTS}/bioage_test.csv",
        checkpoint=f"{MODELS}/best_model.pt",
        breslow=f"{MODELS}/breslow.pkl",
    output:
        f"{RESULTS}/metrics.json",
    log:
        f"{OUTDIR}/logs/evaluate.log",
    script:
        "scripts/evaluate.py"


# ---------------------------------------------------------------------------
# Stage 5: attribution
# ---------------------------------------------------------------------------

rule attribute:
    input:
        train=f"{DERIVED}/train.csv",
        test=f"{DERIVED}/test.csv",
        checkpoint=f"{MODELS}/best_model.pt",
        breslow=f"{MODELS}/breslow.pkl",
        bioage=f"{RESULTS}/bioage_test.csv",
    output:
        shap_g=f"{RESULTS}/shap_g.npz",
        shap_delta=f"{RESULTS}/shap_delta.npz",
    params:
        n_explain=config.get("attribution", {}).get("n_explain", 500),
    log:
        f"{OUTDIR}/logs/attribute.log",
    script:
        "scripts/attribute.py"


# ---------------------------------------------------------------------------
# Stage 6: reporting figures
# ---------------------------------------------------------------------------

rule plot_delta_distribution:
    input:
        f"{RESULTS}/bioage_test.csv",
    output:
        f"{FIGURES}/delta_distribution.pdf",
    script:
        "scripts/reporting/plot_delta_distribution.py"


rule plot_km_by_delta:
    input:
        f"{RESULTS}/bioage_test.csv",
    output:
        f"{FIGURES}/km_by_delta.pdf",
    script:
        "scripts/reporting/plot_km_by_delta.py"


rule plot_importance_contrast:
    input:
        shap_g=f"{RESULTS}/shap_g.npz",
        shap_delta=f"{RESULTS}/shap_delta.npz",
    output:
        f"{FIGURES}/importance_contrast.pdf",
    script:
        "scripts/reporting/plot_importance_contrast.py"


rule plot_group_importance:
    input:
        shap_g=f"{RESULTS}/shap_g.npz",
        shap_delta=f"{RESULTS}/shap_delta.npz",
        groups="configs/variable_groups.yaml",
    output:
        f"{FIGURES}/group_importance.pdf",
    script:
        "scripts/reporting/plot_group_importance.py"


rule plot_age_stratified_importance:
    input:
        shap_delta=f"{RESULTS}/shap_delta.npz",
        bioage=f"{RESULTS}/bioage_test.csv",
    output:
        f"{FIGURES}/age_stratified_importance.pdf",
    script:
        "scripts/reporting/plot_age_stratified_importance.py"


rule plot_shap_interactions:
    input:
        shap_delta=f"{RESULTS}/shap_delta.npz",
    output:
        f"{FIGURES}/shap_interactions.pdf",
    script:
        "scripts/reporting/plot_shap_interactions.py"


rule plot_comparator_benchmark:
    input:
        metrics=f"{RESULTS}/metrics.json",
    output:
        f"{FIGURES}/comparator_benchmark.pdf",
    script:
        "scripts/reporting/plot_comparator_benchmark.py"
