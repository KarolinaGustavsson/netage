"""Train and project Phenotypic Age using Gompertz AFT model.

Implements Modified Levine's Phenotypic Age method (Levine et al. 2018) in Python.
Trains on the same splits as NETAGE for direct comparison.

Uses raw (non-standardized) biomarkers. Gompertz AFT model fitted via scipy MLE.

Usage:
    python scripts/train_phenoage.py --config configs/default.yaml
    python scripts/train_phenoage.py --config configs/hpc.yaml --biomarkers s_alb s_krea tc tg s_k s_urea s_ld s_ca s_urat s_famn s_hapt fs_gluk
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from scipy.optimize import minimize
from scipy.special import gamma as gamma_fn

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Snakemake / standalone dispatch
# ---------------------------------------------------------------------------

if "snakemake" in dir():
    _config_path = snakemake.config["_config_path"]  # type: ignore[name-defined]
    _biomarkers = snakemake.params.get("biomarkers")  # type: ignore[name-defined]
else:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--biomarkers",
        nargs="+",
        default=None,
        help="Subset of biomarkers to use (default: all 17 from schema)",
    )
    args = parser.parse_args()
    _config_path = args.config
    _biomarkers = args.biomarkers

# ---------------------------------------------------------------------------
# Gompertz AFT Model
# ---------------------------------------------------------------------------


class GompertzFit(NamedTuple):
    """Fitted Gompertz AFT model parameters."""

    shape: float  # Gompertz shape parameter (α)
    scale: float  # Scale parameter (σ)
    coef: pd.DataFrame  # Coefficients for biomarkers + intercept
    m_n: float  # Numerator for mortality curve
    m_d: float  # Denominator for mortality curve
    ba_n: float  # Numerator for biological age inversion
    ba_d: float  # Denominator for biological age inversion
    ba_i: float  # Intercept for biological age inversion


def fit_gompertz_aft(
    time: np.ndarray, event: np.ndarray, X: np.ndarray, max_iter: int = 500
) -> tuple[float, float, np.ndarray]:
    """Fit Gompertz AFT model using maximum likelihood estimation.

    Gompertz AFT: h(t|x) = α * exp(β*t) * exp(η(x)) where η(x) = β₀ + β₁X₁ + ...
    log-likelihood summed over observations.

    Args:
        time: Event/censoring times (n,)
        event: Event indicators (n,) - 1=event, 0=censored
        X: Design matrix including intercept column (n, p)
        max_iter: Maximum iterations for optimization

    Returns:
        (shape, scale, coef) where coef includes intercept as first element
    """

    def neg_ll(params):
        """Negative log-likelihood for Gompertz AFT."""
        shape = params[0]
        scale = params[1]
        coef = params[2:]

        if shape <= 0 or scale <= 0:
            return 1e10

        eta = X @ coef  # Linear predictor

        # Gompertz survival: S(t|x) = exp(-α/β * (exp(β*t + η) - exp(η)))
        # where α = shape, β = scale, η = linear predictor
        log_h = np.log(shape) + scale * time + eta  # log hazard
        log_S = -(shape / scale) * (np.exp(scale * time + eta) - np.exp(eta))

        ll = event * log_h + log_S
        return -np.sum(ll)

    # Initialize parameters
    n_features = X.shape[1]
    # Use Cox model to initialize linear predictor
    try:
        cox = CoxPHFitter()
        cox.fit(
            pd.DataFrame(X[:, 1:], columns=[f"X{i}" for i in range(n_features - 1)]),
            duration_col="time",
            event_col="event",
            index_col=None,
            show_progress=False,
        )
    except:
        # Fallback if Cox fails
        cox = None

    if cox is not None:
        init_coef = np.concatenate([[0], cox.params_.values])
    else:
        init_coef = np.zeros(n_features)

    init_params = np.concatenate([[0.1], [0.1], init_coef])

    # Optimize
    result = minimize(
        neg_ll,
        init_params,
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-6, "fatol": 1e-6},
    )

    if not result.success:
        logger.warning("Gompertz MLE optimization did not converge: %s", result.message)

    shape = result.x[0]
    scale = result.x[1]
    coef = result.x[2:]

    return shape, scale, coef


def fit_gompertz_baseline(time: np.ndarray, event: np.ndarray) -> tuple[float, float]:
    """Fit Gompertz model with only time and event (no covariates).

    Used to compute biological age calibration parameters (ba_n, ba_d, ba_i).
    """

    def neg_ll_base(params):
        shape = params[0]
        scale = params[1]

        if shape <= 0 or scale <= 0:
            return 1e10

        log_h = np.log(shape) + scale * time
        log_S = -(shape / scale) * (np.exp(scale * time) - 1)

        ll = event * log_h + log_S
        return -np.sum(ll)

    init_params = np.array([0.1, 0.1])
    result = minimize(
        neg_ll_base,
        init_params,
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-6},
    )

    shape = result.x[0]
    scale = result.x[1]
    return shape, scale


def compute_phenoage(
    X: np.ndarray,
    biomarker_cols: list[str],
    intercept: float,
    coef_dict: dict[str, float],
    fit: GompertzFit,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute phenotypic age and age gap (Δ) from fitted model.

    Args:
        X: Data with biomarkers and age columns
        biomarker_cols: Names of biomarker columns
        intercept: Fitted intercept from Gompertz
        coef_dict: Fitted coefficients dict {colname: coef}
        fit: GompertzFit object with calibration parameters

    Returns:
        (phenoage, phenoage_advance) where phenoage_advance = phenoage - chronological_age
    """
    n = X.shape[0]
    phenoage = np.zeros(n)
    phenoage_advance = np.zeros(n)

    for i in range(n):
        row = X.iloc[i]
        age = row["age_at_baseline"]

        # Linear predictor: sum(biomarker * coef) + intercept
        eta = intercept
        for col in biomarker_cols:
            if col in coef_dict:
                eta += row[col] * coef_dict[col]
        eta += row["age_at_baseline"] * coef_dict.get(
            "age_at_baseline", 0
        )  # Add age coef if present

        # Mortality at 120 years
        t_ref = 120
        exp_xb = np.exp(eta)
        mortality = 1 - np.exp(
            (fit.m_n * exp_xb) / fit.m_d
        )  # m = 1 - exp((m_n * exp(xb)) / m_d)

        # Invert to biological age
        if mortality <= 0 or mortality >= 1:
            phenoage[i] = np.nan
        else:
            log_term = np.log(fit.ba_n * np.log(1 - mortality))
            phenoage[i] = (log_term / fit.ba_d) + fit.ba_i

        phenoage_advance[i] = phenoage[i] - age

    return phenoage, phenoage_advance


def _safe_cindex(
    df: pd.DataFrame, predicted_scores: np.ndarray, label: str
) -> tuple[float, int]:
    """Compute C-index after dropping non-finite rows in all inputs.

    lifelines raises a generic "NaNs detected" ValueError for invalid scoring
    arrays; this helper makes filtering explicit and logs what was removed.
    """
    times = df["age_at_exit"].to_numpy(dtype=float)
    events = df["event"].to_numpy(dtype=float)
    scores = np.asarray(predicted_scores, dtype=float)

    finite_mask = np.isfinite(times) & np.isfinite(events) & np.isfinite(scores)
    valid_n = int(finite_mask.sum())
    total_n = int(len(df))
    dropped_n = total_n - valid_n

    if dropped_n > 0:
        logger.warning(
            "%s: dropped %d/%d rows before C-index due to non-finite values "
            "(time=%d, event=%d, score=%d)",
            label,
            dropped_n,
            total_n,
            int((~np.isfinite(times)).sum()),
            int((~np.isfinite(events)).sum()),
            int((~np.isfinite(scores)).sum()),
        )

    if valid_n == 0:
        logger.warning("%s: no valid rows for C-index after filtering", label)
        return np.nan, 0

    cidx = concordance_index(
        times[finite_mask],
        scores[finite_mask],
        events[finite_mask],
    )
    return float(cidx), valid_n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

from amoris_bioage.config import load_config
from amoris_bioage.data.schema import FEATURE_COLS

cfg = load_config(_config_path)

# Resolve paths relative to project root
config_path = Path(_config_path)
if config_path.is_absolute():
    project_root = config_path.parent.parent
else:
    # Relative path - resolve from cwd
    project_root = Path.cwd()

logger.info("Project root: %s", project_root)

# Construct derived directory path
derived_dir = project_root / cfg.data.derived_dir
if not derived_dir.exists():
    raise FileNotFoundError(f"Derived directory not found: {derived_dir} (looked relative to {project_root})")

logger.info("Loading preprocessed splits from: %s", derived_dir)

# Load the preprocessor to get the same split indices
preprocessor_path = derived_dir / "preprocessor.pkl"
if not preprocessor_path.exists():
    raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

logger.info("Loaded preprocessor (fitted on %d features)", len(preprocessor.feature_cols))

# Load raw data
logger.info("Loading raw data from %s", cfg.data.raw_path)
from amoris_bioage.data.loader import load_raw
df_raw = load_raw(cfg.data.raw_path)

# Check for precomputed R phenoage columns and rename them
og_phenoage_cols = {}
for col in ["phenoage", "phenoage_advance", "phenoage_sd", "phenoage_SD"]:
    if col in df_raw.columns:
        og_col = f"OG_{col}"
        df_raw.rename(columns={col: og_col}, inplace=True)
        og_phenoage_cols[col] = og_col
        logger.info("Found precomputed %s → renamed to %s", col, og_col)

if og_phenoage_cols:
    logger.info("Will compare Python phenoage against R precomputed values")
else:
    logger.info("No precomputed R phenoage columns found")

# Load preprocessed splits to get the same indices
train_pp = pd.read_csv(derived_dir / "train.csv")
val_pp = pd.read_csv(derived_dir / "val.csv")
test_pp = pd.read_csv(derived_dir / "test.csv")

# Get IDs from preprocessed splits to filter raw data
train_ids = set(train_pp["id"].values)
val_ids = set(val_pp["id"].values)
test_ids = set(test_pp["id"].values)

train_raw = df_raw[df_raw["id"].isin(train_ids)].copy()
val_raw = df_raw[df_raw["id"].isin(val_ids)].copy()
test_raw = df_raw[df_raw["id"].isin(test_ids)].copy()

logger.info("Splits from raw data: train=%d val=%d test=%d", 
            len(train_raw), len(val_raw), len(test_raw))

# Determine biomarkers to use
if _biomarkers is not None:
    biomarker_cols = [b.upper() for b in _biomarkers]  # Normalize to uppercase
    logger.info("Using specified biomarkers: %s", biomarker_cols)
else:
    biomarker_cols = FEATURE_COLS
    logger.info("Using all 17 biomarkers from schema")

# Verify biomarkers exist in data
missing = set(biomarker_cols) - set(train_raw.columns)
if missing:
    raise ValueError(f"Missing biomarkers in raw data: {missing}")

# Check for excessive missingness
for col in biomarker_cols:
    miss_train = train_raw[col].isna().mean()
    miss_val = val_raw[col].isna().mean()
    miss_test = test_raw[col].isna().mean()
    if miss_train > 0.5:
        logger.warning(
            "Biomarker %s has %.1f%% missing in training set",
            col,
            miss_train * 100,
        )
    if miss_val > 0.5 or miss_test > 0.5:
        logger.warning(
            "Biomarker %s has %.1f%% missing in val, %.1f%% in test",
            col,
            miss_val * 100,
            miss_test * 100,
        )

# Remove rows with any biomarker NaN (simplified approach; could impute instead)
n_before_train = len(train_raw)
train_raw = train_raw.dropna(subset=biomarker_cols + ["age_at_baseline", "age_at_exit", "event"])
n_after_train = len(train_raw)
if n_after_train < n_before_train:
    logger.warning(
        "Removed %d rows with missing biomarkers from training set",
        n_before_train - n_after_train,
    )

n_before_val = len(val_raw)
val_raw = val_raw.dropna(subset=biomarker_cols + ["age_at_baseline", "age_at_exit", "event"])
n_after_val = len(val_raw)
if n_after_val < n_before_val:
    logger.warning(
        "Removed %d rows with missing biomarkers from validation set",
        n_before_val - n_after_val,
    )

n_before_test = len(test_raw)
test_raw = test_raw.dropna(subset=biomarker_cols + ["age_at_baseline", "age_at_exit", "event"])
n_after_test = len(test_raw)
if n_after_test < n_before_test:
    logger.warning(
        "Removed %d rows with missing biomarkers from test set",
        n_before_test - n_after_test,
    )

# =========================================================================
# TRAINING: Fit Gompertz on training set
# =========================================================================

# Verify required columns
required_cols = biomarker_cols + ["age_at_baseline", "age_at_exit", "event"]
missing_cols = set(required_cols) - set(train_raw.columns)
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

logger.info("Fitting Gompertz AFT model on %d training individuals", len(train_raw))

# Prepare design matrix: [1, biomarkers..., age]
X_train = train_raw[biomarker_cols + ["age_at_baseline"]].values
X_design_train = np.column_stack([np.ones(len(X_train)), X_train])

time_train = train_raw["age_at_exit"].values
event_train = train_raw["event"].values

# Fit Gompertz for biomarkers + age
shape, scale, coef = fit_gompertz_aft(time_train, event_train, X_design_train)
logger.info("Gompertz shape=%.6f scale=%.6f", shape, scale)

# Extract coefficients
coef_df = pd.DataFrame(
    {"coef": coef},
    index=["intercept"] + biomarker_cols + ["age_at_baseline"],
)
logger.info("Fitted coefficients:\n%s", coef_df.to_string())

intercept = coef[0]
coef_dict = {col: coef[i + 1] for i, col in enumerate(biomarker_cols + ["age_at_baseline"])}

# Compute mortality curve calibration parameters
# Fit Gompertz on age only for reference
time_train_ref = train_raw["age_at_exit"].values
event_train_ref = train_raw["event"].values
shape_age, scale_age = fit_gompertz_baseline(time_train_ref, event_train_ref)

# Mortality at t=120
t_ref = 120
m_n = -(np.exp(scale * t_ref) - 1)
m_d = scale

ba_d = scale_age
ba_n = -(np.exp(scale_age * t_ref) - 1)
ba_i = (
    -np.log(np.exp(scale_age * t_ref) - 1) - coef_df.loc["intercept", "coef"]
) / ba_d

logger.info("Calibration: m_n=%.6f m_d=%.6f ba_n=%.6f ba_d=%.6f ba_i=%.6f", m_n, m_d, ba_n, ba_d, ba_i)

fit_obj = GompertzFit(
    shape=shape,
    scale=scale,
    coef=coef_df,
    m_n=m_n,
    m_d=m_d,
    ba_n=ba_n,
    ba_d=ba_d,
    ba_i=ba_i,
)

# =========================================================================
# VALIDATION: Project onto val set and compute phenoage
# =========================================================================

logger.info("Projecting onto validation set (%d individuals)", len(val_raw))
val_phenoage, val_advance = compute_phenoage(
    val_raw, biomarker_cols, intercept, coef_dict, fit_obj
)

val_raw["phenoage"] = val_phenoage
val_raw["phenoage_advance"] = val_advance

# C-index on validation (exclude NaN predictions)
val_cindex, val_valid_n = _safe_cindex(
    val_raw,
    -val_phenoage,
    label="Validation C-index (phenoage)",
)
if val_valid_n > 0:
    logger.info(
        "Validation C-index (phenoage): %.4f (%d valid individuals)",
        val_cindex,
        val_valid_n,
    )

val_adv_mask = np.isfinite(val_advance)
mean_advance_val = val_advance[val_adv_mask].mean() if val_adv_mask.any() else np.nan
std_advance_val = val_advance[val_adv_mask].std() if val_adv_mask.any() else np.nan
logger.info("Validation: mean_advance=%.3f std_advance=%.3f", mean_advance_val, std_advance_val)

# Compare with R phenoage if available
if "OG_phenoage" in val_raw.columns:
    valid_mask = ~(np.isnan(val_phenoage) | val_raw["OG_phenoage"].isna())
    if valid_mask.sum() > 0:
        phenoage_diff = val_raw.loc[valid_mask, "OG_phenoage"].values - val_phenoage[valid_mask]
        logger.info(
            "Validation phenoage_diff (R - Python): mean=%.3f std=%.3f min=%.3f max=%.3f",
            phenoage_diff.mean(),
            phenoage_diff.std(),
            phenoage_diff.min(),
            phenoage_diff.max(),
        )
        val_raw["phenoage_diff"] = val_raw["OG_phenoage"] - val_phenoage

# =========================================================================
# TEST: Project onto test set
# =========================================================================

logger.info("Projecting onto test set (%d individuals)", len(test_raw))
test_phenoage, test_advance = compute_phenoage(
    test_raw, biomarker_cols, intercept, coef_dict, fit_obj
)

test_raw["phenoage"] = test_phenoage
test_raw["phenoage_advance"] = test_advance

# C-index on test (exclude NaN predictions)
test_cindex, test_valid_n = _safe_cindex(
    test_raw,
    -test_phenoage,
    label="Test C-index (phenoage)",
)
if test_valid_n > 0:
    logger.info(
        "Test C-index (phenoage): %.4f (%d valid individuals)",
        test_cindex,
        test_valid_n,
    )

test_adv_mask = np.isfinite(test_advance)
mean_advance_test = test_advance[test_adv_mask].mean() if test_adv_mask.any() else np.nan
std_advance_test = test_advance[test_adv_mask].std() if test_adv_mask.any() else np.nan
logger.info("Test: mean_advance=%.3f std_advance=%.3f", mean_advance_test, std_advance_test)

# Compare with R phenoage if available
if "OG_phenoage" in test_raw.columns:
    valid_mask = ~(np.isnan(test_phenoage) | test_raw["OG_phenoage"].isna())
    if valid_mask.sum() > 0:
        phenoage_diff = test_raw.loc[valid_mask, "OG_phenoage"].values - test_phenoage[valid_mask]
        logger.info(
            "Test phenoage_diff (R - Python): mean=%.3f std=%.3f min=%.3f max=%.3f",
            phenoage_diff.mean(),
            phenoage_diff.std(),
            phenoage_diff.min(),
            phenoage_diff.max(),
        )
        test_raw["phenoage_diff"] = test_raw["OG_phenoage"] - test_phenoage

# =========================================================================
# SAVE RESULTS
# =========================================================================

out_dir = project_root / "outputs" / "results"
out_dir.mkdir(parents=True, exist_ok=True)
logger.info("Output directory: %s", out_dir)

# Determine output suffix based on biomarkers used
if _biomarkers is not None:
    suffix = "_subset"
else:
    suffix = "_all17"

val_raw.to_csv(out_dir / f"phenoage_val{suffix}.csv", index=False)
test_raw.to_csv(out_dir / f"phenoage_test{suffix}.csv", index=False)

# Save fitted model
fit_dict = {
    "shape": fit_obj.shape,
    "scale": fit_obj.scale,
    "coef": fit_obj.coef,
    "m_n": fit_obj.m_n,
    "m_d": fit_obj.m_d,
    "ba_n": fit_obj.ba_n,
    "ba_d": fit_obj.ba_d,
    "ba_i": fit_obj.ba_i,
    "biomarkers": biomarker_cols,
}
with open(out_dir / f"phenoage_fit{suffix}.pkl", "wb") as f:
    pickle.dump(fit_dict, f)

logger.info("Phenoage results saved to %s", out_dir)
logger.info("Files: phenoage_val%s.csv, phenoage_test%s.csv, phenoage_fit%s.pkl", suffix, suffix, suffix)
