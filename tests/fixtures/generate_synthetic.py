#!/usr/bin/env python3
"""Generate synthetic AMORIS-style survival datasets for unit testing.

Simulates the AMORIS structure:
- 15 biomarkers with realistic age/sex effects and distributions
- Survival outcome on the attained-age time scale with left-truncation
  (entry at age_at_baseline, exit at age_at_exit)
- A Gompertz baseline hazard, so event times have a known ground truth
- Selective missingness in three biomarkers

The Cox linear predictor uses known beta coefficients; downstream tests can
use these to verify attribution and inversion logic on synthetic data.

Usage:
    python generate_synthetic.py              # defaults: small=200, medium=2000
    python generate_synthetic.py --n-small 500 --n-medium 5000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent

# fmt: off
# (base_mean, base_std, age_slope_per_year, sex_additive_effect_for_male, log_normal)
# age_slope is relative to reference age 57.5 (midpoint of 40-75 range)
FEATURE_SPECS: dict[str, tuple[float, float, float, float, bool]] = {
    "cholesterol":   (5.50,  1.00,  0.010,  -0.20, False),
    "triglycerides": (1.30,  0.60,  0.010,   0.30, True),
    "hdl":           (1.40,  0.40, -0.005,  -0.30, False),
    "ldl":           (3.50,  0.80,  0.010,   0.00, False),
    "crp":           (2.00,  3.00,  0.050,   0.30, True),
    "wbc":           (6.50,  1.50,  0.010,   0.20, False),
    "albumin":       (43.0,  4.00, -0.100,  -0.50, False),
    "creatinine":    (80.0, 15.00,  0.500,  15.00, False),
    "uric_acid":     (300.0, 60.0,  1.500,  60.00, False),
    "alp":           (80.0, 25.00,  0.500,  -5.00, False),
    "ggt":           (25.0, 20.00,  0.500,  15.00, True),
    "alt":           (25.0, 15.00,  0.100,   8.00, True),
    "glucose":       (5.50,  1.00,  0.030,   0.10, False),
    "hemoglobin":    (14.0,  1.50, -0.020,   1.50, False),
    "iron":          (18.0,  5.00, -0.050,  -2.00, False),
}

# True Cox betas used to generate event times.
# Stored here so tests can verify attribution against ground truth.
TRUE_BETAS: dict[str, float] = {
    "cholesterol":  -0.08,
    "triglycerides": 0.05,
    "hdl":          -0.15,
    "ldl":           0.05,
    "crp":           0.15,
    "wbc":           0.08,
    "albumin":      -0.20,
    "creatinine":    0.12,
    "uric_acid":     0.05,
    "alp":           0.04,
    "ggt":           0.10,
    "alt":           0.06,
    "glucose":       0.10,
    "hemoglobin":   -0.08,
    "iron":         -0.04,
}
# fmt: on

FEATURE_COLS = list(FEATURE_SPECS.keys())

# Columns that receive synthetic missingness (col -> fraction missing).
MISSING_FRACS: dict[str, float] = {
    "crp":  0.08,
    "iron": 0.10,
    "ggt":  0.06,
}

# Gompertz baseline hazard parameters: h0(t) = LAMBDA * exp(GAMMA * t)
GOMPERTZ_LAMBDA: float = 1e-5
GOMPERTZ_GAMMA: float = 0.09

# Age effect on the linear predictor (per year above reference age 57.5).
AGE_BETA: float = 0.07

REF_AGE: float = 57.5
MAX_FOLLOWUP_YEARS: float = 25.0
MAX_ATTAINED_AGE: float = 92.0


def generate_dataset(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic AMORIS-style dataset of size *n*.

    Args:
        n: Number of individuals to simulate.
        seed: Random seed; controls all stochastic draws.

    Returns:
        DataFrame with columns: id, sex, age_at_baseline, age_at_exit, event,
        and the 15 biomarker features (some with missingness).
    """
    rng = np.random.default_rng(seed)

    sex = rng.integers(0, 2, size=n)                     # 0=female, 1=male
    age_at_baseline = rng.uniform(40.0, 75.0, size=n)

    # --- Biomarker generation ---
    raw_features: dict[str, np.ndarray] = {}
    for col, (base_mean, base_std, age_slope, sex_effect, log_normal) in FEATURE_SPECS.items():
        loc = base_mean + age_slope * (age_at_baseline - REF_AGE) + sex_effect * sex
        if log_normal:
            log_loc = np.log(np.maximum(loc, 0.1))
            values = rng.lognormal(log_loc, 0.5, size=n)
        else:
            values = rng.normal(loc, base_std, size=n)
        raw_features[col] = values

    # --- Cox linear predictor ---
    feature_matrix = np.column_stack([raw_features[col] for col in FEATURE_COLS])
    feature_means = np.array([FEATURE_SPECS[col][0] for col in FEATURE_COLS])
    feature_stds = np.array([FEATURE_SPECS[col][1] for col in FEATURE_COLS])
    betas = np.array([TRUE_BETAS[col] for col in FEATURE_COLS])

    # Standardise features before applying betas (consistent with model training).
    z = (feature_matrix - feature_means) / feature_stds
    eta = z @ betas + AGE_BETA * (age_at_baseline - REF_AGE)

    # --- Event time generation via inverse CDF (Gompertz baseline) ---
    # Cumulative baseline hazard: H0(s, t) = (λ/γ)(exp(γt) - exp(γs))
    # Setting H0(entry, T) * exp(eta) = E ~ Exp(1) and solving for T:
    # exp(γT) = exp(γ*entry) + (γ/λ) * E * exp(-eta)
    lam, gam = GOMPERTZ_LAMBDA, GOMPERTZ_GAMMA
    E = rng.exponential(1.0, size=n)
    inner = np.exp(gam * age_at_baseline) + (gam / lam) * E * np.exp(-eta)
    age_at_event = np.log(np.maximum(inner, np.exp(gam * age_at_baseline) + 1e-12)) / gam

    # Administrative censoring: max follow-up OR max attained age.
    censor_age = np.minimum(age_at_baseline + MAX_FOLLOWUP_YEARS, MAX_ATTAINED_AGE)
    age_at_exit = np.minimum(age_at_event, censor_age)
    event = (age_at_event <= censor_age).astype(int)

    # Enforce a minimum follow-up of ~1 month so that rounding to 2 d.p.
    # cannot produce age_at_exit == age_at_baseline.
    age_at_exit = np.maximum(age_at_exit, age_at_baseline + 0.09)

    assert (age_at_exit > age_at_baseline).all(), "BUG: exit age ≤ entry age"

    df = pd.DataFrame(
        {
            "id": [f"SYN{i:07d}" for i in range(n)],
            "sex": sex,
            "age_at_baseline": np.round(age_at_baseline, 2),
            "age_at_exit": np.round(age_at_exit, 2),
            "event": event,
            **{col: np.round(raw_features[col], 3) for col in FEATURE_COLS},
        }
    )

    # --- Introduce missingness ---
    for col, frac in MISSING_FRACS.items():
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-small", type=int, default=200)
    parser.add_argument("--n-medium", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, n in [
        ("synthetic_small", args.n_small),
        ("synthetic_medium", args.n_medium),
    ]:
        df = generate_dataset(n, seed=args.seed)
        out_path = OUTPUT_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(
            f"Wrote {out_path}  n={n}  event_rate={df['event'].mean():.1%}  "
            f"miss_crp={df['crp'].isna().mean():.1%}"
        )


if __name__ == "__main__":
    main()
