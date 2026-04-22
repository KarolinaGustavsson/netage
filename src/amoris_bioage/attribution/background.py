"""Age-stratified cohort-mean background for SHAP attribution.

The background set is constructed by binning the training cohort into age
deciles and computing the per-decile mean of every feature.  This produces a
small reference set (~10 rows) that represents the age-conditional feature
distribution without using individual-level data.

Using an age-stratified background rather than a single cohort mean is
important because the reference biological age g(x, t) is defined relative
to a mortality curve that is itself age-dependent.  Attributing relative to
the age-matched average gives attributions that are comparable across age
strata.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_age_stratified_background(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_age_bins: int = 10,
    age_col: str = "age_at_baseline",
    sex_col: str = "sex",
) -> pd.DataFrame:
    """Compute age-stratified cohort-mean background for SHAP.

    Bins the cohort into equal-frequency age groups and returns the mean
    feature vector within each group.  The result is used as the reference
    distribution (masker background) for SHAP attribution.

    Args:
        df: Preprocessed DataFrame containing ``feature_cols``, ``age_col``,
            and ``sex_col``.
        feature_cols: Ordered biomarker feature column names to average.
        n_age_bins: Number of equal-frequency age bins (default 10 = deciles).
        age_col: Name of the chronological age column.
        sex_col: Name of the binary sex column (0/1).

    Returns:
        DataFrame with one row per age bin and columns
        ``feature_cols + [age_col, sex_col]``, ordered by ascending age.

    Raises:
        ValueError: If any required column is absent from ``df``.
    """
    all_cols = feature_cols + [age_col, sex_col]
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in df: {missing}")

    work = df[all_cols].copy()
    work["_age_bin"] = pd.qcut(
        work[age_col], q=n_age_bins, labels=False, duplicates="drop"
    )

    bg = (
        work.groupby("_age_bin")[all_cols]
        .mean()
        .reset_index(drop=True)
        .sort_values(age_col)
        .reset_index(drop=True)
    )
    # Sex is binary; round the mean to the nearest integer.
    bg[sex_col] = bg[sex_col].round().astype(np.int64)
    return bg
