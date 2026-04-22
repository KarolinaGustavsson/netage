"""Stratified train/validation/test splitting for the AMORIS cohort.

Stratification is on the joint (sex, age_decile) stratum so that both axes
are balanced across splits. The random seed is fixed at the dataset level and
must not be changed after any downstream analysis has begun.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SEED: int = 42
DEFAULT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)
N_AGE_DECILES: int = 10


@dataclass
class SplitResult:
    """Container for the three data splits.

    Attributes:
        train: Training set DataFrame.
        val: Validation set DataFrame.
        test: Test set DataFrame (held out; do not tune on this).
        seed: Random seed used to produce the split.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    seed: int

    def sizes(self) -> dict[str, int]:
        """Return row counts for each split."""
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def make_splits(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> SplitResult:
    """Stratified train/val/test split by sex and age decile.

    Args:
        df: Full dataset; must contain ``sex`` and ``age_at_baseline`` columns.
        ratios: (train, val, test) proportions; must be positive and sum to 1.
        seed: Random seed for reproducibility.

    Returns:
        SplitResult with non-overlapping train, val, and test DataFrames,
        each reset to a contiguous integer index.

    Raises:
        ValueError: If ratios do not sum to 1 or contain non-positive values.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios):.6f}")
    if any(r <= 0 for r in ratios):
        raise ValueError("All ratios must be positive")

    rng = np.random.default_rng(seed)

    work = df.copy()
    work["_age_decile"] = pd.qcut(
        work["age_at_baseline"], q=N_AGE_DECILES, labels=False, duplicates="drop"
    )
    work["_stratum"] = work["sex"].astype(str) + "_" + work["_age_decile"].astype(str)

    train_frac, val_frac = ratios[0], ratios[1]
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for _, group in work.groupby("_stratum", sort=True):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)

        n_train = max(1, round(n * train_frac))
        n_val = max(1, round(n * val_frac))
        # Ensure test receives at least one individual when the stratum is large enough.
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_val = min(n_val, n - n_train - 1)
        elif n == 2:
            n_train, n_val = 1, 1
        else:
            n_train, n_val = 1, 0

        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val :].tolist())

    # Drop temporary columns before returning.
    drop_cols = ["_age_decile", "_stratum"]
    result = SplitResult(
        train=df.loc[train_idx].drop(columns=drop_cols, errors="ignore").reset_index(drop=True),
        val=df.loc[val_idx].drop(columns=drop_cols, errors="ignore").reset_index(drop=True),
        test=df.loc[test_idx].drop(columns=drop_cols, errors="ignore").reset_index(drop=True),
        seed=seed,
    )
    logger.info("Split sizes (seed=%d): %s", seed, result.sizes())
    return result
