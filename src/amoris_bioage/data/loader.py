"""Load and validate AMORIS data from CSV files."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from amoris_bioage.data.schema import ALL_COLS, EXPECTED_DTYPES, FEATURE_COLS

logger = logging.getLogger(__name__)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load and validate a raw AMORIS CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with validated schema and cast dtypes.

    Raises:
        ValueError: If required columns are missing, survival times are
            inconsistent, or event indicators are non-binary.
    """
    path = Path(path)
    logger.info("Loading data from %s", path)

    df = pd.read_csv(path)
    _validate_columns(df)
    _cast_dtypes(df)
    _validate_survival_times(df)

    n_missing = df[FEATURE_COLS].isna().sum()
    high_miss = n_missing[n_missing > 0]
    if not high_miss.empty:
        logger.debug("Missing value counts:\n%s", high_miss.to_string())

    logger.info(
        "Loaded %d individuals, event rate %.1f%%",
        len(df),
        df["event"].mean() * 100,
    )
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    missing = set(ALL_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _cast_dtypes(df: pd.DataFrame) -> None:
    for col, dtype in EXPECTED_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)


def _validate_survival_times(df: pd.DataFrame) -> None:
    bad_times = df["age_at_exit"] <= df["age_at_baseline"]
    if bad_times.any():
        raise ValueError(
            f"{bad_times.sum()} rows have age_at_exit <= age_at_baseline "
            "(violates left-truncation assumption)"
        )
    if not df["event"].isin([0, 1]).all():
        raise ValueError("event column must contain only 0 or 1")
