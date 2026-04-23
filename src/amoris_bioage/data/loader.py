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
    
    # Rename columns from scrambled_b.csv schema to canonical names
    from amoris_bioage.data.schema import CSV_COL_MAPPING
    df = df.rename(columns=CSV_COL_MAPPING)
    
    _validate_columns(df)
    _cast_dtypes(df)
    
    # Filter out rows with invalid survival times (age_at_exit <= age_at_baseline)
    n_before = len(df)
    df = df[df["age_at_exit"] > df["age_at_baseline"]].copy()
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info("Removed %d rows with age_at_exit <= age_at_baseline", n_removed)
    
    _validate_survival_times(df)
    _validate_event_codes(df)

    n_missing = df[FEATURE_COLS].isna().sum()
    high_miss = n_missing[n_missing > 0]
    if not high_miss.empty:
        logger.debug("Missing value counts:\n%s", high_miss.to_string())

    logger.info(
        "Loaded %d individuals, all-cause mortality event rate %.1f%%",
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


def _validate_event_codes(df: pd.DataFrame) -> None:
    """Validate Event column codes for cause-specific outcome derivation."""
    valid_codes = {-10, 10, 20, 30, 40, 50}
    if "Event" in df.columns:
        invalid_codes = set(df["Event"].unique()) - valid_codes
        if invalid_codes:
            logger.warning(
                "Event column contains unexpected codes: %s",
                sorted(invalid_codes),
            )
