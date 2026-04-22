"""Tests for data loading and schema validation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from amoris_bioage.data.loader import load_raw
from amoris_bioage.data.schema import ALL_COLS, FEATURE_COLS, META_COLS, SURVIVAL_COLS


class TestLoadRaw:
    def test_returns_dataframe(self, synthetic_small: pd.DataFrame) -> None:
        assert isinstance(synthetic_small, pd.DataFrame)

    def test_has_all_required_columns(self, synthetic_small: pd.DataFrame) -> None:
        for col in ALL_COLS:
            assert col in synthetic_small.columns, f"Missing column: {col}"

    def test_exactly_15_feature_columns(self) -> None:
        assert len(FEATURE_COLS) == 15

    def test_event_is_binary(self, synthetic_small: pd.DataFrame) -> None:
        assert synthetic_small["event"].isin([0, 1]).all()

    def test_positive_event_rate(self, synthetic_small: pd.DataFrame) -> None:
        rate = synthetic_small["event"].mean()
        assert 0 < rate < 1, f"Unexpected event rate: {rate:.2%}"

    def test_age_at_exit_strictly_greater_than_baseline(
        self, synthetic_small: pd.DataFrame
    ) -> None:
        assert (synthetic_small["age_at_exit"] > synthetic_small["age_at_baseline"]).all()

    def test_sex_values_are_binary(self, synthetic_small: pd.DataFrame) -> None:
        assert synthetic_small["sex"].isin([0, 1]).all()

    def test_age_at_baseline_in_plausible_range(self, synthetic_small: pd.DataFrame) -> None:
        assert synthetic_small["age_at_baseline"].between(18, 100).all()

    def test_feature_dtypes_are_float(self, synthetic_small: pd.DataFrame) -> None:
        for col in FEATURE_COLS:
            assert synthetic_small[col].dtype == "float64", f"{col} has wrong dtype"

    def test_raises_on_missing_column(
        self, synthetic_small_path: Path, tmp_path: Path
    ) -> None:
        df = pd.read_csv(synthetic_small_path).drop(columns=["cholesterol"])
        bad_path = tmp_path / "missing_col.csv"
        df.to_csv(bad_path, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            load_raw(bad_path)

    def test_raises_on_exit_age_le_baseline(
        self, synthetic_small_path: Path, tmp_path: Path
    ) -> None:
        df = pd.read_csv(synthetic_small_path)
        df.loc[0, "age_at_exit"] = df.loc[0, "age_at_baseline"] - 1.0
        bad_path = tmp_path / "bad_times.csv"
        df.to_csv(bad_path, index=False)
        with pytest.raises(ValueError, match="left-truncation"):
            load_raw(bad_path)

    def test_raises_on_non_binary_event(
        self, synthetic_small_path: Path, tmp_path: Path
    ) -> None:
        df = pd.read_csv(synthetic_small_path)
        df.loc[0, "event"] = 2
        bad_path = tmp_path / "bad_event.csv"
        df.to_csv(bad_path, index=False)
        with pytest.raises(ValueError, match="event column"):
            load_raw(bad_path)

    def test_some_features_have_missing_values(self, synthetic_small: pd.DataFrame) -> None:
        # crp, iron, ggt have synthetic missingness by design
        for col in ("crp", "iron", "ggt"):
            assert synthetic_small[col].isna().any(), f"{col} should have missing values"

    def test_meta_cols_have_no_missing_values(self, synthetic_small: pd.DataFrame) -> None:
        for col in META_COLS + SURVIVAL_COLS:
            assert not synthetic_small[col].isna().any(), f"{col} should not have NaNs"
