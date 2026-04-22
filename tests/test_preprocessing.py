"""Tests for the Preprocessor (imputation and standardisation)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amoris_bioage.data.preprocessing import Preprocessor
from amoris_bioage.data.schema import FEATURE_COLS


class TestPreprocessorFit:
    def test_fit_returns_self(self, synthetic_small: pd.DataFrame) -> None:
        p = Preprocessor()
        result = p.fit(synthetic_small)
        assert result is p

    def test_is_fitted_after_fit(self, synthetic_small: pd.DataFrame) -> None:
        p = Preprocessor()
        assert not p._is_fitted
        p.fit(synthetic_small)
        assert p._is_fitted

    def test_medians_computed_for_all_features(self, synthetic_small: pd.DataFrame) -> None:
        p = Preprocessor()
        p.fit(synthetic_small)
        for col in FEATURE_COLS:
            assert col in p._medians

    def test_high_missingness_cols_detected(self) -> None:
        # Use a controlled DataFrame so the test is not sensitive to small-sample
        # variance in the stochastic fixtures.
        n = 200
        df = pd.DataFrame(
            {col: [float(i) for i in range(n)] for col in FEATURE_COLS}
        )
        # Force 15% missingness in crp only.
        df.loc[: int(n * 0.15), "crp"] = float("nan")
        p = Preprocessor()
        p.fit(df)
        assert "crp" in p._high_missingness_cols

    def test_low_missingness_cols_not_flagged(self, synthetic_small: pd.DataFrame) -> None:
        p = Preprocessor()
        p.fit(synthetic_small)
        # cholesterol has no synthetic missingness
        assert "cholesterol" not in p._high_missingness_cols


class TestPreprocessorTransform:
    def test_transform_before_fit_raises(self, synthetic_small: pd.DataFrame) -> None:
        p = Preprocessor()
        with pytest.raises(RuntimeError, match="fit"):
            p.transform(synthetic_small)

    def test_returns_dataframe(self, synthetic_small: pd.DataFrame) -> None:
        p = Preprocessor()
        out = p.fit_transform(synthetic_small)
        assert isinstance(out, pd.DataFrame)

    def test_no_missing_values_after_transform(self, synthetic_small: pd.DataFrame) -> None:
        p = Preprocessor()
        out = p.fit_transform(synthetic_small)
        assert out[FEATURE_COLS].isna().sum().sum() == 0

    def test_does_not_modify_input(self, synthetic_small: pd.DataFrame) -> None:
        original = synthetic_small[FEATURE_COLS].copy()
        p = Preprocessor()
        p.fit_transform(synthetic_small)
        pd.testing.assert_frame_equal(synthetic_small[FEATURE_COLS], original)

    def test_training_set_approximately_zero_mean(self, synthetic_small: pd.DataFrame) -> None:
        p = Preprocessor()
        out = p.fit_transform(synthetic_small)
        means = out[FEATURE_COLS].mean()
        # Small deviation from zero is expected due to median-imputed NAs.
        assert (means.abs() < 0.3).all(), f"Features not approximately zero-mean:\n{means}"

    def test_training_set_approximately_unit_std(self, synthetic_small: pd.DataFrame) -> None:
        p = Preprocessor()
        out = p.fit_transform(synthetic_small)
        stds = out[FEATURE_COLS].std()
        assert (stds > 0.85).all() and (stds < 1.15).all(), (
            f"Features not approximately unit std:\n{stds}"
        )

    def test_missing_indicator_column_added(self) -> None:
        # Controlled DataFrame: crp has 15% missing (above threshold), cholesterol has none.
        n = 200
        df = pd.DataFrame(
            {col: [float(i) for i in range(n)] for col in FEATURE_COLS}
        )
        df.loc[: int(n * 0.15), "crp"] = float("nan")
        p = Preprocessor()
        out = p.fit_transform(df)
        assert "crp_missing" in out.columns

    def test_missing_indicator_is_binary(self) -> None:
        n = 200
        df = pd.DataFrame(
            {col: [float(i) for i in range(n)] for col in FEATURE_COLS}
        )
        df.loc[: int(n * 0.15), "crp"] = float("nan")
        p = Preprocessor()
        out = p.fit_transform(df)
        assert out["crp_missing"].isin([0.0, 1.0]).all()

    def test_no_indicator_for_complete_feature(self) -> None:
        n = 200
        df = pd.DataFrame(
            {col: [float(i) for i in range(n)] for col in FEATURE_COLS}
        )
        df.loc[: int(n * 0.15), "crp"] = float("nan")
        p = Preprocessor()
        out = p.fit_transform(df)
        assert "cholesterol_missing" not in out.columns

    def test_transform_uses_training_statistics_not_test(
        self, synthetic_small: pd.DataFrame, synthetic_medium: pd.DataFrame
    ) -> None:
        p = Preprocessor()
        p.fit(synthetic_small)
        out_train = p.transform(synthetic_small)
        # Training set should be ~zero-mean; test set is shifted by training stats.
        train_mean_max = out_train[FEATURE_COLS].mean().abs().max()
        assert train_mean_max < 0.3

    def test_constant_column_does_not_raise(self) -> None:
        df = pd.DataFrame(
            {
                "id": ["A", "B", "C"],
                "sex": [0, 1, 0],
                "age_at_baseline": [50.0, 55.0, 60.0],
                "age_at_exit": [60.0, 65.0, 70.0],
                "event": [1, 0, 1],
                **{col: [1.0, 1.0, 1.0] for col in FEATURE_COLS},
            }
        )
        p = Preprocessor()
        out = p.fit_transform(df)
        assert (out[FEATURE_COLS] == 0.0).all().all()


class TestPreprocessorFitTransform:
    def test_fit_transform_equals_fit_then_transform(
        self, synthetic_small: pd.DataFrame
    ) -> None:
        p1 = Preprocessor()
        out1 = p1.fit_transform(synthetic_small)

        p2 = Preprocessor()
        p2.fit(synthetic_small)
        out2 = p2.transform(synthetic_small)

        pd.testing.assert_frame_equal(out1, out2)
