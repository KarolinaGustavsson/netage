"""Tests for validation metrics: C-index, calibration, and incremental LRT.

CLAUDE.md requirement:
  - C-index reported on validation and test splits.
  - Incremental C-index and LRT of Δ against a Cox model containing only t.
  - Calibration: predicted vs observed 10-year mortality by decile.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.stats

from amoris_bioage.validation.calibration import calibration_by_decile
from amoris_bioage.validation.concordance import compute_cindex
from amoris_bioage.validation.incremental import incremental_cindex_lrt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_survival_data(n: int = 300, seed: int = 0) -> dict:
    """Synthetic survival data with a known log-hazard signal."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(40.0, 70.0, n)
    log_hz = rng.standard_normal(n)
    # Exponential survival time with hazard exp(log_hz); left-truncation ignored.
    true_surv = rng.exponential(1.0 / np.exp(log_hz))
    censor = rng.uniform(5.0, 20.0, n)
    event = (true_surv <= censor).astype(np.int64)
    event_times = age + np.minimum(true_surv, censor)
    return {
        "log_hazard": log_hz,
        "event_times": event_times,
        "events": event,
        "age": age,
        "follow_up": np.minimum(true_surv, censor),
    }


# ---------------------------------------------------------------------------
# compute_cindex tests
# ---------------------------------------------------------------------------


class TestComputeCindex:
    def test_returns_float(self) -> None:
        d = _make_survival_data(100)
        c = compute_cindex(d["log_hazard"], d["event_times"], d["events"])
        assert isinstance(c, float)

    def test_in_zero_one(self) -> None:
        d = _make_survival_data(100)
        c = compute_cindex(d["log_hazard"], d["event_times"], d["events"])
        assert 0.0 <= c <= 1.0

    def test_perfect_concordance(self) -> None:
        """Log-hazard = -event_time means higher risk for shorter times → C ≈ 1."""
        n = 200
        event_times = np.linspace(1.0, 10.0, n)
        events = np.ones(n, dtype=np.int64)
        # Higher risk for shorter event times.
        log_hazard = -event_times
        c = compute_cindex(log_hazard, event_times, events)
        assert c > 0.95

    def test_anti_concordance_is_near_zero(self) -> None:
        """Reversed predictions → C ≈ 0 (complement of perfect)."""
        n = 200
        event_times = np.linspace(1.0, 10.0, n)
        events = np.ones(n, dtype=np.int64)
        # Lower risk (more negative) for shorter times → anti-concordant.
        log_hazard = event_times
        c = compute_cindex(log_hazard, event_times, events)
        assert c < 0.05

    def test_random_predictions_near_half(self) -> None:
        """Uninformative predictions → C ≈ 0.5."""
        rng = np.random.default_rng(42)
        n = 500
        event_times = rng.uniform(1.0, 20.0, n)
        events = np.ones(n, dtype=np.int64)
        log_hazard = rng.standard_normal(n)  # independent of event times
        c = compute_cindex(log_hazard, event_times, events)
        # Large sample → C close to 0.5; allow ±0.07.
        assert abs(c - 0.5) < 0.07

    def test_informed_predictor_above_half(self) -> None:
        """A predictor correlated with mortality should give C > 0.55.

        Uses pure follow-up time (no age offset) so the log-hazard signal is
        not diluted by the large age-at-baseline variation.
        """
        rng = np.random.default_rng(1)
        n = 500
        log_hz = rng.standard_normal(n)
        true_surv = rng.exponential(1.0 / np.exp(log_hz))
        censor = rng.uniform(2.0, 5.0, n)
        event = (true_surv <= censor).astype(np.int64)
        event_times = np.minimum(true_surv, censor)
        c = compute_cindex(log_hz, event_times, event)
        assert c > 0.55


# ---------------------------------------------------------------------------
# calibration_by_decile tests
# ---------------------------------------------------------------------------


class TestCalibrationByDecile:
    @pytest.fixture(scope="class")
    def calibration_df(self):
        rng = np.random.default_rng(7)
        n = 500
        predicted = rng.uniform(0.05, 0.50, n)
        follow_up = rng.uniform(5.0, 20.0, n)
        events = rng.integers(0, 2, n).astype(np.int64)
        return calibration_by_decile(predicted, follow_up, events, horizon=10.0, n_bins=10)

    def test_returns_dataframe(self, calibration_df) -> None:
        import pandas as pd

        assert isinstance(calibration_df, pd.DataFrame)

    def test_has_expected_columns(self, calibration_df) -> None:
        assert set(calibration_df.columns) == {
            "bin",
            "mean_predicted",
            "observed_mortality",
            "n",
            "n_events",
        }

    def test_n_bins(self, calibration_df) -> None:
        # Should produce at most n_bins rows (duplicates='drop' can merge bins).
        assert len(calibration_df) <= 10

    def test_n_sums_to_total(self, calibration_df) -> None:
        assert calibration_df["n"].sum() == 500

    def test_predicted_in_zero_one(self, calibration_df) -> None:
        assert (calibration_df["mean_predicted"] >= 0.0).all()
        assert (calibration_df["mean_predicted"] <= 1.0).all()

    def test_observed_in_zero_one(self, calibration_df) -> None:
        assert (calibration_df["observed_mortality"] >= 0.0).all()
        assert (calibration_df["observed_mortality"] <= 1.0).all()

    def test_n_events_nonneg(self, calibration_df) -> None:
        assert (calibration_df["n_events"] >= 0).all()

    def test_n_events_le_n(self, calibration_df) -> None:
        assert (calibration_df["n_events"] <= calibration_df["n"]).all()

    def test_higher_predicted_higher_observed(self) -> None:
        """Well-calibrated data: bins with higher mean_predicted should have
        higher observed mortality (monotone trend across bins)."""
        rng = np.random.default_rng(17)
        n = 2000
        # True 10-year mortality proportional to predicted.
        predicted = rng.uniform(0.05, 0.60, n)
        # Simulate events: each individual dies with probability = predicted.
        p_die = predicted
        die = rng.uniform(size=n) < p_die
        # Uniform follow-up; events happen uniformly within 10 years.
        follow_up = np.where(die, rng.uniform(0.1, 10.0, n), rng.uniform(10.0, 20.0, n))
        events = die.astype(np.int64)

        df = calibration_by_decile(predicted, follow_up, events, horizon=10.0, n_bins=5)
        # Correlation between mean predicted and observed should be positive.
        corr = float(
            np.corrcoef(df["mean_predicted"].values, df["observed_mortality"].values)[0, 1]
        )
        assert corr > 0.8, f"Expected positive calibration correlation, got {corr:.3f}"

    def test_custom_n_bins(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        pred = rng.uniform(0, 1, n)
        ft = rng.uniform(1, 15, n)
        ev = rng.integers(0, 2, n).astype(np.int64)
        df = calibration_by_decile(pred, ft, ev, n_bins=5)
        assert len(df) <= 5


# ---------------------------------------------------------------------------
# incremental_cindex_lrt tests
# ---------------------------------------------------------------------------


class TestIncrementalCindexLrt:
    @pytest.fixture(scope="class")
    def result_with_informative_delta(self) -> dict:
        """Generate data where Δ alone drives mortality — age has no survival signal.

        We pass follow-up times (not attained ages) as event_times so the null
        Cox(age) model has C ≈ 0.5, while the full Cox(age, Δ) model captures
        the true log-hazard through Δ and has C >> 0.5.  This guarantees that
        delta_c is reliably positive.
        """
        rng = np.random.default_rng(3)
        n = 600
        age = rng.uniform(40.0, 75.0, n)  # noise covariate — uncorrelated with survival
        delta = rng.normal(0.0, 1.5, n)   # carries the mortality signal
        # True survival depends only on delta, not on age.
        true_surv = rng.exponential(1.0 / np.exp(delta - delta.mean()))
        censor = rng.uniform(3.0, 15.0, n)
        event = (true_surv <= censor).astype(np.int64)
        # Use follow-up time as event_times so age is uncorrelated with ordering.
        follow_up = np.minimum(true_surv, censor)

        return incremental_cindex_lrt(delta, age, follow_up, event)

    def test_returns_expected_keys(self, result_with_informative_delta: dict) -> None:
        assert set(result_with_informative_delta) == {
            "c_null",
            "c_full",
            "delta_c",
            "lrt_stat",
            "p_value",
            "ll_null",
            "ll_full",
        }

    def test_c_null_in_range(self, result_with_informative_delta: dict) -> None:
        assert 0.0 <= result_with_informative_delta["c_null"] <= 1.0

    def test_c_full_in_range(self, result_with_informative_delta: dict) -> None:
        assert 0.0 <= result_with_informative_delta["c_full"] <= 1.0

    def test_lrt_stat_nonneg(self, result_with_informative_delta: dict) -> None:
        assert result_with_informative_delta["lrt_stat"] >= 0.0

    def test_p_value_in_range(self, result_with_informative_delta: dict) -> None:
        assert 0.0 <= result_with_informative_delta["p_value"] <= 1.0

    def test_ll_full_ge_ll_null(self, result_with_informative_delta: dict) -> None:
        """Full model log-likelihood must be ≥ null (more parameters)."""
        assert (
            result_with_informative_delta["ll_full"]
            >= result_with_informative_delta["ll_null"] - 1e-6
        )

    def test_informative_delta_is_significant(
        self, result_with_informative_delta: dict
    ) -> None:
        """An informative Δ must yield a significant LRT (p < 0.05)."""
        assert result_with_informative_delta["p_value"] < 0.05, (
            f"LRT p-value = {result_with_informative_delta['p_value']:.4f}; "
            "expected < 0.05 for informative Δ"
        )

    def test_informative_delta_increases_cindex(
        self, result_with_informative_delta: dict
    ) -> None:
        """Adding an informative Δ must increase the C-index."""
        assert result_with_informative_delta["delta_c"] > 0, (
            f"delta_c = {result_with_informative_delta['delta_c']:.4f}; expected > 0"
        )

    def test_uninformative_delta_not_significant(self) -> None:
        """Random Δ uncorrelated with survival should not be significant."""
        rng = np.random.default_rng(99)
        n = 300
        age = rng.uniform(40.0, 70.0, n)
        delta = rng.standard_normal(n)  # pure noise
        event_times = age + rng.exponential(10.0, n)
        events = rng.integers(0, 2, n).astype(np.int64)

        result = incremental_cindex_lrt(delta, age, event_times, events)
        # p-value should not be small (p > 0.01 for noise predictor at n=300).
        # We use a lenient threshold since this is probabilistic.
        assert result["p_value"] > 0.01 or result["lrt_stat"] < 6.63, (
            "Noise predictor produced a highly significant LRT — check implementation."
        )

    def test_delta_c_equals_c_full_minus_c_null(
        self, result_with_informative_delta: dict
    ) -> None:
        r = result_with_informative_delta
        assert abs(r["delta_c"] - (r["c_full"] - r["c_null"])) < 1e-12
