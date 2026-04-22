"""Tests for the Breslow cumulative baseline hazard estimator.

CLAUDE.md requirement:
  - Breslow baseline hazard: reproduce lifelines output on a shared dataset.
  - Risk set construction under left-truncation correctly excludes
    individuals who have not yet entered.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter

from amoris_bioage.bioage.breslow import BreslowEstimator, _step_interp


# ---------------------------------------------------------------------------
# Step interpolation helper
# ---------------------------------------------------------------------------


class TestStepInterp:
    def test_exact_knot_value(self) -> None:
        knots = np.array([1.0, 2.0, 3.0])
        values = np.array([0.1, 0.3, 0.6])
        result = _step_interp(np.array([2.0]), knots, values)
        assert abs(result[0] - 0.3) < 1e-12

    def test_between_knots_returns_left_value(self) -> None:
        # Right-continuous step: value between knot[0] and knot[1] = values[0]
        knots = np.array([1.0, 3.0])
        values = np.array([0.2, 0.5])
        result = _step_interp(np.array([2.0]), knots, values)
        assert abs(result[0] - 0.2) < 1e-12

    def test_before_first_knot_returns_zero(self) -> None:
        knots = np.array([5.0, 10.0])
        values = np.array([0.1, 0.3])
        result = _step_interp(np.array([3.0]), knots, values)
        assert result[0] == 0.0

    def test_beyond_last_knot_returns_last_value(self) -> None:
        knots = np.array([1.0, 2.0])
        values = np.array([0.4, 0.9])
        result = _step_interp(np.array([100.0]), knots, values)
        assert abs(result[0] - 0.9) < 1e-12


# ---------------------------------------------------------------------------
# BreslowEstimator unit tests
# ---------------------------------------------------------------------------


class TestBreslowEstimator:
    @pytest.fixture(scope="class")
    def fitted_breslow(self) -> BreslowEstimator:
        rng = np.random.default_rng(11)
        n = 300
        sex = rng.integers(0, 2, size=n)
        entry = rng.uniform(40.0, 60.0, size=n)
        follow_up = rng.exponential(10.0, size=n)
        exit_age = entry + follow_up
        censor = entry + 20.0
        event = (exit_age <= censor).astype(int)
        exit_age = np.minimum(exit_age, censor)
        log_hz = rng.normal(0.0, 0.5, size=n)

        breslow = BreslowEstimator()
        breslow.fit(log_hz, exit_age, event, entry, sex)
        return breslow

    def test_fit_sets_is_fitted(self) -> None:
        rng = np.random.default_rng(0)
        n = 50
        sex = rng.integers(0, 2, size=n)
        entry = rng.uniform(40.0, 60.0, size=n)
        exit_age = entry + rng.exponential(10.0, size=n)
        event = rng.integers(0, 2, size=n)
        log_hz = np.zeros(n)
        b = BreslowEstimator()
        assert not b._is_fitted
        b.fit(log_hz, exit_age, event, entry, sex)
        assert b._is_fitted

    def test_raises_before_fit(self) -> None:
        b = BreslowEstimator()
        with pytest.raises(RuntimeError, match="fitted"):
            b.predict_cumhaz(np.array([50.0]), np.array([0]))

    def test_cumhaz_is_non_decreasing(self, fitted_breslow: BreslowEstimator) -> None:
        ages = np.linspace(40.0, 80.0, 100)
        for s in [0, 1]:
            sex = np.full(len(ages), s, dtype=np.int64)
            cumhaz = fitted_breslow.predict_cumhaz(ages, sex)
            diffs = np.diff(cumhaz)
            assert (diffs >= -1e-12).all(), f"Cumhaz decreasing for sex={s}"

    def test_cumhaz_starts_at_zero_before_first_event(
        self, fitted_breslow: BreslowEstimator
    ) -> None:
        # Query at a very young age before any events.
        for s in [0, 1]:
            cumhaz = fitted_breslow.predict_cumhaz(np.array([1.0]), np.array([s]))
            assert cumhaz[0] == 0.0, f"Cumhaz not zero at t=1 for sex={s}"

    def test_cumhaz_positive_at_late_age(self, fitted_breslow: BreslowEstimator) -> None:
        for s in [0, 1]:
            cumhaz = fitted_breslow.predict_cumhaz(np.array([100.0]), np.array([s]))
            assert cumhaz[0] > 0.0, f"Cumhaz is zero at t=100 for sex={s}"

    def test_raises_on_missing_sex_stratum(self) -> None:
        # Provide only sex=0 individuals.
        n = 20
        b = BreslowEstimator()
        with pytest.raises(ValueError, match="sex=1"):
            b.fit(
                log_hazard=np.zeros(n),
                event_times=np.linspace(50, 70, n),
                events=np.ones(n, dtype=np.int64),
                entry_times=np.linspace(40, 60, n),
                sex=np.zeros(n, dtype=np.int64),
            )

    def test_left_truncation_raises_cumhaz(self) -> None:
        """Excluding late entrants from the risk set should give larger cumhaz increments.

        Cohort (sex=0 only to avoid stratum issue):
          A: entry=30, exit=50, event=1, η=0  — in risk at t=50
          B: entry=55, exit=70, event=0, η=0  — NOT in risk at t=50 (entry > 50)

        With B excluded: ΔH(50) = 1 / exp(0) = 1.0
        Without B excluded (if we ignored truncation): ΔH(50) = 1 / 2 = 0.5
        """
        b_truncated = BreslowEstimator()
        b_truncated.fit(
            log_hazard=np.array([0.0, 0.0, 0.0, 0.0]),
            event_times=np.array([50.0, 70.0, 50.0, 70.0]),
            events=np.array([1, 0, 1, 0]),
            entry_times=np.array([30.0, 55.0, 30.0, 55.0]),
            sex=np.array([0, 0, 1, 1]),  # need both sexes
        )
        # At sex=0: only A in risk set at t=50 → ΔH = 1/1 = 1
        cumhaz_truncated = b_truncated.predict_cumhaz(np.array([50.0]), np.array([0]))

        b_no_trunc = BreslowEstimator()
        b_no_trunc.fit(
            log_hazard=np.array([0.0, 0.0, 0.0, 0.0]),
            event_times=np.array([50.0, 70.0, 50.0, 70.0]),
            events=np.array([1, 0, 1, 0]),
            entry_times=np.array([30.0, 0.0, 30.0, 0.0]),  # B now enters at 0
            sex=np.array([0, 0, 1, 1]),
        )
        cumhaz_no_trunc = b_no_trunc.predict_cumhaz(np.array([50.0]), np.array([0]))

        # With truncation: only 1 person at risk → ΔH = 1.0
        # Without: 2 people at risk → ΔH = 0.5
        assert cumhaz_truncated[0] > cumhaz_no_trunc[0]


# ---------------------------------------------------------------------------
# Comparison with lifelines
# ---------------------------------------------------------------------------


class TestBreslowVsLifelines:
    """Reproduce lifelines baseline cumulative hazard on a shared dataset.

    lifelines does not apply left-truncation when computing
    ``baseline_cumulative_hazard_`` (it uses the full-cohort risk set even
    when ``entry_col`` is specified for the partial likelihood). We therefore
    test with ``entry_time = 0`` for all individuals so that the left-
    truncated and non-truncated risk sets are identical, making the
    comparison clean and unambiguous.
    """

    @pytest.fixture(scope="class")
    def no_truncation_dataset(self) -> dict:
        """Dataset where all individuals enter at time 0 (no left truncation)."""
        rng = np.random.default_rng(99)
        n = 200  # individuals per sex stratum
        # Exit ages uniformly distributed; entry at 0 for everyone.
        age_at_exit = rng.uniform(50.0, 80.0, size=n * 2)
        event = rng.integers(0, 2, size=n * 2)
        entry = np.zeros(n * 2, dtype=np.float64)
        sex = np.repeat([0, 1], n).astype(np.int64)
        X1 = rng.standard_normal(n * 2)

        df = pd.DataFrame(
            {
                "age_at_exit": age_at_exit,
                "event": event,
                "age_at_baseline": entry,
                "sex": sex,
                "X1": X1,
            }
        )
        return {"df": df, "n_per_sex": n}

    def test_cumhaz_matches_lifelines_at_event_times(
        self, no_truncation_dataset: dict
    ) -> None:
        df = no_truncation_dataset["df"]

        for sex_val in [0, 1]:
            df_s = df[df["sex"] == sex_val].reset_index(drop=True)
            n_s = len(df_s)

            # Drop the constant sex column before fitting; it has zero variance
            # within each stratum and causes lifelines convergence failure.
            df_fit = df_s.drop(columns=["sex"])
            cph = CoxPHFitter()
            cph.fit(
                df_fit,
                duration_col="age_at_exit",
                event_col="event",
                entry_col="age_at_baseline",
            )

            log_hz_ll = cph.predict_log_partial_hazard(df_fit).values.astype(np.float64)
            ll_cumhaz_df = cph.baseline_cumulative_hazard_
            event_times_ll = ll_cumhaz_df.index.values.astype(np.float64)
            cumhaz_ll = ll_cumhaz_df.iloc[:, 0].values.astype(np.float64)

            # Pad with one dummy of the other sex so the estimator accepts both strata.
            other_sex = 1 - sex_val
            log_hz_ext = np.append(log_hz_ll, [0.0])
            exits_ext = np.append(df_s["age_at_exit"].values, [df_s["age_at_exit"].max() + 1.0])
            events_ext = np.append(df_s["event"].values, [0])
            entries_ext = np.append(df_s["age_at_baseline"].values, [0.0])
            sex_ext = np.append(
                np.full(n_s, sex_val, dtype=np.int64), [np.int64(other_sex)]
            )

            breslow = BreslowEstimator()
            breslow.fit(log_hz_ext, exits_ext, events_ext, entries_ext, sex_ext)

            sex_query = np.full(len(event_times_ll), sex_val, dtype=np.int64)
            my_cumhaz = breslow.predict_cumhaz(event_times_ll, sex_query)

            rel_err = np.abs(my_cumhaz - cumhaz_ll) / (cumhaz_ll + 1e-10)
            abs_err = np.abs(my_cumhaz - cumhaz_ll)
            close = (rel_err < 0.02) | (abs_err < 0.005)
            frac_close = close.mean()
            assert frac_close >= 0.95, (
                f"sex={sex_val}: only {frac_close:.1%} of cumhaz values "
                f"within tolerance of lifelines"
            )
