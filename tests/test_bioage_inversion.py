"""Tests for biological age inversion and the age gap Δ.

CLAUDE.md requirements:
  - Biological age inversion: on synthetic data where the ground-truth mapping
    is known, recover it within tolerance.
  - Δ attribution identity (indirectly exercised via mean-Δ sanity check).
"""
from __future__ import annotations

import numpy as np
import pytest

from amoris_bioage.bioage.breslow import BreslowEstimator
from amoris_bioage.bioage.inversion import BiologicalAgeEstimator, sanity_check


# ---------------------------------------------------------------------------
# Fixtures — Gompertz ground truth
# ---------------------------------------------------------------------------


def _gompertz_cumhaz(t: np.ndarray, lam: float, gam: float) -> np.ndarray:
    """H_0(t) = (λ/γ)(exp(γt) - 1) for a Gompertz baseline hazard."""
    return (lam / gam) * (np.exp(gam * t) - 1.0)


def _make_breslow_from_gompertz(
    lam: float, gam: float, age_grid: np.ndarray
) -> BreslowEstimator:
    """Construct a BreslowEstimator whose cumhaz matches a Gompertz curve.

    We inject the Gompertz values directly into the estimator's internal
    storage, bypassing the fit loop, to create a ground-truth reference
    for the inversion tests.
    """
    cumhaz = _gompertz_cumhaz(age_grid, lam, gam)
    b = BreslowEstimator()
    # Both strata share the same Gompertz curve for simplicity.
    b._event_times = {0: age_grid.copy(), 1: age_grid.copy()}
    b._cumhaz = {0: cumhaz.copy(), 1: cumhaz.copy()}
    b._is_fitted = True
    return b


# Gompertz parameters calibrated so that 10-year mortality at age 60 ≈ 15 %,
# consistent with Swedish population mortality rates in the AMORIS age range.
LAM = 4.5e-5
GAM = 0.09
# The grid must extend to age_grid_max + horizon (100 + 10 = 110) so that
# predict_cumhaz(a + 10) is never clipped at the boundary.
_BRESLOW_GRID_MAX = 115.0
AGE_GRID = np.linspace(20.0, _BRESLOW_GRID_MAX, 5000)


@pytest.fixture(scope="module")
def gompertz_breslow() -> BreslowEstimator:
    return _make_breslow_from_gompertz(LAM, GAM, AGE_GRID)


@pytest.fixture(scope="module")
def bioage_estimator(gompertz_breslow: BreslowEstimator) -> BiologicalAgeEstimator:
    est = BiologicalAgeEstimator(
        horizon_years=10.0,
        age_grid_min=30.0,
        age_grid_max=90.0,   # keep well below _BRESLOW_GRID_MAX - horizon
        age_grid_step=0.05,
    )
    est.fit_reference(gompertz_breslow)
    return est


# ---------------------------------------------------------------------------
# BiologicalAgeEstimator unit tests
# ---------------------------------------------------------------------------


class TestBiologicalAgeEstimatorSetup:
    def test_raises_before_fit_reference(self, gompertz_breslow: BreslowEstimator) -> None:
        est = BiologicalAgeEstimator()
        with pytest.raises(RuntimeError, match="fit_reference"):
            est.transform(
                np.zeros(5),
                np.full(5, 55.0),
                np.zeros(5, dtype=np.int64),
                gompertz_breslow,
            )

    def test_fit_reference_sets_is_fitted(
        self, gompertz_breslow: BreslowEstimator
    ) -> None:
        est = BiologicalAgeEstimator()
        assert not est._is_fitted
        est.fit_reference(gompertz_breslow)
        assert est._is_fitted

    def test_reference_mortality_is_non_decreasing(
        self, bioage_estimator: BiologicalAgeEstimator
    ) -> None:
        for s in [0, 1]:
            m = bioage_estimator._ref_mortality[s]
            assert (np.diff(m) >= -1e-12).all(), f"Reference mortality decreasing for sex={s}"

    def test_reference_mortality_in_zero_one(
        self, bioage_estimator: BiologicalAgeEstimator
    ) -> None:
        for s in [0, 1]:
            m = bioage_estimator._ref_mortality[s]
            assert (m >= 0.0).all() and (m <= 1.0).all()


# ---------------------------------------------------------------------------
# Inversion accuracy tests
# ---------------------------------------------------------------------------


class TestInversionAccuracy:
    """Verify that g(x, t) is recovered correctly on Gompertz ground truth.

    With a Gompertz baseline h_0(t) = λ exp(γt), the 10-year cumulative
    hazard difference is:
        ΔH(a) = H_0(a+10) - H_0(a) = (λ/γ)(exp(γ(a+10)) - exp(γa))
               = (λ/γ) exp(γa) (exp(10γ) - 1)

    ΔH(a) is strictly increasing in a, so the reference mapping a → mortality
    is strictly monotone and the inversion is well-defined.

    For an individual with η = 0, their individual mortality equals the
    reference mortality at their own age → g = t, Δ = 0.

    For η > 0, the individual has higher risk than the reference → g > t.
    The true biological age satisfies ΔH(g) = exp(η) · ΔH(t), which can
    be solved analytically:
        exp(γg) = exp(η) · exp(γt) + (1 − exp(η)) · C      (Gompertz identity)
    where C = 1 / (exp(10γ) - 1).  For large enough ages and small η we can
    verify numerically.
    """

    def test_eta_zero_gives_bioage_equal_to_chronological_age(
        self,
        bioage_estimator: BiologicalAgeEstimator,
        gompertz_breslow: BreslowEstimator,
    ) -> None:
        ages = np.array([45.0, 50.0, 55.0, 60.0, 65.0])  # all within [30, 90] reference grid
        n = len(ages)
        log_hz = np.zeros(n)
        sex = np.zeros(n, dtype=np.int64)

        g, delta = bioage_estimator.transform(log_hz, ages, sex, gompertz_breslow)

        np.testing.assert_allclose(g, ages, atol=0.5)
        np.testing.assert_allclose(delta, 0.0, atol=0.5)

    def test_positive_eta_gives_positive_age_gap(
        self,
        bioage_estimator: BiologicalAgeEstimator,
        gompertz_breslow: BreslowEstimator,
    ) -> None:
        ages = np.full(5, 55.0)
        log_hz = np.full(5, 0.5)  # higher risk than reference
        sex = np.zeros(5, dtype=np.int64)

        _, delta = bioage_estimator.transform(log_hz, ages, sex, gompertz_breslow)
        assert (delta > 0).all(), "Positive η must give Δ > 0"

    def test_negative_eta_gives_negative_age_gap(
        self,
        bioage_estimator: BiologicalAgeEstimator,
        gompertz_breslow: BreslowEstimator,
    ) -> None:
        ages = np.full(5, 55.0)
        log_hz = np.full(5, -0.5)  # lower risk than reference
        sex = np.zeros(5, dtype=np.int64)

        _, delta = bioage_estimator.transform(log_hz, ages, sex, gompertz_breslow)
        assert (delta < 0).all(), "Negative η must give Δ < 0"

    def test_larger_eta_gives_larger_bioage(
        self,
        bioage_estimator: BiologicalAgeEstimator,
        gompertz_breslow: BreslowEstimator,
    ) -> None:
        ages = np.array([55.0, 55.0])
        log_hz = np.array([0.3, 0.8])
        sex = np.zeros(2, dtype=np.int64)

        g, _ = bioage_estimator.transform(log_hz, ages, sex, gompertz_breslow)
        assert g[1] > g[0], "Higher η must give larger biological age"

    def test_bioage_recovers_known_value_on_gompertz(
        self,
        bioage_estimator: BiologicalAgeEstimator,
        gompertz_breslow: BreslowEstimator,
    ) -> None:
        """Numerical inversion must agree with the analytical solution.

        Analytical: find g such that ΔH_0(g) = exp(η) · ΔH_0(t).
        ΔH_0(a) = (λ/γ) exp(γa) (exp(10γ) - 1)
        → exp(γg) = exp(η) exp(γt)
        → g = t + η/γ   (first-order, valid for small η)

        For η=0.09, γ=0.09, t=55: g ≈ 55 + 1 = 56 years.
        """
        t = 55.0
        eta = 0.09  # equals γ so g ≈ t + 1 analytically
        # Exact: exp(γg) = exp(η) * exp(γt)
        # g = (ln(exp(η) * exp(γt))) / γ = t + η/γ = 55 + 1 = 56
        g_true = t + eta / GAM  # = 56.0

        g, _ = bioage_estimator.transform(
            np.array([eta]),
            np.array([t]),
            np.zeros(1, dtype=np.int64),
            gompertz_breslow,
        )
        assert abs(g[0] - g_true) < 0.3, (
            f"Biological age {g[0]:.3f} differs from analytical {g_true:.3f} by > 0.3 years"
        )

    def test_both_sexes_return_values(
        self,
        bioage_estimator: BiologicalAgeEstimator,
        gompertz_breslow: BreslowEstimator,
    ) -> None:
        ages = np.array([50.0, 50.0, 60.0, 60.0])
        log_hz = np.array([0.0, 0.2, -0.1, 0.1])
        sex = np.array([0, 1, 0, 1], dtype=np.int64)

        g, delta = bioage_estimator.transform(log_hz, ages, sex, gompertz_breslow)

        assert not np.isnan(g).any()
        assert not np.isnan(delta).any()

    def test_output_shapes(
        self,
        bioage_estimator: BiologicalAgeEstimator,
        gompertz_breslow: BreslowEstimator,
    ) -> None:
        n = 20
        g, delta = bioage_estimator.transform(
            np.zeros(n),
            np.full(n, 55.0),
            np.zeros(n, dtype=np.int64),
            gompertz_breslow,
        )
        assert g.shape == (n,)
        assert delta.shape == (n,)


# ---------------------------------------------------------------------------
# Sanity check tests
# ---------------------------------------------------------------------------


class TestSanityCheck:
    def test_mean_delta_zero_for_reference_population(
        self,
        bioage_estimator: BiologicalAgeEstimator,
        gompertz_breslow: BreslowEstimator,
    ) -> None:
        """Cohort at η=0 should have mean Δ ≈ 0 by construction."""
        rng = np.random.default_rng(42)
        n = 500
        ages = rng.uniform(35.0, 80.0, n)  # within reference grid [30, 90]
        log_hz = np.zeros(n)
        sex = rng.integers(0, 2, size=n, dtype=np.int64)

        _, delta = bioage_estimator.transform(log_hz, ages, sex, gompertz_breslow)

        result = sanity_check(delta, ages, tolerance_mean=1.0)
        assert result["mean_check_passed"], (
            f"Mean Δ = {result['mean_delta']:.3f} years; expected ≈ 0"
        )

    def test_sanity_check_returns_expected_keys(
        self,
        bioage_estimator: BiologicalAgeEstimator,
        gompertz_breslow: BreslowEstimator,
    ) -> None:
        ages = np.full(50, 55.0)
        log_hz = np.zeros(50)
        sex = np.zeros(50, dtype=np.int64)
        _, delta = bioage_estimator.transform(log_hz, ages, sex, gompertz_breslow)
        result = sanity_check(delta, ages)
        assert set(result) == {
            "mean_delta",
            "std_delta",
            "corr_delta_age",
            "mean_check_passed",
            "corr_warning",
        }

    def test_positive_delta_correlates_with_mortality_risk(
        self,
        bioage_estimator: BiologicalAgeEstimator,
        gompertz_breslow: BreslowEstimator,
    ) -> None:
        """Individuals with larger Δ must have higher η (mortality risk)."""
        rng = np.random.default_rng(7)
        n = 200
        ages = rng.uniform(40.0, 75.0, n)  # within reference grid
        log_hz = rng.normal(0.0, 0.5, n)
        sex = np.zeros(n, dtype=np.int64)

        _, delta = bioage_estimator.transform(log_hz, ages, sex, gompertz_breslow)

        corr = np.corrcoef(log_hz, delta)[0, 1]
        assert corr > 0.9, (
            f"Δ–η correlation {corr:.3f}; expected > 0.9 for Gompertz ground truth"
        )
