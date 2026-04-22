"""Biological age definition via 10-year mortality equivalence.

Biological age g(x, t) is the chronological age at which an individual
in the reference population has the same 10-year mortality probability as
individual (x, t) under the fitted Cox model:

    m_indiv(x, t) = 1 − exp(−exp(η(x,t)) · (Ĥ_0(t+Δ) − Ĥ_0(t)))
    m_ref(a)      = 1 − exp(−(Ĥ_0(a+Δ) − Ĥ_0(a)))          [η=0 reference]

    g(x, t) = m_ref⁻¹( m_indiv(x, t) )
    Δ(x, t) = g(x, t) − t

where Ĥ_0 is the sex-stratified Breslow cumulative baseline hazard and
Δ is the mortality prediction horizon (default 10 years).

The reference population uses η = 0, corresponding to the individual at
the average log-hazard in the training cohort. Because the Cox partial
likelihood is translation-invariant, this average is absorbed into Ĥ_0
by the Breslow estimator, so η = 0 is the natural reference.

Sanity checks (must hold before downstream analysis)
-----------------------------------------------------
- Mean Δ across the cohort ≈ 0 (verified by ``sanity_check``).
- Δ correlates positively with 10-year mortality in held-out data.
- Δ is approximately symmetric within each age decile.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
from numpy.typing import NDArray

from amoris_bioage.bioage.breslow import BreslowEstimator

logger = logging.getLogger(__name__)

# Individuals whose predicted mortality falls outside the reference range are
# extrapolated to the boundary; a warning is emitted when this happens.
_BOUNDARY_TOL = 1e-6


class BiologicalAgeEstimator:
    """Compute biological age and age gap from a fitted Breslow estimator.

    Call ``fit_reference`` once after Breslow is estimated on the training
    set, then use ``transform`` on any split.

    Args:
        horizon_years: Mortality prediction horizon Δ (default 10 years).
        age_grid_min: Lower bound of the reference age grid.
        age_grid_max: Upper bound of the reference age grid.
        age_grid_step: Spacing of the reference age grid in years.
    """

    def __init__(
        self,
        horizon_years: float = 10.0,
        age_grid_min: float = 30.0,
        age_grid_max: float = 100.0,
        age_grid_step: float = 0.1,
    ) -> None:
        self.horizon_years = horizon_years
        self.age_grid_min = age_grid_min
        self.age_grid_max = age_grid_max
        self.age_grid_step = age_grid_step

        # Populated by fit_reference().
        # Per-sex: reference age grid and corresponding 10-year mortality.
        self._ref_ages: dict[int, NDArray[np.float64]] = {}
        self._ref_mortality: dict[int, NDArray[np.float64]] = {}
        self._is_fitted: bool = False

    def fit_reference(
        self, breslow: BreslowEstimator
    ) -> "BiologicalAgeEstimator":
        """Precompute the sex-stratified reference mortality mapping.

        m_ref^s(a) = 1 − exp(−(Ĥ_0^s(a + horizon) − Ĥ_0^s(a)))

        for a fine age grid, for each sex stratum s ∈ {0, 1}.

        Args:
            breslow: A fitted ``BreslowEstimator``.

        Returns:
            self
        """
        age_grid = np.arange(
            self.age_grid_min, self.age_grid_max, self.age_grid_step
        )

        for s in [0, 1]:
            sex_col = np.full(len(age_grid), s, dtype=np.int64)
            h0_a = breslow.predict_cumhaz(age_grid, sex_col)
            h0_a_h = breslow.predict_cumhaz(age_grid + self.horizon_years, sex_col)
            cumhaz_diff = h0_a_h - h0_a
            mortality = 1.0 - np.exp(-cumhaz_diff)

            self._ref_ages[s] = age_grid
            self._ref_mortality[s] = mortality
            logger.debug(
                "Reference sex=%d: mortality range [%.4f, %.4f] over ages [%.1f, %.1f]",
                s,
                mortality.min(),
                mortality.max(),
                age_grid[0],
                age_grid[-1],
            )

        self._is_fitted = True
        return self

    def transform(
        self,
        log_hazard: NDArray[np.float64],
        age_at_baseline: NDArray[np.float64],
        sex: NDArray[np.int64],
        breslow: BreslowEstimator,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute biological age g and age gap Δ for a cohort.

        Args:
            log_hazard: (N,) η(x, t) from the trained network.
            age_at_baseline: (N,) chronological age at baseline (years).
            sex: (N,) sex indicator (0 or 1).
            breslow: The same fitted BreslowEstimator used for ``fit_reference``.

        Returns:
            Tuple (biological_age, age_gap), each (N,) array in years.

        Raises:
            RuntimeError: If called before ``fit_reference``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "BiologicalAgeEstimator.fit_reference() must be called before transform()."
            )

        log_hazard = np.asarray(log_hazard, dtype=np.float64)
        age_at_baseline = np.asarray(age_at_baseline, dtype=np.float64)
        sex = np.asarray(sex, dtype=np.int64)

        # Individual 10-year mortality.
        h0_t = breslow.predict_cumhaz(age_at_baseline, sex)
        h0_t_h = breslow.predict_cumhaz(age_at_baseline + self.horizon_years, sex)
        indiv_mortality = 1.0 - np.exp(-np.exp(log_hazard) * (h0_t_h - h0_t))

        biological_age = np.full_like(log_hazard, np.nan)

        for s in [0, 1]:
            mask = sex == s
            if not mask.any():
                continue
            biological_age[mask] = self._invert(
                indiv_mortality[mask], stratum=s
            )

        age_gap = biological_age - age_at_baseline
        return biological_age, age_gap

    def _invert(
        self,
        target_mortality: NDArray[np.float64],
        stratum: int,
    ) -> NDArray[np.float64]:
        """Invert the reference mortality mapping via linear interpolation.

        For each target mortality value, finds the reference age at which
        the reference population has that same 10-year mortality.

        Values outside [min_ref, max_ref] are extrapolated to the boundary
        ages, and a warning is issued.
        """
        ref_ages = self._ref_ages[stratum]
        ref_mortality = self._ref_mortality[stratum]

        # ref_mortality is not guaranteed to be strictly monotone at all
        # points (Breslow is a step function; adjacent grid points may share
        # a value). Use the first occurrence of each mortality level.
        # np.interp requires x (ref_mortality) to be non-decreasing.
        # Since cumhaz is non-decreasing, ref_mortality is non-decreasing.

        min_ref = ref_mortality[0]
        max_ref = ref_mortality[-1]

        n_below = (target_mortality < min_ref - _BOUNDARY_TOL).sum()
        n_above = (target_mortality > max_ref + _BOUNDARY_TOL).sum()
        if n_below > 0 or n_above > 0:
            warnings.warn(
                f"sex={stratum}: {n_below} individuals below and {n_above} above "
                "the reference mortality range; biological age extrapolated to grid boundary. "
                "Consider extending age_grid_min / age_grid_max.",
                UserWarning,
                stacklevel=3,
            )

        # np.interp clamps to boundary values for out-of-range queries.
        return np.interp(target_mortality, ref_mortality, ref_ages)


def sanity_check(
    age_gap: NDArray[np.float64],
    age_at_baseline: NDArray[np.float64],
    tolerance_mean: float = 1.0,
) -> dict[str, float]:
    """Run cohort-level sanity checks on the age gap Δ.

    Checks that must pass before any downstream attribution analysis:
    1. Mean Δ is close to zero (≤ tolerance_mean years).
    2. Δ does not correlate strongly with chronological age at baseline
       within the cohort (|r| < 0.5 is a soft warning threshold).

    Args:
        age_gap: (N,) Δ(x, t) = g(x, t) − t values.
        age_at_baseline: (N,) chronological age at baseline.
        tolerance_mean: Allowed absolute deviation of mean Δ from zero.

    Returns:
        Dict with keys "mean_delta", "std_delta", "corr_delta_age",
        "mean_check_passed", "corr_warning".
    """
    mean_delta = float(np.mean(age_gap))
    std_delta = float(np.std(age_gap))
    corr = float(np.corrcoef(age_gap, age_at_baseline)[0, 1])

    mean_ok = abs(mean_delta) <= tolerance_mean
    corr_warning = abs(corr) >= 0.5

    if not mean_ok:
        logger.warning(
            "SANITY FAIL: mean Δ = %.3f years (tolerance ±%.1f). "
            "Check Breslow estimation and reference mapping.",
            mean_delta,
            tolerance_mean,
        )
    if corr_warning:
        logger.warning(
            "SANITY WARNING: correlation of Δ with chronological age = %.3f. "
            "A large value may indicate miscalibration of the inversion.",
            corr,
        )

    return {
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "corr_delta_age": corr,
        "mean_check_passed": mean_ok,
        "corr_warning": corr_warning,
    }
