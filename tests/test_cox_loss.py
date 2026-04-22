"""Tests for the Cox partial likelihood with Efron tie correction.

CLAUDE.md requirement:
  - Cox partial likelihood with and without left-truncation, tested against
    lifelines on synthetic data with known ground truth.
  - Risk set construction under left-truncation: at least one test with a
    constructed cohort where the correct risk set is computed by hand.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch
from lifelines import CoxPHFitter

from amoris_bioage.models.cox_loss import cox_partial_likelihood_efron


# ---------------------------------------------------------------------------
# Hand-computed cohort tests
# ---------------------------------------------------------------------------


class TestRiskSetAndEfronHandComputed:
    """Verify risk set membership and Efron correction against manual calculation."""

    def _make_batch(
        self,
        entries: list[float],
        exits: list[float],
        events: list[int],
        log_hazards: list[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(log_hazards, dtype=torch.float64),
            torch.tensor(exits, dtype=torch.float64),
            torch.tensor(events, dtype=torch.float64),
            torch.tensor(entries, dtype=torch.float64),
        )

    def test_single_event_no_ties(self) -> None:
        """One event, one censored; risk set = both; log PL = η_event - log(S_R)."""
        # At t=50: R = {A (event), B (censored)}; d=1.
        # log PL = 0.0 - log(exp(0.0) + exp(0.0)) = -log(2)
        log_hz, times, events, entries = self._make_batch(
            entries=[30.0, 30.0],
            exits=[50.0, 60.0],
            events=[1, 0],
            log_hazards=[0.0, 0.0],
        )
        loss = cox_partial_likelihood_efron(log_hz, times, events, entries)
        expected = math.log(2)  # negative mean log PL = log(2) / 1 event
        assert abs(loss.item() - expected) < 1e-10

    def test_two_tied_events_efron_correction(self) -> None:
        """Two tied events at t=50; Efron denominator must be (S_R)(S_R - S_D/2).

        Cohort:
          A: entry=30, exit=50, event=1, η=0
          B: entry=35, exit=50, event=1, η=0
          C: entry=40, exit=60, event=0, η=0
          D: entry=45, exit=50, event=0, η=0

        Risk set at t=50: all four (entry < 50 ≤ exit for every individual).
        d=2, S_R = 4*exp(0)=4, S_D = 2*exp(0)=2.

        Efron log-denominator = log(S_R - 0/2*S_D) + log(S_R - 1/2*S_D)
                               = log(4) + log(3)

        Numerator = η_A + η_B = 0.

        Total log PL = 0 - (log 4 + log 3) = -log 12.
        Negative mean per event = log(12) / 2.
        """
        log_hz, times, events, entries = self._make_batch(
            entries=[30.0, 35.0, 40.0, 45.0],
            exits=[50.0, 50.0, 60.0, 50.0],
            events=[1, 1, 0, 0],
            log_hazards=[0.0, 0.0, 0.0, 0.0],
        )
        loss = cox_partial_likelihood_efron(log_hz, times, events, entries)
        expected = math.log(12) / 2
        assert abs(loss.item() - expected) < 1e-10

    def test_left_truncation_removes_late_entrant_from_risk_set(self) -> None:
        """Individual who enters AFTER an event time must not be in the risk set.

        Cohort:
          A: entry=30, exit=50, event=1, η=0   ← event at t=50
          B: entry=55, exit=70, event=0, η=0   ← enters after t=50; NOT in risk set
          C: entry=40, exit=60, event=0, η=0   ← in risk set at t=50

        At t=50: R = {A, C} (B has entry 55 > 50).
        log PL = 0 - log(exp(0) + exp(0)) = -log(2).
        Negative mean per event = log(2) / 1.
        """
        log_hz, times, events, entries = self._make_batch(
            entries=[30.0, 55.0, 40.0],
            exits=[50.0, 70.0, 60.0],
            events=[1, 0, 0],
            log_hazards=[0.0, 0.0, 0.0],
        )
        loss = cox_partial_likelihood_efron(log_hz, times, events, entries)
        expected = math.log(2)
        assert abs(loss.item() - expected) < 1e-10

    def test_without_left_truncation_includes_late_entrant(self) -> None:
        """Control: same data but late entrant set to entry=0 IS included.

        All three individuals in risk set at t=50; log PL = -log(3).
        """
        log_hz, times, events, entries = self._make_batch(
            entries=[30.0, 0.0, 40.0],  # B now enters at 0
            exits=[50.0, 70.0, 60.0],
            events=[1, 0, 0],
            log_hazards=[0.0, 0.0, 0.0],
        )
        loss = cox_partial_likelihood_efron(log_hz, times, events, entries)
        expected = math.log(3)
        assert abs(loss.item() - expected) < 1e-10

    def test_left_truncation_changes_risk_set_value(self) -> None:
        """Left-truncation materially changes the loss value vs no truncation."""
        # With left-truncation: risk set at t=50 excludes late entrant → loss = log(2)
        # Without: risk set includes late entrant → loss = log(3)
        log_hz = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        times = torch.tensor([50.0, 70.0, 60.0], dtype=torch.float64)
        events = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

        entries_truncated = torch.tensor([30.0, 55.0, 40.0], dtype=torch.float64)
        entries_no_trunc = torch.tensor([30.0, 0.0, 40.0], dtype=torch.float64)

        loss_trunc = cox_partial_likelihood_efron(log_hz, times, events, entries_truncated)
        loss_no_trunc = cox_partial_likelihood_efron(log_hz, times, events, entries_no_trunc)

        # Smaller risk set (2 vs 3) → smaller denominator → each death is "more expected"
        # → higher log-PL → lower negative loss.  log(2) < log(3).
        assert loss_trunc.item() < loss_no_trunc.item()

    def test_non_zero_log_hazard_changes_loss(self) -> None:
        """Higher η for the event individual should reduce the loss."""
        # Risk set = {A, B}; event at A.
        # η_A = 1.0, η_B = 0.0
        # log PL = 1.0 - log(exp(1) + exp(0)) = 1 - log(e + 1)
        log_hz = torch.tensor([1.0, 0.0], dtype=torch.float64)
        times = torch.tensor([50.0, 60.0], dtype=torch.float64)
        events = torch.tensor([1.0, 0.0], dtype=torch.float64)
        entries = torch.tensor([30.0, 30.0], dtype=torch.float64)

        loss = cox_partial_likelihood_efron(log_hz, times, events, entries)
        expected = -(1.0 - math.log(math.e + 1))  # negated
        assert abs(loss.item() - expected) < 1e-10

    def test_raises_on_no_events(self) -> None:
        log_hz = torch.tensor([0.0, 0.0])
        times = torch.tensor([50.0, 60.0])
        events = torch.tensor([0.0, 0.0])
        entries = torch.tensor([30.0, 30.0])
        with pytest.raises(ValueError, match="no events"):
            cox_partial_likelihood_efron(log_hz, times, events, entries)

    def test_gradients_flow_through_log_hazard(self) -> None:
        log_hz = torch.tensor([0.5, -0.3, 0.1], dtype=torch.float64, requires_grad=True)
        times = torch.tensor([50.0, 60.0, 55.0], dtype=torch.float64)
        events = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
        entries = torch.tensor([30.0, 30.0, 30.0], dtype=torch.float64)

        loss = cox_partial_likelihood_efron(log_hz, times, events, entries)
        loss.backward()

        assert log_hz.grad is not None
        assert not torch.isnan(log_hz.grad).any()


# ---------------------------------------------------------------------------
# Comparison with lifelines on no-tie data
# ---------------------------------------------------------------------------


class TestAgainstLifelines:
    """Verify that my partial likelihood matches lifelines on tie-free data.

    When all event times are unique (d=1 everywhere), Efron and Breslow
    corrections are identical, so any correct implementation must agree.
    """

    @pytest.fixture(scope="class")
    def no_tie_dataset(self) -> dict:
        rng = np.random.default_rng(7)
        n = 80

        X1 = rng.standard_normal(n)
        X2 = rng.standard_normal(n)
        age_at_baseline = rng.uniform(40.0, 65.0, size=n)

        # Generate unique continuous follow-up times from Exponential.
        raw_times = rng.exponential(scale=12.0, size=n)
        # Make unique to avoid ties.
        raw_times = raw_times + np.arange(n) * 1e-4
        age_at_exit = age_at_baseline + raw_times

        # Censor at 20 years of follow-up.
        censor_time = age_at_baseline + 20.0
        event_mask = age_at_exit <= censor_time
        age_at_exit = np.minimum(age_at_exit, censor_time)
        event = event_mask.astype(int)

        # Confirm no ties among event times.
        event_ages = age_at_exit[event == 1]
        assert len(event_ages) == len(np.unique(np.round(event_ages, 8)))

        df = pd.DataFrame(
            {
                "age_at_exit": age_at_exit,
                "event": event,
                "age_at_baseline": age_at_baseline,
                "X1": X1,
                "X2": X2,
            }
        )
        return {"df": df, "n": n}

    def test_log_likelihood_matches_lifelines(self, no_tie_dataset: dict) -> None:
        df = no_tie_dataset["df"]

        cph = CoxPHFitter()
        cph.fit(
            df,
            duration_col="age_at_exit",
            event_col="event",
            entry_col="age_at_baseline",
        )
        ll_lifelines = float(cph.log_likelihood_)

        log_hz = torch.tensor(
            cph.predict_log_partial_hazard(df).values, dtype=torch.float64
        )
        t = torch.tensor(df["age_at_exit"].values, dtype=torch.float64)
        e = torch.tensor(df["event"].values, dtype=torch.float64)
        s = torch.tensor(df["age_at_baseline"].values, dtype=torch.float64)

        n_events = int(df["event"].sum())
        my_loss = cox_partial_likelihood_efron(log_hz, t, e, s)
        my_total_ll = -my_loss.item() * n_events

        assert abs(my_total_ll - ll_lifelines) < 0.05, (
            f"Total log-likelihood: mine={my_total_ll:.6f}, lifelines={ll_lifelines:.6f}"
        )

    def test_loss_decreases_toward_fitted_betas(self, no_tie_dataset: dict) -> None:
        """Loss at the MLE (lifelines betas) must be ≤ loss at zero betas."""
        df = no_tie_dataset["df"]

        cph = CoxPHFitter()
        cph.fit(
            df,
            duration_col="age_at_exit",
            event_col="event",
            entry_col="age_at_baseline",
        )

        t = torch.tensor(df["age_at_exit"].values, dtype=torch.float64)
        e = torch.tensor(df["event"].values, dtype=torch.float64)
        s = torch.tensor(df["age_at_baseline"].values, dtype=torch.float64)

        log_hz_mle = torch.tensor(
            cph.predict_log_partial_hazard(df).values, dtype=torch.float64
        )
        log_hz_zero = torch.zeros(len(df), dtype=torch.float64)

        loss_mle = cox_partial_likelihood_efron(log_hz_mle, t, e, s)
        loss_zero = cox_partial_likelihood_efron(log_hz_zero, t, e, s)

        assert loss_mle.item() <= loss_zero.item() + 1e-6, (
            f"MLE loss ({loss_mle.item():.4f}) must be ≤ zero-β loss ({loss_zero.item():.4f})"
        )
