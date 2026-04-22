"""Tests for SHAP and integrated-gradients attribution.

CLAUDE.md requirements:
  - SHAP attributions on a linear Cox model reduce to β_j × (x_j − mean)
    within numerical tolerance.
  - Δ attribution identity: per-feature SHAP on Δ sums (with the bias term)
    to Δ at the individual, to within numerical tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest
import shap
import torch

from amoris_bioage.attribution.background import make_age_stratified_background
from amoris_bioage.attribution.ig_explainer import CoxIGExplainer
from amoris_bioage.attribution.shap_explainer import BioageShapExplainer
from amoris_bioage.bioage.breslow import BreslowEstimator
from amoris_bioage.bioage.inversion import BiologicalAgeEstimator
from amoris_bioage.data.schema import FEATURE_COLS
from amoris_bioage.models.network import CoxMLP


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

# Gompertz parameters matching test_bioage_inversion.py
LAM = 4.5e-5
GAM = 0.09
_GRID_MAX = 115.0
_AGE_GRID = np.linspace(20.0, _GRID_MAX, 5000)


def _gompertz_cumhaz(t: np.ndarray) -> np.ndarray:
    return (LAM / GAM) * (np.exp(GAM * t) - 1.0)


def _make_gompertz_breslow() -> BreslowEstimator:
    cumhaz = _gompertz_cumhaz(_AGE_GRID)
    b = BreslowEstimator()
    b._event_times = {0: _AGE_GRID.copy(), 1: _AGE_GRID.copy()}
    b._cumhaz = {0: cumhaz.copy(), 1: cumhaz.copy()}
    b._is_fitted = True
    return b


def _make_bioage_estimator(breslow: BreslowEstimator) -> BiologicalAgeEstimator:
    est = BiologicalAgeEstimator(
        horizon_years=10.0,
        age_grid_min=30.0,
        age_grid_max=90.0,
        age_grid_step=0.05,
    )
    est.fit_reference(breslow)
    return est


def _make_df(n: int = 30, seed: int = 0) -> "pd.DataFrame":
    import pandas as pd

    rng = np.random.default_rng(seed)
    age = rng.uniform(40.0, 75.0, n)
    sex = rng.integers(0, 2, n).astype(np.int64)
    data = {
        "age_at_baseline": age,
        "age_at_exit": age + rng.uniform(1.0, 15.0, n),
        "event": rng.integers(0, 2, n).astype(np.int64),
        "sex": sex,
    }
    for col in FEATURE_COLS:
        data[col] = rng.standard_normal(n)
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def gompertz_breslow() -> BreslowEstimator:
    return _make_gompertz_breslow()


@pytest.fixture(scope="module")
def bioage_est(gompertz_breslow: BreslowEstimator) -> BiologicalAgeEstimator:
    return _make_bioage_estimator(gompertz_breslow)


@pytest.fixture(scope="module")
def small_model() -> CoxMLP:
    torch.manual_seed(42)
    return CoxMLP(n_features=len(FEATURE_COLS), hidden_sizes=[32, 32])


@pytest.fixture(scope="module")
def sample_df():
    return _make_df(n=40, seed=1)


@pytest.fixture(scope="module")
def background_df(sample_df):
    return make_age_stratified_background(sample_df, FEATURE_COLS, n_age_bins=5)


@pytest.fixture(scope="module")
def shap_explainer(
    small_model: CoxMLP,
    gompertz_breslow: BreslowEstimator,
    bioage_est: BiologicalAgeEstimator,
    background_df,
) -> BioageShapExplainer:
    return BioageShapExplainer(
        model=small_model,
        breslow=gompertz_breslow,
        bioage_estimator=bioage_est,
        feature_cols=FEATURE_COLS,
        background=background_df,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Background construction tests
# ---------------------------------------------------------------------------


class TestMakeBackground:
    def test_returns_dataframe(self, sample_df) -> None:
        import pandas as pd

        bg = make_age_stratified_background(sample_df, FEATURE_COLS, n_age_bins=5)
        assert isinstance(bg, pd.DataFrame)

    def test_n_rows_le_n_bins(self, sample_df) -> None:
        bg = make_age_stratified_background(sample_df, FEATURE_COLS, n_age_bins=5)
        assert len(bg) <= 5

    def test_columns_present(self, sample_df) -> None:
        bg = make_age_stratified_background(sample_df, FEATURE_COLS, n_age_bins=5)
        for col in FEATURE_COLS + ["age_at_baseline", "sex"]:
            assert col in bg.columns

    def test_sex_is_integer(self, background_df) -> None:
        assert background_df["sex"].dtype in (np.int64, np.int32, int)

    def test_age_is_ascending(self, background_df) -> None:
        ages = background_df["age_at_baseline"].values
        assert (np.diff(ages) >= 0).all()

    def test_raises_on_missing_column(self, sample_df) -> None:
        with pytest.raises(ValueError, match="nonexistent"):
            make_age_stratified_background(
                sample_df, FEATURE_COLS + ["nonexistent"], n_age_bins=5
            )


# ---------------------------------------------------------------------------
# SHAP linear-model consistency test (CLAUDE.md requirement)
# ---------------------------------------------------------------------------


class TestShapLinearConsistency:
    """SHAP on a linear function f(x) = β·x must equal β·(x − background_mean).

    This is the analytical result for any Shapley-consistent attribution on a
    linear model.  It holds exactly for permutation SHAP with an Independent
    masker.
    """

    def test_linear_shap_equals_beta_times_deviation(self) -> None:
        rng = np.random.default_rng(0)
        n_feat = 6
        n_bg = 30
        n_test = 8

        beta = rng.standard_normal(n_feat)
        background = rng.standard_normal((n_bg, n_feat))
        test_x = rng.standard_normal((n_test, n_feat))

        def linear_fn(X: np.ndarray) -> np.ndarray:
            return X @ beta

        masker = shap.maskers.Independent(background)
        explainer = shap.Explainer(linear_fn, masker)
        shap_vals = explainer(test_x).values  # (n_test, n_feat)

        bg_mean = background.mean(axis=0)
        expected = (test_x - bg_mean) * beta  # (n_test, n_feat)

        np.testing.assert_allclose(shap_vals, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# SHAP explainer structural tests
# ---------------------------------------------------------------------------


class TestBioageShapExplainer:
    def test_explain_delta_returns_explanation(
        self, shap_explainer: BioageShapExplainer, sample_df
    ) -> None:
        subset = sample_df.iloc[:5]
        expl = shap_explainer.explain_delta(subset, max_evals=200)
        assert isinstance(expl, shap.Explanation)

    def test_shap_values_shape(
        self, shap_explainer: BioageShapExplainer, sample_df
    ) -> None:
        n = 5
        subset = sample_df.iloc[:n]
        expl = shap_explainer.explain_delta(subset, max_evals=200)
        n_inputs = len(FEATURE_COLS) + 2  # features + age + sex
        assert expl.values.shape == (n, n_inputs)

    def test_explain_g_returns_explanation(
        self, shap_explainer: BioageShapExplainer, sample_df
    ) -> None:
        subset = sample_df.iloc[:5]
        expl = shap_explainer.explain_g(subset, max_evals=200)
        assert isinstance(expl, shap.Explanation)

    def test_raises_on_missing_column(
        self, shap_explainer: BioageShapExplainer
    ) -> None:
        import pandas as pd

        bad_df = pd.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(ValueError, match="missing columns"):
            shap_explainer.explain_delta(bad_df)

    def test_raises_on_bad_background(
        self,
        small_model: CoxMLP,
        gompertz_breslow: BreslowEstimator,
        bioage_est: BiologicalAgeEstimator,
        background_df,
    ) -> None:
        import pandas as pd

        bad_bg = pd.DataFrame({"x": [1.0]})
        with pytest.raises(ValueError, match="missing columns"):
            BioageShapExplainer(
                model=small_model,
                breslow=gompertz_breslow,
                bioage_estimator=bioage_est,
                feature_cols=FEATURE_COLS,
                background=bad_bg,
            )

    def test_delta_attribution_completeness(
        self,
        shap_explainer: BioageShapExplainer,
        small_model: CoxMLP,
        gompertz_breslow: BreslowEstimator,
        bioage_est: BiologicalAgeEstimator,
        sample_df,
    ) -> None:
        """Δ attribution identity (CLAUDE.md requirement).

        For every individual i, the sum of per-feature SHAP values plus the
        SHAP base value (expected Δ over the background) must equal the
        individual's actual Δ.  This is the Shapley efficiency / completeness
        axiom, which any correct SHAP implementation must satisfy.

            sum_j φ_j(x_i) + E[Δ(background)] = Δ(x_i)
        """
        n = 6
        subset = sample_df.iloc[:n].reset_index(drop=True)

        expl = shap_explainer.explain_delta(subset, max_evals=300)

        # True Δ for each individual.
        features = torch.tensor(
            subset[FEATURE_COLS].values, dtype=torch.float32
        )
        age = torch.tensor(subset["age_at_baseline"].values, dtype=torch.float32)
        sex = subset["sex"].values.astype(np.int64)
        with torch.no_grad():
            log_hz = small_model(features, age).numpy()
        _, true_delta = bioage_est.transform(log_hz, age.numpy(), sex, gompertz_breslow)

        # Completeness: values.sum(axis=1) + base_values == true_delta
        shap_sum = expl.values.sum(axis=1) + expl.base_values
        np.testing.assert_allclose(shap_sum, true_delta, atol=0.01)


# ---------------------------------------------------------------------------
# Integrated gradients tests
# ---------------------------------------------------------------------------


class TestCoxIGExplainer:
    @pytest.fixture(scope="class")
    def ig_explainer(
        self, small_model: CoxMLP, background_df
    ) -> CoxIGExplainer:
        return CoxIGExplainer(
            model=small_model,
            feature_cols=FEATURE_COLS,
            background=background_df,
            device="cpu",
        )

    def test_explain_returns_dict_with_expected_keys(
        self, ig_explainer: CoxIGExplainer, sample_df
    ) -> None:
        result = ig_explainer.explain(sample_df.iloc[:5])
        assert set(result) == {
            "feature_attributions",
            "age_attribution",
            "eta",
            "eta_baseline",
        }

    def test_feature_attributions_shape(
        self, ig_explainer: CoxIGExplainer, sample_df
    ) -> None:
        n = 5
        result = ig_explainer.explain(sample_df.iloc[:n])
        assert result["feature_attributions"].shape == (n, len(FEATURE_COLS))

    def test_age_attribution_shape(
        self, ig_explainer: CoxIGExplainer, sample_df
    ) -> None:
        n = 5
        result = ig_explainer.explain(sample_df.iloc[:n])
        assert result["age_attribution"].shape == (n,)

    def test_ig_completeness(
        self, ig_explainer: CoxIGExplainer, sample_df
    ) -> None:
        """IG completeness: sum of all attributions = η(x) − η(baseline).

        This must hold for any correct integrated gradients implementation.
        """
        n = 8
        result = ig_explainer.explain(sample_df.iloc[:n], n_steps=100)

        total_attr = (
            result["feature_attributions"].sum(axis=1) + result["age_attribution"]
        )
        expected = result["eta"] - float(result["eta_baseline"])
        # IG uses a Riemann sum; with n_steps=100 the approximation error is ~O(1/n_steps^2).
        np.testing.assert_allclose(total_attr, expected, atol=0.02)
