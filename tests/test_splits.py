"""Tests for train/val/test splitting."""
from __future__ import annotations

import pandas as pd
import pytest

from amoris_bioage.data.splits import DEFAULT_RATIOS, DEFAULT_SEED, SplitResult, make_splits


class TestMakeSplits:
    def test_returns_split_result(self, synthetic_medium: pd.DataFrame) -> None:
        result = make_splits(synthetic_medium)
        assert isinstance(result, SplitResult)

    def test_split_sizes_sum_to_total(self, synthetic_medium: pd.DataFrame) -> None:
        result = make_splits(synthetic_medium)
        total = len(synthetic_medium)
        assert len(result.train) + len(result.val) + len(result.test) == total

    def test_approximate_train_ratio(self, synthetic_medium: pd.DataFrame) -> None:
        result = make_splits(synthetic_medium)
        actual = len(result.train) / len(synthetic_medium)
        assert abs(actual - DEFAULT_RATIOS[0]) < 0.02

    def test_approximate_val_ratio(self, synthetic_medium: pd.DataFrame) -> None:
        result = make_splits(synthetic_medium)
        actual = len(result.val) / len(synthetic_medium)
        assert abs(actual - DEFAULT_RATIOS[1]) < 0.02

    def test_approximate_test_ratio(self, synthetic_medium: pd.DataFrame) -> None:
        result = make_splits(synthetic_medium)
        actual = len(result.test) / len(synthetic_medium)
        assert abs(actual - DEFAULT_RATIOS[2]) < 0.02

    def test_no_overlap_train_val(self, synthetic_medium: pd.DataFrame) -> None:
        result = make_splits(synthetic_medium)
        assert set(result.train["id"]).isdisjoint(set(result.val["id"]))

    def test_no_overlap_train_test(self, synthetic_medium: pd.DataFrame) -> None:
        result = make_splits(synthetic_medium)
        assert set(result.train["id"]).isdisjoint(set(result.test["id"]))

    def test_no_overlap_val_test(self, synthetic_medium: pd.DataFrame) -> None:
        result = make_splits(synthetic_medium)
        assert set(result.val["id"]).isdisjoint(set(result.test["id"]))

    def test_all_ids_present(self, synthetic_medium: pd.DataFrame) -> None:
        result = make_splits(synthetic_medium)
        all_ids = (
            set(result.train["id"]) | set(result.val["id"]) | set(result.test["id"])
        )
        assert all_ids == set(synthetic_medium["id"])

    def test_both_sexes_in_train(self, synthetic_medium: pd.DataFrame) -> None:
        result = make_splits(synthetic_medium)
        assert set(result.train["sex"].unique()) == {0, 1}

    def test_both_sexes_in_val(self, synthetic_medium: pd.DataFrame) -> None:
        result = make_splits(synthetic_medium)
        assert set(result.val["sex"].unique()) == {0, 1}

    def test_both_sexes_in_test(self, synthetic_medium: pd.DataFrame) -> None:
        result = make_splits(synthetic_medium)
        assert set(result.test["sex"].unique()) == {0, 1}

    def test_age_range_covered_in_all_splits(self, synthetic_medium: pd.DataFrame) -> None:
        result = make_splits(synthetic_medium)
        overall_min = synthetic_medium["age_at_baseline"].min()
        overall_max = synthetic_medium["age_at_baseline"].max()
        age_span = overall_max - overall_min
        for split_name, split_df in [
            ("train", result.train),
            ("val", result.val),
            ("test", result.test),
        ]:
            span = split_df["age_at_baseline"].max() - split_df["age_at_baseline"].min()
            assert span > 0.5 * age_span, f"{split_name} has a narrow age range"

    def test_reproducible_with_same_seed(self, synthetic_medium: pd.DataFrame) -> None:
        r1 = make_splits(synthetic_medium, seed=99)
        r2 = make_splits(synthetic_medium, seed=99)
        assert list(r1.train["id"]) == list(r2.train["id"])
        assert list(r1.val["id"]) == list(r2.val["id"])
        assert list(r1.test["id"]) == list(r2.test["id"])

    def test_different_seeds_produce_different_splits(
        self, synthetic_medium: pd.DataFrame
    ) -> None:
        r1 = make_splits(synthetic_medium, seed=1)
        r2 = make_splits(synthetic_medium, seed=2)
        assert list(r1.train["id"]) != list(r2.train["id"])

    def test_seed_stored_in_result(self, synthetic_medium: pd.DataFrame) -> None:
        result = make_splits(synthetic_medium, seed=77)
        assert result.seed == 77

    def test_sizes_dict_keys(self, synthetic_medium: pd.DataFrame) -> None:
        result = make_splits(synthetic_medium)
        assert set(result.sizes()) == {"train", "val", "test"}

    def test_raises_on_ratios_not_summing_to_one(
        self, synthetic_medium: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            make_splits(synthetic_medium, ratios=(0.5, 0.3, 0.1))

    def test_raises_on_zero_ratio(self, synthetic_medium: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="positive"):
            make_splits(synthetic_medium, ratios=(0.85, 0.15, 0.0))

    def test_raises_on_negative_ratio(self, synthetic_medium: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="positive"):
            make_splits(synthetic_medium, ratios=(0.90, 0.20, -0.10))

    def test_input_dataframe_not_modified(self, synthetic_medium: pd.DataFrame) -> None:
        original_cols = list(synthetic_medium.columns)
        original_len = len(synthetic_medium)
        make_splits(synthetic_medium)
        assert list(synthetic_medium.columns) == original_cols
        assert len(synthetic_medium) == original_len
