"""Tests for typed configuration dataclasses."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from amoris_bioage.config import (
    AMORISConfig,
    BiologicalAgeConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
)


class TestDataConfig:
    def test_valid_config(self, tmp_path: Path) -> None:
        cfg = DataConfig(raw_path=tmp_path / "raw.csv", derived_dir=tmp_path)
        assert cfg.train_frac == 0.70
        assert cfg.val_frac == 0.15
        assert abs(cfg.test_frac - 0.15) < 1e-9

    def test_test_frac_is_complement(self, tmp_path: Path) -> None:
        cfg = DataConfig(
            raw_path=tmp_path / "raw.csv",
            derived_dir=tmp_path,
            train_frac=0.60,
            val_frac=0.20,
        )
        assert abs(cfg.test_frac - 0.20) < 1e-9

    def test_fracs_summing_to_one_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="test_frac"):
            DataConfig(
                raw_path=tmp_path / "raw.csv",
                derived_dir=tmp_path,
                train_frac=0.85,
                val_frac=0.15,
            )

    def test_negative_train_frac_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            DataConfig(
                raw_path=tmp_path / "raw.csv",
                derived_dir=tmp_path,
                train_frac=-0.1,
                val_frac=0.15,
            )


class TestModelConfig:
    def test_defaults(self) -> None:
        cfg = ModelConfig()
        assert cfg.hidden_sizes == [128, 64]
        assert cfg.dropout == 0.2
        assert cfg.activation == "relu"
        assert cfg.n_features == 15

    def test_three_hidden_layers_accepted(self) -> None:
        cfg = ModelConfig(hidden_sizes=[128, 64, 32])
        assert len(cfg.hidden_sizes) == 3

    def test_one_hidden_layer_raises(self) -> None:
        with pytest.raises(ValidationError, match="hidden_sizes"):
            ModelConfig(hidden_sizes=[128])

    def test_four_hidden_layers_raises(self) -> None:
        with pytest.raises(ValidationError, match="hidden_sizes"):
            ModelConfig(hidden_sizes=[128, 64, 32, 16])

    def test_invalid_activation_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(activation="tanh")

    def test_dropout_gte_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(dropout=1.0)


class TestTrainingConfig:
    def test_defaults(self) -> None:
        cfg = TrainingConfig()
        assert 1e-4 <= cfg.learning_rate <= 1e-3
        assert cfg.max_epochs > 0
        assert cfg.patience > 0

    def test_zero_learning_rate_raises(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0.0)


class TestBiologicalAgeConfig:
    def test_defaults(self) -> None:
        cfg = BiologicalAgeConfig()
        assert cfg.horizon_years == 10.0

    def test_invalid_grid_range_raises(self) -> None:
        with pytest.raises(ValidationError, match="age_grid_min"):
            BiologicalAgeConfig(age_grid_min=80.0, age_grid_max=30.0)


class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        yaml_content = """\
data:
  raw_path: /data/raw.csv
  derived_dir: /data/derived
model:
  hidden_sizes: [128, 64]
  dropout: 0.3
training:
  learning_rate: 0.001
bioage:
  horizon_years: 10.0
"""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml_content)
        cfg = load_config(cfg_path)
        assert isinstance(cfg, AMORISConfig)
        assert cfg.model.dropout == 0.3
        assert cfg.training.learning_rate == 0.001

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        yaml_content = "model:\n  dropout: 0.2\n"  # no 'data' key
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text(yaml_content)
        with pytest.raises(ValidationError):
            load_config(cfg_path)
