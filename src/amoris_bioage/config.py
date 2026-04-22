"""Typed configuration dataclasses for the AMORIS pipeline.

All mutable configuration lives here. Configs are loaded from YAML and
validated by pydantic; no magic strings appear elsewhere in the codebase.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    """Paths and split parameters."""

    raw_path: Path
    derived_dir: Path
    split_seed: int = Field(default=42, ge=0)
    train_frac: float = Field(default=0.70, gt=0, lt=1)
    val_frac: float = Field(default=0.15, gt=0, lt=1)

    @model_validator(mode="after")
    def fracs_sum_to_less_than_one(self) -> "DataConfig":
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError(
                f"train_frac ({self.train_frac}) + val_frac ({self.val_frac}) "
                "must be < 1.0 so that test_frac > 0"
            )
        return self

    @property
    def test_frac(self) -> float:
        return round(1.0 - self.train_frac - self.val_frac, 10)


class ModelConfig(BaseModel):
    """Neural network architecture."""

    # Number of biomarker features (excluding chronological age).
    n_features: int = Field(default=15, ge=1)
    hidden_sizes: list[int] = Field(default=[128, 64])
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    activation: Literal["relu", "elu", "selu"] = "relu"

    @model_validator(mode="after")
    def hidden_sizes_valid(self) -> "ModelConfig":
        if len(self.hidden_sizes) < 2 or len(self.hidden_sizes) > 3:
            raise ValueError("hidden_sizes must have 2 or 3 entries per CLAUDE.md spec")
        if any(h < 32 or h > 256 for h in self.hidden_sizes):
            raise ValueError("Each hidden layer must be between 32 and 256 units")
        return self


class TrainingConfig(BaseModel):
    """Optimiser and training loop parameters."""

    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    max_epochs: int = Field(default=200, ge=1)
    # Early stopping patience measured in epochs of no validation C-index improvement.
    patience: int = Field(default=20, ge=1)
    batch_size: int = Field(default=4096, ge=1)
    seed: int = Field(default=0, ge=0)


class BiologicalAgeConfig(BaseModel):
    """Parameters for the biological age computation and inversion."""

    # Prediction horizon for the mortality equivalence definition.
    horizon_years: float = Field(default=10.0, gt=0)
    # Age grid over which the reference mortality mapping is precomputed.
    age_grid_min: float = Field(default=30.0)
    age_grid_max: float = Field(default=100.0)
    age_grid_step: float = Field(default=0.1, gt=0)

    @model_validator(mode="after")
    def grid_range_valid(self) -> "BiologicalAgeConfig":
        if self.age_grid_min >= self.age_grid_max:
            raise ValueError("age_grid_min must be less than age_grid_max")
        return self


class AMORISConfig(BaseModel):
    """Top-level config; load from YAML via pydantic model_validate."""

    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    bioage: BiologicalAgeConfig = Field(default_factory=BiologicalAgeConfig)


def load_config(path: str | Path) -> AMORISConfig:
    """Load and validate an AMORIS config from a YAML file.

    Args:
        path: Path to a YAML config file.

    Returns:
        Validated AMORISConfig.
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    return AMORISConfig.model_validate(raw)
