"""Shared pytest fixtures for the AMORIS test suite."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def synthetic_small_path() -> Path:
    return FIXTURES_DIR / "synthetic_small.csv"


@pytest.fixture(scope="session")
def synthetic_medium_path() -> Path:
    return FIXTURES_DIR / "synthetic_medium.csv"


@pytest.fixture(scope="session")
def synthetic_small(synthetic_small_path: Path) -> pd.DataFrame:
    from amoris_bioage.data.loader import load_raw

    return load_raw(synthetic_small_path)


@pytest.fixture(scope="session")
def synthetic_medium(synthetic_medium_path: Path) -> pd.DataFrame:
    from amoris_bioage.data.loader import load_raw

    return load_raw(synthetic_medium_path)
