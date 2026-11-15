"""Shared fixtures: session-scoped match arrays (first use builds the cache)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]  # repo root from this file
if str(ROOT) not in sys.path:  # skip if already on path
    sys.path.insert(0, str(ROOT))  # make repo root importable


@pytest.fixture(scope="session")
def arrays():
    from src import data as D

    return D.load_arrays()  # build or reuse cache
