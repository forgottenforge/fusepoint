"""
Smoke tests for the public API. Kept fast and deterministic for CI.
"""
import numpy as np
import pandas as pd
import pytest

import fusepoint
from fusepoint import analyze, scan, compare, StabilityResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def logistic():
    """Smooth sigmoid with a sharp transition at x = 5.0."""
    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, 120)
    y = 1.0 / (1.0 + np.exp(-2.0 * (x - 5.0))) + rng.normal(0, 0.02, 120)
    return x, y


@pytest.fixture
def df_logistic(logistic):
    x, y = logistic
    return pd.DataFrame({"input": x, "output": y})


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_version_is_published():
    assert fusepoint.__version__ == "1.1.0"


def test_paper_doi_exposed():
    assert fusepoint.__paper_doi__ == "10.5281/zenodo.20548818"
    assert fusepoint.__paper_url__.startswith("https://doi.org/")


# ---------------------------------------------------------------------------
# Statistical layer
# ---------------------------------------------------------------------------

def test_analyze_array_mode(logistic):
    x, y = logistic
    r = analyze(x, y, n_boot=200, n_perm=500)
    assert isinstance(r, StabilityResult)
    assert 0 <= r.score <= 100
    assert r.grade in {"STABLE", "MODERATE", "WARNING", "CRITICAL"}
    # Tipping point should land near 5.0
    assert 3.5 < r.critical_x < 6.5
    assert r.chi.shape == x.shape


def test_analyze_dataframe_mode(df_logistic):
    r = analyze(df_logistic, x="input", y="output", n_boot=200, n_perm=500)
    assert r.x_name == "Input"
    assert r.y_name == "Output"


def test_compare_returns_delta(logistic):
    x, y = logistic
    r = compare(x, y, x, y * 1.1, n_boot=200, n_perm=500)
    assert hasattr(r, "delta_score")
    assert isinstance(r.summary(), str)


def test_scan_multiple_columns(df_logistic):
    df_logistic = df_logistic.copy()
    df_logistic["noise"] = np.random.default_rng(0).normal(0, 1, len(df_logistic))
    results = scan(df_logistic, n_boot=100, n_perm=200)
    assert len(results) >= 1
    assert all(isinstance(r, StabilityResult) for r in results)


# ---------------------------------------------------------------------------
# Theorem-anchored layer (deep=True)
# ---------------------------------------------------------------------------

def test_deep_mode_populates_v4_fields(logistic):
    x, y = logistic
    r = analyze(x, y, deep=True, n_boot=200, n_perm=500)
    # Theorem-anchored fields are populated
    assert r.regime in {"I_geom", "II_geom", "III_geom"}
    assert r.citations is not None and len(r.citations) >= 2
    assert "def:sigmac" in r.citations
    assert r.paper_doi == "10.5281/zenodo.20548818"
    # Statistical fields are still there
    assert 0 <= r.score <= 100


def test_shallow_mode_leaves_v4_fields_none(logistic):
    x, y = logistic
    r = analyze(x, y, n_boot=100, n_perm=200)  # deep defaults to False
    assert r.regime is None
    assert r.citations is None
    assert r.paper_doi is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_too_few_points_raises():
    with pytest.raises(ValueError):
        analyze(np.array([1, 2, 3]), np.array([1, 2, 3]))


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        analyze(np.array([1, 2, 3, 4, 5]), np.array([1, 2]))
