import numpy as np
import pandas as pd
import pytest
from src.data_analysis_lab.correlation_analyzer import CorrelationAnalyzer


@pytest.fixture
def correlated_df():
    np.random.seed(42)
    n = 100
    x = np.random.normal(0, 1, n)
    return pd.DataFrame({
        "x": x,
        "y_high": x * 2 + np.random.normal(0, 0.1, n),   # high correlation with x
        "y_low": np.random.normal(0, 1, n),                # low correlation with x
        "cat": ["A"] * 50 + ["B"] * 50,                    # non-numeric, should be ignored
    })


def test_compute_pearson_shape(correlated_df):
    analyzer = CorrelationAnalyzer()
    corr = analyzer.compute_pearson(correlated_df)
    # 3 numeric columns: x, y_high, y_low
    assert corr.shape == (3, 3)


def test_pearson_diagonal_is_one(correlated_df):
    analyzer = CorrelationAnalyzer()
    corr = analyzer.compute_pearson(correlated_df)
    for col in corr.columns:
        assert corr.loc[col, col] == pytest.approx(1.0)


def test_pearson_detects_high_correlation(correlated_df):
    analyzer = CorrelationAnalyzer()
    corr = analyzer.compute_pearson(correlated_df)
    assert abs(corr.loc["x", "y_high"]) > 0.95


def test_pearson_low_correlation(correlated_df):
    analyzer = CorrelationAnalyzer()
    corr = analyzer.compute_pearson(correlated_df)
    assert abs(corr.loc["x", "y_low"]) < 0.4


def test_spearman_returns_matrix(correlated_df):
    analyzer = CorrelationAnalyzer()
    corr = analyzer.compute_spearman(correlated_df)
    assert corr.shape == (3, 3)


def test_find_top_correlations_above_threshold(correlated_df):
    analyzer = CorrelationAnalyzer()
    corr = analyzer.compute_pearson(correlated_df)
    pairs = analyzer.find_top_correlations(corr, n=10, threshold=0.9)
    assert len(pairs) >= 1
    for pair in pairs:
        assert abs(pair["correlation"]) >= 0.9


def test_find_top_correlations_no_duplicates(correlated_df):
    analyzer = CorrelationAnalyzer()
    corr = analyzer.compute_pearson(correlated_df)
    pairs = analyzer.find_top_correlations(corr, n=10, threshold=0.0)
    seen = set()
    for pair in pairs:
        key = tuple(sorted([pair["col1"], pair["col2"]]))
        assert key not in seen
        seen.add(key)


def test_find_highly_correlated_pairs(correlated_df):
    analyzer = CorrelationAnalyzer()
    pairs = analyzer.find_highly_correlated_pairs(correlated_df, method="pearson", threshold=0.9)
    pair_names = [(p[0], p[1]) for p in pairs]
    # x and y_high should appear as a highly correlated pair
    assert any("x" in p and "y_high" in p for p in pair_names)
