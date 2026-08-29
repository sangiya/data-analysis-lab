import numpy as np
import pandas as pd
import pytest
from src.data_analysis_lab.outlier_detector import OutlierDetector


@pytest.fixture
def df_with_outlier():
    data = list(range(1, 51))  # 1 to 50
    data.append(10000)  # obvious outlier at index 50
    return pd.DataFrame({"value": data})


def test_zscore_detects_outlier(df_with_outlier):
    detector = OutlierDetector()
    result = detector.detect_zscore(df_with_outlier, ["value"], threshold=2.0)
    assert 50 in result["value"]


def test_zscore_no_false_positives_on_normal_data():
    df = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
    detector = OutlierDetector()
    result = detector.detect_zscore(df, ["x"], threshold=5.0)
    # Very few (if any) outliers at 5-sigma in 100 samples
    assert len(result["x"]) <= 2


def test_iqr_detects_outlier(df_with_outlier):
    detector = OutlierDetector()
    result = detector.detect_iqr(df_with_outlier, ["value"], factor=1.5)
    assert 50 in result["value"]


def test_iqr_symmetric_data_no_outliers():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    detector = OutlierDetector()
    result = detector.detect_iqr(df, ["x"], factor=1.5)
    assert len(result["x"]) == 0


def test_isolation_forest_returns_boolean_mask(df_with_outlier):
    detector = OutlierDetector()
    mask = detector.detect_isolation_forest(df_with_outlier, ["value"], contamination=0.05)
    assert mask.dtype == bool
    assert len(mask) == len(df_with_outlier)
    # The extreme outlier should be flagged
    assert mask.iloc[50]


def test_summary_dataframe_structure():
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    detector = OutlierDetector()
    results = {"a": [1, 2, 3], "b": []}
    summary = detector.summary(df, results)
    assert "column" in summary.columns
    assert "n_outliers" in summary.columns
    assert "pct" in summary.columns
    assert summary[summary["column"] == "a"]["n_outliers"].iloc[0] == 3
    assert summary[summary["column"] == "b"]["n_outliers"].iloc[0] == 0


def test_zscore_constant_column():
    df = pd.DataFrame({"x": [5.0, 5.0, 5.0, 5.0]})
    detector = OutlierDetector()
    result = detector.detect_zscore(df, ["x"], threshold=3.0)
    assert result["x"] == []
