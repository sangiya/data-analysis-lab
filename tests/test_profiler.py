import numpy as np
import pandas as pd
import pytest
from src.data_analysis_lab.profiler import DataProfiler, DataProfile


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25.0, 30.0, None, 45.0, 22.0],
        "salary": [50000.0, 60000.0, 55000.0, None, 48000.0],
        "department": ["Engineering", "Sales", "Engineering", "HR", "Sales"],
    })


def test_profile_returns_dataprofile(sample_df):
    profiler = DataProfiler()
    profile = profiler.profile(sample_df)
    assert isinstance(profile, DataProfile)


def test_profile_shape(sample_df):
    profiler = DataProfiler()
    profile = profiler.profile(sample_df)
    assert profile.shape == (5, 3)


def test_profile_missing_counts(sample_df):
    profiler = DataProfiler()
    profile = profiler.profile(sample_df)
    assert profile.missing_counts["age"] == 1
    assert profile.missing_counts["salary"] == 1
    assert profile.missing_counts["department"] == 0


def test_profile_missing_pct(sample_df):
    profiler = DataProfiler()
    profile = profiler.profile(sample_df)
    assert profile.missing_pct["age"] == pytest.approx(20.0)


def test_profile_numeric_stats_keys(sample_df):
    profiler = DataProfiler()
    profile = profiler.profile(sample_df)
    assert "age" in profile.numeric_stats
    assert "salary" in profile.numeric_stats
    assert "mean" in profile.numeric_stats["age"]


def test_profile_categorical_stats(sample_df):
    profiler = DataProfiler()
    profile = profiler.profile(sample_df)
    assert "department" in profile.categorical_stats
    assert "Engineering" in profile.categorical_stats["department"]


def test_profile_no_duplicates(sample_df):
    profiler = DataProfiler()
    profile = profiler.profile(sample_df)
    assert profile.duplicated_rows == 0


def test_profile_with_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 10, 20]})
    profiler = DataProfiler()
    profile = profiler.profile(df)
    assert profile.duplicated_rows == 1


def test_to_markdown_contains_header(sample_df):
    profiler = DataProfiler()
    profile = profiler.profile(sample_df)
    md = profiler.to_markdown(profile)
    assert "## Dataset Profile" in md
    assert "Missing Values" in md
    assert "age" in md
