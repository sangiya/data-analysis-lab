from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class OutlierDetector:
    def detect_zscore(
        self,
        df: pd.DataFrame,
        columns: list[str],
        threshold: float = 3.0,
    ) -> dict[str, list[int]]:
        """Return row indices that exceed the Z-score threshold for each column."""
        result = {}
        for col in columns:
            s = df[col].dropna()
            std = s.std()
            if std == 0:
                result[col] = []
                continue
            zscores = (s - s.mean()) / std
            result[col] = list(s[zscores.abs() > threshold].index)
        return result

    def detect_iqr(
        self,
        df: pd.DataFrame,
        columns: list[str],
        factor: float = 1.5,
    ) -> dict[str, list[int]]:
        """Return row indices outside the IQR fence for each column."""
        result = {}
        for col in columns:
            s = df[col].dropna()
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            result[col] = list(s[(s < lower) | (s > upper)].index)
        return result

    def detect_isolation_forest(
        self,
        df: pd.DataFrame,
        columns: list[str],
        contamination: float = 0.05,
    ) -> pd.Series:
        """Return a boolean mask where True indicates an outlier row."""
        X = df[columns].dropna()
        model = IsolationForest(contamination=contamination, random_state=42)
        preds = model.fit_predict(X)
        mask = pd.Series(False, index=df.index)
        mask.loc[X.index] = preds == -1
        return mask

    def summary(self, df: pd.DataFrame, results: dict[str, list[int]]) -> pd.DataFrame:
        """Summarise outlier detection results as a DataFrame."""
        rows = []
        for col, indices in results.items():
            rows.append({
                "column": col,
                "n_outliers": len(indices),
                "pct": round(len(indices) / len(df) * 100, 2) if len(df) > 0 else 0.0,
            })
        return pd.DataFrame(rows)
