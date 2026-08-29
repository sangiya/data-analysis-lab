from __future__ import annotations
import pandas as pd


class CorrelationAnalyzer:
    def compute_pearson(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pearson correlation matrix for numeric columns."""
        return df.select_dtypes(include="number").corr(method="pearson")

    def compute_spearman(self, df: pd.DataFrame) -> pd.DataFrame:
        """Spearman rank correlation matrix for numeric columns."""
        return df.select_dtypes(include="number").corr(method="spearman")

    def find_top_correlations(
        self,
        corr: pd.DataFrame,
        n: int = 10,
        threshold: float = 0.5,
    ) -> list[dict]:
        """Return the top N correlated column pairs above the given absolute threshold."""
        results = []
        seen: set[tuple[str, str]] = set()
        for col1 in corr.columns:
            for col2 in corr.columns:
                if col1 == col2 or (col2, col1) in seen:
                    continue
                seen.add((col1, col2))
                val = float(corr.loc[col1, col2])
                if abs(val) >= threshold:
                    results.append({"col1": col1, "col2": col2, "correlation": val})
        return sorted(results, key=lambda x: abs(x["correlation"]), reverse=True)[:n]

    def find_highly_correlated_pairs(
        self,
        df: pd.DataFrame,
        method: str = "pearson",
        threshold: float = 0.9,
    ) -> list[tuple[str, str, float]]:
        """Return (col1, col2, correlation) tuples where |corr| >= threshold.

        Useful for identifying redundant features before modelling.
        """
        corr = df.select_dtypes(include="number").corr(method=method)
        pairs = []
        seen: set[tuple[str, str]] = set()
        for col1 in corr.columns:
            for col2 in corr.columns:
                if col1 == col2 or (col2, col1) in seen:
                    continue
                seen.add((col1, col2))
                val = float(corr.loc[col1, col2])
                if abs(val) >= threshold:
                    pairs.append((col1, col2, val))
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
