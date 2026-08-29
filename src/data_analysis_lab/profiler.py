from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class DataProfile:
    shape: tuple
    dtypes: dict
    missing_counts: dict
    missing_pct: dict
    numeric_stats: dict
    categorical_stats: dict
    duplicated_rows: int


class DataProfiler:
    def profile(self, df: pd.DataFrame) -> DataProfile:
        """Generate a comprehensive profile of a DataFrame."""
        numeric_stats = {}
        for col in df.select_dtypes(include="number").columns:
            numeric_stats[col] = df[col].describe().to_dict()

        categorical_stats = {}
        for col in df.select_dtypes(include=["object", "category"]).columns:
            categorical_stats[col] = df[col].value_counts().head(10).to_dict()

        return DataProfile(
            shape=df.shape,
            dtypes={c: str(t) for c, t in df.dtypes.items()},
            missing_counts={c: int(df[c].isna().sum()) for c in df.columns},
            missing_pct={c: float(df[c].isna().mean() * 100) for c in df.columns},
            numeric_stats=numeric_stats,
            categorical_stats=categorical_stats,
            duplicated_rows=int(df.duplicated().sum()),
        )

    def to_markdown(self, profile: DataProfile) -> str:
        """Render a DataProfile as a Markdown report."""
        lines = [
            "## Dataset Profile",
            f"- Shape: {profile.shape[0]} rows x {profile.shape[1]} columns",
            f"- Duplicated rows: {profile.duplicated_rows}",
            "",
            "### Missing Values",
            "| Column | Count | % |",
            "|--------|-------|---|",
        ]
        for col, count in profile.missing_counts.items():
            lines.append(f"| {col} | {count} | {profile.missing_pct[col]:.1f}% |")

        if profile.numeric_stats:
            lines += [
                "",
                "### Numeric Statistics",
                "| Column | Mean | Std | Min | Max |",
                "|--------|------|-----|-----|-----|",
            ]
            for col, stats in profile.numeric_stats.items():
                lines.append(
                    f"| {col} | {stats.get('mean', 0):.2f} | {stats.get('std', 0):.2f} "
                    f"| {stats.get('min', 0):.2f} | {stats.get('max', 0):.2f} |"
                )

        return "\n".join(lines)
