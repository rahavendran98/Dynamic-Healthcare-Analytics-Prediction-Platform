"""EDA Module."""

import os
import base64
from io import BytesIO
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utilities import setup_logger

logger = setup_logger(__name__)


def statistical_summary(df: pd.DataFrame) -> dict:

    try:
        numeric_cols = df.select_dtypes(include="number").columns
        categorical_cols = df.select_dtypes(include="object").columns

        summary = {
            "numeric": df[numeric_cols].describe().T if len(numeric_cols) else None,
            "categorical": df[categorical_cols].describe().T if len(categorical_cols) else None,
            "n_numeric": len(numeric_cols),
            "n_categorical": len(categorical_cols),
        }
        logger.info(f"Statistical summary: {summary['n_numeric']} numeric, "
                    f"{summary['n_categorical']} categorical")
        return summary
    except Exception as e:
        logger.error(f"Error in statistical_summary: {e}")
        return {"numeric": None, "categorical": None, "n_numeric": 0, "n_categorical": 0}


def plot_missing_values(df: pd.DataFrame) -> Optional[plt.Figure]:

    try:
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if missing.empty:
            logger.info("No missing values to plot.")
            return None

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=missing.values, y=missing.index.astype(str), ax=ax)
        ax.set_title("Missing values per column")
        ax.set_xlabel("Missing Count")
        for i, v in enumerate(missing.values):
            ax.text(v, i, f" {v}", va="center")
        fig.tight_layout()

        logger.info(f"Missing values found in {len(missing)} columns")
        return fig
    except Exception as e:
        logger.error(f"Error in plot_missing_values: {e}")
        return None


def plot_target_distribution(df: pd.DataFrame, target_col: str) -> Optional[plt.Figure]:

    try:
        fig, ax = plt.subplots(figsize=(7, 4))

        counts = df[target_col].value_counts()
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)

        ax.set_title(f"Target Distribution: {target_col}")
        ax.set_xlabel(target_col)
        ax.set_ylabel("Count")

        for i, v in enumerate(counts.values):
            ax.text(i, v, str(v), ha="center", va="bottom")
        fig.tight_layout()

        pct = (counts / counts.sum() * 100).round(2)
        logger.info(f"Class balance for '{target_col}': {pct.to_dict()}")
        return fig
    except Exception as e:
        logger.error(f"Error in plot_target_distribution: {e}")
        return None


def plot_numeric_distributions(
    df: pd.DataFrame,
    min_unique: int = 15,
    max_unique: int = 10000,
    max_cols: int = 6,
) -> Optional[plt.Figure]:

    try:
        numeric_cols = [
            c for c in df.select_dtypes(include="number").columns
            if min_unique <= df[c].nunique() <= max_unique
        ]

        if not numeric_cols:
            logger.info("No continuous numeric columns to plot.")
            return None

        numeric_cols = numeric_cols[:max_cols]
        n = len(numeric_cols)
        n_rows = (n + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde=True, ax=axes[i])
            axes[i].set_title(f"{col}")

        for j in range(n, len(axes)):
            axes[j].set_visible(False)
        fig.tight_layout()

        logger.info(f"Plotted numeric distributions for {n} columns")
        return fig
    except Exception as e:
        logger.error(f"Error in plot_numeric_distributions: {e}")
        return None


def plot_categorical_distribution(
    df: pd.DataFrame,
    max_unique: int = 15,
    max_cols: int = 6,
) -> Optional[plt.Figure]:

    try:
        cat_cols = [
            c for c in df.select_dtypes(include="object").columns
            if df[c].nunique() <= max_unique
        ]

        if not cat_cols:
            logger.info("No suitable categorical columns to plot.")
            return None

        cat_cols = cat_cols[:max_cols]
        n = len(cat_cols)
        n_rows = (n + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(cat_cols):
            counts = df[col].value_counts()
            sns.barplot(x=counts.values, y=counts.index.astype(str), ax=axes[i])
            axes[i].set_title(f"{col}")
            axes[i].set_xlabel("Count")

        for j in range(n, len(axes)):
            axes[j].set_visible(False)
        fig.tight_layout()

        skipped = [
            c for c in df.select_dtypes(include="object").columns
            if df[c].nunique() > max_unique
        ]
        if skipped:
            logger.info(f"Skipped high-cardinality categorical columns: {skipped}")

        logger.info(f"Plotted categorical distributions for {n} columns")
        return fig
    except Exception as e:
        logger.error(f"Error in plot_categorical_distribution: {e}")
        return None


def plot_correlation(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    min_unique: int = 15,
) -> Optional[plt.Figure]:

    try:
        df_corr = df.copy()

        numeric_cols = [
            c for c in df_corr.select_dtypes(include="number").columns
            if df_corr[c].nunique() >= min_unique
        ]

        if target_col and target_col not in numeric_cols:
            if df_corr[target_col].nunique() <= 10:
                df_corr[target_col] = df_corr[target_col].astype("category").cat.codes
                numeric_cols.append(target_col)

        if len(numeric_cols) < 2:
            logger.info("Not enough numeric columns for correlation.")
            return None

        corr = df_corr[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()

        if target_col in corr.columns:
            target_corr = corr[target_col].drop(target_col).abs().sort_values(ascending=False)
            logger.info(f"Strongest correlations with '{target_col}': "
                        f"{target_corr.head(5).round(3).to_dict()}")

        return fig
    except Exception as e:
        logger.error(f"Error in plot_correlation: {e}")
        return None


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib Figure to a base64 <img> tag for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=80)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;"/>'


def generate_eda_report(
    df: pd.DataFrame,
    target_col: str,
    output_path: str = "reports/eda_report.html",
) -> Optional[str]:

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        parts = []

        parts.append(f"""
        <html><head><title>EDA Report</title>
        <style>
            body {{ font-family: Arial; margin: 30px; color: #333; }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; }}
            h2 {{ color: #2980b9; margin-top: 30px; }}
            table {{ border-collapse: collapse; margin: 15px 0; font-size: 12px; }}
            th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
            th {{ background: #3498db; color: white; }}
            .box {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }}
        </style></head><body>
        <h1>EDA Report</h1>
        <div class="box">
            <b>Rows:</b> {df.shape[0]} &nbsp;|&nbsp;
            <b>Columns:</b> {df.shape[1]} &nbsp;|&nbsp;
            <b>Target:</b> {target_col}
        </div>
        """)

        # Target distribution
        fig = plot_target_distribution(df, target_col)
        if fig is not None:
            parts.append("<h2>Target Distribution</h2>")
            parts.append(_fig_to_base64(fig))

        # Statistical summary
        summary = statistical_summary(df)
        parts.append("<h2>Statistical Summary</h2>")
        if summary["numeric"] is not None:
            parts.append("<h3>Numeric Features</h3>")
            parts.append(summary["numeric"].round(2).to_html())
        if summary["categorical"] is not None:
            parts.append("<h3>Categorical Features</h3>")
            parts.append(summary["categorical"].to_html())

        # Correlation heatmap
        fig = plot_correlation(df, target_col)
        if fig is not None:
            parts.append("<h2>Correlation Heatmap</h2>")
            parts.append(_fig_to_base64(fig))

        # Missing values
        fig = plot_missing_values(df)
        parts.append("<h2>Missing Values</h2>")
        if fig is not None:
            parts.append(_fig_to_base64(fig))
        else:
            parts.append("<p>No missing values.</p>")

        # Numeric distributions
        fig = plot_numeric_distributions(df)
        if fig is not None:
            parts.append("<h2>Numerical Feature Distributions</h2>")
            parts.append(_fig_to_base64(fig))

        # Categorical distributions
        fig = plot_categorical_distribution(df)
        if fig is not None:
            parts.append("<h2>Categorical Feature Distributions</h2>")
            parts.append(_fig_to_base64(fig))

        parts.append("</body></html>")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))

        logger.info(f"EDA report saved -> {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error in generate_eda_report: {e}")
        return None


