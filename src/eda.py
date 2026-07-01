"""EDA Module.

Generates statistical summaries, key charts, and a downloadable HTML report
for any healthcare dataset dynamically.
"""

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

        return {
            "numeric": df[numeric_cols].describe().T if len(numeric_cols) else None,
            "categorical": df[categorical_cols].describe().T if len(categorical_cols) else None,
            "n_numeric": len(numeric_cols),
            "n_categorical": len(categorical_cols),
        }
    except Exception as e:
        logger.error(f"Error in statistical_summary: {e}")
        return {"numeric": None, "categorical": None, "n_numeric": 0, "n_categorical": 0}


def plot_target_distribution(df: pd.DataFrame, target_col: str) -> Optional[plt.Figure]:
    
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df[target_col].value_counts()
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#3498db")
        ax.set_title(f"Target Distribution: {target_col}")
        ax.set_ylabel("Count")
        fig.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error in plot_target_distribution: {e}")
        return None


def plot_correlation(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    min_unique: int = 15,
) -> Optional[plt.Figure]:
    
    try:
        df_c = df.copy()
        numeric_cols = [
            c for c in df_c.select_dtypes(include="number").columns
            if df_c[c].nunique() >= min_unique
        ]

        if target_col and target_col not in numeric_cols:
            if df_c[target_col].nunique() <= 10:
                df_c[target_col] = df_c[target_col].astype("category").cat.codes
                numeric_cols.append(target_col)

        if len(numeric_cols) < 2:
            logger.info("Not enough numeric columns for correlation.")
            return None

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_c[numeric_cols].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error in plot_correlation: {e}")
        return None


def plot_missing_values(df: pd.DataFrame) -> Optional[plt.Figure]:
    
    try:
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            logger.info("No missing values to plot.")
            return None

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=missing.values, y=missing.index.astype(str), ax=ax, color="#e74c3c")
        ax.set_title("Missing Values per Column")
        ax.set_xlabel("Missing Count")
        fig.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error in plot_missing_values: {e}")
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
    
    import os
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

        # Numeric summary table
        summary = statistical_summary(df)
        if summary["numeric"] is not None:
            parts.append("<h2>Numeric Summary</h2>")
            parts.append(summary["numeric"].round(2).to_html())

        # Target distribution
        fig = plot_target_distribution(df, target_col)
        if fig is not None:
            parts.append("<h2>Target Distribution</h2>")
            parts.append(_fig_to_base64(fig))

        # Correlation
        fig = plot_correlation(df, target_col)
        if fig is not None:
            parts.append("<h2>Correlation Heatmap</h2>")
            parts.append(_fig_to_base64(fig))

        # Missing values
        fig = plot_missing_values(df)
        if fig is not None:
            parts.append("<h2>Missing Values</h2>")
            parts.append(_fig_to_base64(fig))

        parts.append("</body></html>")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))

        logger.info(f"EDA report saved -> {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error in generate_eda_report: {e}")
        return None