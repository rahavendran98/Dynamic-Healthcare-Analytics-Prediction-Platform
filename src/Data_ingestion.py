"""Data Ingestion Module.

Handles loading, validating, and analyzing healthcare datasets dynamically.
Works with any CSV regardless of column names, sizes, or data types.
"""

import pandas as pd
from typing import Optional


def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A DataFrame if loading succeeds, otherwise None.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded successfully: {df.shape[0]} rows x {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File not found -> {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except Exception as e:
        print(f"Error while loading file: {e}")
        return None


def validate_dataset(df: pd.DataFrame) -> tuple[bool, list]:
    """Check whether the dataset is usable for analysis.

    Args:
        df: The input DataFrame.

    Returns:
        A tuple (is_valid, issues) where issues is a list of problem messages.
    """
    issues = []

    if df is None:
        return False, ["Dataset is None (loading failed)."]

    if df.shape[0] == 0:
        issues.append("Dataset is empty (0 rows).")

    if df.shape[1] < 2:
        issues.append("At least 2 columns required (features + target).")

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        issues.append(f"Fully empty columns found: {empty_cols}")

    is_valid = len(issues) == 0
    return is_valid, issues


def analyze_dataset(df: pd.DataFrame) -> Optional[dict]:
    """Generate a complete summary of the dataset.

    Args:
        df: The input DataFrame.

    Returns:
        A dictionary with dataset-level and column-level statistics, or None on error.
    """
    try:
        analysis = {
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
            "columns": [],
        }

        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "missing_pct": round(df[col].isna().mean() * 100, 2),
                "unique": int(df[col].nunique()),
            }
            analysis["columns"].append(col_info)

        return analysis

    except Exception as e:
        print(f"Error during analysis: {e}")
        return None