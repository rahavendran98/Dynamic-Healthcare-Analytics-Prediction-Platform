"""Data Cleaning Module.

Provides dynamic cleaning for any healthcare dataset:
missing value treatment, duplicate removal, and outlier capping.
Works regardless of column names or data types.
"""

import pandas as pd
from utilities import setup_logger

logger = setup_logger(__name__)


def treat_missing(df: pd.DataFrame, drop_threshold: float = 0.5) -> tuple[pd.DataFrame, dict]:
    """Handle missing values dynamically.

    Columns above the drop_threshold are dropped. Numeric columns are filled
    with the median; categorical columns are filled with the mode.

    Args:
        df: Input DataFrame.
        drop_threshold: Missing fraction above which a column is dropped.

    Returns:
        Cleaned DataFrame and a report dict.
    """
    df = df.copy()
    report = {"dropped_columns": [], "filled_columns": {}}

    try:
        for col in df.columns:
            miss_frac = df[col].isna().mean()
            if miss_frac == 0:
                continue

            if miss_frac > drop_threshold:
                df = df.drop(columns=[col])
                report["dropped_columns"].append(col)
                logger.info(f"Dropped column '{col}' ({miss_frac:.1%} missing)")
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median()
                method = "median"
            else:
                fill_value = df[col].mode()[0]
                method = "mode"

            df[col] = df[col].fillna(fill_value)
            report["filled_columns"][col] = {
                "method": method,
                "fill_value": fill_value,
                "pct_filled": round(miss_frac * 100, 2),
            }
            logger.info(f"Filled '{col}' with {method} ({miss_frac:.1%} missing)")

    except Exception as e:
        logger.error(f"Error in treat_missing: {e}")

    return df, report


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    
    df = df.copy()
    n_before = df.shape[0]
    n_duplicates = int(df.duplicated().sum())

    df = df.drop_duplicates().reset_index(drop=True)
    n_after = df.shape[0]

    logger.info(f"Removed {n_before - n_after} duplicate rows")

    report = {
        "rows_before": n_before,
        "duplicates_found": n_duplicates,
        "rows_after": n_after,
        "rows_removed": n_before - n_after,
    }
    return df, report


def treat_outliers(
    df: pd.DataFrame,
    factor: float = 1.5,
    min_unique: int = 15,
) -> tuple[pd.DataFrame, dict]:
    
    df = df.copy()
    report = {}

    try:
        numeric_cols = df.select_dtypes(include="number").columns

        for col in numeric_cols:
            # Skip binary / low-cardinality columns (likely categorical in disguise)
            if df[col].nunique() < min_unique:
                logger.info(
                    f"Skipped '{col}' for outliers ({df[col].nunique()} unique values)"
                )
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR

            n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
            if n_outliers > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                report[col] = {
                    "outliers": n_outliers,
                    "lower_bound": round(lower, 2),
                    "upper_bound": round(upper, 2),
                }
                logger.info(f"Capped {n_outliers} outliers in '{col}'")

    except Exception as e:
        logger.error(f"Error in treat_outliers: {e}")

    return df, report


def clean_dataset(
    df: pd.DataFrame,
    drop_threshold: float = 0.5,
    outlier_factor: float = 1.5,
    min_unique: int = 15,
) -> tuple[pd.DataFrame, dict]:
    
    logger.info(f"Cleaning started. Input shape: {df.shape}")
    report = {"original_shape": df.shape}

    df, report["missing"] = treat_missing(df, drop_threshold=drop_threshold)
    df, report["duplicates"] = remove_duplicates(df)
    df, report["outliers"] = treat_outliers(df, factor=outlier_factor, min_unique=min_unique)

    report["final_shape"] = df.shape
    logger.info(f"Cleaning finished. Output shape: {df.shape}")
    return df, report