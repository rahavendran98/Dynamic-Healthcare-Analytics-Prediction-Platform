"""Feature Engineering Module.

Dynamic feature processing for any healthcare dataset:
1. Feature engineering (drop useless columns, expand datetime)
2. Encoding (label for binary/target, one-hot for low-cardinality)
3. Scaling (StandardScaler for continuous features)

Saves all fitted artifacts (encoders, scaler, column order) as one bundle
so the exact same transforms can be reused at prediction time.
"""

import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utilities import setup_logger

logger = setup_logger(__name__)



def engineer_features(
    df: pd.DataFrame,
    target_col: str,
    id_threshold: float = 0.95,
    cardinality_threshold: int = 50,
) -> tuple[pd.DataFrame, dict]:
    
    df = df.copy()
    n_rows = len(df)
    report = {"dropped": {}, "datetime_expanded": []}

    try:
        for col in list(df.columns):
            if col == target_col:
                continue

            n_unique = df[col].nunique(dropna=False)

            if n_unique <= 1:
                df = df.drop(columns=[col])
                report["dropped"][col] = "constant"
                continue

            if n_unique / n_rows >= id_threshold:
                df = df.drop(columns=[col])
                report["dropped"][col] = "id-like"
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.9:
                    df[f"{col}_year"] = parsed.dt.year
                    df[f"{col}_month"] = parsed.dt.month
                    df[f"{col}_quarter"] = parsed.dt.quarter
                    df[f"{col}_weekday"] = parsed.dt.weekday
                    df = df.drop(columns=[col])
                    report["datetime_expanded"].append(col)
                    continue

                if n_unique > cardinality_threshold:
                    df = df.drop(columns=[col])
                    report["dropped"][col] = f"high-cardinality ({n_unique})"
                    continue

        logger.info(f"Feature engineering: dropped {len(report['dropped'])} columns")
    except Exception as e:
        logger.error(f"Error in engineer_features: {e}")

    return df, report



def encode_features(
    df: pd.DataFrame,
    target_col: str,
    low_card_max: int = 15,
) -> tuple[pd.DataFrame, dict, dict]:
    df = df.copy()

    report = {"target_encoded": None, "binary_encoded": [], "onehot_encoded": []}
    encoders = {}

    try:
        # Target -> label
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col])
            encoders["__target__"] = le
            report["target_encoded"] = {c: int(i) for c, i in
                                        zip(le.classes_, le.transform(le.classes_))}

        cat_cols = [c for c in df.columns
                    if c != target_col and not pd.api.types.is_numeric_dtype(df[c])]
        binary_cols = [c for c in cat_cols if df[c].nunique() == 2]
        onehot_cols = [c for c in cat_cols if 2 < df[c].nunique() <= low_card_max]

        # Binary -> label
        for col in binary_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            report["binary_encoded"].append(col)

        # Low-cardinality -> one-hot
        if onehot_cols:
            df = pd.get_dummies(df, columns=onehot_cols, drop_first=False)
            bool_cols = df.select_dtypes(include="bool").columns
            df[bool_cols] = df[bool_cols].astype(int)
            report["onehot_encoded"] = onehot_cols

        logger.info(f"Encoding: {len(binary_cols)} binary, {len(onehot_cols)} one-hot")
    except Exception as e:
        logger.error(f"Error in encode_features: {e}")

    return df, report, encoders



def scale_features(
    df: pd.DataFrame,
    target_col: str,
    min_unique: int = 15,
) -> tuple[pd.DataFrame, dict, StandardScaler]:
    
    df = df.copy()
    scaler = None
    report = {"scaled_columns": [], "skipped": []}

    try:
        continuous_cols = [
            c for c in df.columns
            if c != target_col
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].nunique() >= min_unique
        ]
        report["scaled_columns"] = continuous_cols

        if continuous_cols:
            scaler = StandardScaler()
            df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

        logger.info(f"Scaling: {len(continuous_cols)} continuous columns scaled")
    except Exception as e:
        logger.error(f"Error in scale_features: {e}")

    return df, report, scaler



def process_features(
    df: pd.DataFrame,
    target_col: str,
    artifacts_path: str = "models/preprocessing_artifacts.pkl",
) -> tuple[pd.DataFrame, dict]:
    
    logger.info(f"Feature processing started. Input shape: {df.shape}")
    report = {"input_shape": df.shape}

    df, report["engineering"] = engineer_features(df, target_col)
    df, report["encoding"], encoders = encode_features(df, target_col)
    df, report["scaling"], scaler = scale_features(df, target_col)

    # Feature column order (exclude target) — crucial for prediction alignment
    feature_columns = [c for c in df.columns if c != target_col]

    artifacts = {
        "encoders": encoders,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "target_col": target_col,
        "scaled_columns": report["scaling"]["scaled_columns"],
    }

    try:
        os.makedirs(os.path.dirname(artifacts_path), exist_ok=True)
        joblib.dump(artifacts, artifacts_path)
        logger.info(f"Artifacts saved -> {artifacts_path}")
        report["artifacts_path"] = artifacts_path
    except Exception as e:
        logger.error(f"Error saving artifacts: {e}")

    report["final_shape"] = df.shape
    logger.info(f"Feature processing complete. Final shape: {df.shape}")
    return df, report