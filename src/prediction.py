"""Prediction Module."""


import os
import joblib
import pandas as pd

from utilities import setup_logger

logger = setup_logger(__name__)



#  LOAD SAVED MODEL + PREPROCESSING ARTIFACTS

def load_artifacts(models_dir: str = "../models", target_col: str = "diabetes") -> tuple[dict, dict]:
    model_path = os.path.join(models_dir, f"{target_col}_best_model.pkl")
    artifacts_path = os.path.join(models_dir, f"{target_col}_preprocessing_artifacts.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. Run model_training.ipynb first."
        )
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(
            f"No preprocessing artifacts found at '{artifacts_path}'. "
            "Run feature_engineering.ipynb first."
        )

    model_bundle = joblib.load(model_path)
    artifacts = joblib.load(artifacts_path)

    logger.info(
        f"Loaded model '{model_bundle['model_name']}' and preprocessing artifacts "
        f"({len(artifacts['feature_columns'])} features)."
    )
    return model_bundle, artifacts



#  PREPROCESS UNSEEN DATA (transform-only, using saved artifacts)

def preprocess_for_prediction(df_raw: pd.DataFrame, artifacts: dict) -> tuple[pd.DataFrame, dict]:
    
    df = df_raw.copy()
    target_col = artifacts["target_col"]
    report = {
        "dropped_target": False,
        "missing_filled": {},
        "unseen_categories": {},
        "missing_raw_columns": [],
        "extra_raw_columns": [],
    }

    if target_col in df.columns:
        df = df.drop(columns=[target_col])
        report["dropped_target"] = True

    try:
        # Missing values: dynamic median/mode imputation (columns are never
        # dropped here — the model expects a fixed schema).
        for col in df.columns:
            miss_frac = df[col].isna().mean()
            if miss_frac > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = df[col].median()
                else:
                    mode = df[col].mode()
                    fill_value = mode.iloc[0] if not mode.empty else "missing"
                df[col] = df[col].fillna(fill_value)
                report["missing_filled"][col] = round(miss_frac * 100, 2)

        encoders = artifacts.get("encoders", {})

        # Frequency-encoded (high-cardinality) columns
        for col, freq in encoders.get("__freq_maps__", {}).items():
            if col in df.columns:
                df[col] = df[col].map(freq).fillna(0)

        # Label-encoded (binary) columns — map unseen categories to a known class
        for col, encoder in encoders.items():
            if col in ("__target__", "__freq_maps__") or col not in df.columns:
                continue
            known = set(encoder.classes_)
            values = df[col].astype(str)
            unseen_mask = ~values.isin(known)
            if unseen_mask.any():
                report["unseen_categories"][col] = int(unseen_mask.sum())
                values = values.where(~unseen_mask, encoder.classes_[0])
            df[col] = encoder.transform(values)

        # Remaining categorical columns -> one-hot (same scheme used at training)
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols)
            bool_cols = df.select_dtypes(include="bool").columns
            df[bool_cols] = df[bool_cols].astype(int)

        # Align to the exact feature columns/order the model was trained on
        feature_columns = artifacts["feature_columns"]
        report["missing_raw_columns"] = [c for c in feature_columns if c not in df.columns]
        report["extra_raw_columns"] = [c for c in df.columns if c not in feature_columns]
        df = df.reindex(columns=feature_columns, fill_value=0)

        # Scale continuous columns with the scaler fit during training
        scaler = artifacts.get("scaler")
        scaled_columns = [c for c in (artifacts.get("scaled_columns") or []) if c in df.columns]
        if scaler is not None and scaled_columns:
            df[scaled_columns] = scaler.transform(df[scaled_columns])

        logger.info(f"Preprocessed unseen data for prediction -> {df.shape}")
    except Exception as e:
        logger.error(f"Error in preprocess_for_prediction: {e}")
        raise

    return df, report



#  GENERATE PREDICTIONS

def generate_predictions(df_raw: pd.DataFrame, model_bundle: dict, artifacts: dict) -> tuple[pd.DataFrame, dict]:
    
    X, prep_report = preprocess_for_prediction(df_raw, artifacts)
    model = model_bundle["model"]
    target_col = artifacts["target_col"]

    raw_predictions = model.predict(X)
    target_encoder = artifacts.get("encoders", {}).get("__target__")
    predictions = (
        target_encoder.inverse_transform(raw_predictions) if target_encoder is not None else raw_predictions
    )

    results = df_raw.reset_index(drop=True).copy()
    results[f"predicted_{target_col}"] = predictions

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = (
            target_encoder.inverse_transform(model.classes_) if target_encoder is not None else model.classes_
        )
        for cls, col_values in zip(classes, proba.T):
            results[f"probability_{cls}"] = col_values
        results["prediction_confidence"] = proba.max(axis=1)

    logger.info(
        f"Generated predictions for {len(results)} rows using '{model_bundle['model_name']}'."
    )
    return results, prep_report



#  SAVE / EXPORT PREDICTIONS

def save_predictions(results: pd.DataFrame, reports_dir: str = "../reports",
                      filename: str = "predictions.csv") -> str:
    """Save prediction results to CSV and return the file path."""
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, filename)
    results.to_csv(path, index=False)
    logger.info(f"Saved predictions -> {path}")
    return path
