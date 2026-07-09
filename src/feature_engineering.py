"""Feature Engineering Module."""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

from utilities import setup_logger

logger = setup_logger(__name__)



# FEATURE ENGINEERING

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

            # Constant column
            if n_unique <= 1:
                df = df.drop(columns=[col])
                report["dropped"][col] = "constant"
                continue

            # ID-like column
            if n_unique / n_rows >= id_threshold:
                df = df.drop(columns=[col])
                report["dropped"][col] = "id-like"
                continue

            # Text columns: datetime detection, then high-cardinality
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
                    report["dropped"][col] = f"high-cardinality ({n_unique} unique)"
                    continue

        logger.info(f"Feature engineering: dropped {len(report['dropped'])} columns, "
                    f"expanded {len(report['datetime_expanded'])} datetime -> {df.shape}")
    except Exception as e:
        logger.error(f"Error in engineer_features: {e}")

    report["final_shape"] = df.shape
    return df, report



#  ENCODING

def encode_features(
    df: pd.DataFrame,
    target_col: str,
    low_card_max: int = 15,
) -> tuple[pd.DataFrame, dict, dict]:
   
    df = df.copy()
    report = {"target_encoded": None, "binary_encoded": [], "onehot_encoded": []}
    encoders = {}

    try:
        # Target variable encoding
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col])
            encoders["__target__"] = le
            report["target_encoded"] = dict(zip(le.classes_, le.transform(le.classes_)))

        # Split categorical columns 
        cat_cols = [
            c for c in df.columns
            if c != target_col and not pd.api.types.is_numeric_dtype(df[c])
        ]
        binary_cols = [c for c in cat_cols if df[c].nunique() == 2]
        onehot_cols = [c for c in cat_cols if 2 < df[c].nunique() <= low_card_max]
        highcard_cols = [c for c in cat_cols if df[c].nunique() > low_card_max]

        # High-cardinality -> frequency encoding
        freq_maps = {}
        for col in highcard_cols:
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq).fillna(0)
            freq_maps[col] = freq.to_dict()

        if freq_maps:
            encoders["__freq_maps__"] = freq_maps
            report["freq_encoded"] = highcard_cols

        # Binary -> label encoding
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

        logger.info(f"Encoding: {len(binary_cols)} binary, {len(onehot_cols)} one-hot, "
                    f"{len(highcard_cols)} freq -> {df.shape}")
    except Exception as e:
        logger.error(f"Error in encode_features: {e}")

    report["final_shape"] = df.shape
    return df, report, encoders



#  TRAIN / TEST SPLIT

def split_dataset(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Auto-decide stratification
    n_unique = y.nunique()
    is_classification = (not pd.api.types.is_numeric_dtype(y)) or (n_unique <= 20)
    min_class_count = y.value_counts().min() if is_classification else 0

    if is_classification and min_class_count >= 2:
        stratify = y
        strat_note = "stratified"
    else:
        stratify = None
        if is_classification and min_class_count < 2:
            strat_note = "NOT stratified (a class has <2 rows)"
        else:
            strat_note = "NOT stratified (continuous target)"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=random_state
    )

    logger.info(f"Split ({strat_note}): X_train {X_train.shape}, X_test {X_test.shape}")
    if verbose:
        print(f"Split: {strat_note}")
        print("X_train:", X_train.shape, "| X_test:", X_test.shape)
        if is_classification:
            print("Train ratio:", y_train.value_counts(normalize=True).round(3).to_dict())
            print("Test  ratio:", y_test.value_counts(normalize=True).round(3).to_dict())

    return X_train, X_test, y_train, y_test



# SCALING (train/test split-aware)

def scale_features_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    min_unique: int = 15,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, list]:
    
    X_train = X_train.copy()
    X_test = X_test.copy()

    continuous_cols = [
        c for c in X_train.columns
        if pd.api.types.is_numeric_dtype(X_train[c]) and X_train[c].nunique() >= min_unique
    ]

    if not continuous_cols:
        logger.info("Scaling skipped: no continuous columns.")
        if verbose:
            print("No continuous columns to scale — skipped.")
        return X_train, X_test, None, []

    scaler = StandardScaler()
    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

    logger.info(f"Scaling (fit on train): {len(continuous_cols)} continuous columns")
    if verbose:
        print("Scaled (fit on train):", continuous_cols)
        print("X_train:", X_train.shape, "| X_test:", X_test.shape)

    return X_train, X_test, scaler, continuous_cols



# SCALING (single-DataFrame variant) — kept from notebook

def scale_features(
    df: pd.DataFrame,
    target_col: str,
    min_unique: int = 15,
) -> tuple[pd.DataFrame, dict, StandardScaler]:
    
    df = df.copy()

    continuous_cols = [
        c for c in df.columns
        if c != target_col
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique() >= min_unique
    ]

    report = {"scaled_columns": continuous_cols, "skipped": []}
    report["skipped"] = [c for c in df.columns if c != target_col and c not in continuous_cols]

    scaler = None
    if continuous_cols:
        scaler = StandardScaler()
        df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    report["final_shape"] = df.shape
    return df, report, scaler



#  FEATURE SELECTION (Mutual Information)

def select_features_mi(
    X_train_fs: pd.DataFrame,
    X_test_fs: pd.DataFrame,
    y_train: pd.Series,
    mi_threshold: float = 0.0,
    min_features: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if X_train_fs.shape[1] == 0:
        logger.info("MI selection skipped: no columns.")
        return X_train_fs, X_test_fs, {
            "selected_features": [], "fallback_used": False,
            "n_informative": 0, "dropped": [],
        }

    try:
        # Discrete features mask (binary/dummy columns)
        discrete_mask = [X_train_fs[c].nunique() <= 2 for c in X_train_fs.columns]

        mi_scores = mutual_info_classif(
            X_train_fs, y_train,
            discrete_features=discrete_mask,
            random_state=random_state,
        )

        mi_table = pd.DataFrame({
            "feature": X_train_fs.columns, "mi_score": mi_scores
        }).sort_values("mi_score", ascending=False).reset_index(drop=True)

        informative = mi_table.loc[mi_table["mi_score"] > mi_threshold, "feature"].tolist()

        # Fallback: too few survive -> keep all
        if len(informative) >= min_features:
            selected_features, fallback_used = informative, False
        else:
            selected_features, fallback_used = X_train_fs.columns.tolist(), True

        dropped = [c for c in X_train_fs.columns if c not in selected_features]
        mi_report = {
            "mi_threshold": mi_threshold,
            "min_features": min_features,
            "n_informative": len(informative),
            "selected_features": selected_features,
            "dropped": dropped,
            "fallback_used": fallback_used,
        }

        X_train_fs = X_train_fs[selected_features]
        X_test_fs = X_test_fs[selected_features]

        logger.info(f"MI selection: {len(informative)} informative (MI>{mi_threshold}), "
                    f"fallback={fallback_used} -> X_train {X_train_fs.shape}")
        if verbose:
            print("MI table (top 10):")
            print(mi_table.head(10).to_string(index=False))
            print(f"\nInformative (MI>{mi_threshold}): {len(informative)} | "
                  f"fallback (kept all, <{min_features})? {fallback_used}")
            print("X_train_fs:", X_train_fs.shape, "| X_test_fs:", X_test_fs.shape)

        return X_train_fs, X_test_fs, mi_report

    except Exception as e:
        logger.error(f"Error in select_features_mi: {e}")
        return X_train_fs, X_test_fs, {
            "selected_features": X_train_fs.columns.tolist(), "fallback_used": True,
            "n_informative": 0, "dropped": [],
        }



#  ORCHESTRATOR

def run_pipeline(
    df: pd.DataFrame,
    target_col: str,
    processed_dir: str = "../data/processed",
    models_dir: str = "../models",
    test_size: float = 0.2,
    random_state: int = 42,
    min_unique: int = 15,
    mi_threshold: float = 0.0,
    min_features: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict, dict]:
    
    logger.info(f"Pipeline started. Input shape: {df.shape}, target: {target_col}")
    reports = {}

    # engineer -> encode
    df, reports["engineering"] = engineer_features(df, target_col)
    df, reports["encoding"], encoders = encode_features(df, target_col)

    # split BEFORE scaling (leakage-safe)
    X_train, X_test, y_train, y_test = split_dataset(
        df, target_col, test_size=test_size, random_state=random_state, verbose=False
    )

    # scale (fit on train) -> MI selection
    X_train, X_test, scaler, scaled_cols = scale_features_split(
        X_train, X_test, min_unique=min_unique, verbose=False
    )
    X_train_fs, X_test_fs, reports["mutual_info"] = select_features_mi(
        X_train, X_test, y_train,
        mi_threshold=mi_threshold, min_features=min_features,
        random_state=random_state, verbose=False,
    )

    # artifacts bundle
    artifacts = {
        "encoders": encoders,
        "scaler": scaler,
        "scaled_columns": scaled_cols,
        "feature_columns": list(X_train_fs.columns),  # final selected order
        "target_col": target_col,
    }

    # persist splits + artifacts + reports
    try:
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        X_train_fs.to_pickle(os.path.join(processed_dir, "X_train.pkl"))
        X_test_fs.to_pickle(os.path.join(processed_dir, "X_test.pkl"))
        y_train.to_pickle(os.path.join(processed_dir, "y_train.pkl"))
        y_test.to_pickle(os.path.join(processed_dir, "y_test.pkl"))

        
        joblib.dump(artifacts, os.path.join(models_dir, f"{target_col}_preprocessing_artifacts.pkl"))
        joblib.dump(reports, os.path.join(models_dir, f"{target_col}_feature_selection_report.pkl"))


        logger.info(f"Saved splits -> {processed_dir}")
        logger.info(f"Saved artifacts -> {models_dir}/preprocessing_artifacts.pkl "
                    f"({len(artifacts['feature_columns'])} features)")
    except Exception as e:
        logger.error(f"Error saving pipeline outputs: {e}")

    logger.info(f"Pipeline complete. Final: X_train {X_train_fs.shape}, X_test {X_test_fs.shape}")
    return X_train_fs, X_test_fs, y_train, y_test, artifacts, reports