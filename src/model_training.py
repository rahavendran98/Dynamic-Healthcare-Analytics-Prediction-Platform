"""Model Training Module."""


import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)

from utilities import setup_logger

logger = setup_logger(__name__)


IMBALANCE_THRESHOLD = 0.40


# CLASS IMBALANCE DETECTION

def detect_imbalance(y: pd.Series, threshold: float = IMBALANCE_THRESHOLD) -> dict:

    counts = y.value_counts()
    props = y.value_counts(normalize=True)
    minority_share = props.min()
    is_imbalanced = bool(minority_share < threshold)

    report = {
        "class_counts": counts.to_dict(),
        "class_proportions": props.round(3).to_dict(),
        "n_classes": len(counts),
        "minority_share": round(minority_share, 3),
        "is_imbalanced": is_imbalanced,
        "class_weight": "balanced" if is_imbalanced else None,
    }

    logger.info(f"Imbalance check: minority_share={report['minority_share']}, "
                f"is_imbalanced={is_imbalanced}, class_weight={report['class_weight']}")
    return report



#  SAVE IMBALANCE REPORT

def save_imbalance_report(
    imbalance_report: dict,
    output_path: str = "reports/imbalance_report.json",
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    serializable = {
        "class_counts": {str(k): int(v) for k, v in imbalance_report["class_counts"].items()},
        "class_proportions": {str(k): v for k, v in imbalance_report["class_proportions"].items()},
        "n_classes": imbalance_report["n_classes"],
        "minority_share": imbalance_report["minority_share"],
        "is_imbalanced": imbalance_report["is_imbalanced"],
        "class_weight": imbalance_report["class_weight"],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"Imbalance report saved -> {output_path}")
    return output_path



# 2. MODEL BUILDING

def build_models(class_weight: str = None, random_state: int = 42) -> dict:

    return {
        "LogisticRegression": LogisticRegression(
            class_weight=class_weight, random_state=random_state, max_iter=1000
        ),
        "DecisionTree": DecisionTreeClassifier(
            class_weight=class_weight, random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            class_weight=class_weight, random_state=random_state
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=random_state
        ),
    }



# TRAINING

def train_models(models: dict, X_train: pd.DataFrame, y_train: pd.Series,
                  imb_report: dict) -> tuple[dict, dict]:

    gb_sample_weight = None
    if imb_report["class_weight"] == "balanced":
        gb_sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    trained_models = {}
    training_report = {}

    for name, model in models.items():
        try:
            if name == "GradientBoosting" and gb_sample_weight is not None:
                model.fit(X_train, y_train, sample_weight=gb_sample_weight)
                weight_status = "sample_weight=balanced"
            else:
                model.fit(X_train, y_train)
                weight_status = f"class_weight={imb_report['class_weight']}" \
                                if name != "GradientBoosting" else "none"
            trained_models[name] = model
            training_report[name] = {"trained": True, "weighting": weight_status}
        except Exception as e:
            training_report[name] = {"trained": False, "error": str(e)}
            logger.error(f"Error training '{name}': {e}")

    logger.info(f"Training complete: {list(trained_models.keys())}")
    return trained_models, training_report



# EVALUATION

def evaluate_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series,
                     target: str = None, report_name: str = "model_comparison",
                     reports_dir: str = "../reports") -> tuple[pd.DataFrame, dict]:

    results = []
    confusion_matrices = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        roc_auc = np.nan
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            try:
                if len(np.unique(y_test)) == 2:
                    roc_auc = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    roc_auc = roc_auc_score(
                        y_test, y_prob, multi_class="ovr", average="weighted",
                    )
            except Exception as e:
                logger.error(f"ROC-AUC failed for '{name}': {e}")
                roc_auc = np.nan

        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="F1 Score", ascending=False).reset_index(drop=True)

    if target is not None:
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, f"{target}_{report_name}.csv")
        results_df.to_csv(report_path, index=False)
        logger.info(f"Saved comparison report -> {report_path}")

    return results_df, confusion_matrices



#  HYPERPARAMETER TUNING 

def tune_models(results_df: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series,
                 imb_report: dict, top_n: int = 2, cv_folds: int = 5) -> tuple[dict, dict]:

    is_binary = (y_train.nunique() == 2)
    scoring = "f1" if is_binary else "f1_weighted"
    cw = "balanced" if imb_report["is_imbalanced"] else None

    sample_weight = None
    if imb_report["is_imbalanced"]:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    grids = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, class_weight=cw),
            {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs", "liblinear"]},
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=42, class_weight=cw),
            {"max_depth": [5, 10, None], "min_samples_split": [2, 10]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42, class_weight=cw),
            {"n_estimators": [100, 200], "max_depth": [10, None],
             "min_samples_split": [2, 5]},
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
             "max_depth": [3, 5]},
        ),
    }

    candidates = results_df["Model"].head(top_n).tolist()
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    tuned_models = {}
    best_params = {}

    try:
        for name in candidates:
            estimator, param_grid = grids[name]
            gs = GridSearchCV(estimator, param_grid, scoring=scoring, cv=skf, n_jobs=-1)
            if name == "GradientBoosting" and sample_weight is not None:
                gs.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                gs.fit(X_train, y_train)
            tuned_models[name] = gs.best_estimator_
            best_params[name] = gs.best_params_
            logger.info(f"Tuned '{name}': best_params={gs.best_params_}")
    except Exception as e:
        logger.error(f"Error in tune_models: {e}")

    return tuned_models, best_params



#  CONFUSION MATRIX VISUALIZATION (baseline vs tuned)

def get_class_labels(artifacts: dict, target_col: str) -> list:
    """Get class names from the saved target encoder, else fall back to numbers."""
    target_encoder = artifacts.get("encoders", {}).get("__target__")
    if target_encoder is not None:
        return list(target_encoder.classes_)
    return None


def plot_baseline_vs_tuned(cm_baseline: dict, cm_tuned: dict, labels: list = None,
                            prefix: str = "model", reports_dir: str = "../reports") -> None:
    """Plot baseline vs tuned confusion matrices side by side and save each."""
    os.makedirs(reports_dir, exist_ok=True)

    for name in [m for m in cm_tuned if m in cm_baseline]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, stage, cm_dict in zip(axes, ["Baseline", "Tuned"], [cm_baseline, cm_tuned]):
            ConfusionMatrixDisplay(cm_dict[name], display_labels=labels).plot(
                ax=ax, cmap="Blues", colorbar=False
            )
            ax.set_title(f"{name} - {stage}")

        fig.tight_layout()
        path = os.path.join(reports_dir, f"{prefix}_{name}_confusion_matrix.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved confusion matrix -> {path}")



#  BEST MODEL SELECTION

def save_best_model(results_df: pd.DataFrame, models_dict: dict, feature_columns: list,
                     target_col: str, models_dir: str, best_params: dict = None,
                     metric: str = "F1 Score") -> tuple[dict, str]:

    best_row = results_df.sort_values(metric, ascending=False).iloc[0]
    best_name = best_row["Model"]

    metrics = {
        col: (None if pd.isna(best_row[col]) else round(float(best_row[col]), 4))
        for col in results_df.columns if col != "Model"
    }

    bundle = {
        "model": models_dict[best_name],
        "model_name": best_name,
        "feature_columns": list(feature_columns),
        "target_col": target_col,
        "best_params": (best_params or {}).get(best_name),
        "metrics": metrics,
    }

    os.makedirs(models_dir, exist_ok=True)
    
    path = os.path.join(models_dir, f"{target_col}_best_model.pkl")

    joblib.dump(bundle, path)

    logger.info(f"Best model: {best_name} | metrics={metrics} | saved -> {path}")
    return bundle, path



#  ORCHESTRATOR

def run_training_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_columns: list,
    artifacts: dict,
    target_col: str,
    models_dir: str = "../models",
    reports_dir: str = "../reports",
    top_n: int = 2,
    random_state: int = 42,
) -> dict:
    
    logger.info(f"Model training pipeline started. Target: {target_col}")

    imbalance_report = detect_imbalance(y_train)
    save_imbalance_report(imbalance_report, os.path.join(reports_dir, f"{target_col}_imbalance_report.json"))

    models = build_models(class_weight=imbalance_report["class_weight"], random_state=random_state)
    trained_models, training_report = train_models(models, X_train, y_train, imbalance_report)

    results_df, confusion_matrices = evaluate_models(
        trained_models, X_test, y_test, target=target_col,
        report_name="model_comparison", reports_dir=reports_dir,
    )

    tuned_models, best_params = tune_models(
        results_df, X_train, y_train, imbalance_report, top_n=top_n,
    )
    tuned_results, confusion_matrices_tuned = evaluate_models(
        tuned_models, X_test, y_test, target=target_col,
        report_name="tuned_model_comparison", reports_dir=reports_dir,
    )

    class_labels = get_class_labels(artifacts, target_col)
    plot_baseline_vs_tuned(
        confusion_matrices, confusion_matrices_tuned,
        labels=class_labels, prefix=target_col, reports_dir=reports_dir,
    )

    bundle, best_model_path = save_best_model(
        tuned_results, tuned_models, feature_columns, target_col, models_dir, best_params,
    )

    logger.info("Model training pipeline complete.")
    return {
        "imbalance_report": imbalance_report,
        "training_report": training_report,
        "results_df": results_df,
        "tuned_results": tuned_results,
        "best_params": best_params,
        "confusion_matrices": confusion_matrices,
        "confusion_matrices_tuned": confusion_matrices_tuned,
        "best_model_bundle": bundle,
        "best_model_path": best_model_path,
    }
