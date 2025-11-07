"""Hyperparameter tuning workflows for customer purchase prediction models."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from models.mlflow_utils import (
    log_confusion_matrix,
    log_dict_as_artifact,
    log_roc_curve,
    set_experiment,
)


EXPERIMENT_NAME = "Customer Purchase Tuning"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "customer_data.csv"
BEST_MODEL_PATH = Path(__file__).resolve().parent / "last_best.json"
REGISTERED_MODEL_NAME = "best-customer-classifier"


def _load_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and target series from CSV."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Generate it via `python data/make_data.py` first."
        )

    df = pd.read_csv(path)
    feature_cols = [col for col in df.columns if col.startswith("f")]
    if "target" not in df.columns:
        raise ValueError("Dataset must contain a 'target' column.")

    return df[feature_cols], df["target"]


def _load_baseline() -> Dict[str, Any]:
    """Load metadata about the best known model from disk."""

    if BEST_MODEL_PATH.exists():
        with BEST_MODEL_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            data.setdefault("metric", "f1")
            return data
    return {"metric": "f1", "score": -np.inf, "run_id": None, "model_version": None}


def _save_best(payload: Dict[str, Any]) -> None:
    """Persist details of the best model."""

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BEST_MODEL_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _log_model_version(pipeline: Pipeline, run_id: str) -> str | None:
    """Log and register a model version, returning its registry version."""

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    client = MlflowClient()
    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}' and run_id='{run_id}'"
    )
    if not versions:
        return None
    latest_version = max(int(v.version) for v in versions)
    return str(latest_version)


def _log_top_results(cv_results: Dict[str, Iterable[Any]], artifact_name: str, top_k: int = 5) -> None:
    """Log the top-k hyperparameter configurations as a CSV artifact."""

    results_df = pd.DataFrame(cv_results)
    results_df = results_df.sort_values("rank_test_score").head(top_k)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / artifact_name
        results_df.to_csv(temp_path, index=False)
        mlflow.log_artifact(str(temp_path), artifact_path="reports")


def _evaluate_model(pipeline: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series) -> Dict[str, float]:
    """Compute evaluation metrics on the validation split."""

    y_pred = pipeline.predict(X_valid)
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_valid)[:, 1]
    else:
        scores = pipeline.decision_function(X_valid)
        y_proba = 1 / (1 + np.exp(-scores))

    return {
        "accuracy": accuracy_score(y_valid, y_pred),
        "f1": f1_score(y_valid, y_pred),
        "roc_auc": roc_auc_score(y_valid, y_proba),
    }


def _log_evaluation_artifacts(
    pipeline: Pipeline,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    artifact_prefix: str,
) -> None:
    """Log confusion matrix and ROC curve artifacts for the provided model."""

    y_pred = pipeline.predict(X_valid)
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_valid)[:, 1]
    else:
        scores = pipeline.decision_function(X_valid)
        y_proba = 1 / (1 + np.exp(-scores))

    log_confusion_matrix(
        y_valid,
        y_pred,
        artifact_path=f"plots/{artifact_prefix}_confusion_matrix.png",
    )
    log_roc_curve(
        y_valid,
        y_proba,
        artifact_path=f"plots/{artifact_prefix}_roc_curve.png",
    )


def run_tuning() -> None:
    """Conduct sequential hyperparameter tuning rounds and update MLflow registry."""

    X, y = _load_dataset(DATA_PATH)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    baseline = _load_baseline()
    current_best_score = baseline.get("score", -np.inf)

    set_experiment(EXPERIMENT_NAME)

    tuning_rounds = [
        {
            "name": "random_forest_tuning",
            "estimator": Pipeline(
                steps=[
                    ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
                ]
            ),
            "param_distributions": {
                "model__n_estimators": [100, 200, 300, 400, 500],
                "model__max_depth": [None, 10, 20, 30, 40],
                "model__max_features": ["sqrt", "log2", None],
            },
            "rationale": "Tăng n_estimators và max_depth để học quan hệ phi tuyến sâu hơn.",
            "artifact_name": "rf_top_configs.csv",
        },
        {
            "name": "xgboost_tuning",
            "estimator": Pipeline(
                steps=[
                    (
                        "model",
                        XGBClassifier(
                            eval_metric="logloss",
                            n_jobs=-1,
                            random_state=42,
                            use_label_encoder=False,
                        ),
                    )
                ]
            ),
            "param_distributions": {
                "model__n_estimators": [200, 300, 400, 500],
                "model__max_depth": [3, 4, 5, 6, 7],
                "model__learning_rate": [0.03, 0.05, 0.07, 0.1],
                "model__subsample": [0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
            },
            "rationale": "Điều chỉnh XGBoost để cân bằng bias-variance và khai thác tương tác đặc trưng.",
            "artifact_name": "xgb_top_configs.csv",
        },
    ]

    for round_cfg in tuning_rounds:
        with mlflow.start_run(run_name=round_cfg["name"]) as run:
            mlflow.set_tag("phase", "tuning")
            mlflow.set_tag("rationale", round_cfg["rationale"])
            mlflow.log_metric("baseline_f1", current_best_score if np.isfinite(current_best_score) else -1.0)

            log_dict_as_artifact(round_cfg["param_distributions"], f"configs/{round_cfg['name']}_search_space.json")

            search = RandomizedSearchCV(
                estimator=round_cfg["estimator"],
                param_distributions=round_cfg["param_distributions"],
                n_iter=min(20, np.prod([len(v) for v in round_cfg["param_distributions"].values()])),
                scoring="f1",
                cv=5,
                n_jobs=-1,
                random_state=42,
                refit=True,
                verbose=0,
            )

            search.fit(X_train, y_train)

            _log_top_results(search.cv_results_, round_cfg["artifact_name"], top_k=5)

            best_estimator: Pipeline = search.best_estimator_
            best_params = search.best_params_
            # Flatten parameter keys for MLflow logging
            flattened_params = {k.replace("model__", ""): v for k, v in best_params.items()}
            mlflow.log_params(flattened_params)
            mlflow.log_metric("cv_best_f1", search.best_score_)

            evaluation_metrics = _evaluate_model(best_estimator, X_valid, y_valid)
            mlflow.log_metrics({f"valid_{k}": v for k, v in evaluation_metrics.items()})

            delta_f1 = evaluation_metrics["f1"] - current_best_score
            mlflow.log_metric("valid_f1_improvement", delta_f1)

            _log_evaluation_artifacts(
                pipeline=best_estimator,
                X_valid=X_valid,
                y_valid=y_valid,
                artifact_prefix=round_cfg["name"],
            )

            comparison_payload = {
                "baseline_f1": current_best_score,
                "tuned_valid_f1": evaluation_metrics["f1"],
                "cv_best_f1": search.best_score_,
                "delta_f1": delta_f1,
            }
            log_dict_as_artifact(comparison_payload, f"reports/{round_cfg['name']}_comparison.json")

            improved = evaluation_metrics["f1"] > current_best_score

            model_version = None
            if improved:
                model_version = _log_model_version(best_estimator, run.info.run_id)
                current_best_score = evaluation_metrics["f1"]
                _save_best(
                    {
                        "metric": "f1",
                        "score": evaluation_metrics["f1"],
                        "run_id": run.info.run_id,
                        "model_version": model_version,
                    }
                )

            summary_payload = {
                "model": round_cfg["name"],
                "best_params": flattened_params,
                "validation_metrics": evaluation_metrics,
                "cv_best_f1": search.best_score_,
                "baseline_f1": baseline.get("score", -1.0),
                "improved_over_baseline": improved,
                "model_version": model_version,
            }
            log_dict_as_artifact(summary_payload, f"reports/{round_cfg['name']}_summary.json")

            if improved:
                baseline = {
                    "metric": "f1",
                    "score": evaluation_metrics["f1"],
                    "run_id": run.info.run_id,
                    "model_version": model_version,
                }


if __name__ == "__main__":
    run_tuning()

