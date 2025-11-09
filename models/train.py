"""Training pipeline for the customer purchase prediction task."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from models.mlflow_utils import (
    log_confusion_matrix,
    log_dict_as_artifact,
    log_feature_importance,
    log_roc_curve,
    set_experiment,
)

logger = logging.getLogger(__name__)


EXPERIMENT_NAME = "Customer Purchase Prediction"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "customer_data.csv"
BEST_MODEL_PATH = Path(__file__).resolve().parent / "last_best.json"
REGISTERED_MODEL_NAME = "best-customer-classifier"


def _load_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the customer dataset and separate features/target."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    feature_cols = [col for col in df.columns if col.startswith("f")]
    if "target" not in df.columns:
        raise ValueError("Dataset must contain 'target' column.")

    features = df[feature_cols]
    target = df["target"]
    logger.info("Loaded dataset with shape=%s from %s", df.shape, path)
    return features, target


def _prepare_models(random_state: int = 42) -> List[Dict[str, Any]]:
    """Define candidate models and their metadata."""

    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
        ]
    )

    xgboost_classifier = Pipeline(
        steps=[
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1,
                    use_label_encoder=False,
                ),
            )
        ]
    )

    return [
        {
            "name": "logistic_regression",
            "estimator": logistic_regression,
            "params": {
                "penalty": "l2",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
            },
            "note": "Baseline tuyến tính với chuẩn hóa đặc trưng.",
        },
        {
            "name": "random_forest",
            "estimator": random_forest,
            "params": {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_leaf": 2,
            },
            "note": "Mô hình rừng ngẫu nhiên để nắm bắt quan hệ phi tuyến.",
        },
        {
            "name": "xgboost",
            "estimator": xgboost_classifier,
            "params": {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 5,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
            },
            "note": "Gradient boosting cho hiệu năng cao với cấu hình ổn định.",
        },
    ]


def _log_model(
    pipeline: Pipeline,
    run_id: str,
    registered_model_name: str,
) -> str | None:
    """Log and register the trained model, returning the model version if available."""

    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name=registered_model_name,
    )

    client = MlflowClient()
    versions = client.search_model_versions(
        f"name='{registered_model_name}' and run_id='{run_id}'"
    )
    if not versions:
        return None

    # In case multiple versions match, take the latest numeric version.
    latest_version = max(int(v.version) for v in versions)
    return str(latest_version)


def run_training() -> None:
    """Execute training for multiple models and log results to MLflow."""

    features, target = _load_dataset(DATA_PATH)
    feature_names = features.columns.tolist()

    X_train, X_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    models_config = _prepare_models()
    set_experiment(EXPERIMENT_NAME)
    logger.info("Starting training pipeline with %d candidate models.", len(models_config))
    best_result: Dict[str, Any] | None = None
    overall_summary: List[Dict[str, Any]] = []

    for config in models_config:
        estimator: Pipeline = config["estimator"]
        with mlflow.start_run(run_name=config["name"]) as run:
            logger.info("Training model=%s", config["name"])
            mlflow.set_tag("model_name", config["name"])
            mlflow.set_tag("note", config["note"])
            mlflow.log_params(config["params"])

            estimator.fit(X_train, y_train)
            logger.info("Completed fit for model=%s", config["name"])

            y_pred = estimator.predict(X_valid)
            if hasattr(estimator, "predict_proba"):
                y_proba = estimator.predict_proba(X_valid)[:, 1]
            else:
                # Fall back to decision_function when predict_proba unavailable.
                y_scores = estimator.decision_function(X_valid)
                y_proba = 1 / (1 + np.exp(-y_scores))

            metrics = {
                "accuracy": accuracy_score(y_valid, y_pred),
                "f1": f1_score(y_valid, y_pred),
                "roc_auc": roc_auc_score(y_valid, y_proba),
            }

            mlflow.log_metrics(metrics)
            logger.info(
                "Validation metrics for %s -> accuracy=%.4f, f1=%.4f, roc_auc=%.4f",
                config["name"],
                metrics["accuracy"],
                metrics["f1"],
                metrics["roc_auc"],
            )

            log_confusion_matrix(y_valid, y_pred, artifact_path="plots/confusion_matrix.png")
            log_roc_curve(y_valid, y_proba, artifact_path="plots/roc_curve.png")

            core_model = estimator.named_steps.get("model") if hasattr(estimator, "named_steps") else estimator
            if hasattr(core_model, "feature_importances_"):
                log_feature_importance(
                    feature_names,
                    core_model.feature_importances_,
                    artifact_path=f"plots/{config['name']}_feature_importance.png",
                    top_k=10,
                )

            summary_payload = {
                "model_name": config["name"],
                "metrics": metrics,
                "params": config["params"],
            }
            log_dict_as_artifact(summary_payload, f"reports/{config['name']}_summary.json")

            model_version = _log_model(
                pipeline=estimator,
                run_id=run.info.run_id,
                registered_model_name=REGISTERED_MODEL_NAME,
            )

            run_summary = {
                "run_id": run.info.run_id,
                "model_name": config["name"],
                "metrics": metrics,
                "params": config["params"],
                "model_version": model_version,
            }
            overall_summary.append(run_summary)

            if (
                best_result is None
                or metrics["f1"] > best_result["metrics"]["f1"]
                or (
                    np.isclose(metrics["f1"], best_result["metrics"]["f1"])
                    and metrics["roc_auc"] > best_result["metrics"]["roc_auc"]
                )
            ):
                best_result = run_summary

    if best_result is None:
        raise RuntimeError("Training did not produce any results.")

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Best model identified: %s with F1=%.4f (run_id=%s, version=%s)",
        best_result["model_name"],
        best_result["metrics"]["f1"],
        best_result["run_id"],
        best_result["model_version"],
    )
    with BEST_MODEL_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metric": "f1",
                "score": best_result["metrics"]["f1"],
                "run_id": best_result["run_id"],
                "model_version": best_result["model_version"],
            },
            handle,
            indent=2,
        )

    print(
        "Best model registered",
        json.dumps(
            {
                "model_name": best_result["model_name"],
                "metric": "f1",
                "score": best_result["metrics"]["f1"],
                "roc_auc": best_result["metrics"]["roc_auc"],
                "run_id": best_result["run_id"],
                "model_version": best_result["model_version"],
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    run_training()