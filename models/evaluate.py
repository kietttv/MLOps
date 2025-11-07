"""Evaluate MLflow runs, manage model registry stages, and persist best metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import pandas as pd


DEFAULT_EXPERIMENT_NAME = "Customer Purchase Prediction"
REGISTERED_MODEL_NAME = "best-customer-classifier"
BEST_MODEL_PATH = Path(__file__).resolve().parent / "last_best.json"


def _load_last_best() -> Dict[str, Any]:
    """Load metadata for the current best model from disk."""

    if BEST_MODEL_PATH.exists():
        with BEST_MODEL_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def _save_last_best(payload: Dict[str, Any]) -> None:
    """Persist metadata about the latest best model."""

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BEST_MODEL_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _ensure_model_version(client: MlflowClient, run_id: str) -> str:
    """Ensure the run has a registered model version and return the version string."""

    search_filter = f"name='{REGISTERED_MODEL_NAME}' and run_id='{run_id}'"
    versions = client.search_model_versions(search_filter)
    if versions:
        latest_version = max(int(v.version) for v in versions)
        return str(latest_version)

    model_uri = f"runs:/{run_id}/model"
    model_info = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    return str(model_info.version)


def _transition_to_production(client: MlflowClient, version: str) -> None:
    """Transition the given model version through staging into production."""

    try:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=version,
            stage="Staging",
        )
    except MlflowException:
        # Stage may already be set; continue to production transition.
        pass

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )


def evaluate_best_run(metric: str, experiment_name: str) -> Dict[str, Any]:
    """Identify the best run by metric, register/transition its model, and update metadata."""

    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found.")

    order_by = [f"metrics.{metric} DESC"]
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        order_by=order_by,
        max_results=1,
    )

    if runs_df.empty:
        raise RuntimeError(f"No active runs found for experiment '{experiment_name}'.")

    top_run = runs_df.iloc[0]
    top_run_id = top_run["run_id"]
    metric_column = f"metrics.{metric}"
    top_metric_value = top_run.get(metric_column)
    if pd.isna(top_metric_value):
        raise RuntimeError(f"Top run does not have metric '{metric}'.")

    last_best = _load_last_best()
    last_best_run_id = last_best.get("run_id")

    version = _ensure_model_version(client, top_run_id)
    _transition_to_production(client, version)

    best_payload = {
        "metric": metric,
        "score": float(top_metric_value),
        "run_id": top_run_id,
        "model_version": version,
    }
    _save_last_best(best_payload)

    transition_info = {
        "model_name": REGISTERED_MODEL_NAME,
        "version": version,
        "stage": "Production",
        "run_id": top_run_id,
        "metric": metric,
        "score": float(top_metric_value),
        "previous_best_run_id": last_best_run_id,
    }

    print(json.dumps(transition_info, indent=2))
    return transition_info


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate MLflow runs and manage model registry stages.")
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        help="Primary metric used to rank runs (default: f1).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help=f"MLflow experiment to evaluate (default: {DEFAULT_EXPERIMENT_NAME}).",
    )
    return parser.parse_args()


def run_evaluation() -> None:
    """CLI entry point for evaluating runs and updating the model registry."""

    args = _parse_args()
    evaluate_best_run(metric=args.metric, experiment_name=args.experiment_name)


if __name__ == "__main__":
    run_evaluation()

