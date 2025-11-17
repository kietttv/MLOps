"""Evaluate MLflow runs, manage model registry stages, and persist best metadata."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import pandas as pd


DEFAULT_EXPERIMENT_NAME = "Customer Purchase Prediction"
REGISTERED_MODEL_NAME = "best-customer-classifier"
BEST_MODEL_PATH = Path(__file__).resolve().parent / "last_best.json"
PRODUCTION_MODEL_DIR = Path(__file__).resolve().parent / "production"


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


def _copy_production_model(client: MlflowClient, version: str) -> None:
    """Copy the production model to models/production/ for git tracking."""

    try:
        import mlflow.pyfunc
        
        # Load model from MLflow registry
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{version}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Get the actual model path from MLflow
        # Try to find model in mlruns directory structure
        tracking_uri = mlflow.get_tracking_uri()
        
        # Skip copy if using SQLite backend (e.g., in Docker where model is already in image)
        if tracking_uri.startswith("sqlite://"):
            print(f"Skipping copy: SQLite backend detected (model should already be in models/production/)")
            return
        
        if not tracking_uri.startswith("file://"):
            raise ValueError(f"Unsupported tracking URI format: {tracking_uri}")
        
        import urllib.parse
        # Parse file:// URI correctly (file:///D:/path -> D:/path)
        tracking_path_str = tracking_uri.replace("file://", "")
        # Remove leading slash if present (file:///D: -> /D: -> D:)
        if tracking_path_str.startswith("/"):
            tracking_path_str = tracking_path_str[1:]
        mlruns_path = Path(urllib.parse.unquote(tracking_path_str))
        
        # Get model version info
        model_version_info = client.get_model_version(REGISTERED_MODEL_NAME, version)
        model_id = model_version_info.model_id  # e.g., m-5bff8c804e6b441c87825513cdde7820
        
        # Read storage_location from meta.yaml file directly (Python API doesn't expose it)
        meta_yaml_path = mlruns_path / "models" / REGISTERED_MODEL_NAME / f"version-{version}" / "meta.yaml"
        
        if meta_yaml_path.exists():
            import yaml
            with meta_yaml_path.open("r", encoding="utf-8") as f:
                meta_data = yaml.safe_load(f)
                storage_location = meta_data.get("storage_location")
                
                if storage_location and storage_location.startswith("file://"):
                    # Parse storage_location URI
                    storage_path_str = storage_location.replace("file://", "")
                    if storage_path_str.startswith("/"):
                        storage_path_str = storage_path_str[1:]
                    storage_path_str = urllib.parse.unquote(storage_path_str)
                    storage_path = Path(storage_path_str)
                    # storage_location points to artifacts/ directory, model files are directly in it
                    if storage_path.exists():
                        source_path = storage_path
                    else:
                        raise FileNotFoundError(f"Storage path does not exist: {storage_path}")
                else:
                    raise ValueError(f"Invalid storage_location in meta.yaml: {storage_location}")
        else:
            # Fallback: Try to find model using model_id
            # Model structure: mlruns/{experiment_id}/models/{model_id}/artifacts/model
            source_path = None
            for exp_dir in mlruns_path.iterdir():
                if exp_dir.is_dir() and exp_dir.name != "models" and exp_dir.name != "0":
                    model_dir = exp_dir / "models" / model_id / "artifacts" / "model"
                    if model_dir.exists():
                        source_path = model_dir
                        break
            
            if source_path is None:
                # Last fallback: try registry directory
                registry_model_path = mlruns_path / "models" / REGISTERED_MODEL_NAME / f"version-{version}" / "artifacts" / "model"
                if registry_model_path.exists():
                    source_path = registry_model_path
                else:
                    raise FileNotFoundError(f"Could not find model for version {version} (model_id={model_id})")
        
        # Create production directory
        PRODUCTION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        target_path = PRODUCTION_MODEL_DIR / "model"
        
        # Remove existing model if present
        if target_path.exists():
            shutil.rmtree(target_path)
        
        # Copy model directory
        shutil.copytree(source_path, target_path)
        
        # Create .gitkeep to ensure directory is tracked
        (PRODUCTION_MODEL_DIR / ".gitkeep").touch(exist_ok=True)
        
        print(f"Copied production model (version {version}) to {target_path}")
    except Exception as exc:
        print(f"Warning: Failed to copy production model: {exc}")
        import traceback
        traceback.print_exc()
        # Don't fail the whole evaluation if copy fails


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
    
    # Copy production model to models/production/ for git tracking
    _copy_production_model(client, version)

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

