"""Helper utilities for interacting with MLflow tracking and registry APIs."""

from __future__ import annotations

import json
import tempfile
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve

logger = logging.getLogger(__name__)


def set_experiment(name: str) -> None:
    """Configure MLflow to use the target experiment.

    Parameters
    ----------
    name:
        Name of the experiment. The experiment is created if it does not exist.
    """

    logger.info("Setting MLflow experiment to '%s'", name)
    mlflow.set_experiment(name)


def log_dict_as_artifact(payload: Mapping[str, Any], artifact_path: str) -> None:
    """Persist a dictionary as a JSON artifact in the active MLflow run.

    Parameters
    ----------
    payload:
        Dictionary content to serialize.
    artifact_path:
        Artifact destination path (e.g., ``reports/summary.json``).
    """

    artifact_path_obj = Path(artifact_path)
    artifact_dir = artifact_path_obj.parent.as_posix() if artifact_path_obj.parent != Path(".") else None

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / artifact_path_obj.name
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        mlflow.log_artifact(str(temp_path), artifact_path=artifact_dir)
        logger.info("Logged artifact at path=%s", artifact_path)


def log_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str] | None = None,
    artifact_path: str = "plots/confusion_matrix.png",
) -> None:
    """Plot a confusion matrix and log it as a PNG artifact."""

    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(matrix, display_labels=labels)

    artifact_path_obj = Path(artifact_path)
    artifact_dir = artifact_path_obj.parent.as_posix() if artifact_path_obj.parent != Path(".") else None

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / artifact_path_obj.name
        fig, ax = plt.subplots(figsize=(6, 6))
        display.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.tight_layout()
        fig.savefig(temp_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(temp_path), artifact_path=artifact_dir)
        logger.info("Logged confusion matrix to %s", artifact_path)


def log_roc_curve(
    y_true: Sequence[int],
    y_score: Sequence[float],
    artifact_path: str = "plots/roc_curve.png",
) -> None:
    """Plot a ROC curve and log it as a PNG artifact."""

    fpr, tpr, _ = roc_curve(y_true, y_score)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr)

    artifact_path_obj = Path(artifact_path)
    artifact_dir = artifact_path_obj.parent.as_posix() if artifact_path_obj.parent != Path(".") else None

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / artifact_path_obj.name
        fig, ax = plt.subplots(figsize=(6, 6))
        display.plot(ax=ax)
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
        ax.set_title("ROC Curve")
        plt.tight_layout()
        fig.savefig(temp_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(temp_path), artifact_path=artifact_dir)
        logger.info("Logged ROC curve to %s", artifact_path)


def log_feature_importance(
    feature_names: Sequence[str],
    importance_values: Sequence[float],
    artifact_path: str = "plots/feature_importance.png",
    top_k: int | None = None,
) -> None:
    """Plot feature importances and log them as an artifact."""

    if top_k is not None and top_k > 0:
        pairs = sorted(zip(feature_names, importance_values), key=lambda pair: pair[1], reverse=True)[:top_k]
        names, importances = zip(*pairs)
    else:
        names, importances = feature_names, importance_values

    artifact_path_obj = Path(artifact_path)
    artifact_dir = artifact_path_obj.parent.as_posix() if artifact_path_obj.parent != Path(".") else None

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / artifact_path_obj.name
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(names, importances)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(temp_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(temp_path), artifact_path=artifact_dir)
        logger.info("Logged feature importance plot to %s", artifact_path)