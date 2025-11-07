"""Flask application exposing prediction endpoints for customer purchase scoring."""

from __future__ import annotations

from typing import Any, Dict, List

import mlflow.pyfunc
import numpy as np
from flask import Flask, jsonify, render_template, request

MODEL_URI = "models:/best-customer-classifier/Production"
FEATURE_NAMES = [f"f{idx}" for idx in range(1, 11)]


def _load_model() -> mlflow.pyfunc.PyFuncModel:
    """Load the production model from the MLflow Model Registry."""

    return mlflow.pyfunc.load_model(MODEL_URI)


def _parse_input(payload: Dict[str, Any]) -> np.ndarray:
    """Validate and transform incoming payload into a model-ready numpy array."""

    values: List[float] = []
    for feature in FEATURE_NAMES:
        if feature not in payload:
            raise ValueError(f"Missing feature '{feature}'.")
        try:
            values.append(float(payload[feature]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for '{feature}': {payload[feature]}") from exc

    return np.array(values, dtype=float).reshape(1, -1)


def create_app() -> Flask:
    """Instantiate and configure the Flask application."""

    app = Flask(__name__)
    model = _load_model()

    @app.route("/", methods=["GET"])
    def index() -> str:
        """Render the interactive prediction form."""

        return render_template("index.html", feature_names=FEATURE_NAMES)

    @app.route("/predict", methods=["POST"])
    def predict() -> Any:
        """Generate predictions based on incoming JSON or form payloads."""

        if request.is_json:
            payload = request.get_json(silent=True) or {}
        else:
            payload = request.form.to_dict()

        try:
            features = _parse_input(payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        prediction = model.predict(features)
        pred_label = int(prediction[0]) if isinstance(prediction, (list, np.ndarray)) else int(prediction)
        response = {
            "prediction": pred_label,
            "label": "Likely to Purchase" if pred_label == 1 else "Unlikely to Purchase",
        }
        return jsonify(response)

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        """Health check endpoint."""

        return jsonify({"status": "ok"})

    return app
app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)

