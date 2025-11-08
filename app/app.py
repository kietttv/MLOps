"""Flask application exposing prediction endpoints for customer purchase scoring."""

from __future__ import annotations

from typing import Any, Dict, List

import mlflow.pyfunc
import numpy as np
from flask import Flask, jsonify, render_template, request

MODEL_URI = "models:/best-customer-classifier/Production"
FEATURE_SCHEMA = [
    {"id": "f1", "label": "Customer Age", "hint": "Tuổi của khách hàng (năm)"},
    {"id": "f2", "label": "Annual Income (USD)", "hint": "Thu nhập hằng năm (đô la Mỹ)"},
    {"id": "f3", "label": "Purchase Frequency", "hint": "Số đơn hàng trung bình mỗi tháng"},
    {"id": "f4", "label": "Average Order Value", "hint": "Giá trị đơn hàng trung bình (USD)"},
    {"id": "f5", "label": "Days Since Last Purchase", "hint": "Số ngày kể từ lần mua cuối"},
    {"id": "f6", "label": "Website Visits per Month", "hint": "Số lượt truy cập website mỗi tháng"},
    {"id": "f7", "label": "Email Engagement Rate", "hint": "Tỷ lệ tương tác email marketing (0-1)"},
    {"id": "f8", "label": "Loyalty Score", "hint": "Điểm trung thành/tương tác tổng hợp"},
    {"id": "f9", "label": "Customer Tenure (Months)", "hint": "Số tháng khách hàng gắn bó"},
    {"id": "f10", "label": "Discount Usage Rate", "hint": "Tỷ lệ sử dụng mã giảm giá (0-1)"},
]
FEATURE_NAMES = [field["id"] for field in FEATURE_SCHEMA]


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
    model_cache: Dict[str, mlflow.pyfunc.PyFuncModel] = {}

    def _get_model() -> mlflow.pyfunc.PyFuncModel:
        """Load the production model lazily and cache it for subsequent requests."""

        cached = model_cache.get("model")
        if cached is None:
            cached = _load_model()
            model_cache["model"] = cached
        return cached

    @app.route("/", methods=["GET"])
    def index() -> str:
        """Render the interactive prediction form."""

        return render_template("index.html", feature_schema=FEATURE_SCHEMA)

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

        model = _get_model()
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

