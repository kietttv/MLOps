#!/bin/sh
set -e

export PYTHONPATH="${PYTHONPATH:-/app}"

if [ ! -f "/app/data/customer_data.csv" ]; then
    echo "Generating customer dataset..."
    python data/make_data.py
fi

if [ ! -f "/app/mlflow.db" ]; then
    echo "Bootstrapping MLflow model registry..."
    python models/train.py
    python models/evaluate.py
fi

exec gunicorn -b 0.0.0.0:8080 --workers 2 --timeout 120 app.app:app

