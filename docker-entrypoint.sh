#!/bin/sh
set -e

export PYTHONPATH="${PYTHONPATH:-/app}"

exec gunicorn -b 0.0.0.0:8080 --workers 2 --timeout 120 app.app:app

