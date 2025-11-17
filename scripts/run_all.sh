#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)"

python data/make_data.py
python models/train.py
python models/tune.py
python models/evaluate.py
python app/app.py

