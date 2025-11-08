# MLOps Starter Project

This repository provides a lightweight scaffold for exploring end-to-end MLOps workflows, from data preparation and model experimentation to containerized serving and continuous integration.

## Quickstart

- Create a virtual environment and install dependencies:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
  pip install -r requirements.txt
  ```
- Generate placeholder data and train (implement logic inside the scripts before running):
  ```bash
  python data/make_data.py
  python models/train.py
  ```
- Launch the Flask app locally:
  ```bash
  python app/app.py
  ```
- Optional make targets (when a `Makefile` is added) to streamline workflows:
  ```bash
  make data   # calls data/make_data.py
  make train  # calls models/train.py
  make app    # runs the Flask development server
  ```
- Build and run with Docker:
  ```bash
  docker build -t mlops-app .
  docker run --rm -p 8000:8000 mlops-app
  ```
- Publishable image workflow:
  ```bash
  docker build -t tvtkiet2002/mlops:latest .
  docker run -p 8080:8080 --name mlops tvtkiet2002/mlops:latest
  ```
  Build step sẽ tự động sinh dữ liệu, train, và evaluate để đăng ký model Production trong image, nên container khởi động có thể dự đoán ngay.
  Nếu tự chạy script trong một môi trường khác, nhớ set `PYTHONPATH=/app` (ở Dockerfile được cấu hình sẵn) để tránh lỗi import.
- GitHub Actions CI/CD (push to Docker Hub):
  1. Trong repo GitHub, vào **Settings → Secrets and variables → Actions**.
  2. Tạo secret `DOCKERHUB_USERNAME` (tên tài khoản Docker Hub).
  3. Tạo secret `DOCKERHUB_TOKEN` (Access Token từ Docker Hub > Security > New Access Token).
  4. Commit/push vào nhánh `main` để workflow `.github/workflows/docker-ci.yml` build & push image với tag `latest` và `SHA`.
- Run the automated test suite:
  ```bash
  pytest -q
  ```

## Project Structure

```text
MLOps/
├─ data/
│  └─ make_data.py
├─ models/
│  ├─ train.py
│  ├─ tune.py
│  └─ evaluate.py
├─ app/
│  ├─ app.py
│  └─ templates/
│     └─ index.html
├─ tests/
│  └─ test_app.py
├─ requirements.txt
├─ Dockerfile
├─ .gitignore
├─ .github/
│  └─ workflows/
│     └─ docker-ci.yml
├─ MLproject
├─ README.md
└─ repo_link.txt
```

## Running the CI Workflow

The GitHub Actions workflow (`.github/workflows/docker-ci.yml`) triggers automatically on pushes or pull requests targeting the `main` branch. It installs dependencies, runs tests, and builds the Docker image to ensure the project remains production-ready.

## MLflow Setup

Configure MLflow to store both tracking metadata and the model registry in a local SQLite database. Set the following environment variables before running experiments:

- **Linux / macOS (bash/zsh):**
  ```bash
  export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
  export MLFLOW_REGISTRY_URI=sqlite:///mlflow.db
  ```
- **Windows (PowerShell):**
  ```powershell
  $env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
  $env:MLFLOW_REGISTRY_URI = "sqlite:///mlflow.db"
  ```
- **Windows (Command Prompt):**
  ```cmd
  set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
  set MLFLOW_REGISTRY_URI=sqlite:///mlflow.db
  ```

Launch the MLflow UI to inspect runs and registered models:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Artifacts and runs are stored locally under the `mlruns/` directory (ignored by Git). Ensure the helper utilities in `models/mlflow_utils.py` are used to set experiments and log artifacts consistently.

## Contributing

Each script currently raises `NotImplementedError` to mark areas where data processing, training, tuning, and evaluation logic should be added. Implement the relevant workflow, add automated tests, and open a pull request to share improvements.


