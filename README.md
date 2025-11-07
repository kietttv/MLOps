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

## Contributing

Each script currently raises `NotImplementedError` to mark areas where data processing, training, tuning, and evaluation logic should be added. Implement the relevant workflow, add automated tests, and open a pull request to share improvements.


