FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV MLFLOW_REGISTRY_URI=sqlite:///mlflow.db
ENV PYTHONPATH=/app

RUN python data/make_data.py \
    && python models/train.py \
    && python models/evaluate.py

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "app.app:app"]

