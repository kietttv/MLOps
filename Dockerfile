FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV MLFLOW_REGISTRY_URI=sqlite:///mlflow.db
ENV PYTHONPATH=/app

COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["./docker-entrypoint.sh"]