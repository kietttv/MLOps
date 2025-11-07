FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app/app.py
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["gunicorn", "app.app:create_app", "--bind", "0.0.0.0:8000", "--worker-class", "sync"]

