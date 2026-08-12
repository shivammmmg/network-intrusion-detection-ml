FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY demo/api/requirements.txt ./demo/api/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt -r demo/api/requirements.txt

# The API intentionally reads the project's frozen artifacts in place.  Copy
# only the serving runtime and the exact immutable files it requires.
COPY src/ ./src/
COPY demo/api/app/ ./demo/api/app/
COPY demo/fixtures/ ./demo/fixtures/
COPY artifacts/ ./artifacts/
COPY experiments/standardized_evaluation/selected_thresholds.json ./experiments/standardized_evaluation/selected_thresholds.json
COPY experiments/xgboost/xgboost_model.json ./experiments/xgboost/xgboost_model.json

CMD ["sh", "-c", "uvicorn demo.api.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
