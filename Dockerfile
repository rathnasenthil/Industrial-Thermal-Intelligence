# Production image for Railway: FastAPI + in-process FIRMS NRT scheduler.
# Build context must be the repository root so both backend/ and aiml/ are included.
# Do not bake secrets into this image — inject DATABASE_URL / FIRMS_MAP_KEY at runtime.

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend:/app/aiml

# Minimal OS libs commonly needed by geospatial Python wheels at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libexpat1 \
        libgeos-c1v5 \
        libproj25 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/backend/requirements.txt

COPY backend /app/backend
COPY aiml /app/aiml

WORKDIR /app/backend

# Railway injects PORT. Exactly one Uvicorn worker (APScheduler is in-process).
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
