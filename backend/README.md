# Backend — Industrial Fire Intelligence Platform

FastAPI service that will expose thermal event, alert and analytics APIs to
the frontend, backed by PostgreSQL/PostGIS, and will eventually call into
the `aiml/` inference pipeline.

## Tech stack

- Python 3.12+
- FastAPI + Uvicorn
- SQLAlchemy 2.x + GeoAlchemy2 (PostGIS-ready ORM)
- PostgreSQL + PostGIS
- Pydantic v2 + pydantic-settings
- Alembic (migrations)
- psycopg (v3)
- httpx, pytest

## Folder structure

```
app/
├── api/
│   ├── routes/     # one module per resource (health, events, alerts, ...)
│   └── router.py   # aggregates all route modules under /api
├── core/
│   ├── config.py   # environment-driven settings
│   └── logging.py  # logging setup
├── db/
│   ├── database.py # engine/session
│   └── base.py     # declarative base for ORM models
├── models/         # SQLAlchemy models (PostGIS-ready via GeoAlchemy2)
├── schemas/        # Pydantic request/response schemas
├── services/       # business logic, eventually incl. AIML client
└── main.py         # FastAPI app entrypoint
```

## Local development

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`, with the health check
at `http://localhost:8000/api/health`.

## Database migrations (Alembic)

Alembic is configured (`alembic.ini`, `alembic/env.py`) to read the database
URL from application settings. No migrations have been generated yet since
the schema is not final:

```bash
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

## Tests

```bash
pytest
```

## Notes

- The database schema is intentionally not implemented yet.
- Backend will call AIML inference through a clean service interface in
  `app/services/` once the model exists — it must not duplicate ML logic.
