# Backend — Industrial Fire Intelligence Platform

FastAPI service exposing thermal event, facility, alert, and dashboard APIs
backed by PostgreSQL/PostGIS. The backend **consumes frozen AIML Stage VI
outputs as data** — it does not retrain models or re-implement AIML stages.

## Tech stack

- Python 3.12+
- FastAPI + Uvicorn
- SQLAlchemy 2.x + GeoAlchemy2 (PostGIS)
- PostgreSQL + PostGIS
- Pydantic v2 + pydantic-settings
- Alembic
- psycopg (v3)
- httpx, pytest

## Folder structure

```
app/
├── api/routes/     # health, events, facilities, alerts, dashboard
├── core/           # settings, logging
├── db/             # engine, session, declarative base
├── models/         # ThermalEvent, Facility, EventFacilityCandidate
├── schemas/        # Pydantic response models
├── services/       # ingestion + query logic
└── main.py
scripts/
└── ingest_stage_vi.py
alembic/versions/   # schema migrations
tests/
```

## Data sources (read-only)

| Dataset | Path | Approx. rows |
|---------|------|--------------|
| Stage VI events | `aiml/data/processed/thermal_events_with_risk_prioritization.csv` | 179,740 |
| Stage I.1 facilities | `aiml/data/processed/osm_facilities.csv` | 112,956 |
| Stage I.2 candidates | `aiml/data/processed/thermal_event_facility_candidates.csv` | 268,020 |

Do not modify files under `aiml/`.

### Important column names (Stage VI)

Source CSV uses AIML names (not informal aliases):

- Temporal: `event_start`, `event_end`, `observed_duration_hours`
- Location: `centroid_latitude`, `centroid_longitude`, `centroid_wkt`, `footprint_wkt`
- Persistence: `persistence_label`
- Facility relation: `facility_association_method`
- Priority: `investigation_priority`
- Severity: `thermal_severity_band`

API query aliases map to these columns (`priority` → `investigation_priority`,
`persistence_class` → `persistence_label`).

## Data semantics

1. A thermal anomaly is **not** automatically an industrial fire.
2. Facility association is spatial attribution, **not** source classification.
3. `NO_FACILITY_ASSOCIATION` does **not** mean natural.
4. `AMBIGUOUS` does **not** mean industrial or non-industrial.
5. Missing STA evidence is **unavailable**, not negative evidence.
6. Missing environmental evidence is **unavailable**, not negative evidence.
7. `risk_score` is a **decision-support prioritization score**, not a probability.
8. `CRITICAL` means highest investigation priority under configured rules, not “confirmed fire”.
9. Candidate labels are interpretations, not ground truth.
10. Stage V produced **no validated performance claim**.

`GET /api/alerts` is an **investigation-priority view** of HIGH/CRITICAL events —
not an emergency dispatch system.

## Local development

### 1. Start PostGIS

From the repository root:

```bash
docker compose up -d db
```

Requires Docker. The `postgis/postgis:16-3.4` service enables PostGIS via
`docker/init-postgis.sql`.

### 2. Backend environment

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

### 3. Migrate schema

```powershell
alembic upgrade head
```

### 4. Ingest frozen AIML CSVs

```powershell
python scripts/ingest_stage_vi.py
```

Optional flags:

```powershell
python scripts/ingest_stage_vi.py --skip-candidates --report-json ingestion_report.json
```

Ingestion uses bulk inserts (batched), preserves nulls, rejects invalid
coordinates / duplicate IDs with a report, and is idempotent via
`mode=replace` (truncate + reload).

### 5. Run API

```powershell
uvicorn app.main:app --reload
```

- API: http://localhost:8000
- OpenAPI: http://localhost:8000/docs
- Health: http://localhost:8000/api/health

## Example API queries

```text
GET /api/events?page=1&page_size=50&priority=HIGH&min_risk_score=40
GET /api/events?bbox=72,8,97,37&date_from=2023-01-01T00:00:00Z
GET /api/events/EVT_0000001
GET /api/events/EVT_0000001/evidence
GET /api/events/EVT_0000001/timeline
GET /api/facilities?search=port&facility_type=INDUSTRIAL_AREA
GET /api/facilities/{facility_id}/history?priority=CRITICAL
GET /api/alerts?page=1
GET /api/dashboard/statistics
```

Spatial filters use PostGIS (`ST_Intersects` + `ST_MakeEnvelope`), not
in-Python distance loops.

Dashboard statistics are live SQL aggregates — never hardcoded Stage VI counts.

## Tests

```powershell
pytest
```

Unit/API contract tests run without PostGIS. Integration tests that ingest
the full Stage VI CSV are skipped automatically when Docker/PostGIS is down,
and marked `slow` when present:

```powershell
pytest -m "not slow"
pytest tests/test_integration_db.py -m slow
```

## Environment variables

See `.env.example`:

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | SQLAlchemy URL (`postgresql+psycopg://...`) |
| `POSTGRES_*` | Docker Compose database credentials |
| `CORS_ORIGINS` | Comma-separated allowed origins |

Do not commit `.env`, passwords, or large datasets.

## Limitations

- Detection-level FIRMS timelines are **not** in the Stage VI backend dataset;
  `/timeline` returns event-level aggregates only.
- STA / environmental domains may be entirely unavailable in the frozen export;
  the API reports `status: unavailable` rather than fabricating values.
- Full production ingestion (~180k events + ~113k facilities + ~268k candidates)
  requires a running PostGIS instance and several minutes of load time.
- No ML inference is invoked by these read APIs.
