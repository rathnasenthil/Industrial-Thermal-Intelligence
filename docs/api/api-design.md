# API Design

## Conventions

- Base path: `/api`
- JSON request/response bodies, validated with Pydantic v2 schemas.
- CORS is restricted to the frontend dev origin (`http://localhost:5173`),
  configurable via the `CORS_ORIGINS` environment variable.
- Route handlers live in `backend/app/api/routes/`, one module per resource,
  and are aggregated in `backend/app/api/router.py`.

## Current endpoints

### `GET /api/health`

Liveness check used by the frontend and local development setup.

**Response `200`**

```json
{
  "status": "ok"
}
```

## Planned endpoints (not implemented yet)

These are illustrative only — schemas and routes will be defined once the
underlying data model and ML pipeline exist.

| Method | Path                    | Purpose                                           |
|--------|-------------------------|----------------------------------------------------|
| GET    | `/api/events`           | List thermal events (filter by bbox, date, class)  |
| GET    | `/api/events/{id}`      | Get a single thermal event with full details       |
| GET    | `/api/events/{id}/history` | Historical thermal observations for an event    |
| GET    | `/api/alerts`           | List active anomaly/risk alerts                    |
| POST   | `/api/events/{id}/classify` | Trigger (re)classification via the AIML pipeline |

## Backend → AIML integration

The backend will call into the AIML package through a narrow, well-defined
interface (see `aiml/src/inference/predict.py`) from a backend service
module (e.g. `backend/app/services/`), rather than duplicating feature
engineering or model logic inside the API layer.
