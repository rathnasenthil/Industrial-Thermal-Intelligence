# API Design

## Conventions

- Base path: `/api`
- JSON request/response bodies, validated with Pydantic v2 schemas.
- CORS is restricted to the frontend dev origin (`http://localhost:5173`),
  configurable via the `CORS_ORIGINS` environment variable.
- Route handlers live in `backend/app/api/routes/`, one module per resource,
  and are aggregated in `backend/app/api/router.py`.
- The backend consumes **frozen AIML Stage VI** outputs. It does not retrain
  models or invent STA/environmental evidence.

## Data semantics (API-wide)

- `risk_score` is an engineering **decision-support** prioritization score
  (0–100), **not** a probability of industrial fire.
- `investigation_priority` / query alias `priority` is investigation priority.
  `CRITICAL` ≠ confirmed fire.
- Missing STA / environmental evidence is returned as **unavailable**, never
  as negative evidence.
- Facility association is spatial attribution, not source classification.
- Stage V produced **no validated performance claim**.

## Endpoints

### `GET /api/health`

Liveness check.

### `GET /api/events`

Paginated thermal events.

| Query | Maps to DB column / behaviour |
|-------|-------------------------------|
| `page`, `page_size` | Pagination (default page_size=50, max 500) |
| `priority` | `investigation_priority` |
| `industrial_context` | `industrial_context` |
| `facility_type` | `facility_type` |
| `persistence_class` | `persistence_label` |
| `anomaly_status` | `anomaly_status` |
| `date_from`, `date_to` | filter on `event_start` |
| `min_risk_score`, `max_risk_score` | `risk_score` |
| `bbox` | `min_lon,min_lat,max_lon,max_lat` via PostGIS `ST_Intersects` |

Response: `{ items, total, page, page_size, total_pages }`.

### `GET /api/events/{event_id}`

Full event detail including facility association, persistence, anomaly,
evidence fusion summary fields, risk prioritization, explanation fields,
geometry, and I.2 facility candidates when ingested.

### `GET /api/events/{event_id}/evidence`

Evidence organized by family:

```json
{
  "temporal": { "available": true, "status": "available", "score": 1.0, "details": {} },
  "infrastructure": { "...": "..." },
  "historical": { "...": "..." },
  "anomaly": { "...": "..." },
  "sta": { "available": false, "status": "unavailable", "score": null },
  "environmental": { "available": false, "status": "unavailable", "score": null },
  "fusion": { "...": "..." }
}
```

### `GET /api/events/{event_id}/timeline`

Event-level temporal aggregates only. Per-detection FIRMS timelines are
**not** fabricated (`detection_level_timeline_available: false`).

### `GET /api/facilities`

Paginated facilities (`facility_type`, `search`, `bbox`).

### `GET /api/facilities/{facility_id}`

Facility metadata, geometry, and a historical thermal summary derived from
associated Stage VI events (counts / max risk — not ML performance metrics).

### `GET /api/facilities/{facility_id}/history`

Paginated events for that `facility_id` (optional `date_from` / `date_to` /
`priority`).

### `GET /api/alerts`

Investigation-priority view: events where `investigation_priority IN
('HIGH','CRITICAL')`. **Not** an emergency dispatch or push-notification
system.

### `GET /api/dashboard/statistics`

Live PostgreSQL aggregates: totals, priority / context / persistence /
severity / anomaly / facility-type distributions, association counts,
HIGH/CRITICAL counts, and date range. Values are never hardcoded.

## Backend → AIML integration

Read APIs load persisted Stage VI rows. Future inference (if any) should call
AIML through a narrow service interface rather than duplicating feature
engineering inside the API layer. Phase 1 does **not** invoke ML inference.
