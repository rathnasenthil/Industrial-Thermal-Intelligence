# Railway backend deployment (SIH)

Production runs **one** Railway web service from the repository-root `Dockerfile`:
FastAPI + in-process APScheduler (FIRMS NRT). Use **exactly one** replica and
**exactly one** Uvicorn worker (already set in the Dockerfile `CMD`).

## Required environment variables

Set these in the Railway service variables UI. **Never** commit their values.

| Name | Notes |
|------|--------|
| `DATABASE_URL` | Supabase Postgres URL for SQLAlchemy. Must use the `psycopg` driver and SSL, e.g. `postgresql+psycopg://postgres:<PASSWORD>@db.<PROJECT_REF>.supabase.co:5432/postgres?sslmode=require`. Prefer the **direct** DB host on port **5432** (not the transaction pooler) while the in-process scheduler holds sessions. Do **not** rely on `POSTGRES_*` alone — the app reads `database_url` / `DATABASE_URL` only. |
| `FIRMS_MAP_KEY` | NASA FIRMS Area API map key. |
| `FIRMS_NRT_ENABLED` | Set to `true` in production so the lifespan starts the scheduler. |
| `CORS_ORIGINS` | Comma-separated allowed browser origins for the deployed frontend (no spaces required; trimming is supported). Example shape: `https://your-frontend.example`. |
| `ENVIRONMENT` | Set to `production`. |

## Optional FIRMS variables (code defaults apply if unset)

| Name | Default in code |
|------|-----------------|
| `FIRMS_BASE_URL` | `https://firms.modaps.eosdis.nasa.gov/api/area/csv` |
| `FIRMS_PRODUCT` | `VIIRS_NOAA20_NRT` |
| `FIRMS_BBOX` | `68.0,6.0,98.0,37.5` |
| `FIRMS_DAY_RANGE` | `2` |
| `FIRMS_TIMEOUT_SECONDS` | `60` |
| `FIRMS_NRT_INTERVAL_MINUTES` | `15` |

## Local development

Local Docker Compose (`docker-compose.yml`) continues to run **only** PostGIS for
development. It is unchanged by this Railway setup. Keep using `backend/.env`
locally (gitignored).
