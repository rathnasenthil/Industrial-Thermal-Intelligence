# Industrial Fire Intelligence Platform

**Smart India Hackathon — Problem Statement ID: 26162**

AI-Based Detection and Classification of Industrial Fires and Persistent
Thermal Sources Using NASA FIRMS, OSM & Satellite Data.

## Project

An AI-powered geospatial intelligence system for detecting and classifying
industrial fires and persistent thermal sources, and visualizing them on a
GIS dashboard.

## Problem

NASA FIRMS detects thermal anomalies from satellite data, but does not by
itself reliably distinguish between industrial fires, wildfires,
agricultural burning, gas flares, mining activity, and other persistent
thermal sources. This makes it hard to act on FIRMS data directly for
industrial safety and environmental monitoring use cases.

## Proposed approach

```
FIRMS + OSM + Satellite/Land Cover + Historical Data
            │
            ▼
   Geospatial Analysis + Machine Learning
            │
            ▼
   Anomaly Detection + Risk Scoring
            │
            ▼
        GIS Visualization
```

See `docs/architecture/system-architecture.md` for the full data flow.

## Technology stack

**Frontend** — React + TypeScript + Vite + Tailwind CSS + MapLibre GL JS +
React Router + Axios + Recharts + Lucide React

**Backend** — FastAPI + Python + SQLAlchemy 2.x + GeoAlchemy2 + PostgreSQL +
PostGIS + Alembic + Pydantic v2

**AIML** — Python + Pandas + GeoPandas + Scikit-learn + XGBoost + Rasterio +
Shapely/PyProj

**Infrastructure** — Docker Compose (PostgreSQL/PostGIS for local dev)

## Repository structure

```
industrial-fire-intelligence/
├── frontend/   # React + TypeScript GIS dashboard (see frontend/README.md)
├── backend/    # FastAPI REST API (see backend/README.md)
├── aiml/       # Data ingestion, feature engineering, ML models (see aiml/README.md)
├── docs/       # Architecture, API design and ML planning docs
│   ├── architecture/
│   ├── api/
│   └── ml/
├── docker-compose.yml   # PostgreSQL + PostGIS for local development
└── .gitignore
```

The three development areas (`frontend/`, `backend/`, `aiml/`) are
independent — each has its own dependency management and can be developed
on separate branches — but are designed to integrate through well-defined
interfaces (HTTP API between frontend/backend, a service-layer call between
backend/aiml).

## Local development

### 1. Clone the repository

```bash
git clone <repository-url>
cd industrial-fire-intelligence
```

### 2. Start PostgreSQL/PostGIS with Docker

```bash
docker compose up -d
```

This starts a PostGIS-enabled PostgreSQL instance on `localhost:5432` with
database `industrial_fire_db`.

### 3. Backend setup

**Windows (PowerShell)**

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

**macOS / Linux**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`, with a health check
at `http://localhost:8000/api/health`.

### 4. Frontend setup

**Windows (PowerShell) / macOS / Linux**

```bash
cd frontend
npm install
copy .env.example .env    # Windows
cp .env.example .env      # macOS/Linux
npm run dev
```

The app will be available at `http://localhost:5173`.

### 5. AIML setup

**Windows (PowerShell)**

```powershell
cd aiml
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

**macOS / Linux**

```bash
cd aiml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Architectural principles

1. Frontend communicates with the backend only through its HTTP API.
2. Frontend never directly accesses PostgreSQL.
3. AIML contains no frontend or backend framework code.
4. Backend contains no frontend code.
5. ML/data-processing logic stays inside `aiml/`.
6. Backend calls AIML inference through a clean interface (no duplicated
   ML logic in the API layer).
7. Geospatial operations stay in the appropriate backend/AIML modules.
8. No duplicated models or database logic across services.
9. Configuration is via environment variables — no secrets in source code.
10. Python code uses type hints; TypeScript is configured in strict mode.
11. The architecture is modular so frontend, backend and AIML can be
    developed independently and in parallel (e.g. on separate branches).

## Project status

This repository currently contains the initial development environment
only: working scaffolds for the frontend, backend, and AIML package, a
Dockerized PostGIS database, and baseline documentation. No datasets have
been downloaded, no ML model has been trained, and the GIS dashboard/API
features are not yet implemented.
