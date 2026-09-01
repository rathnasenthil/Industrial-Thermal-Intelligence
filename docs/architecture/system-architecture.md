# System Architecture

## Overview

The Industrial Fire Intelligence Platform combines NASA FIRMS thermal
anomaly data, OpenStreetMap infrastructure data, and satellite/land-cover
data with machine learning to detect and classify industrial fires and
persistent thermal sources, distinguishing them from wildfires and
agricultural burning.

## High-level data flow

```
NASA FIRMS
   │  (thermal hotspot detections: lat/lon, brightness, FRP, confidence, time)
   ▼
Thermal observations
   │  (cleaned, deduplicated, standardized)
   ▼
Event formation
   │  (spatio-temporal clustering of repeated detections into discrete events)
   ▼
OSM + satellite + land-cover context
   │  (nearby infrastructure, land use, terrain)
   ▼
Feature engineering
   │  (spatial, temporal, thermal features)
   ▼
AI classification
   │  (industrial fire / wildfire / agricultural burning / gas flare / ...)
   ▼
Persistence analysis
   │  (is this a recurring / long-lived thermal source?)
   ▼
Historical baseline
   │  (compare against the location's own detection history)
   ▼
Anomaly detection
   │  (flag deviations from the expected baseline)
   ▼
Risk scoring
   │  (combine classification, persistence and anomaly signals)
   ▼
FastAPI
   │  (serves events, classifications, risk scores, alerts via REST API)
   ▼
React GIS dashboard
   (map visualization, event feed, alerts, analytics)
```

## Components

### Frontend (`frontend/`)

React + TypeScript + Vite single-page app. Renders the GIS map (MapLibre GL
JS), thermal event feed, alerts and analytics (Recharts). Communicates with
the backend exclusively via its HTTP API — it never accesses the database
directly.

### Backend (`backend/`)

FastAPI service exposing REST endpoints under `/api`. Owns the PostgreSQL +
PostGIS database via SQLAlchemy 2.x / GeoAlchemy2, and will eventually call
into the AIML inference pipeline through a clean service-layer interface.

### AIML (`aiml/`)

Standalone Python package responsible for data ingestion (FIRMS, OSM),
preprocessing, feature engineering, model training/evaluation, and
inference. Has no dependency on the frontend or backend frameworks, so it
can be developed, tested and (eventually) deployed independently.

### Database

PostgreSQL with the PostGIS extension, run via Docker Compose
(`docker-compose.yml`) for local development. Stores thermal events,
classifications, and derived analytics with proper geospatial types.

## Design principles

- Frontend never talks to PostgreSQL directly — only through the backend API.
- AIML contains no frontend/backend framework code; it is a pure data/ML
  package.
- The backend will call AIML functionality through a clean interface
  (e.g. `aiml/src/inference/predict.py`) rather than duplicating ML logic.
- All three areas (`frontend/`, `backend/`, `aiml/`) are independently
  runnable and can be developed on separate branches.
