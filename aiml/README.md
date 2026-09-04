# AIML — Industrial Fire Intelligence Platform

Python package for data ingestion, preprocessing, feature engineering,
model training, evaluation and inference for classifying industrial fires
and persistent thermal sources.

## Tech stack

- Python 3.12+
- pandas, numpy, scipy
- scikit-learn, xgboost
- geopandas, shapely, pyproj, rasterio (geospatial)
- requests, httpx (data ingestion)
- matplotlib, jupyter (exploration)

## Folder structure

```
aiml/
├── data/
│   ├── raw/         # untouched source data (gitignored)
│   ├── processed/   # cleaned/merged datasets (gitignored)
│   └── external/    # third-party reference data (gitignored)
├── notebooks/        # exploratory analysis
├── src/
│   ├── data_ingestion/       # firms.py, osm.py
│   ├── preprocessing/        # cleaning, event formation
│   ├── feature_engineering/  # spatial/temporal/thermal features
│   ├── models/                # classifier
│   ├── evaluation/            # metrics
│   └── inference/             # prediction entrypoint for the backend
└── tests/             # environment/import sanity tests
```

## Local development

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Tests

```bash
python -m pytest
```

Most tests still only verify the Python environment and that modules
import correctly. The exception is the FIRMS ingestion/preprocessing
stage (see below), which has real unit + integration test coverage.

## FIRMS ingestion & data-quality preprocessing

The only pipeline stage currently implemented is **FIRMS ingestion and
data-quality preprocessing** (everything upstream of Geospatial Event
Formation in the GIFT framework). It lives in:

- `src/data_ingestion/firms_csv.py` — loads raw FIRMS VIIRS CSV exports,
  validates the schema, combines multiple files with a `source_file`
  provenance column.
- `src/preprocessing/coordinates.py` — lat/lon range validation.
- `src/preprocessing/numeric_fields.py` — numeric conversion + FRP
  validation (never fabricates missing values).
- `src/preprocessing/timestamps.py` — builds a UTC `acq_datetime` from
  `acq_date`/`acq_time` (auto-detects `YYYY-MM-DD` vs `DD-MM-YYYY`, parses
  `acq_time` as HHMM).
- `src/preprocessing/categorical_fields.py` — whitespace/case-only
  normalization of `confidence`/`daynight` (no reclassification).
- `src/preprocessing/duplicates.py` — exact-duplicate detection (not
  lat/lon-based — see module docstring for why).
- `src/preprocessing/firms_pipeline.py` — orchestrates the above.
- `src/preprocessing/run_firms_pipeline.py` — CLI entry point.

Run it (from this `aiml/` directory, with raw CSVs in `data/raw/`):

```bash
python -m src.preprocessing.run_firms_pipeline
```

This writes `data/processed/firms_viirs_india_2023_2024_clean.csv` and
`data/processed/firms_preprocessing_report.json`. Pass `--input`
(repeatable), `--output`, `--report`, or `--raw-dir` to override defaults.

This stage intentionally does **not** perform ST-DBSCAN event formation,
OSM/STA/Sentinel enrichment, facility fingerprinting, anomaly detection,
or risk scoring — it only produces a trustworthy, cleaned detection-level
dataset for those later stages to consume.

## GIFT Stage G — Geospatial Event Formation

Groups cleaned FIRMS detections into spatio-temporal "thermal events"
using ST-DBSCAN (haversine spatial distance + a temporal window), without
classifying what caused them. Lives in `src/event_formation/`:

- `config.py` — `STDBSCANConfig` (spatial/temporal epsilon, min_samples;
  documented, tunable, NOT scientifically validated).
- `spatial_index.py` — haversine `BallTree` helpers (great-circle
  distance, not naive Euclidean lat/lon).
- `st_dbscan.py` — batched spatial-index neighbor-graph construction +
  `sklearn.cluster.DBSCAN(metric="precomputed")`; avoids an O(n^2)
  distance matrix.
- `geometry.py` — centroid/bounding-box/convex-hull (WKT) per event.
- `event_features.py` — vectorized per-event thermal/temporal/confidence/
  day-night aggregation.
- `noise.py` — annotates (never deletes) unclustered detections.
- `event_pipeline.py` / `event_report.py` — orchestration + JSON report.
- `run_event_formation.py` — CLI entry point.
- `benchmark_st_dbscan.py` — subset benchmark, run before any full-dataset
  run.

Run it (from this `aiml/` directory):

```bash
python -m src.event_formation.benchmark_st_dbscan   # sanity-check performance first
python -m src.event_formation.run_event_formation
```

This writes `data/processed/thermal_events.csv`,
`data/processed/thermal_event_detections.csv`,
`data/processed/thermal_event_noise.csv` and
`data/processed/event_formation_report.json`.

Events are called **"thermal events"**, never "fires" — source
classification (industrial/wildfire/agricultural/etc.) happens in later
GIFT stages (I, F, T), not here.

## GIFT Stage G.1 — Persistence & Recurrence Characterization

Adds deterministic, rule-based persistence metrics on top of the
**immutable** Stage G `thermal_events.csv` — it never re-runs ST-DBSCAN
and never changes `event_id`/detection assignments; it only reads that
table and appends columns. Lives in `src/persistence/`:

- `config.py` — `PersistenceConfig` thresholds (`min_detections_for_classification`,
  `short_lived_max_duration_hours`, `persistent_min_duty_cycle`,
  `persistent_max_gap_hours`), calibrated against the real full Stage G
  run and documented as engineering defaults, **not** scientifically
  validated. Notably, `persistent_max_gap_hours` must stay below Stage
  G's `temporal_eps_hours` (36h) — a single ST-DBSCAN event can never
  have an internal gap larger than that by construction.
- `classification.py` — computes `span_days` (calendar-day span) and
  `duty_cycle` (`distinct_detection_days / span_days`), then labels each
  event `INSUFFICIENT_OBSERVATIONS`, `SHORT_LIVED`, `PERSISTENT`, or
  `RECURRING`. `PERSISTENT` is an OR of "high duty cycle" or "no long
  internal gap", so one isolated long pause doesn't downgrade an
  otherwise consistently-detected source.
- `persistence_pipeline.py` / `persistence_report.py` — orchestration +
  JSON report (label counts/percentages, duty-cycle & duration stats,
  a spot-checkable sample of the longest events).
- `run_persistence_characterization.py` — CLI entry point.

Run it (from this `aiml/` directory, after Stage G has produced
`thermal_events.csv`):

```bash
python -m src.persistence.run_persistence_characterization
```

This writes `data/processed/thermal_events_with_persistence.csv` (all
Stage G columns plus `span_days`, `duty_cycle`, `persistence_label`,
`persistence_basis`) and
`data/processed/persistence_characterization_report.json`.

`observed_duration_hours`/`span_days`/`duty_cycle` describe the
*observed* detection pattern only — FIRMS is discrete satellite
overpasses, not continuous monitoring, so they are not claims about the
true physical start/end/duration of the underlying thermal source.

## Notes

- No trained models or prediction outputs are included.
- Ingestion/feature/model/inference functions outside the FIRMS
  preprocessing stage above are still placeholders with `TODO` comments
  and raise `NotImplementedError`.
- The backend will eventually call into `src/inference/predict.py` through
  a clean interface — avoid duplicating this logic elsewhere.
