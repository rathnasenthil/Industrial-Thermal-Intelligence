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

## GIFT Stage I.1 — OSM Facility Ingestion & Normalization

**Why OSM is used, and why it is NOT ground truth:** OpenStreetMap is a
free, crowd-sourced source of *contextual* infrastructure evidence
(refineries, power plants, mines, industrial zones, ...) that can later
be spatially associated with thermal events (Stage I.2, not implemented
here). A facility record existing — or not existing — near a thermal
event says nothing on its own about that event's cause. OSM coverage is
also known to be incomplete and inconsistent, especially in parts of
India, so **missing OSM data must never be used to automatically rule
out an industrial cause** for a thermal event, and **OSM presence must
never be used to automatically confirm one** either.

Lives in `src/infrastructure/`:

- `config.py` — `InfrastructureConfig` (where to look for a static
  extract, source label, LNG name-keyword evidence).
- `osm_loader.py` — loads a **static** OSM extract (GeoJSON, CSV, or a
  real OSM PBF file) from disk. No live Overpass API dependency, no
  network access, no fabricated data. `discover_default_osm_input()`
  looks for a file in `data/raw/` and returns `None` (not an error) if
  none exists.
- `osm_pbf_loader.py` — **streaming** OSM PBF ingestion (see "Real OSM
  PBF support" below). Dispatches to the same canonical intermediate
  representation as the GeoJSON/CSV loaders, so normalization,
  validation and reporting are shared, unduplicated code.
- `facility_schema.py` — controlled `facility_type` vocabulary
  (`REFINERY`, `POWER_PLANT`, `MINE`, `INDUSTRIAL_AREA`, `LNG_TERMINAL`,
  `OTHER_INDUSTRIAL`, `UNKNOWN`), canonical output columns, and
  deterministic facility-id generation (`osm_<type>_<id>`, or a
  content-hash `fallback_<hash>` id when no stable OSM id is available).
- `osm_normalization.py` — maps raw OSM tags to `facility_type` using
  actual, documented OSM tagging conventions as evidence (e.g.
  `industrial=refinery`, `power=plant`, `industrial=mine`/
  `landuse=quarry`, `landuse=industrial`); builds the canonical facility
  table, preserving original geometry (Point/Polygon/MultiPolygon) and
  tags, with a centroid computed only as a convenience "representative
  point" for later spatial association.
- `facility_validation.py` — flags (never silently drops) missing ids,
  invalid/unsupported geometry, invalid coordinates, and unsupported
  types; detects duplicate facilities conservatively (only exact
  `facility_id` repeats, never "nearby therefore the same").
- `facility_report.py` / `infrastructure_pipeline.py` — orchestration +
  JSON report.
- `run_osm_ingestion.py` — CLI entry point.

Run it (from this `aiml/` directory):

```bash
python -m src.infrastructure.run_osm_ingestion
```

This auto-discovers a static extract in `data/raw/` (filename must
contain "osm", "facility", "facilities" or "industrial", case-insensitive,
`.geojson`/`.json`/`.csv`). **No such file ships with this repository** —
running the command above today produces an explicitly empty facility
layer and a report whose `input.status` is
`"no_production_osm_input_found"`, not a fabricated dataset.

**To provide real coverage:** place a GeoJSON export (e.g. from
`overpass-turbo`/`osmtogeojson`/QGIS's OSM plugin — a `FeatureCollection`
whose feature `properties` carry the OSM tags), a flattened CSV (with
either a `geometry_wkt` column or `latitude`/`longitude` columns, plus a
JSON `tags` column or arbitrary tag columns), **or a real OSM PBF
extract** (see below) into `aiml/data/raw/`, e.g.
`aiml/data/raw/osm_facilities_india.geojson` or
`aiml/data/raw/india-260904.osm.pbf`, then re-run the command.

### Real OSM PBF support

A real, India-wide `.osm.pbf` extract (e.g. from Geofabrik) is supported
directly — place it at `aiml/data/raw/<name>.osm.pbf` (filename must
contain "osm" for auto-discovery, e.g. `india-260904.osm.pbf`) and run:

```bash
python -m src.infrastructure.run_osm_ingestion --input data/raw/india-260904.osm.pbf
# or, if the filename is auto-discoverable in data/raw/:
python -m src.infrastructure.run_osm_ingestion
```

**Dependency:** [pyosmium](https://github.com/osmcode/pyosmium) (PyPI
package `osmium`), the standard Python binding for libosmium (the C++
library used by most production OSM tooling, e.g. `osm2pgsql`). Chosen
because it ships official pre-built wheels for Windows/Linux/macOS (no
compiler needed) and is a genuine streaming, callback-based parser.

**Streaming / memory-conscious design:** the file is scanned once with
`osmium.SimpleHandler` callbacks (`node`/`way`/`relation`), firing one
OSM object at a time. A cheap tag-key/value check
(`osm_pbf_loader._is_relevant_tags`) rejects the overwhelming majority
of objects (plain roads, buildings, amenities — untagged or
non-industrial) before any Python dict or geometry is built. Only
matched candidates (a small fraction of the country-wide object count)
are kept in memory and fed into the same `normalize_osm_facilities` /
`validate_facilities` / `build_facility_report` used for GeoJSON/CSV.
The whole country is never materialized as a single GeoDataFrame.

**Object types supported:**
- **Nodes** with a relevant tag → `Point` directly from the node's own
  location (always available, no index lookup needed).
- **Ways** with a relevant tag → geometry built from the way's own
  member-node locations (resolved via libosmium's `flex_mem` location
  index, populated as nodes stream by). A closed way with ≥4 resolved
  points → `Polygon` (the **original** boundary is preserved, never
  replaced by a centroid); an open way → `LineString`, which is not a
  supported facility geometry type and is therefore consistently
  flagged `invalid_geometry` and preserved in the rejected output
  (never silently dropped). A way referencing a node whose location
  never resolved (e.g. clipped at the extract's bounding box) is
  likewise preserved with `geometry=None` and flagged.
- **Relations** with a relevant tag → **known limitation:** full
  multipolygon geometry reconstruction (assembling outer/inner member
  ways via libosmium's two-pass `AreaManager`) is **not implemented**.
  That machinery requires a second, indiscriminate pass assembling
  areas for every closed way/relation in the country (not just
  industrial candidates), for a benefit limited to the minority of
  facilities mapped as true multi-ring relations (most real-world
  industrial/power/mining polygons are simple closed ways, which ARE
  fully supported). Candidate relations are instead preserved with
  their `osm_id`/`osm_type`/`osm_tags`/name but `geometry=None`, which
  the existing (unmodified) validation logic correctly flags as
  `invalid_geometry` and preserves — with a `rejection_reason` — in the
  rejected-records output, never silently dropped.

**Candidate tag filter (conservative, by design):** mirrors the
load-bearing tag evidence `classify_facility_type` actually inspects
(`industrial=*`, `landuse=industrial`/`quarry`, `power=plant`/
`generator`, `man_made=works`/`wastewater_plant`/`storage_tank`/`tank`,
`content=lng`/`gas`/`natural_gas`). One deliberate narrowing: `power=*`
is restricted to `{plant, generator}` rather than "any power tag" —
otherwise every transmission tower/pole/line/substation in India (a
huge, facility-scale-irrelevant object count) would become a candidate.
This is a scope limitation of *this loader's* early filter only; the
frozen `classify_facility_type` mapping itself is unchanged.

**Real production smoke-test results** (run against
`aiml/data/raw/india-260904.osm.pbf`, 1,706,172,077 bytes / ~1.7 GB):

| Metric | Value |
|---|---|
| OSM objects scanned | 291,162,792 (260,962,901 nodes / 29,994,525 ways / 205,366 relations) |
| Candidate objects | 115,575 (40,650 nodes / 72,314 ways / 2,611 relations) |
| Normalized facilities (valid) | 112,956 |
| Rejected (preserved, not deleted) | 2,619 — 2,611 relations (geometry unavailable, documented limitation) + 8 open ways (unsupported `LineString`) |
| Facility-type distribution | REFINERY 35, POWER_PLANT 4,029, MINE 10,626, INDUSTRIAL_AREA 25,383, LNG_TERMINAL 0, OTHER_INDUSTRIAL 65,846, UNKNOWN 9,656 |
| Geometry-type distribution | mostly `Point` (nodes) and `Polygon` (closed ways) |
| Processing time | ~1,478 seconds (~25 minutes), single-threaded |
| Peak process memory | reported per-run in `osm_facility_report.json` → `input.pbf_scan_stats.peak_memory_mb` (dominated by the `flex_mem` location index needed to resolve way-node coordinates, not by the small candidate set retained) |

Manually spot-checked normalized records confirm plausibility: e.g.
"Gujarat Refinery" / "BPCL Refinery" / "MRPL" (`industrial=refinery`),
"Varahi underground Powerhouse" / "Chibro Power Plant" (`power=plant`),
various `landuse=quarry` mines, and `power=generator` wind/solar
installations correctly bucketed as `OTHER_INDUSTRIAL` per the existing
(unmodified) normalization rules. Original OSM tags — including
non-Latin-script `name:*` variants — are preserved verbatim with no
`"nan"`-string leakage into `osm_tags`.

Outputs (always written, even when empty):
`data/processed/osm_facilities.csv`, `data/processed/osm_facilities.geojson`,
`data/processed/osm_facility_report.json`, and — only when there is
something to report — `data/processed/osm_facilities_rejected.csv`
(invalid/duplicate records, preserved with a `rejection_reason`, never
silently deleted).

**Limitations:** OSM coverage is whatever the supplied extract contains
— this stage makes no claim about completeness for the study area.
`UNKNOWN`/`OTHER_INDUSTRIAL` are expected outcomes for objects that can't
be confidently mapped, not errors. `LNG_TERMINAL` in particular has no
single universal OSM tag, so it is only assigned given gas-industrial
tag evidence *combined with* an explicit "LNG" name match — otherwise
such objects fall back to `OTHER_INDUSTRIAL`. For PBF input specifically,
multipolygon **relation** geometry is not reconstructed (see "Real OSM
PBF support" above) — such relations are preserved with their id/tags
but flagged `invalid_geometry` rather than contributing a usable
polygon; simple closed-way facility boundaries (the majority of
real-world cases) are unaffected.

**How this differs from Stage I.2:** this stage (I.1) only builds the
canonical facility layer. It does **not** compute `distance_to_facility`,
`facility_association`, `is_within_facility_boundary` or
`attribution_confidence`, and it never reads or modifies
`thermal_events.csv` / `thermal_events_with_persistence.csv`. Associating
thermal events with nearby facilities is Stage I.2, not implemented here.

## Notes

- No trained models or prediction outputs are included.
- Ingestion/feature/model/inference functions outside the FIRMS
  preprocessing stage above are still placeholders with `TODO` comments
  and raise `NotImplementedError`.
- The backend will eventually call into `src/inference/predict.py` through
  a clean interface — avoid duplicating this logic elsewhere.
