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

Most placeholder modules still only have tests that verify the Python
environment and that they import correctly. The implemented GIFT stages
(FIRMS ingestion/preprocessing, Stage G, Stage G.1, Stage I.1, Stage I.2,
Stage I.3, Stage I.4 — see below) all have real unit + integration test
coverage (395 tests total as of Stage I.4).

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
thermal events with nearby facilities is Stage I.2 (see below).

## GIFT Stage I.2 — Thermal Event ↔ Facility Association

**Purpose:** a strictly *geospatial association* layer, and nothing more.
Given the Stage G.1 event table and the Stage I.1 facility layer, it
answers "which normalized OSM facility record(s), if any, are spatially
plausible neighbors of this thermal event, and how confident is that
*spatial* match?" — it never answers "what caused this thermal event?".

> **Facility association is NOT source classification.** A
> `WITHIN_FACILITY` or `NEAR_FACILITY` result means an event is
> spatially plausible near a facility — it is **not** proof the facility
> caused it. No field produced by this stage (or documented anywhere in
> this section) should be read as an industrial/wildfire/agricultural
> source determination; that is a later, not-yet-implemented GIFT stage.
> Likewise, `NO_FACILITY_ASSOCIATION` only means no suitable OSM record
> was found nearby — OSM coverage is incomplete, so this is never a
> claim that the event is confirmed non-industrial.

**Inputs (both read-only, never modified or regenerated by this stage):**
`data/processed/thermal_events_with_persistence.csv` (preferred over the
bare Stage G table because it carries persistence characterization too)
and `data/processed/osm_facilities.geojson`.

Lives in `src/infrastructure/` alongside Stage I.1:

- `association_config.py` — `AssociationConfig`: `association_radius_km`
  (default **5.0 km** — an engineering search-radius threshold, not a
  scientifically validated causal distance; set above Stage G's own
  `spatial_eps_km` of 1.5 km so this stage's event-to-facility search is
  never tighter than the intra-event clustering radius already applied
  upstream), `ambiguity_distance_tolerance_km` (default 0.5 km — how
  close two top candidates' distances must be before the pipeline
  refuses to pick one), `max_candidates_per_event` (default 10 — caps
  only the *detail* file's size, never the main selection logic).
- `association_geometry.py` — spatial-index candidate search. Persisted
  coordinates stay in **EPSG:4326** throughout (matching Stage G / I.1);
  all buffering/distance/containment computation happens internally in
  a custom **India-centered Albers Equal-Area Conic** projection
  (`+proj=aea +lat_1=8 +lat_2=37 +lat_0=22 +lon_0=82 ...`, defined by a
  self-contained PROJ string, not an EPSG lookup — see the module
  docstring for full rationale). Candidates are found via a
  `geopandas`/`STRtree` spatial-index join (buffer each event's Stage G
  convex-hull footprint by the radius, then `sjoin` against facility
  geometries) — **never** an events × facilities distance matrix or
  nested loop.
- `facility_association.py` — deterministic candidate ranking (spatial
  relationship tier → distance → facility geometry quality →
  facility_type → facility_id, in that exact order — see module
  docstring) and the explicit **"do not blindly select the nearest
  facility"** rule: if the top two ranked candidates are in the same
  relationship tier and within `ambiguity_distance_tolerance_km` of each
  other, the event is marked `AMBIGUOUS` and **no single facility is
  selected** in the main output, even though candidates exist.
- `association_pipeline.py` / `association_report.py` — orchestration +
  JSON report. Append-only: every original event column is preserved
  unchanged; new columns are only ever added.
- `run_facility_association.py` — CLI entry point.

Run it (from this `aiml/` directory, after Stage G.1 and Stage I.1 have
produced their outputs):

```bash
python -m src.infrastructure.run_facility_association
```

**Spatial relationship categories** (`facility_association_method`):

| Value | Meaning |
|---|---|
| `WITHIN_FACILITY` | Event centroid lies inside/on a facility geometry. |
| `INTERSECTS_FACILITY` | Event footprint (Stage G convex hull) intersects a facility geometry, but the centroid does not. |
| `NEAR_FACILITY` | No containment/intersection, but centroid-to-facility distance ≤ `association_radius_km`. |
| `AMBIGUOUS` | Top candidates too similar to confidently distinguish — no facility selected. |
| `NO_FACILITY_ASSOCIATION` | No candidate facility satisfies any of the above. |

**Attribution confidence** (`facility_attribution_confidence`, spatial
confidence only — never a probability, never a source-classification
score): `HIGH` (`WITHIN_FACILITY`/`INTERSECTS_FACILITY`), `MEDIUM`
(single unambiguous `NEAR_FACILITY` candidate), `LOW` (`AMBIGUOUS`, or
`NEAR_FACILITY` with multiple candidates in a dense cluster), `NONE`
(`NO_FACILITY_ASSOCIATION`).

**Outputs:**

- `data/processed/thermal_events_with_facility_association.csv` — every
  original event/persistence column, plus `facility_id`,
  `facility_name`, `facility_type`, `facility_association_method`,
  `facility_attribution_confidence`, `facility_distance_km`,
  `candidate_facility_count`, `candidate_facility_ids`. Same row count
  and `event_id` set as the input — no event is ever dropped, even with
  zero facility candidates.
- `data/processed/thermal_event_facility_candidates.csv` — every
  retained (event, facility) candidate pair (capped at
  `max_candidates_per_event` per event), ranked with `candidate_rank`
  and a `candidate_score` (a deterministic sort key, explicitly **not**
  a probability). Preserves ambiguity that the main table collapses.
- `data/processed/facility_association_report.json` — input/association/
  confidence/distance/candidate statistics, runtime, full configuration,
  CRS strategy, and explicit limitations (see report `limitations`).

**Real production run** (179,740 events × 112,956 facilities,
`association_radius_km=5.0`):

| Metric | Value |
|---|---|
| Events with a facility association | 40,079 (22.3%) |
| — `WITHIN_FACILITY` | 10,972 |
| — `INTERSECTS_FACILITY` | 2,089 |
| — `NEAR_FACILITY` | 27,018 |
| Events `AMBIGUOUS` (candidates found, none confidently selected) | 17,542 |
| Events with `NO_FACILITY_ASSOCIATION` | 122,119 |
| Confidence `HIGH` / `MEDIUM` / `LOW` / `NONE` | 13,061 / 15,128 / 29,432 / 122,119 |
| `facility_distance_km` (associated events) | min 0.0, median 1.84, mean 1.94, max 8.29 |
| Candidate count | median 3 (excluding zero-candidate events), max 6,111 (one event's 5 km buffer overlapped a very dense industrial cluster of thousands of individually-mapped OSM objects) |
| Runtime | ~33–62 s end-to-end (load + spatial join + ranking, single-threaded) |

Manually spot-checked: real refineries — Bongaigaon, Reliance, Gujarat,
Guru Gobind Singh — correctly `WITHIN_FACILITY`/`HIGH`, backed by actual
`industrial=refinery` OSM tags (not an arbitrary nearby object); several
coal mines correctly `INTERSECTS_FACILITY`/`WITHIN_FACILITY`; power
plants mostly `NEAR_FACILITY` (OSM often maps them as a small point,
not the full site boundary, so exact containment is less common);
`AMBIGUOUS` events checked by hand had genuinely near-identical
candidate distances (e.g. 4.62 km vs. 4.75 km to two separate mine
polygons) — correctly refusing to guess between them.

**Determinism:** given the same event table, facility table and
configuration, output is byte-for-byte identical across runs (explicit
tie-breaking chain in `facility_association.py`; no reliance on
unordered sets/dicts or input row order).

**Limitations** (also recorded verbatim in every report's
`limitations` field): association ≠ causation; OSM is contextual,
incomplete, crowd-sourced evidence, not ground truth (`NO_FACILITY_
ASSOCIATION` ≠ confirmed non-industrial); facility geometry quality
varies (Stage I.1 does not reconstruct multipolygon relation geometry);
`association_radius_km` is an engineering threshold pending future
calibration against labeled outcomes; `AMBIGUOUS` events deliberately
have no selected `facility_id` even though candidates exist (see the
candidates file for the full ranked list); event `footprint_wkt` is the
Stage G *observed-detection envelope*, not the true physical fire
perimeter, so containment/intersection results inherit that same
approximation.

## GIFT Stage I.3 — Facility Fingerprinting & Historical Thermal Baseline

**Purpose:** a purely *descriptive historical baseline*, and nothing
more. Given every event Stage I.2 confidently associated with a single
facility, it answers "what thermal behaviour has historically been
observed around/within this facility?" — it never answers "is this
facility's current activity unusual?" (that comparison is GIFT Stage
I.4, not implemented here), and it never assigns any anomaly score,
industrial/wildfire/agricultural source label, or risk score.

> **A fingerprint is a summary, not a judgment.** `event_count = 175`
> and `peak_frp_median = 3.68` describe what has been observed near a
> facility historically; they are not a claim that the facility is
> "risky", "normal" or anything else. No comparison against any other
> facility, threshold, or future event happens anywhere in this stage.

**Which events count as a confirmed historical observation:** only
events where Stage I.2 selected exactly one facility (`facility_id` not
null — i.e. `WITHIN_FACILITY`/`INTERSECTS_FACILITY`/`NEAR_FACILITY`).
`AMBIGUOUS` events (Stage I.2 explicitly declined to pick a single
facility) and `NO_FACILITY_ASSOCIATION` events are **never** treated as
a confirmed observation for any facility — deliberately conservative,
per the same "don't guess" principle as Stage I.2. An informational-only
`ambiguous_candidate_opportunity_count` separately records how often a
facility appeared as an *unresolved* candidate for an ambiguous event,
without ever inflating `event_count`.

**Inputs (both read-only, never modified or regenerated by this
stage):** `data/processed/thermal_events_with_facility_association.csv`
(Stage I.2's output — already carries every Stage G/G.1 event column
plus the facility association columns, so this is the *only* event
table Stage I.3 reads) and `data/processed/osm_facilities.csv` (the full
Stage I.1 facility universe, so that facilities with zero associated
events still get a fingerprint row). `thermal_event_detections.csv`
(1.17M rows) is intentionally never read — Stage G's own per-event
`detection_count` already counts exactly the FIRMS detections
contributing to that event, so summing it per facility is equivalent
and far cheaper than re-reading the detection-level table.

Lives in `src/fingerprinting/`:

- `robust_stats.py` — median / MAD (raw, **not** scaled to a
  normal-consistent std) / quantile helpers that drop NaNs and return
  `None` (never a fabricated `0.0`) for empty/all-missing input, plus a
  fully vectorized `grouped_summary_stats()` (pandas `groupby` +
  `transform`, no per-group Python loop) used for the real ~8k-facility
  aggregation.
- `fingerprint_config.py` — `FingerprintConfig`: `min_observations_for_
  limited_history` (default **3**) and `min_observations_for_
  established_baseline` (default **10**) — both explicit engineering
  thresholds, not scientifically validated minimum sample sizes (see
  module docstring for rationale); fully configurable.
- `facility_fingerprint.py` — the core per-facility aggregation
  (observation volume, temporal/seasonal/day-night, thermal-intensity,
  event-size, duration, persistence, spatial and confidence fingerprints
  — see module docstring for the exact, deterministic definition of
  every field, including the day/night classification rule and the
  single-start-month attribution rule for long-running events).
- `monthly_profile.py` — the companion long-format
  `(facility_id, month)` activity table.
- `fingerprint_report.py` / `fingerprint_pipeline.py` — JSON report +
  orchestration. Both event and facility inputs are loaded exactly once;
  no spatial join is repeated (Stage I.2 already did the expensive
  part).
- `run_facility_fingerprinting.py` — CLI entry point.

Run it (from this `aiml/` directory, after Stage I.2 has produced its
output):

```bash
python -m src.fingerprinting.run_facility_fingerprinting
```

**Fingerprint status** (`fingerprint_status`, based on confirmed
`event_count`):

| Value | `event_count` | Meaning |
|---|---|---|
| `NO_OBSERVATIONS` | 0 | No confirmed historical association at all. |
| `INSUFFICIENT_HISTORY` | 1–2 | Too few to distinguish a coincidence from a pattern. |
| `LIMITED_HISTORY` | 3–9 | Some history; not yet a robust baseline. |
| `ESTABLISHED_BASELINE` | 10+ | Enough confirmed events for the median/MAD/quantile statistics to be a meaningful historical summary. |

For a `NO_OBSERVATIONS` facility, every *count* field (`event_count`,
`detection_count`, `active_month_count`, …) is a legitimate `0`, while
every *statistical* field (`peak_frp_median`, `distance_km_max`,
`day_event_fraction`, …) is `null` — "zero observations" is never
confused with "an observed value of zero".

**Fingerprint fields** (one row per Stage I.1 facility in
`facility_thermal_fingerprints.csv`): observation volume
(`event_count`, `detection_count`, `observation_day_count`,
`active_month_count`); temporal (`first_observation_date`,
`last_observation_date`, `observation_span_days`, `day_event_count`,
`night_event_count`, `day_event_fraction`, `night_event_fraction` —
day/night classified per event from Stage G's own
`day_detection_count`/`night_detection_count` as `DAY`/`NIGHT`/`MIXED`/
`UNKNOWN`; `MIXED`/`UNKNOWN` events count toward the fraction
denominators but neither numerator, so the two fractions are **not**
guaranteed to sum to 1); thermal intensity (`peak_frp_{median,mad,p25,
p75,p90,max}`); event size (`event_size_{...}`, detection count per
event); duration (`duration_hours_{...}`, Stage G's observed-detection
span, not physical burn duration); persistence (`persistent_event_
count`/`_fraction`, `recurring_event_count`/`_fraction`,
`short_lived_event_count`/`_fraction`, `insufficient_observations_
event_count`, from Stage G.1 labels — always summing back to
`event_count`); spatial (`distance_km_{...}` from Stage I.2's selected-
facility distance, plus `within_facility_count`/`intersects_facility_
count`/`near_facility_count`); confidence composition
(`high_confidence_event_count`/`medium_confidence_event_count`/
`low_confidence_event_count`, from Stage I.2's spatial confidence, never
a probability); `ambiguous_candidate_opportunity_count` (informational
only); `fingerprint_observation_count` (== `event_count`) and
`fingerprint_status`.

**Monthly profile** (`facility_monthly_thermal_profile.csv`,
`facility_id, month, event_count, detection_count, event_fraction`):
up to 12 rows per facility with at least one confirmed event; facilities
with zero confirmed events contribute no rows (their `NO_OBSERVATIONS`
status is already fully captured in the main table). Each event is
attributed to the single UTC calendar month of its `event_start` — a
long-running persistent event is not split/duplicated across every
month it spans.

**Outputs:**

- `data/processed/facility_thermal_fingerprints.csv` — one row per
  Stage I.1 facility (112,956 rows in production), sorted by
  `facility_id`.
- `data/processed/facility_monthly_thermal_profile.csv` — the monthly
  activity table, sorted by `(facility_id, month)`.
- `data/processed/facility_fingerprinting_report.json` — input/coverage/
  observation/persistence/type/confidence/temporal statistics, full
  configuration + rationale, runtime, and explicit limitations.

**Real production run** (179,740 events, 40,079 confirmed-associated,
112,956 facilities, default thresholds):

| Metric | Value |
|---|---|
| Facilities with ≥1 confirmed observation | 7,812 (6.9%) |
| — `INSUFFICIENT_HISTORY` (1–2 events) | 4,568 |
| — `LIMITED_HISTORY` (3–9 events) | 2,348 |
| — `ESTABLISHED_BASELINE` (10+ events) | 896 |
| Facilities with zero observations (`NO_OBSERVATIONS`) | 105,144 (93.1%) |
| Events per observed facility | min 1, median 2, mean 5.13, max 175 |
| Runtime | ~2 s end-to-end (no spatial join repeated; two CSV reads) |

Manually spot-checked `ESTABLISHED_BASELINE` facilities: **Reliance
Refinery** (98 events) and **Guru Gobind Singh Refinery** (79 events)
both `REFINERY`; **Kusmunda Open Cast Mine**, **Kathara coal mine**,
**Belpahar Coal Mine** (110–169 events each) all real, well-known Indian
coal mines correctly typed `MINE`; **Angul Steel Plant** and **Tata
Steel Meramandali** (`INDUSTRIAL_AREA`, 153–175 events, spans of
~727–730 days matching the full 2023–2024 FIRMS window) — all plausible.
Zero-observation facilities spot-checked (e.g. a cashew factory, a
corporate R&D center) correctly show `event_count = 0` and every
statistical field `null`, not `0`.

**Determinism:** given the same inputs and configuration, every output
value is byte-for-byte identical across runs (all aggregation via
vectorized pandas `groupby`, sorted deterministically by `facility_id`
— never unordered set/dict iteration).

**Limitations** (also recorded verbatim in the report's `limitations`
field): purely descriptive, not anomaly detection or source
classification (that is Stage I.4 and beyond, not implemented here);
historical FIRMS observations are satellite detections, not confirmed
facility fires; the underlying Stage I.2 spatial association is
contextual evidence, not causal proof; OSM coverage is incomplete, so a
`NO_OBSERVATIONS`/`INSUFFICIENT_HISTORY` facility may simply reflect
sparse coverage rather than genuine absence of activity; `PERSISTENT`/
`RECURRING` labels describe an observed satellite-detection pattern, not
continuous physical burning; `LIMITED_HISTORY`/`INSUFFICIENT_HISTORY`
facilities do not have a statistically robust baseline; the two history
thresholds are engineering choices pending future calibration; no
anomaly score, source classification, or ML model is computed anywhere
in this stage.

## GIFT Stage I.4 — Temporal Deviation & Anomaly Detection

**Purpose:** decide whether a thermal event with a *confirmed* Stage I.2
facility association is **unusual relative to that facility's own prior
confirmed associations**.

> **Anomaly ≠ industrial fire.** An `ANOMALOUS` / `ELEVATED` result means
> the event's thermal/spatial/persistence pattern differs from what this
> facility has looked like historically. It is **not** a source
> classification, risk score, or claim of illegal/hazardous activity.

**Walk-forward / prior-only baseline (mandatory):** for each facility,
confirmed events are sorted by `(event_start ASC, event_id ASC)`. Event
N is scored using **only** events 1..N−1. The current event is appended
to history **after** scoring. This avoids temporal leakage from building
a baseline that includes the observation being scored. Stage I.3's
full-history fingerprints are loaded for provenance only and are **never**
used as the scoring baseline.

**Ambiguous / unassociated events:** `AMBIGUOUS` and
`NO_FACILITY_ASSOCIATION` events are preserved in the output with
`anomaly_status=INSUFFICIENT_HISTORY`. They never enter any facility's
confirmed history and are never silently assigned to a facility.

Lives in `src/anomaly_detection/`:

- `config.py` — engineering thresholds/weights (history cutoffs matching
  I.3's 3/10 defaults; `normal_max_score=2.0`, `elevated_max_score=3.5`;
  feature weights; zero-MAD constant-mismatch deviation=3.0).
- `robust_deviation.py` — robust deviation index
  `|x − median| / MAD` (raw MAD, **not** a z-score, never ×1.4826);
  zero-MAD handling (same→0; different→IQR fallback or documented
  constant-mismatch deviation).
- `temporal_baseline.py` — walk-forward per-facility history + feature
  deviations (peak FRP, detection count, duration, distance, persistence
  rarity from G.1 labels, prior-only same-month peak FRP).
- `anomaly_scoring.py` — weighted mean of *available* feature deviations
  (missing features excluded, not treated as zero); status + confidence.
- `anomaly_explanation.py` — deterministic template explanations (no LLM).
- `anomaly_report.py` / `anomaly_pipeline.py` / `run_anomaly_detection.py`.

Run it (from this `aiml/` directory, after Stage I.2):

```bash
python -m src.anomaly_detection.run_anomaly_detection
```

**Anomaly status** (engineering thresholds on the robust-deviation score):

| Status | Rule |
|---|---|
| `INSUFFICIENT_HISTORY` | <3 prior confirmed observations at the facility, or no confirmed association / ambiguous |
| `NORMAL` | score < 2.0 |
| `ELEVATED` | 2.0 ≤ score < 3.5 |
| `ANOMALOUS` | score ≥ 3.5 |

**Confidence** (`NONE` / `LOW` / `MEDIUM` / `HIGH`) reflects evidence
quality (prior history depth + features evaluated) — **not** a fire
probability.

**Outputs:**

- `data/processed/thermal_events_with_anomaly_detection.csv` — all
  original event columns plus I.4 fields (`anomaly_score`,
  `anomaly_status`, `anomaly_confidence`, per-feature deviations,
  baseline medians/MADs, `anomaly_explanation`). Same row count as
  input; no event deleted.
- `data/processed/anomaly_detection_report.json` — coverage, status/
  confidence counts, feature availability, configuration, leakage
  validation note, limitations.

**Real production run** (179,740 events; walk-forward over 40,079
confirmed associations):

| Metric | Value |
|---|---|
| `INSUFFICIENT_HISTORY` | 155,353 |
| `NORMAL` | 16,610 |
| `ELEVATED` | 3,292 |
| `ANOMALOUS` | 4,485 |
| Confidence `NONE` / `MEDIUM` / `HIGH` | 155,353 / 10,362 / 14,025 |
| Runtime | ~274 s |

**Limitations:** most facilities lack confirmed history, so most events
cannot be strongly scored; OSM association ≠ source identity; FIRMS is
not continuous ground truth; G.1 persistence is an observation pattern,
not physical burn persistence; long Stage G events are not split;
2023–2024 may miss some operational regimes; near-zero historical MAD
can produce very large robust-deviation ratios (status thresholds still
apply); thresholds/weights are engineering choices without independent
ground-truth validation — **no accuracy claims are reported**.

## Notes

- No trained models or prediction outputs are included.
- Ingestion/feature/model/inference functions outside the FIRMS
  preprocessing stage above are still placeholders with `TODO` comments
  and raise `NotImplementedError`.
- The backend will eventually call into `src/inference/predict.py` through
  a clean interface — avoid duplicating this logic elsewhere.
