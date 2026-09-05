"""Integration tests for the Stage I.1 OSM facility ingestion pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from src.infrastructure.config import InfrastructureConfig
from src.infrastructure.facility_report import LOADED_STATUS, NO_PRODUCTION_INPUT_STATUS, PRODUCTION_PBF_STATUS
from src.infrastructure.infrastructure_pipeline import (
    InfrastructureResult,
    run_infrastructure_ingestion,
    save_outputs,
)
from src.infrastructure.osm_loader import discover_default_osm_input, load_osm_extract


def _write_geojson(path: Path, features: list[dict]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")


def _synthetic_extract(path: Path) -> None:
    _write_geojson(
        path,
        [
            {
                "id": "way/111",
                "type": "Feature",
                "properties": {"industrial": "refinery", "name": "Test Refinery"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[80.0, 20.0], [80.01, 20.0], [80.01, 20.01], [80.0, 20.01], [80.0, 20.0]]],
                },
            },
            {
                "id": "node/222",
                "type": "Feature",
                "properties": {"power": "plant", "plant:source": "coal", "name": "Test Power Plant"},
                "geometry": {"type": "Point", "coordinates": [77.0, 12.0]},
            },
            {
                "id": "node/333",
                "type": "Feature",
                "properties": {"shop": "bakery", "name": "Random Shop"},
                "geometry": {"type": "Point", "coordinates": [78.0, 13.0]},
            },
            # Exact duplicate of node/222 (same stable osm id) -- a common
            # artifact of overlapping bounding-box Overpass exports.
            {
                "id": "node/222",
                "type": "Feature",
                "properties": {"power": "plant", "plant:source": "coal", "name": "Test Power Plant"},
                "geometry": {"type": "Point", "coordinates": [77.0, 12.0]},
            },
            # Invalid coordinates.
            {
                "id": "node/444",
                "type": "Feature",
                "properties": {"landuse": "industrial", "name": "Bad Coords"},
                "geometry": {"type": "Point", "coordinates": [999.0, 999.0]},
            },
        ],
    )


# ---------------------------------------------------------------------------
# "No production input" behavior
# ---------------------------------------------------------------------------


def test_no_input_path_does_not_fail_and_reports_missing_coverage() -> None:
    result = run_infrastructure_ingestion(None, InfrastructureConfig())

    assert isinstance(result, InfrastructureResult)
    assert len(result.facilities_gdf) == 0
    assert result.report["input"]["status"] == NO_PRODUCTION_INPUT_STATUS
    assert result.report["input"]["raw_record_count"] == 0
    assert "NO REAL OSM COVERAGE" in result.report["coverage_status"]


def test_no_input_path_never_fabricates_records() -> None:
    result = run_infrastructure_ingestion(None, InfrastructureConfig())
    assert result.facilities_gdf.empty
    assert result.rejected_df.empty


def test_discover_default_osm_input_returns_none_in_a_fresh_directory(tmp_path: Path) -> None:
    """Confirms the pipeline's upstream discovery step also correctly
    reports 'nothing found' rather than inventing a file."""
    assert discover_default_osm_input(tmp_path) is None


# ---------------------------------------------------------------------------
# Synthetic-fixture ingestion (end to end)
# ---------------------------------------------------------------------------


def test_synthetic_fixture_end_to_end_counts(tmp_path: Path) -> None:
    path = tmp_path / "osm_facilities_synthetic.geojson"
    _synthetic_extract(path)

    result = run_infrastructure_ingestion(path, InfrastructureConfig())

    assert result.report["input"]["status"] == LOADED_STATUS
    assert result.report["input"]["raw_record_count"] == 5
    # 1 duplicate + 1 invalid-coordinate record excluded -> 3 valid facilities.
    assert len(result.facilities_gdf) == 3
    assert len(result.rejected_df) == 2


def test_synthetic_fixture_facility_type_counts(tmp_path: Path) -> None:
    path = tmp_path / "osm_facilities_synthetic.geojson"
    _synthetic_extract(path)

    result = run_infrastructure_ingestion(path, InfrastructureConfig())
    types = set(result.facilities_gdf["facility_type"])
    assert "REFINERY" in types
    assert "POWER_PLANT" in types
    assert "UNKNOWN" in types  # the bakery


def test_synthetic_fixture_duplicate_and_invalid_are_preserved_with_reasons(tmp_path: Path) -> None:
    path = tmp_path / "osm_facilities_synthetic.geojson"
    _synthetic_extract(path)

    result = run_infrastructure_ingestion(path, InfrastructureConfig())
    reasons = set(result.rejected_df["rejection_reason"])
    assert any("duplicate_facility_id" in r for r in reasons)
    assert any("invalid_coordinates" in r for r in reasons)
    # Never silently deleted -- every rejected row keeps its own data.
    assert "osm_id" in result.rejected_df.columns


def test_pipeline_never_touches_stage_g_or_g1_outputs(tmp_path: Path) -> None:
    """Stage I.1 must not read or modify thermal_events*.csv."""
    path = tmp_path / "osm_facilities_synthetic.geojson"
    _synthetic_extract(path)

    # Sentinel files representing the frozen Stage G/G.1 outputs.
    events_path = tmp_path / "thermal_events.csv"
    persistence_path = tmp_path / "thermal_events_with_persistence.csv"
    events_path.write_text("event_id\nEVT_0000001\n", encoding="utf-8")
    persistence_path.write_text("event_id\nEVT_0000001\n", encoding="utf-8")
    events_before = events_path.read_bytes()
    persistence_before = persistence_path.read_bytes()

    run_infrastructure_ingestion(path, InfrastructureConfig())

    assert events_path.read_bytes() == events_before
    assert persistence_path.read_bytes() == persistence_before


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_pipeline_is_deterministic_across_runs(tmp_path: Path) -> None:
    path = tmp_path / "osm_facilities_synthetic.geojson"
    _synthetic_extract(path)

    result_1 = run_infrastructure_ingestion(path, InfrastructureConfig(), source_version="fixed")
    result_2 = run_infrastructure_ingestion(path, InfrastructureConfig(), source_version="fixed")

    pd.testing.assert_frame_equal(
        pd.DataFrame(result_1.facilities_gdf.drop(columns=["geometry"])),
        pd.DataFrame(result_2.facilities_gdf.drop(columns=["geometry"])),
    )
    assert result_1.report["output"]["final_facility_count"] == result_2.report["output"]["final_facility_count"]


def test_repeated_cli_style_runs_produce_identical_files(tmp_path: Path) -> None:
    path = tmp_path / "osm_facilities_synthetic.geojson"
    _synthetic_extract(path)

    out_dir_1 = tmp_path / "out1"
    out_dir_2 = tmp_path / "out2"

    for out_dir in (out_dir_1, out_dir_2):
        result = run_infrastructure_ingestion(path, InfrastructureConfig(), source_version="fixed")
        save_outputs(
            result,
            out_dir / "osm_facilities.csv",
            out_dir / "osm_facilities.geojson",
            out_dir / "osm_facilities_rejected.csv",
        )

    csv_1 = pd.read_csv(out_dir_1 / "osm_facilities.csv")
    csv_2 = pd.read_csv(out_dir_2 / "osm_facilities.csv")
    pd.testing.assert_frame_equal(csv_1, csv_2)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def test_report_contains_required_summary_fields(tmp_path: Path) -> None:
    path = tmp_path / "osm_facilities_synthetic.geojson"
    _synthetic_extract(path)

    result = run_infrastructure_ingestion(path, InfrastructureConfig())
    report = result.report

    assert report["pipeline_stage"].startswith("GIFT Stage I.1")
    assert "input" in report
    assert "normalization" in report
    assert "validation" in report
    assert "duplicate_detection" in report
    assert "output" in report
    assert "coverage_status" in report
    assert "performance" in report
    assert report["performance"]["processing_seconds"] >= 0
    assert report["reproducibility"]["deterministic"] is True
    assert any("Stage I.2" in note for note in report["notes"])
    assert any("NOT ground truth" in note or "not ground truth" in note.lower() for note in report["notes"])


def test_report_facility_type_counts_include_zero_categories(tmp_path: Path) -> None:
    path = tmp_path / "osm_facilities_synthetic.geojson"
    _synthetic_extract(path)

    result = run_infrastructure_ingestion(path, InfrastructureConfig())
    counts = result.report["normalization"]["facility_type_counts"]
    # MINE and LNG_TERMINAL are absent from the fixture but must still be
    # reported as zero, not omitted.
    assert counts["MINE"] == 0
    assert counts["LNG_TERMINAL"] == 0
    assert counts["REFINERY"] == 1


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def test_save_outputs_writes_csv_and_geojson(tmp_path: Path) -> None:
    path = tmp_path / "osm_facilities_synthetic.geojson"
    _synthetic_extract(path)
    result = run_infrastructure_ingestion(path, InfrastructureConfig())

    csv_path = tmp_path / "out" / "osm_facilities.csv"
    geojson_path = tmp_path / "out" / "osm_facilities.geojson"
    rejected_path = tmp_path / "out" / "osm_facilities_rejected.csv"

    save_outputs(result, csv_path, geojson_path, rejected_path)

    assert csv_path.exists()
    assert geojson_path.exists()
    assert rejected_path.exists()

    reloaded_csv = pd.read_csv(csv_path)
    assert len(reloaded_csv) == len(result.facilities_gdf)

    reloaded_geojson = gpd.read_file(geojson_path)
    assert len(reloaded_geojson) == len(result.facilities_gdf)
    assert reloaded_geojson.geometry.iloc[0] is not None


def test_save_outputs_skips_rejected_file_when_nothing_rejected(tmp_path: Path) -> None:
    # A fixture with no duplicates/invalid records.
    path = tmp_path / "clean.geojson"
    _write_geojson(
        path,
        [
            {
                "id": "node/1",
                "type": "Feature",
                "properties": {"power": "plant"},
                "geometry": {"type": "Point", "coordinates": [77.0, 12.0]},
            }
        ],
    )
    result = run_infrastructure_ingestion(path, InfrastructureConfig())
    assert result.rejected_df.empty

    rejected_path = tmp_path / "out" / "osm_facilities_rejected.csv"
    save_outputs(result, tmp_path / "out" / "f.csv", tmp_path / "out" / "f.geojson", rejected_path)
    assert not rejected_path.exists()


def test_load_osm_extract_used_by_pipeline_does_not_mutate_source_file(tmp_path: Path) -> None:
    path = tmp_path / "osm_facilities_synthetic.geojson"
    _synthetic_extract(path)
    original = path.read_bytes()

    run_infrastructure_ingestion(path, InfrastructureConfig())
    load_osm_extract(path)

    assert path.read_bytes() == original


# ---------------------------------------------------------------------------
# OSM PBF input (real-format extension of Stage I.1 -- see osm_pbf_loader.py)
# ---------------------------------------------------------------------------


def _write_pbf_fixture(path: Path) -> None:
    import osmium
    import osmium.osm.mutable as mutable

    common = {"version": 1, "changeset": 1, "timestamp": "2020-01-01T00:00:00Z", "uid": 1}
    writer = osmium.SimpleWriter(str(path))
    try:
        writer.add_node(
            mutable.Node(
                id=101,
                location=(77.0, 12.0),
                tags={"power": "plant", "plant:source": "coal", "name": "Test Power Plant"},
                **common,
            )
        )
        writer.add_node(
            mutable.Node(id=102, location=(78.0, 13.0), tags={"shop": "bakery", "name": "Random Shop"}, **common)
        )
        ring = [(80.0, 20.0), (80.01, 20.0), (80.01, 20.01), (80.0, 20.01)]
        for i, (lon, lat) in enumerate(ring):
            writer.add_node(mutable.Node(id=200 + i, location=(lon, lat), tags={}, **common))
        writer.add_way(
            mutable.Way(
                id=300,
                nodes=[200, 201, 202, 203, 200],
                tags={"industrial": "refinery", "name": "Test Refinery"},
                **common,
            )
        )
        writer.add_relation(
            mutable.Relation(
                id=400,
                members=[("w", 300, "outer")],
                tags={"type": "multipolygon", "landuse": "industrial", "name": "Unreconstructed Relation"},
                **common,
            )
        )
    finally:
        writer.close()


def test_pbf_input_reports_production_pbf_status(tmp_path: Path) -> None:
    path = tmp_path / "fixture.osm.pbf"
    _write_pbf_fixture(path)

    result = run_infrastructure_ingestion(path, InfrastructureConfig())

    assert result.report["input"]["status"] == PRODUCTION_PBF_STATUS
    assert result.report["input"]["pbf_scan_stats"] is not None
    assert result.report["input"]["pbf_scan_stats"]["osm_objects_scanned"] >= 7
    assert result.report["input"]["file_size_bytes"] > 0


def test_pbf_input_produces_expected_facility_types(tmp_path: Path) -> None:
    path = tmp_path / "fixture.osm.pbf"
    _write_pbf_fixture(path)

    result = run_infrastructure_ingestion(path, InfrastructureConfig())

    types = set(result.facilities_gdf["facility_type"])
    assert "POWER_PLANT" in types
    assert "REFINERY" in types


def test_pbf_input_relation_preserved_in_rejected_with_reason(tmp_path: Path) -> None:
    """The unreconstructed multipolygon relation must be preserved (never
    silently dropped) with an explicit rejection reason."""
    path = tmp_path / "fixture.osm.pbf"
    _write_pbf_fixture(path)

    result = run_infrastructure_ingestion(path, InfrastructureConfig())

    relation_rows = result.rejected_df[result.rejected_df["osm_type"] == "relation"]
    assert len(relation_rows) == 1
    assert "invalid_or_unsupported_geometry" in relation_rows.iloc[0]["rejection_reason"]
    assert relation_rows.iloc[0]["osm_id"] == "400"


def test_pbf_input_is_deterministic_across_runs(tmp_path: Path) -> None:
    path = tmp_path / "fixture.osm.pbf"
    _write_pbf_fixture(path)

    result_1 = run_infrastructure_ingestion(path, InfrastructureConfig(), source_version="fixed")
    result_2 = run_infrastructure_ingestion(path, InfrastructureConfig(), source_version="fixed")

    pd.testing.assert_frame_equal(
        pd.DataFrame(result_1.facilities_gdf.drop(columns=["geometry"])),
        pd.DataFrame(result_2.facilities_gdf.drop(columns=["geometry"])),
    )
    stats_1 = dict(result_1.report["input"]["pbf_scan_stats"])
    stats_2 = dict(result_2.report["input"]["pbf_scan_stats"])
    stats_1.pop("processing_seconds", None)
    stats_2.pop("processing_seconds", None)
    assert stats_1 == stats_2  # exact scan/candidate counts must match; timing may vary slightly.


def test_pbf_input_does_not_touch_stage_g_or_g1_outputs(tmp_path: Path) -> None:
    path = tmp_path / "fixture.osm.pbf"
    _write_pbf_fixture(path)

    events_path = tmp_path / "thermal_events.csv"
    events_path.write_text("event_id\nEVT_0000001\n", encoding="utf-8")
    events_before = events_path.read_bytes()

    run_infrastructure_ingestion(path, InfrastructureConfig())

    assert events_path.read_bytes() == events_before
