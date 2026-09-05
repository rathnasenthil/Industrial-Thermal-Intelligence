"""Tests for src.infrastructure.osm_pbf_loader (streaming OSM PBF ingestion).

Uses tiny, in-memory-constructed synthetic `.osm.pbf` fixtures (built with
`osmium.SimpleWriter` + `osmium.osm.mutable`) -- never the real India-wide
extract. The real extract is exercised separately via a manual smoke test
(see aiml/README.md, GIFT Stage I.1), not in this automated suite.
"""

from __future__ import annotations

from pathlib import Path

import osmium
import osmium.osm.mutable as mutable
import pytest

from src.infrastructure.osm_loader import load_osm_extract
from src.infrastructure.osm_pbf_loader import PbfScanStats, load_osm_pbf

_COMMON_ATTRS = {"version": 1, "changeset": 1, "timestamp": "2020-01-01T00:00:00Z", "uid": 1}


def _write_pbf_fixture(
    path: Path,
    nodes: list[dict],
    ways: list[dict] | None = None,
    relations: list[dict] | None = None,
) -> None:
    """Build a tiny synthetic `.osm.pbf` file for tests.

    Args:
        path: Output file path (must not already exist).
        nodes: Each dict needs ``id``, ``lon``, ``lat``, and optional ``tags``.
        ways: Each dict needs ``id``, ``node_ids`` (list[int]), and optional ``tags``.
        relations: Each dict needs ``id``, ``members`` (list of (type, ref, role)
            tuples), and optional ``tags``.
    """
    writer = osmium.SimpleWriter(str(path))
    try:
        for n in nodes:
            writer.add_node(
                mutable.Node(
                    id=n["id"],
                    location=(n["lon"], n["lat"]),
                    tags=n.get("tags", {}),
                    **_COMMON_ATTRS,
                )
            )
        for w in ways or []:
            writer.add_way(
                mutable.Way(id=w["id"], nodes=w["node_ids"], tags=w.get("tags", {}), **_COMMON_ATTRS)
            )
        for r in relations or []:
            writer.add_relation(
                mutable.Relation(
                    id=r["id"], members=r["members"], tags=r.get("tags", {}), **_COMMON_ATTRS
                )
            )
    finally:
        writer.close()


def _refinery_ring_way(way_id: int = 300, first_node_id: int = 200) -> tuple[list[dict], dict]:
    """Four distinct boundary nodes + a closed way reusing the first node id."""
    coords = [(80.0, 20.0), (80.01, 20.0), (80.01, 20.01), (80.0, 20.01)]
    nodes = [
        {"id": first_node_id + i, "lon": lon, "lat": lat} for i, (lon, lat) in enumerate(coords)
    ]
    node_ids = [n["id"] for n in nodes] + [first_node_id]  # closed: last ref == first ref
    way = {"id": way_id, "node_ids": node_ids, "tags": {"industrial": "refinery", "name": "Test Refinery"}}
    return nodes, way


# ---------------------------------------------------------------------------
# Node extraction / relevant-tag filtering
# ---------------------------------------------------------------------------


def test_pbf_loader_extracts_candidate_node_as_point(tmp_path: Path) -> None:
    path = tmp_path / "fixture.osm.pbf"
    _write_pbf_fixture(
        path,
        nodes=[
            {
                "id": 101,
                "lon": 77.0,
                "lat": 12.0,
                "tags": {"power": "plant", "plant:source": "coal", "name": "Test Power Plant"},
            }
        ],
    )

    gdf, stats = load_osm_pbf(path)

    assert len(gdf) == 1
    row = gdf.iloc[0]
    assert row["osm_id"] == "101"
    assert row["osm_type"] == "node"
    assert row["name"] == "Test Power Plant"
    assert row["raw_tags"]["power"] == "plant"
    assert row["geometry"].geom_type == "Point"
    assert row["geometry"].x == pytest.approx(77.0)
    assert row["geometry"].y == pytest.approx(12.0)
    assert stats.nodes_scanned == 1
    assert stats.candidate_nodes == 1


def test_pbf_loader_filters_out_irrelevant_node(tmp_path: Path) -> None:
    path = tmp_path / "fixture.osm.pbf"
    _write_pbf_fixture(
        path,
        nodes=[{"id": 102, "lon": 78.0, "lat": 13.0, "tags": {"shop": "bakery", "name": "Random Shop"}}],
    )

    gdf, stats = load_osm_pbf(path)

    assert len(gdf) == 0
    assert stats.nodes_scanned == 1
    assert stats.candidate_nodes == 0


def test_pbf_loader_filters_out_untagged_node(tmp_path: Path) -> None:
    path = tmp_path / "fixture.osm.pbf"
    _write_pbf_fixture(path, nodes=[{"id": 103, "lon": 79.0, "lat": 14.0}])

    gdf, stats = load_osm_pbf(path)

    assert len(gdf) == 0
    assert stats.nodes_scanned == 1
    assert stats.candidate_nodes == 0


def test_pbf_loader_no_network_required(tmp_path: Path) -> None:
    """Loading a local PBF file must never attempt any network access --
    exercised implicitly (no mocking needed) since this test suite has no
    network available in CI; a hang or connection error would fail it."""
    path = tmp_path / "fixture.osm.pbf"
    _write_pbf_fixture(
        path, nodes=[{"id": 1, "lon": 0.0, "lat": 0.0, "tags": {"industrial": "mine"}}]
    )
    gdf, _stats = load_osm_pbf(path)
    assert len(gdf) == 1


# ---------------------------------------------------------------------------
# Way extraction / geometry handling
# ---------------------------------------------------------------------------


def test_pbf_loader_extracts_closed_way_as_polygon(tmp_path: Path) -> None:
    path = tmp_path / "fixture.osm.pbf"
    nodes, way = _refinery_ring_way()
    _write_pbf_fixture(path, nodes=nodes, ways=[way])

    gdf, stats = load_osm_pbf(path)

    assert len(gdf) == 1
    row = gdf.iloc[0]
    assert row["osm_id"] == "300"
    assert row["osm_type"] == "way"
    assert row["geometry"].geom_type == "Polygon"
    assert row["geometry"].is_valid
    assert stats.ways_scanned == 1
    assert stats.candidate_ways == 1
    assert stats.ways_geometry_unresolved == 0


def test_pbf_loader_filters_out_irrelevant_way(tmp_path: Path) -> None:
    path = tmp_path / "fixture.osm.pbf"
    nodes = [{"id": 1, "lon": 0.0, "lat": 0.0}, {"id": 2, "lon": 1.0, "lat": 1.0}]
    way = {"id": 10, "node_ids": [1, 2], "tags": {"highway": "residential"}}
    _write_pbf_fixture(path, nodes=nodes, ways=[way])

    gdf, stats = load_osm_pbf(path)

    assert len(gdf) == 0
    assert stats.ways_scanned == 1
    assert stats.candidate_ways == 0


def test_pbf_loader_open_way_produces_unsupported_linestring_not_silently_dropped(tmp_path: Path) -> None:
    """An open (non-closed) candidate way cannot be a facility polygon.
    It must still be preserved (not silently dropped) so downstream
    validation can flag it explicitly as an unsupported geometry type."""
    path = tmp_path / "fixture.osm.pbf"
    nodes = [{"id": 1, "lon": 0.0, "lat": 0.0}, {"id": 2, "lon": 1.0, "lat": 1.0}]
    way = {"id": 11, "node_ids": [1, 2], "tags": {"landuse": "industrial"}}
    _write_pbf_fixture(path, nodes=nodes, ways=[way])

    gdf, stats = load_osm_pbf(path)

    assert len(gdf) == 1
    assert gdf.iloc[0]["geometry"].geom_type == "LineString"
    assert stats.candidate_ways == 1
    assert stats.ways_geometry_unresolved == 0


def test_pbf_loader_way_with_unresolved_node_location_is_preserved_without_geometry(tmp_path: Path) -> None:
    """A way referencing a node id that never appears in the file (e.g. the
    extract's bounding box clipped it) cannot have its geometry resolved.
    It must still be preserved (osm_id/tags), with geometry=None, so
    validation flags -- rather than silently drops -- it."""
    path = tmp_path / "fixture.osm.pbf"
    nodes = [{"id": 1, "lon": 0.0, "lat": 0.0}]  # node 2 is deliberately absent
    way = {"id": 12, "node_ids": [1, 2], "tags": {"landuse": "industrial"}}
    _write_pbf_fixture(path, nodes=nodes, ways=[way])

    gdf, stats = load_osm_pbf(path)

    assert len(gdf) == 1
    assert gdf.iloc[0]["geometry"] is None
    assert gdf.iloc[0]["osm_id"] == "12"
    assert stats.ways_geometry_unresolved == 1


# ---------------------------------------------------------------------------
# Relation handling (documented limitation: geometry unavailable, preserved)
# ---------------------------------------------------------------------------


def test_pbf_loader_relation_is_preserved_with_tags_but_no_geometry(tmp_path: Path) -> None:
    path = tmp_path / "fixture.osm.pbf"
    nodes, way = _refinery_ring_way()
    relation = {
        "id": 400,
        "members": [("w", way["id"], "outer")],
        "tags": {"type": "multipolygon", "landuse": "industrial", "name": "Test Multipolygon"},
    }
    _write_pbf_fixture(path, nodes=nodes, ways=[way], relations=[relation])

    gdf, stats = load_osm_pbf(path)

    relation_rows = gdf[gdf["osm_type"] == "relation"]
    assert len(relation_rows) == 1
    row = relation_rows.iloc[0]
    assert row["osm_id"] == "400"
    assert row["name"] == "Test Multipolygon"
    assert row["raw_tags"]["landuse"] == "industrial"
    assert row["geometry"] is None  # documented limitation, not silently dropped
    assert stats.candidate_relations == 1
    assert stats.relations_geometry_unavailable == 1


def test_pbf_loader_filters_out_irrelevant_relation(tmp_path: Path) -> None:
    path = tmp_path / "fixture.osm.pbf"
    nodes, way = _refinery_ring_way()
    way["tags"] = {"building": "yes"}
    relation = {"id": 401, "members": [("w", way["id"], "outer")], "tags": {"type": "multipolygon", "building": "yes"}}
    _write_pbf_fixture(path, nodes=nodes, ways=[way], relations=[relation])

    gdf, stats = load_osm_pbf(path)

    assert len(gdf) == 0
    assert stats.relations_scanned == 1
    assert stats.candidate_relations == 0


# ---------------------------------------------------------------------------
# Facility-type mapping downstream of the loader (feeds the same
# canonical normalization used for CSV/GeoJSON -- see osm_normalization.py)
# ---------------------------------------------------------------------------


def test_pbf_candidate_feeds_refinery_classification(tmp_path: Path) -> None:
    from src.infrastructure.config import InfrastructureConfig
    from src.infrastructure.osm_normalization import normalize_osm_facilities

    path = tmp_path / "fixture.osm.pbf"
    nodes, way = _refinery_ring_way()
    _write_pbf_fixture(path, nodes=nodes, ways=[way])

    raw_gdf, _stats = load_osm_pbf(path)
    normalized = normalize_osm_facilities(raw_gdf, InfrastructureConfig(), source_version="test")

    assert normalized.iloc[0]["facility_type"] == "REFINERY"


def test_pbf_candidate_feeds_power_plant_classification(tmp_path: Path) -> None:
    from src.infrastructure.config import InfrastructureConfig
    from src.infrastructure.osm_normalization import normalize_osm_facilities

    path = tmp_path / "fixture.osm.pbf"
    _write_pbf_fixture(
        path,
        nodes=[{"id": 1, "lon": 77.0, "lat": 12.0, "tags": {"power": "plant", "plant:source": "gas"}}],
    )
    raw_gdf, _stats = load_osm_pbf(path)
    normalized = normalize_osm_facilities(raw_gdf, InfrastructureConfig(), source_version="test")

    assert normalized.iloc[0]["facility_type"] == "POWER_PLANT"


def test_pbf_candidate_feeds_mine_classification(tmp_path: Path) -> None:
    from src.infrastructure.config import InfrastructureConfig
    from src.infrastructure.osm_normalization import normalize_osm_facilities

    path = tmp_path / "fixture.osm.pbf"
    _write_pbf_fixture(
        path, nodes=[{"id": 1, "lon": 83.0, "lat": 23.0, "tags": {"industrial": "mine", "name": "Test Mine"}}]
    )
    raw_gdf, _stats = load_osm_pbf(path)
    normalized = normalize_osm_facilities(raw_gdf, InfrastructureConfig(), source_version="test")

    assert normalized.iloc[0]["facility_type"] == "MINE"


def test_pbf_candidate_feeds_industrial_area_classification(tmp_path: Path) -> None:
    from src.infrastructure.config import InfrastructureConfig
    from src.infrastructure.osm_normalization import normalize_osm_facilities

    path = tmp_path / "fixture.osm.pbf"
    nodes, way = _refinery_ring_way()
    way["tags"] = {"landuse": "industrial", "name": "Test Industrial Area"}
    _write_pbf_fixture(path, nodes=nodes, ways=[way])

    raw_gdf, _stats = load_osm_pbf(path)
    normalized = normalize_osm_facilities(raw_gdf, InfrastructureConfig(), source_version="test")

    assert normalized.iloc[0]["facility_type"] == "INDUSTRIAL_AREA"


# ---------------------------------------------------------------------------
# Deterministic IDs
# ---------------------------------------------------------------------------


def test_pbf_loader_produces_deterministic_facility_ids_across_runs(tmp_path: Path) -> None:
    from src.infrastructure.config import InfrastructureConfig
    from src.infrastructure.osm_normalization import normalize_osm_facilities

    path = tmp_path / "fixture.osm.pbf"
    _write_pbf_fixture(
        path, nodes=[{"id": 555, "lon": 77.0, "lat": 12.0, "tags": {"power": "plant"}}]
    )

    ids = set()
    for _ in range(2):
        raw_gdf, _stats = load_osm_pbf(path)
        normalized = normalize_osm_facilities(raw_gdf, InfrastructureConfig(), source_version="test")
        ids.add(normalized.iloc[0]["facility_id"])

    assert ids == {"osm_node_555"}


def test_pbf_loader_deterministic_repeated_scan_stats(tmp_path: Path) -> None:
    path = tmp_path / "fixture.osm.pbf"
    nodes, way = _refinery_ring_way()
    _write_pbf_fixture(path, nodes=nodes, ways=[way])

    _gdf1, stats1 = load_osm_pbf(path)
    _gdf2, stats2 = load_osm_pbf(path)

    assert stats1.nodes_scanned == stats2.nodes_scanned
    assert stats1.candidate_ways == stats2.candidate_ways == 1
    assert stats1.osm_objects_scanned == stats2.osm_objects_scanned


# ---------------------------------------------------------------------------
# Missing file / dispatch integration
# ---------------------------------------------------------------------------


def test_pbf_loader_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_osm_pbf(tmp_path / "missing.osm.pbf")


def test_load_osm_extract_dispatches_pbf_extension(tmp_path: Path) -> None:
    path = tmp_path / "fixture.osm.pbf"
    _write_pbf_fixture(
        path, nodes=[{"id": 1, "lon": 77.0, "lat": 12.0, "tags": {"power": "plant"}}]
    )

    gdf = load_osm_extract(path)

    assert len(gdf) == 1
    assert gdf.iloc[0]["osm_type"] == "node"


def test_pbf_scan_stats_to_dict_shape(tmp_path: Path) -> None:
    path = tmp_path / "fixture.osm.pbf"
    _write_pbf_fixture(
        path, nodes=[{"id": 1, "lon": 77.0, "lat": 12.0, "tags": {"power": "plant"}}]
    )
    _gdf, stats = load_osm_pbf(path)

    assert isinstance(stats, PbfScanStats)
    d = stats.to_dict()
    for key in (
        "input_file",
        "file_size_bytes",
        "osm_objects_scanned",
        "candidate_objects",
        "candidate_nodes",
        "candidate_ways",
        "candidate_relations",
        "ways_geometry_unresolved",
        "relations_geometry_unavailable",
        "processing_seconds",
        "peak_memory_mb",
    ):
        assert key in d
