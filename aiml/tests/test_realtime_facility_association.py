"""
Phase 5: realtime I.2 must match batch find/rank/select semantics.

Does not invoke ``run_facility_association()`` over a full events table.
"""

from __future__ import annotations

from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from realtime.facility_association import (
    FacilityRecord,
    batch_pipeline_invocation_count,
    process_event_facility_association,
)
from src.infrastructure.association_config import AssociationConfig
from src.infrastructure.association_geometry import (
    build_event_geometries,
    find_candidate_pairs,
)
from src.infrastructure.facility_association import (
    AMBIGUOUS,
    NEAR_FACILITY,
    NO_FACILITY_ASSOCIATION,
    WITHIN_FACILITY,
    rank_candidates,
    select_association,
)

_CFG = AssociationConfig()


def _point_wkt(lon: float, lat: float) -> str:
    return f"POINT ({lon} {lat})"


def _poly_wkt(lon: float, lat: float, half: float = 0.01) -> str:
    # ~1 km-ish box at India latitudes (engineering fixture).
    return Polygon(
        [
            (lon - half, lat - half),
            (lon + half, lat - half),
            (lon + half, lat + half),
            (lon - half, lat + half),
        ]
    ).wkt


def _facility(fid: str, lon: float, lat: float, *, polygon: bool = False, ftype: str = "MINE") -> FacilityRecord:
    if polygon:
        return FacilityRecord(
            facility_id=fid,
            facility_name=f"Name {fid}",
            facility_type=ftype,
            geometry_type="Polygon",
            geometry_wkt=_poly_wkt(lon, lat),
        )
    return FacilityRecord(
        facility_id=fid,
        facility_name=f"Name {fid}",
        facility_type=ftype,
        geometry_type="Point",
        geometry_wkt=_point_wkt(lon, lat),
    )


def _batch_one_event(event_id: str, centroid_wkt: str, footprint_wkt: str, facilities: list[FacilityRecord]):
    events_df = pd.DataFrame(
        [{"event_id": event_id, "centroid_wkt": centroid_wkt, "footprint_wkt": footprint_wkt}]
    )
    fac_rows = [
        {
            "facility_id": f.facility_id,
            "facility_name": f.facility_name,
            "facility_type": f.facility_type,
            "geometry_type": f.geometry_type,
        }
        for f in facilities
    ]
    geoms = [__import__("shapely.wkt", fromlist=["loads"]).loads(f.geometry_wkt) for f in facilities]
    facilities_gdf = gpd.GeoDataFrame(fac_rows, geometry=geoms, crs="EPSG:4326")
    events_gdf = build_event_geometries(events_df)
    pairs = find_candidate_pairs(events_gdf, facilities_gdf, _CFG.association_radius_km)
    ranked = rank_candidates(pairs)
    selected = select_association(events_df["event_id"], ranked, _CFG).iloc[0]
    return ranked, selected


def test_no_candidate_outside_radius() -> None:
    # Event near Bhubaneswar; facility ~50 km away.
    event_wkt = _point_wkt(85.8, 20.3)
    fac = _facility("F_FAR", 86.3, 20.3)
    rt = process_event_facility_association(
        "E_NONE",
        centroid_wkt=event_wkt,
        footprint_wkt=event_wkt,
        facilities=[fac],
        config=_CFG,
    )
    assert rt.facility_association_method == NO_FACILITY_ASSOCIATION
    assert rt.facility_id is None
    assert rt.candidate_facility_count == 0
    assert rt.candidates == ()


def test_one_near_candidate() -> None:
    event_wkt = _point_wkt(85.80, 20.30)
    fac = _facility("F_NEAR", 85.81, 20.30)  # ~1 km east
    rt = process_event_facility_association(
        "E_ONE",
        centroid_wkt=event_wkt,
        footprint_wkt=event_wkt,
        facilities=[fac],
        config=_CFG,
    )
    assert rt.facility_id == "F_NEAR"
    assert rt.facility_association_method == NEAR_FACILITY
    assert rt.candidate_facility_count == 1
    assert len(rt.candidates) == 1
    assert rt.facility_distance_km is not None
    assert 0.5 < rt.facility_distance_km < 2.0


def test_multiple_candidates_ranking_matches_batch() -> None:
    event_wkt = _point_wkt(85.80, 20.30)
    facilities = [
        _facility("F_far", 85.83, 20.30),  # farther NEAR
        _facility("F_close", 85.81, 20.30),  # closer NEAR
        _facility("F_within", 85.80, 20.30, polygon=True),  # WITHIN
    ]
    rt = process_event_facility_association(
        "E_MULTI",
        centroid_wkt=event_wkt,
        footprint_wkt=event_wkt,
        facilities=facilities,
        config=_CFG,
    )
    ranked, selected = _batch_one_event("E_MULTI", event_wkt, event_wkt, facilities)
    assert rt.facility_id == selected["facility_id"]
    assert rt.facility_association_method == selected["facility_association_method"]
    assert rt.candidate_facility_count == int(selected["candidate_facility_count"])
    assert [c.facility_id for c in rt.candidates] == ranked.sort_values("candidate_rank")[
        "facility_id"
    ].tolist()
    # WITHIN must beat closer NEAR
    assert rt.candidates[0].facility_id == "F_within"
    assert rt.facility_association_method == WITHIN_FACILITY


def test_boundary_just_inside_and_outside() -> None:
    event_wkt = _point_wkt(85.0, 20.0)
    # ~4.7 km at this latitude (inside 5 km)
    inside = _facility("F_IN", 85.045, 20.0)
    # ~5.5 km (outside)
    outside = _facility("F_OUT", 85.053, 20.0)
    rt_in = process_event_facility_association(
        "E_IN", centroid_wkt=event_wkt, footprint_wkt=event_wkt, facilities=[inside], config=_CFG
    )
    rt_out = process_event_facility_association(
        "E_OUT", centroid_wkt=event_wkt, footprint_wkt=event_wkt, facilities=[outside], config=_CFG
    )
    assert rt_in.candidate_facility_count == 1
    assert rt_out.candidate_facility_count == 0


def test_ambiguous_when_two_near_too_close() -> None:
    event_wkt = _point_wkt(85.80, 20.30)
    facilities = [
        _facility("F_a", 85.810, 20.30),
        _facility("F_b", 85.811, 20.30),  # ~0.1 km apart
    ]
    rt = process_event_facility_association(
        "E_AMB",
        centroid_wkt=event_wkt,
        footprint_wkt=event_wkt,
        facilities=facilities,
        config=AssociationConfig(ambiguity_distance_tolerance_km=0.5),
    )
    assert rt.facility_association_method == AMBIGUOUS
    assert rt.facility_id is None
    assert rt.candidate_facility_count == 2


def test_idempotent_reprocess() -> None:
    event_wkt = _point_wkt(85.80, 20.30)
    fac = _facility("F1", 85.81, 20.30)
    a = process_event_facility_association(
        "E_ID", centroid_wkt=event_wkt, footprint_wkt=event_wkt, facilities=[fac]
    )
    b = process_event_facility_association(
        "E_ID", centroid_wkt=event_wkt, footprint_wkt=event_wkt, facilities=[fac]
    )
    assert a.to_dict() == b.to_dict()


def test_null_geometry_no_association() -> None:
    rt = process_event_facility_association(
        "E_NULL",
        centroid_wkt=None,
        footprint_wkt=None,
        facilities=[_facility("F1", 85.81, 20.30)],
    )
    assert rt.facility_association_method == NO_FACILITY_ASSOCIATION
    assert rt.facility_id is None


def test_invalid_wkt_no_association() -> None:
    rt = process_event_facility_association(
        "E_BAD",
        centroid_wkt="NOT_WKT",
        footprint_wkt="NOT_WKT",
        facilities=[_facility("F1", 85.81, 20.30)],
    )
    assert rt.facility_association_method == NO_FACILITY_ASSOCIATION


def test_batch_realtime_parity_full_primitives() -> None:
    event_wkt = _point_wkt(76.30, 10.35)
    facilities = [
        _facility("P5_F1", 76.301, 10.35),
        _facility("P5_F2", 76.32, 10.35),
        _facility("P5_POLY", 76.30, 10.35, polygon=True, ftype="INDUSTRIAL_AREA"),
    ]
    rt = process_event_facility_association(
        "E_PARITY",
        centroid_wkt=event_wkt,
        footprint_wkt=event_wkt,
        facilities=facilities,
        config=_CFG,
    )
    ranked, selected = _batch_one_event("E_PARITY", event_wkt, event_wkt, facilities)
    assert rt.facility_id == (None if pd.isna(selected["facility_id"]) else selected["facility_id"])
    assert rt.facility_association_method == selected["facility_association_method"]
    assert rt.facility_attribution_confidence == selected["facility_attribution_confidence"]
    assert rt.candidate_facility_count == int(selected["candidate_facility_count"])
    if rt.facility_distance_km is not None:
        assert rt.facility_distance_km == pytest.approx(float(selected["facility_distance_km"]), rel=1e-5)
    assert {c.facility_id for c in rt.candidates} == set(ranked["facility_id"])


def test_realtime_does_not_call_batch_orchestrator() -> None:
    before = batch_pipeline_invocation_count()

    def _boom(*_a, **_k):
        raise AssertionError("run_facility_association must not be called")

    with patch(
        "src.infrastructure.association_pipeline.run_facility_association",
        side_effect=_boom,
    ):
        process_event_facility_association(
            "E_X",
            centroid_wkt=_point_wkt(85.8, 20.3),
            footprint_wkt=_point_wkt(85.8, 20.3),
            facilities=[_facility("F1", 85.81, 20.3)],
        )
    assert batch_pipeline_invocation_count() == before
    # Batch pipeline module remains importable for offline use.
    import src.infrastructure.association_pipeline as ap

    assert hasattr(ap, "run_facility_association")
    assert callable(ap.run_facility_association)
