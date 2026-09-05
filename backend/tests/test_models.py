"""Model metadata and schema integrity tests (no live DB required)."""

from geoalchemy2 import Geometry

from app.models.facility import Facility
from app.models.thermal_event import ThermalEvent


def test_thermal_event_geometry_srid_4326() -> None:
    geom_col = ThermalEvent.__table__.c.geometry
    assert isinstance(geom_col.type, Geometry)
    assert geom_col.type.srid == 4326
    assert geom_col.type.geometry_type == "POINT"


def test_facility_geometry_srid_4326() -> None:
    geom_col = Facility.__table__.c.geometry
    assert isinstance(geom_col.type, Geometry)
    assert geom_col.type.srid == 4326


def test_nullable_evidence_columns_exist() -> None:
    cols = ThermalEvent.__table__.c
    for name in (
        "sta_domain_available",
        "environmental_domain_available",
        "sta_evidence_score",
        "environmental_evidence_score",
        "anomaly_score",
        "facility_id",
    ):
        assert cols[name].nullable is True


def test_event_id_unique_constraint() -> None:
    names = {c.name for c in ThermalEvent.__table__.constraints}
    assert "uq_thermal_events_event_id" in names
