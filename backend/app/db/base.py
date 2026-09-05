from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Shared declarative base for all ORM models.

    GeoAlchemy2 column types (e.g. `geoalchemy2.Geometry`) can be used directly
    on models that inherit from this base.
    """


# Import model modules for Alembic metadata discovery (avoid circular package import).
from app.models import facility as _facility  # noqa: E402, F401
from app.models import thermal_event as _thermal_event  # noqa: E402, F401
from app.models import event_facility_candidate as _event_facility_candidate  # noqa: E402, F401
