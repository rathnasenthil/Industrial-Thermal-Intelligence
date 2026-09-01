from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Shared declarative base for all ORM models.

    GeoAlchemy2 column types (e.g. `geoalchemy2.Geometry`) can be used directly
    on models that inherit from this base once the actual data models are
    defined in `app/models/`.
    """


# TODO: Import model modules here once they exist, e.g.:
#   from app.models.thermal_event import ThermalEvent  # noqa: F401
# so Alembic autogenerate can discover them via `Base.metadata`.
