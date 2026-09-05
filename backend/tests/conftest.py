"""
Pytest configuration and shared fixtures.

Integration tests requiring PostGIS are skipped unless DATABASE_URL is
reachable (Docker db service). Unit tests do not need a database.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


def _database_available() -> bool:
    settings = get_settings()
    try:
        engine = create_engine(settings.database_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            conn.execute(text("SELECT PostGIS_Version()"))
        engine.dispose()
        return True
    except Exception:
        return False


REQUIRES_POSTGIS = pytest.mark.skipif(
    not _database_available(),
    reason="PostgreSQL/PostGIS not available (start docker-compose db service)",
)


@pytest.fixture(scope="session")
def db_engine():
    if not _database_available():
        pytest.skip("PostgreSQL/PostGIS not available")
    settings = get_settings()
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine) -> Session:
    SessionLocal = sessionmaker(bind=db_engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
