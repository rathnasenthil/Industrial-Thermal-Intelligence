"""Unit tests for Events list is_active filter and sort clause helpers."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.models.thermal_event import ThermalEvent
from app.services.events import _apply_event_filters, event_sort_clauses


def test_default_sort_preserves_risk_then_start_then_id() -> None:
    clauses = event_sort_clauses()
    rendered = [str(c) for c in clauses]
    assert len(clauses) == 3
    assert "risk_score" in rendered[0]
    assert "DESC" in rendered[0].upper()
    assert "event_start" in rendered[1]
    assert "DESC" in rendered[1].upper()
    assert "event_id" in rendered[2]
    assert "ASC" in rendered[2].upper() or "asc" in rendered[2]


def test_sort_by_last_detection_at_defaults_to_desc() -> None:
    clauses = event_sort_clauses(sort_by="last_detection_at")
    rendered = [str(c) for c in clauses]
    assert "last_detection_at" in rendered[0]
    assert "DESC" in rendered[0].upper()
    assert "event_id" in rendered[1]


def test_sort_by_event_start_asc() -> None:
    clauses = event_sort_clauses(sort_by="event_start", sort_order="asc")
    rendered = [str(c) for c in clauses]
    assert "event_start" in rendered[0]
    assert "ASC" in rendered[0].upper() or "asc" in rendered[0]


def test_invalid_sort_by_raises() -> None:
    with pytest.raises(ValueError, match="sort_by"):
        event_sort_clauses(sort_by="updated_at")


def test_is_active_filter_true_and_false() -> None:
    base = select(ThermalEvent)
    active = _apply_event_filters(base, is_active=True)
    inactive = _apply_event_filters(base, is_active=False)
    omitted = _apply_event_filters(base)

    def where_sql(stmt) -> str:
        compiled = stmt.compile(compile_kwargs={"literal_binds": True})
        text = str(compiled)
        return text.split("WHERE", 1)[1] if "WHERE" in text else ""

    assert "is_active" in where_sql(active)
    assert "true" in where_sql(active).lower()
    assert "is_active" in where_sql(inactive)
    assert "false" in where_sql(inactive).lower()
    assert where_sql(omitted) == ""
