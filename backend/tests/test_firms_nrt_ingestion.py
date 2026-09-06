"""Unit tests for FIRMS NRT ingestion (mocked HTTP — no live NASA calls)."""

from __future__ import annotations

import httpx
import pytest

from app.services.firms_nrt_ingestion import (
    REQUIRED_FIRMS_COLUMNS,
    FirmsNRTConfigError,
    FirmsNRTHttpError,
    FirmsNRTParseError,
    build_firms_area_url,
    fetch_firms_nrt_observations,
    normalize_firms_nrt_dataframe,
    parse_bbox,
    parse_firms_csv_text,
    redact_secrets,
)

SAMPLE_CSV = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight,type
18.97,83.80,330.1,0.39,0.36,2026-09-05,0655,N20,VIIRS,n,2.0NRT,296.5,2.41,D,0
20.83,86.96,340.2,0.40,0.36,2026-09-05,0710,N20,VIIRS,n,2.0NRT,300.1,5.10,N,0
"""

HEADER_ONLY_CSV = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight,type\n"
)


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_missing_api_key_raises() -> None:
    with pytest.raises(FirmsNRTConfigError, match="FIRMS_MAP_KEY"):
        fetch_firms_nrt_observations(map_key="")


def test_invalid_bbox_raises() -> None:
    with pytest.raises(FirmsNRTConfigError, match="bbox"):
        parse_bbox("1,2,3")
    with pytest.raises(FirmsNRTConfigError, match="west"):
        parse_bbox("100.0,6.0,90.0,37.0")


def test_build_url_redacts_key_in_helper() -> None:
    url = build_firms_area_url(
        map_key="secret-key-value",
        product="VIIRS_NOAA20_NRT",
        bbox="68.0,6.0,98.0,37.5",
        day_range=1,
    )
    assert "secret-key-value" in url  # raw builder keeps key for the request
    assert "VIIRS_NOAA20_NRT" in url
    assert "/1" in url
    safe = redact_secrets(url, "secret-key-value")
    assert "secret-key-value" not in safe
    assert "***" in safe


def test_successful_csv_parsing_with_mocked_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "VIIRS_NOAA20_NRT" in str(request.url)
        return httpx.Response(200, text=SAMPLE_CSV)

    client = _mock_client(handler)
    df = fetch_firms_nrt_observations(
        map_key="fake-map-key",
        product="VIIRS_NOAA20_NRT",
        bbox="68.0,6.0,98.0,37.5",
        day_range=1,
        client=client,
    )
    assert len(df) == 2
    for col in REQUIRED_FIRMS_COLUMNS:
        assert col in df.columns
    assert "source_file" in df.columns
    assert df.iloc[0]["latitude"] == "18.97"
    assert str(df.iloc[0]["source_file"]).startswith("firms_nrt_")


def test_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Forbidden")

    with pytest.raises(FirmsNRTHttpError, match="HTTP 403"):
        fetch_firms_nrt_observations(
            map_key="fake-map-key",
            client=_mock_client(handler),
        )


def test_http_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(FirmsNRTHttpError, match="timed out"):
        fetch_firms_nrt_observations(
            map_key="fake-map-key",
            client=_mock_client(handler),
        )


def test_empty_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="   ")

    with pytest.raises(FirmsNRTParseError, match="empty"):
        fetch_firms_nrt_observations(
            map_key="fake-map-key",
            client=_mock_client(handler),
        )


def test_header_only_csv_returns_empty_normalized_frame() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=HEADER_ONLY_CSV)

    df = fetch_firms_nrt_observations(
        map_key="fake-map-key",
        client=_mock_client(handler),
    )
    assert len(df) == 0
    for col in REQUIRED_FIRMS_COLUMNS:
        assert col in df.columns


def test_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not csv</html>")

    with pytest.raises(FirmsNRTParseError, match="does not look like a CSV"):
        fetch_firms_nrt_observations(
            map_key="fake-map-key",
            client=_mock_client(handler),
        )


def test_malformed_missing_required_columns() -> None:
    raw = parse_firms_csv_text("latitude,longitude\n1,2\n")
    with pytest.raises(FirmsNRTParseError, match="missing required"):
        normalize_firms_nrt_dataframe(raw, product="VIIRS_NOAA20_NRT")


def test_required_normalized_fields_present() -> None:
    df = normalize_firms_nrt_dataframe(
        parse_firms_csv_text(SAMPLE_CSV),
        product="VIIRS_NOAA20_NRT",
    )
    expected = set(REQUIRED_FIRMS_COLUMNS) | {"source_file"}
    assert expected.issubset(set(df.columns))
    # No invented observation id column
    assert "observation_id" not in df.columns
    assert "detection_id" not in df.columns
