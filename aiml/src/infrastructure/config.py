"""
Configuration for GIFT Stage I.1 (OSM Facility Ingestion & Normalization).

Unlike Stage G / Stage G.1, this stage has no clustering or classification
thresholds to tune — its "configuration" is mostly about *where* to look
for a user-supplied static OSM extract and how to record provenance in
the report. Nothing here is a scientific parameter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InfrastructureConfig:
    """Configuration for the OSM facility ingestion pipeline.

    Attributes:
        raw_dir: Directory searched for a user-supplied static OSM
            extract (see `osm_loader.discover_default_osm_input`). This
            pipeline never queries the live Overpass API and never
            downloads or fabricates data — if nothing is found here, the
            pipeline reports that explicitly rather than pretending
            coverage exists.
        source_label: Recorded in every normalized facility's `source`
            column and in the report, so it is always clear the facility
            layer came from a static, user-supplied extract — not a live
            query, not an authoritative government registry, and not
            ground truth for whether a nearby thermal event is
            industrial.
        lng_name_keywords: Keywords (case-insensitive substring match)
            required, in addition to gas-industrial tag evidence, before
            a facility is mapped to LNG_TERMINAL (see
            `osm_normalization.classify_facility_type`). OSM has no
            single universal tag for LNG terminals specifically, so this
            stage deliberately requires this extra name-based evidence
            rather than mapping every generic gas-industrial tag to
            LNG_TERMINAL.
    """

    raw_dir: Path = Path("data/raw")
    source_label: str = "osm_static_extract"
    lng_name_keywords: tuple[str, ...] = ("lng", "liquefied natural gas")

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain, JSON-serializable dict."""
        d = asdict(self)
        d["raw_dir"] = str(d["raw_dir"])
        d["lng_name_keywords"] = list(d["lng_name_keywords"])
        return d


DEFAULT_CONFIG = InfrastructureConfig()
