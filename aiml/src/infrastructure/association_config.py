"""
Configuration for GIFT Stage I.2 (Thermal Event <-> Facility Association).

Every threshold below is an ENGINEERING default for a *spatial* matching
step, not a scientifically validated causal threshold. None of them
imply that a thermal event within range of a facility was actually
caused by it -- see `facility_association.py` module docstring for the
mandatory distinction between "facility association" and "source
classification" (the latter is explicitly out of scope for this stage).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AssociationConfig:
    """Tunable parameters for Stage I.2 spatial candidate search and ranking.

    Attributes:
        association_radius_km: Maximum distance (kilometers) between a
            thermal event's centroid and a facility's geometry for that
            facility to be considered a `NEAR_FACILITY` candidate at
            all. Facilities the event's footprint geometry already
            contains/intersects are always candidates regardless of this
            radius (see `facility_association.SpatialRelation`).

            Rationale for the default (5.0 km): an engineering starting
            point, not a calibrated causal radius. It is set well above
            Stage G's own `spatial_eps_km` (1.5 km, the distance used to
            *link individual detections into one event*) so that this
            stage's *event-to-facility* search radius is not accidentally
            tighter than the intra-event clustering radius already
            applied upstream, while still being small enough that a
            "NEAR" association plausibly reflects genuine local spatial
            proximity rather than a coincidental country-scale match.
            This value has NOT been validated against confirmed
            industrial-source ground truth and must not be read as "an
            event within 5 km of a facility was caused by it" -- it only
            defines the search radius for *considering* a facility as a
            spatial candidate. It is fully configurable and expected to
            be recalibrated once labeled outcomes are available.
        ambiguity_distance_tolerance_km: Two (or more) top-ranked
            candidates within this distance of each other (and in the
            same spatial-relation tier -- see `SPATIAL_RELATION_PRIORITY`)
            are considered indistinguishable, and the event's association
            is marked `AMBIGUOUS` rather than confidently assigned to
            either one.

            Rationale for the default (0.5 km): comfortably larger than
            typical facility-geometry representative-point/centroid
            jitter (a facility boundary polygon's own extent is commonly
            on the order of hundreds of meters), so two candidates this
            close are genuinely hard to tell apart spatially -- this is
            NOT a claim that facilities more than 0.5 km apart are always
            unambiguous, just an engineering tie-breaking margin.
        max_candidates_per_event: Hard cap on how many ranked candidates
            are retained per event in the detailed candidates output
            (`thermal_event_facility_candidates.csv`). Only affects the
            *candidates* file, never which facility (if any) is selected
            as the main association -- that always uses the single
            best-ranked candidate among all found, capped output is
            purely to bound file size in unusually dense industrial
            clusters. ``None`` disables the cap.

            Rationale for the default (10): comfortably above the
            largest expected genuine local cluster of distinct
            facilities, while bounding output size if a data anomaly
            (e.g. many overlapping/duplicate-looking OSM polygons) were
            to produce an extreme candidate count for one event.
        events_path: Default input event table (the persistence-enriched
            Stage G.1 output, preferred over the bare Stage G table
            because it carries all Stage G columns plus persistence
            characterization -- see `association_pipeline.py`).
        facilities_path: Default input facility table (Stage I.1's
            normalized GeoJSON output).
    """

    association_radius_km: float = 5.0
    ambiguity_distance_tolerance_km: float = 0.5
    max_candidates_per_event: int | None = 10
    events_path: Path = Path("data/processed/thermal_events_with_persistence.csv")
    facilities_path: Path = Path("data/processed/osm_facilities.geojson")

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain, JSON-serializable dict."""
        d = asdict(self)
        d["events_path"] = str(d["events_path"])
        d["facilities_path"] = str(d["facilities_path"])
        return d

    def describe_rationale(self) -> dict[str, str]:
        """Human-readable rationale strings for each threshold (for reports)."""
        return {
            "association_radius_km": (
                "Engineering search radius (not a calibrated causal distance) "
                "for considering a facility a spatial candidate at all. Set "
                "above Stage G's own spatial_eps_km (1.5 km, used to link "
                "individual detections into one event) so this stage's "
                "event-to-facility search is not tighter than the intra-event "
                "clustering radius already applied upstream. A facility within "
                "this radius is NOT proven to be the event's cause."
            ),
            "ambiguity_distance_tolerance_km": (
                "Engineering tie-breaking margin: top candidates within this "
                "distance of each other (in the same spatial-relation tier) "
                "are treated as indistinguishable and the association is "
                "marked AMBIGUOUS rather than confidently picking one."
            ),
            "max_candidates_per_event": (
                "Bounds the size of the detailed candidates output in "
                "unusually dense facility clusters; does not affect which "
                "facility (if any) is selected as the main association."
            ),
        }


DEFAULT_CONFIG = AssociationConfig()
