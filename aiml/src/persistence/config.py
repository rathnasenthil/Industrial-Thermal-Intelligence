"""
Configuration for GIFT Stage G.1 — Persistence & Recurrence Characterization.

Every threshold below is an ENGINEERING default, chosen from basic
domain reasoning about satellite revisit cadence and event scale — NOT a
scientifically validated threshold. None of them have been tuned against
labeled ground truth (e.g. confirmed industrial flares vs. confirmed
one-off agricultural burns). They are recorded verbatim in every run's
report so they can be revisited later.

This stage only ever *reads* the event-level table produced by GIFT
Stage G (`thermal_events.csv`) and adds new columns to a copy of it. It
never re-clusters detections, never changes `event_id` assignments, and
never splits or merges rows — the ST-DBSCAN output from Stage G is
treated as an immutable baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PersistenceConfig:
    """Tunable thresholds for persistence/recurrence classification.

    Attributes:
        min_detections_for_classification: Minimum `detection_count` an
            event must have before this stage will assign it SHORT_LIVED,
            PERSISTENT or RECURRING at all.

            Rationale for the default (3): with only 2 detections (the
            minimum possible cluster size given Stage G's
            `min_samples=2`), there is exactly one observed gap and no
            way to tell "a brief, one-off pair of overpasses" apart from
            "the first two observations of what might become a long
            persistent source". Rather than guess, events below this
            count are labeled INSUFFICIENT_OBSERVATIONS rather than
            forced into one of the other three categories.
        short_lived_max_duration_hours: Events with
            `observed_duration_hours` at or below this value are
            SHORT_LIVED regardless of how densely they were detected.

            Rationale for the default (48.0, i.e. 2 days): this is
            comfortably above Stage G's own `temporal_eps_hours` (36h),
            so it does not simply relabel "whatever ST-DBSCAN was willing
            to link in one hop" — it requires the *observed* activity to
            span a genuinely multi-day window before being considered for
            PERSISTENT/RECURRING at all.
        persistent_min_duty_cycle: An event qualifies as PERSISTENT if it
            was detected on at least this fraction of the calendar days
            it spans (`distinct_detection_days / span_days`) — evaluated
            with OR against `persistent_max_gap_hours` (see below), i.e.
            either condition alone is enough.

            Rationale for the default (0.85): chosen empirically from the
            full 1.17M-detection Stage G run, not guessed in the
            abstract. Among the ~10k events that survive the
            `min_detections`/`short_lived` filters, `duty_cycle` is
            almost always either very high (>=0.95, the bulk of cases,
              including the known ~166-day coal-seam-fire-like event at
              duty_cycle=0.994) or clearly low (<0.85); there is little
            real data in between, so 0.85 sits in that empirical gap
            rather than at an arbitrary round number. It should still be
            read as "detected on a large majority of spanned days", not
            as a scientifically derived cutoff.
        persistent_max_gap_hours: An event also qualifies as PERSISTENT
            (regardless of duty cycle) if its longest gap between
            consecutive (time-sorted) detections never exceeds this
            value.

            IMPORTANT STRUCTURAL CONSTRAINT discovered while calibrating
            this stage: because Stage G's ST-DBSCAN requires every edge
            in its spatio-temporal neighbor graph to be within
            `temporal_eps_hours` (36h by default), the longest
            time-sorted gap *within a single connected event* can never
            exceed `temporal_eps_hours` — if it did, the event would
            already have been split into two separate Stage G events.
            Concretely, in the real dataset the maximum observed
            `max_gap_hours` across all events is ~35h, i.e. right up
            against that ceiling. Setting this threshold at or above
            `temporal_eps_hours` (e.g. the original draft value of 72h)
            would therefore be structurally unreachable and vacuous.

            Rationale for the default (24.0, i.e. ~2x the ~12h VIIRS
            day/night overpass interval): most persistent sources are
            caught on essentially every overpass opportunity (gap <=
            ~24-25h); a small tail of otherwise-persistent events has
            exactly one longer pause approaching the ~35h ceiling
            (likely one missed overpass pair, e.g. cloud cover) — those
            are still treated as PERSISTENT via the duty-cycle OR-branch
            above rather than being penalized for a single gap.
        combination_logic: This field is documentation-only (not an
            actual dataclass attribute — see the class docstring above):
            PERSISTENT = `duty_cycle >= persistent_min_duty_cycle` OR
            `max_gap_hours <= persistent_max_gap_hours`. OR (not AND) was
            chosen deliberately so that one long-but-isolated pause does
            not downgrade an otherwise consistently-detected, genuinely
            persistent source (this is exactly what would happen to the
            real ~166-day event under an AND rule, since its one pause is
            ~35h). Events failing BOTH conditions (low duty cycle *and*
            at least one long pause) are RECURRING instead.
    """

    min_detections_for_classification: int = 3
    short_lived_max_duration_hours: float = 48.0
    persistent_min_duty_cycle: float = 0.85
    persistent_max_gap_hours: float = 24.0

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain, JSON-serializable dict."""
        return asdict(self)

    def describe_rationale(self) -> dict[str, str]:
        """Human-readable rationale strings for each threshold (for reports)."""
        return {
            "min_detections_for_classification": (
                "Below this count there is at most one observed gap, which is "
                "not enough to distinguish a brief coincidence from the start "
                "of a persistent source; such events are labeled "
                "INSUFFICIENT_OBSERVATIONS rather than guessed at."
            ),
            "short_lived_max_duration_hours": (
                "Set above Stage G's own temporal_eps_hours (36h) so this "
                "threshold reflects genuinely multi-day observed activity, "
                "not just whatever ST-DBSCAN was able to link in one hop."
            ),
            "persistent_min_duty_cycle": (
                "Empirically-informed bar (85% of spanned days detected), "
                "chosen because real Stage G events cluster into a "
                "high-duty-cycle group (>=0.95) and a clearly-lower group "
                "(<0.85), with little data in between. Combined with "
                "persistent_max_gap_hours via OR: either condition alone "
                "is enough to call an event PERSISTENT."
            ),
            "persistent_max_gap_hours": (
                "Since Stage G's ST-DBSCAN cannot produce a single-event "
                "internal gap longer than its own temporal_eps_hours "
                "(36h) without splitting into two events, this threshold "
                "must sit below that structural ceiling to mean anything; "
                "24h (~2x the ~12h overpass interval) tolerates a single "
                "missed overpass. Combined with persistent_min_duty_cycle "
                "via OR, so one longer pause does not downgrade an "
                "otherwise consistently-detected source."
            ),
        }


DEFAULT_CONFIG = PersistenceConfig()
