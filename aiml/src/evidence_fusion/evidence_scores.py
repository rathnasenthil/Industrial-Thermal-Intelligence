"""
Ordinal multi-family evidence scoring for GIFT Stage I.7.

Scores are ENGINEERING ORDINAL SUPPORT levels — not probabilities,
accuracies, or scientifically validated weights.

Family scale (unless noted):
  0 = no supporting evidence
  1 = weak supporting evidence
  2 = moderate supporting evidence
  3 = strong supporting evidence

Aggregate industrial_evidence_score uses documented weights and caps to
limit double-counting correlated infrastructure/history signals.
Missing STA/environmental evidence contributes 0 and is never treated as
negative evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.evidence_fusion.config import EvidenceFusionConfig

# Facility association method sets (mirrors config; kept local to avoid import cycles).
CONFIRMED_FACILITY_METHODS: frozenset[str] = frozenset(
    {"WITHIN_FACILITY", "INTERSECTS_FACILITY", "NEAR_FACILITY"}
)
STRONG_FACILITY_METHODS: frozenset[str] = frozenset(
    {"WITHIN_FACILITY", "INTERSECTS_FACILITY"}
)

# Facility types with stronger industrial-context prior (engineering list).
STRONG_FACILITY_TYPES: frozenset[str] = frozenset(
    {"REFINERY", "POWER_PLANT", "MINE", "LNG_TERMINAL"}
)
WEAKER_FACILITY_TYPES: frozenset[str] = frozenset(
    {"OTHER_INDUSTRIAL", "INDUSTRIAL_AREA", "UNKNOWN"}
)

# Aggregate industrial score = infra_weight * infra + temporal + hist_eff + anom_eff + sta
# Max theoretical (documented): 2*3 + 2 + 2 + 2 + 2 = 14
INFRA_AGGREGATE_WEIGHT = 2
HISTORICAL_EFFECTIVE_CAP = 2
ANOMALY_EFFECTIVE_CAP = 2
TEMPORAL_EFFECTIVE_CAP = 2
STA_EFFECTIVE_CAP = 2
INDUSTRIAL_SCORE_MAX = (
    INFRA_AGGREGATE_WEIGHT * 3
    + TEMPORAL_EFFECTIVE_CAP
    + HISTORICAL_EFFECTIVE_CAP
    + ANOMALY_EFFECTIVE_CAP
    + STA_EFFECTIVE_CAP
)

STRENGTH_NONE = "NONE"
STRENGTH_WEAK = "WEAK"
STRENGTH_MODERATE = "MODERATE"
STRENGTH_STRONG = "STRONG"


def score_infrastructure(method: str, confidence: str, facility_type: str | None) -> int:
    """Ordinal infrastructure support 0-3. Cap at 3. NEAR never reaches 3."""
    method = method or "UNAVAILABLE"
    confidence = confidence or "NONE"
    ftype = facility_type or ""

    if method in STRONG_FACILITY_METHODS:
        score = 3
        if confidence == "LOW":
            score = 2
        if ftype in WEAKER_FACILITY_TYPES and confidence != "HIGH":
            score = min(score, 2)
        if ftype in STRONG_FACILITY_TYPES and confidence in ("HIGH", "MEDIUM"):
            score = 3
        return int(max(0, min(3, score)))

    if method == "NEAR_FACILITY":
        score = 1
        if confidence in ("MEDIUM", "HIGH"):
            score = 2
        # Strong facility type can reinforce NEAR up to moderate, never strong/3.
        if ftype in STRONG_FACILITY_TYPES:
            score = max(score, 1)
            score = min(2, score + (1 if confidence == "LOW" else 0))
            score = min(2, score)
        return int(max(0, min(2, score)))

    # AMBIGUOUS / NO_FACILITY / other → 0 for industrial infra support
    return 0


def score_temporal(persistence: str) -> int:
    """Ordinal temporal/persistence support 0-2 (persistence is not industrial proof)."""
    if persistence in ("PERSISTENT", "RECURRING"):
        return 2
    if persistence == "SHORT_LIVED":
        return 0
    # INSUFFICIENT_OBSERVATIONS / UNAVAILABLE
    return 0


def score_historical(history: str, method: str) -> int:
    """Historical support only when a confirmed facility association exists."""
    if method not in CONFIRMED_FACILITY_METHODS:
        return 0
    if history == "ESTABLISHED_BASELINE":
        return 3
    if history == "LIMITED_HISTORY":
        return 2
    if history == "INSUFFICIENT_HISTORY":
        return 1
    # NO_PRIOR_OBSERVATIONS / NOT_APPLICABLE / UNAVAILABLE
    return 0


def score_anomaly(anomaly: str) -> int:
    """Ordinal anomaly deviation support. Not industrial origin."""
    if anomaly == "ANOMALOUS":
        return 2
    if anomaly == "ELEVATED":
        return 1
    return 0


def score_sta(sta_available: bool, sta_signal: str) -> tuple[int, bool]:
    """Return (score, available_flag). Unavailable → (0, False), not negative."""
    if not sta_available or sta_signal == "UNAVAILABLE":
        return 0, False
    if sta_signal == "STA_ASSOCIATED":
        return 2, True
    if sta_signal == "AMBIGUOUS":
        return 1, True
    # NO_STA_ASSOCIATION → 0 contribution, domain still available
    return 0, True


def score_environmental(
    env_available: bool,
    veg: str,
    ag: str,
    built: str,
) -> tuple[int, int, bool]:
    """Return (environmental_support_score, builtup_context_bonus, available).

    environmental_support_score: vegetation/agriculture natural-context support 0-3.
    builtup_context_bonus: optional weak industrial-context cue 0-1 (not used alone).
    Unavailable → (0, 0, False) — never negative industrial evidence.
    """
    if not env_available:
        return 0, 0, False
    env_score = 0
    if veg == "PRESENT":
        env_score += 2
    if ag == "PRESENT":
        env_score += 2
    env_score = min(3, env_score)
    built_bonus = 1 if built == "PRESENT" else 0
    return env_score, built_bonus, True


def aggregate_industrial_score(
    infra: int,
    temporal: int,
    historical: int,
    anomaly: int,
    sta: int,
    *,
    config: "EvidenceFusionConfig | None" = None,
) -> tuple[int, int, int, int]:
    """Aggregate industrial evidence with correlation caps.

    Returns:
        (industrial_evidence_score, temporal_eff, historical_eff, anomaly_eff)

    Rules:
    - Infrastructure is weighted more heavily (spatial association).
    - Historical only contributes when infrastructure > 0, capped to avoid
      double-counting facility association + facility history.
    - Anomaly only contributes to *industrial* aggregate when infrastructure > 0
      (deviation without a facility is not industrial support).
    - Temporal can contribute even without facility (behavioral), but alone
      cannot produce an industrial candidate (enforced downstream).
    - Missing STA (score 0) is not a penalty.
    """
    _ = config  # reserved for future weight overrides
    temporal_eff = min(int(temporal), TEMPORAL_EFFECTIVE_CAP)
    if infra <= 0:
        historical_eff = 0
        anomaly_eff = 0
    else:
        historical_eff = min(int(historical), HISTORICAL_EFFECTIVE_CAP)
        anomaly_eff = min(int(anomaly), ANOMALY_EFFECTIVE_CAP)
    sta_eff = min(int(sta), STA_EFFECTIVE_CAP)
    total = (
        INFRA_AGGREGATE_WEIGHT * int(infra)
        + temporal_eff
        + historical_eff
        + anomaly_eff
        + sta_eff
    )
    return int(total), temporal_eff, historical_eff, anomaly_eff


def corroboration_score(
    temporal_eff: int,
    historical_eff: int,
    anomaly_eff: int,
    sta_eff: int,
) -> int:
    """Non-infrastructure corroboration for candidate gates."""
    return int(temporal_eff + historical_eff + anomaly_eff + sta_eff)


def evidence_strength_label(industrial_score: int, infra: int) -> str:
    """Map aggregate industrial score to an ordinal strength label (not probability)."""
    if industrial_score <= 0 and infra <= 0:
        return STRENGTH_NONE
    if industrial_score <= 3:
        return STRENGTH_WEAK
    if industrial_score <= 7:
        return STRENGTH_MODERATE
    return STRENGTH_STRONG


def evidence_coverage_label(present_count: int, total_domains: int = 4) -> str:
    """Simple coverage string: present/total domains."""
    return f"{int(present_count)}/{int(total_domains)}"
