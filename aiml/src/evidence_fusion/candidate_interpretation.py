"""
Deterministic multi-family candidate interpretation for Stage I.7.

Uses ordinal evidence scores from available families (infrastructure,
temporal, historical, anomaly; optional STA / environmental).

Candidates are NOT ground truth. Scores are NOT probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evidence_fusion.config import (
    CANDIDATE_AGRICULTURE_CONTEXT,
    CANDIDATE_AMBIGUOUS,
    CANDIDATE_INDUSTRIAL,
    CANDIDATE_INSUFFICIENT,
    CANDIDATE_MIXED,
    CANDIDATE_POSSIBLE_INDUSTRIAL,
    CANDIDATE_VEGETATION_CONTEXT,
    CONFIRMED_FACILITY_METHODS,
    EvidenceFusionConfig,
    SUFFICIENCY_INSUFFICIENT,
    SUFFICIENCY_PARTIAL,
    SUFFICIENCY_SUFFICIENT,
    UNCERTAINTY_HIGH,
    UNCERTAINTY_LOW,
    UNCERTAINTY_MEDIUM,
    UNCERTAINTY_VERY_HIGH,
)
from src.evidence_fusion.evidence_scores import (
    INDUSTRIAL_SCORE_MAX,
    aggregate_industrial_score,
    corroboration_score,
    evidence_coverage_label,
    evidence_strength_label,
    score_anomaly,
    score_environmental,
    score_historical,
    score_infrastructure,
    score_sta,
    score_temporal,
)


def _s(values: pd.Series | np.ndarray, n: int, default: str = "") -> np.ndarray:
    arr = np.asarray(values, dtype=object)
    if len(arr) != n:
        raise ValueError("aligned length mismatch")
    out = np.empty(n, dtype=object)
    for i, v in enumerate(arr):
        if v is None or (isinstance(v, float) and v != v):
            out[i] = default
        else:
            out[i] = str(v)
    return out


def _join(codes: list[str]) -> str:
    return ";".join(codes) if codes else ""


def interpret_candidates(
    events_df: pd.DataFrame,
    temporal: pd.DataFrame,
    infrastructure: pd.DataFrame,
    sta: pd.DataFrame,
    environmental: pd.DataFrame,
    availability: pd.DataFrame,
    conflicts: pd.DataFrame,
    config: EvidenceFusionConfig,
) -> pd.DataFrame:
    """Assign ordinal scores + explainable candidates for every event."""
    n = len(events_df)
    event_ids = events_df["event_id"].astype(str).to_numpy()

    temp = temporal.set_index("event_id").reindex(event_ids)
    infra_f = infrastructure.set_index("event_id").reindex(event_ids)
    sta_f = sta.set_index("event_id").reindex(event_ids)
    env = environmental.set_index("event_id").reindex(event_ids)
    avail = availability.set_index("event_id").reindex(event_ids)
    conf = conflicts.set_index("event_id").reindex(event_ids)

    methods = (
        _s(events_df["facility_association_method"], n, "UNAVAILABLE")
        if "facility_association_method" in events_df.columns
        else np.full(n, "UNAVAILABLE", dtype=object)
    )
    fac_types = (
        _s(events_df["facility_type"], n, "")
        if "facility_type" in events_df.columns
        else np.full(n, "", dtype=object)
    )
    fac_conf = _s(infra_f["infrastructure_confidence_signal"], n, "NONE")
    hist = _s(infra_f["infrastructure_history_signal"], n, "NOT_APPLICABLE")
    persist = _s(temp["temporal_persistence_signal"], n, "UNAVAILABLE")
    anomaly = _s(temp["temporal_anomaly_signal"], n, "UNAVAILABLE")
    temp_ok = temp["temporal_evidence_available"].fillna(False).astype(bool).to_numpy()
    infra_ok = infra_f["infrastructure_evidence_available"].fillna(False).astype(bool).to_numpy()
    assoc = _s(infra_f["infrastructure_association_signal"], n, "UNAVAILABLE")
    sta_ok = sta_f["sta_domain_available"].fillna(False).astype(bool).to_numpy()
    sta_sig = _s(sta_f["sta_association_signal"], n, "UNAVAILABLE")
    env_ok = env["environmental_domain_available"].fillna(False).astype(bool).to_numpy()
    veg = _s(env["environmental_vegetation_signal"], n, "")
    ag = _s(env["environmental_agriculture_signal"], n, "")
    built = _s(env["environmental_builtup_signal"], n, "")
    conflict_flag = conf["evidence_conflict_flag"].fillna(False).astype(bool).to_numpy()
    conflict_codes = _s(conf["evidence_conflict_codes"], n, "")
    present_count = avail["evidence_sources_present_count"].fillna(0).astype(int).to_numpy()
    missing = _s(avail["evidence_sources_missing"], n, "")

    # Output arrays
    infra_scores = np.zeros(n, dtype=np.int64)
    temporal_scores = np.zeros(n, dtype=np.int64)
    historical_scores = np.zeros(n, dtype=np.int64)
    anomaly_scores = np.zeros(n, dtype=np.int64)
    sta_scores = np.zeros(n, dtype=np.int64)
    env_scores = np.zeros(n, dtype=np.int64)
    industrial_scores = np.zeros(n, dtype=np.int64)
    env_support_scores = np.zeros(n, dtype=np.int64)
    fusion_scores = np.zeros(n, dtype=np.int64)
    coverage = np.empty(n, dtype=object)
    strength = np.empty(n, dtype=object)

    profile_codes = np.empty(n, dtype=object)
    supporting = np.empty(n, dtype=object)
    ambiguous_codes = np.empty(n, dtype=object)
    limiting = np.empty(n, dtype=object)
    candidates = np.empty(n, dtype=object)
    rationales = np.empty(n, dtype=object)
    sufficiency = np.empty(n, dtype=object)
    uncertainty = np.empty(n, dtype=object)
    interp_conf = np.empty(n, dtype=object)

    for i in range(n):
        method = methods[i]
        ftype = fac_types[i] if fac_types[i] else None
        confidence = fac_conf[i]

        infra_s = score_infrastructure(method, confidence, ftype)
        temporal_s = score_temporal(persist[i]) if temp_ok[i] else 0
        historical_s = score_historical(hist[i], method)
        anomaly_s = score_anomaly(anomaly[i]) if temp_ok[i] else 0
        sta_s, sta_domain = score_sta(bool(sta_ok[i]), sta_sig[i])
        env_s, built_bonus, env_domain = score_environmental(
            bool(env_ok[i]), veg[i], ag[i], built[i]
        )

        industrial, temporal_eff, historical_eff, anomaly_eff = aggregate_industrial_score(
            infra_s, temporal_s, historical_s, anomaly_s, sta_s, config=config
        )
        # Built-up is weak optional context only when env available; never alone.
        if env_domain and built_bonus and infra_s > 0:
            industrial = min(INDUSTRIAL_SCORE_MAX, industrial + built_bonus)

        corr = corroboration_score(temporal_eff, historical_eff, anomaly_eff, sta_s)
        env_support = env_s
        fusion = industrial  # documented alias: aggregate industrial evidence used for gates

        infra_scores[i] = infra_s
        temporal_scores[i] = temporal_s
        historical_scores[i] = historical_s
        anomaly_scores[i] = anomaly_s
        sta_scores[i] = sta_s
        env_scores[i] = env_s
        industrial_scores[i] = industrial
        env_support_scores[i] = env_support
        fusion_scores[i] = fusion
        coverage[i] = evidence_coverage_label(int(present_count[i]))
        strength[i] = evidence_strength_label(industrial, infra_s)

        # --- explanatory codes ---
        p_codes: list[str] = []
        s_codes: list[str] = []
        a_codes: list[str] = []
        lim_codes: list[str] = []

        if temp_ok[i]:
            p_codes.append(f"PERSISTENCE:{persist[i]}")
            p_codes.append(f"ANOMALY:{anomaly[i]}")
            if temporal_s > 0:
                s_codes.append(f"TEMPORAL_{persist[i]}")
            if anomaly_s > 0:
                s_codes.append(f"TEMPORAL_{anomaly[i]}_DEVIATION")
        else:
            lim_codes.append("TEMPORAL_UNAVAILABLE")

        if infra_ok[i]:
            p_codes.append(f"FACILITY:{assoc[i]}")
            p_codes.append(f"INFRA_SCORE:{infra_s}")
            if infra_s > 0:
                s_codes.append(f"FACILITY_{method}")
                s_codes.append(f"FACILITY_CONFIDENCE_{confidence}")
                if ftype:
                    s_codes.append(f"FACILITY_TYPE_{ftype}")
            if assoc[i] == "AMBIGUOUS":
                a_codes.append("FACILITY_AMBIGUOUS")
            elif assoc[i] == "NONE":
                p_codes.append("FACILITY_NONE_NOT_NATURAL")
                lim_codes.append("NO_FACILITY_ASSOCIATION")
            if hist[i] not in ("", "NOT_APPLICABLE", "UNAVAILABLE"):
                p_codes.append(f"HISTORY:{hist[i]}")
                if historical_s > 0:
                    s_codes.append(f"HISTORICAL_{hist[i]}")
                elif method in CONFIRMED_FACILITY_METHODS:
                    lim_codes.append("NO_ESTABLISHED_HISTORY")
        else:
            lim_codes.append("INFRASTRUCTURE_UNAVAILABLE")

        if sta_domain:
            p_codes.append(f"STA:{sta_sig[i]}")
            if sta_s > 0:
                s_codes.append(f"STA_{sta_sig[i]}")
            elif sta_sig[i] == "NO_STA_ASSOCIATION":
                p_codes.append("STA_NONE_NOT_ANTI_INDUSTRIAL")
                lim_codes.append("NO_STA_MATCH")
        else:
            lim_codes.append("STA_UNAVAILABLE")

        if env_domain:
            if veg[i] == "PRESENT":
                p_codes.append("ENV_VEGETATION_PRESENT")
                s_codes.append("ENV_VEGETATION_PRESENT")
            if ag[i] == "PRESENT":
                p_codes.append("ENV_AGRICULTURE_PRESENT")
                s_codes.append("ENV_AGRICULTURE_PRESENT")
            if built[i] == "PRESENT":
                p_codes.append("ENV_BUILTUP_PRESENT")
                s_codes.append("ENV_BUILTUP_PRESENT")
        else:
            lim_codes.append("ENVIRONMENTAL_CONTEXT_UNAVAILABLE")

        if conflict_flag[i]:
            a_codes.append("EVIDENCE_CONFLICT")
            for code in conflict_codes[i].split(";"):
                if code:
                    a_codes.append(code)

        # Soft notes (not MIXED): behavioral evidence without facility
        if infra_s <= 0 and temporal_s > 0:
            a_codes.append("PERSISTENT_OR_TEMPORAL_WITHOUT_FACILITY")
        if infra_s <= 0 and anomaly_s > 0:
            a_codes.append("ANOMALY_WITHOUT_FACILITY_NOT_INDUSTRIAL")

        # --- candidate decision (multi-family) ---
        candidate = CANDIDATE_INSUFFICIENT
        rationale = (
            "Insufficient converging available evidence for a source-intelligence candidate."
        )
        suff = SUFFICIENCY_INSUFFICIENT
        unc = UNCERTAINTY_VERY_HIGH if present_count[i] <= 1 else UNCERTAINTY_HIGH
        ic = "NONE"

        env_veg = env_domain and veg[i] == "PRESENT"
        env_ag = env_domain and ag[i] == "PRESENT"
        meaningful_env = env_support >= 2
        meaningful_industrial = industrial >= 6 and infra_s >= 2

        if assoc[i] == "AMBIGUOUS" or (sta_domain and sta_sig[i] == "AMBIGUOUS" and infra_s <= 0):
            candidate = CANDIDATE_AMBIGUOUS
            rationale = (
                "Ambiguous facility and/or STA association; competing infrastructure "
                "evidence is retained without forcing a primary industrial candidate. "
                f"industrial_evidence_score={industrial} (ordinal, not probability)."
            )
            suff = SUFFICIENCY_PARTIAL
            unc = UNCERTAINTY_HIGH
            ic = "LOW"
        elif conflict_flag[i] and infra_s >= 2 and meaningful_env:
            candidate = CANDIDATE_MIXED
            rationale = (
                "Meaningful infrastructure/industrial evidence coexists with available "
                f"environmental context ({conflict_codes[i]}). "
                f"industrial_evidence_score={industrial}; environmental_support_score={env_support}. "
                "Not ground truth."
            )
            suff = SUFFICIENCY_PARTIAL
            unc = UNCERTAINTY_HIGH
            ic = "LOW"
        elif meaningful_industrial and meaningful_env and not conflict_flag[i]:
            # Cross-family tension even if STRtree conflict codes missed a case
            candidate = CANDIDATE_MIXED
            rationale = (
                "Combined industrial-support and environmental-support scores are both "
                f"material (industrial={industrial}, environmental={env_support}). "
                "Mixed interpretation retained; not ground truth."
            )
            suff = SUFFICIENCY_PARTIAL
            unc = UNCERTAINTY_HIGH
            ic = "LOW"
        elif env_ag and infra_s <= 0:
            candidate = CANDIDATE_AGRICULTURE_CONTEXT
            rationale = (
                "Available agricultural context without infrastructure support. "
                "Environmental context evidence only - not an agricultural-fire label."
            )
            suff = SUFFICIENCY_PARTIAL
            unc = UNCERTAINTY_HIGH
            ic = "LOW"
        elif env_veg and infra_s <= 0:
            candidate = CANDIDATE_VEGETATION_CONTEXT
            rationale = (
                "Available vegetation context without infrastructure support. "
                "Environmental context evidence only - not a wildfire label."
            )
            suff = SUFFICIENCY_PARTIAL
            unc = UNCERTAINTY_HIGH
            ic = "LOW"
        elif infra_s >= 3 and (corr >= 1 or infra_s >= 3):
            # Strong infrastructure: INDUSTRIAL if corroboration OR sufficiently strong infra alone.
            # infra_s >= 3 already means exceptionally strong spatial association.
            candidate = CANDIDATE_INDUSTRIAL
            if corr >= 1:
                rationale = (
                    f"Strong infrastructure evidence (score={infra_s}, {method}) with "
                    f"corroborating temporal/history/anomaly/STA support (corr={corr}). "
                    f"industrial_evidence_score={industrial}/{INDUSTRIAL_SCORE_MAX} "
                    "(ordinal engineering score, not probability). "
                    "Candidate interpretation only - not ground truth. "
                    "ANOMALOUS!=INDUSTRIAL_FIRE; facility proximity!=proof of industrial origin."
                )
            else:
                rationale = (
                    f"Exceptionally strong infrastructure evidence alone (score={infra_s}, {method}) "
                    f"without temporal/history/anomaly corroboration (corr=0). "
                    f"industrial_evidence_score={industrial}/{INDUSTRIAL_SCORE_MAX}. "
                    "Candidate interpretation only - not ground truth; missing history limits strength."
                )
                lim_codes.append("NO_BEHAVIORAL_CORROBORATION")
            suff = SUFFICIENCY_SUFFICIENT
            unc = UNCERTAINTY_MEDIUM if corr >= 2 and present_count[i] >= 2 else UNCERTAINTY_HIGH
            if corr >= 3 and sta_s > 0:
                unc = UNCERTAINTY_LOW
            ic = "MEDIUM" if corr >= 1 else "LOW"
        elif infra_s >= 2 and corr >= 1:
            candidate = CANDIDATE_POSSIBLE_INDUSTRIAL
            rationale = (
                f"Moderate infrastructure evidence (score={infra_s}, {method}) with "
                f"corroboration (corr={corr}). "
                f"industrial_evidence_score={industrial}/{INDUSTRIAL_SCORE_MAX}. "
                "Possible industrial-activity candidate only - not ground truth."
            )
            suff = SUFFICIENCY_PARTIAL
            unc = UNCERTAINTY_HIGH
            ic = "LOW"
        elif infra_s >= 1 and corr >= 2:
            # Weak/near infra needs stronger corroboration for POSSIBLE
            candidate = CANDIDATE_POSSIBLE_INDUSTRIAL
            rationale = (
                f"Weak/moderate infrastructure evidence (score={infra_s}, {method}) with "
                f"material corroboration (corr={corr}). "
                f"industrial_evidence_score={industrial}/{INDUSTRIAL_SCORE_MAX}. "
                "Possible industrial-activity candidate only - not ground truth."
            )
            suff = SUFFICIENCY_PARTIAL
            unc = UNCERTAINTY_HIGH
            ic = "LOW"
        elif infra_s >= 1 and corr < 2:
            # NEAR (or weak infra) without enough corroboration → insufficient, not auto-POSSIBLE
            candidate = CANDIDATE_INSUFFICIENT
            rationale = (
                f"Infrastructure signal present (score={infra_s}, {method}) but corroborating "
                f"temporal/history/anomaly evidence is insufficient (corr={corr}). "
                f"NEAR_FACILITY alone is not treated as POSSIBLE_INDUSTRIAL_ACTIVITY. "
                f"industrial_evidence_score={industrial}/{INDUSTRIAL_SCORE_MAX}."
            )
            lim_codes.append("INSUFFICIENT_CORROBORATION")
            suff = SUFFICIENCY_INSUFFICIENT
            unc = UNCERTAINTY_HIGH
            ic = "NONE"
        else:
            candidate = CANDIDATE_INSUFFICIENT
            rationale = (
                "No converging available evidence for a source-intelligence candidate. "
                "NO_FACILITY_ASSOCIATION!=NATURAL; missing STA/env!=negative evidence. "
                f"industrial_evidence_score={industrial}/{INDUSTRIAL_SCORE_MAX}."
            )
            if anomaly_s > 0 and infra_s <= 0:
                rationale += (
                    f" Temporal anomaly status {anomaly[i]} is deviation evidence only "
                    "and does not imply INDUSTRIAL_FIRE."
                )
            if temporal_s > 0 and infra_s <= 0:
                rationale += (
                    f" Persistence {persist[i]} is behavioral evidence only "
                    "and does not imply industrial origin."
                )
            suff = SUFFICIENCY_INSUFFICIENT
            unc = UNCERTAINTY_VERY_HIGH if present_count[i] <= 2 else UNCERTAINTY_HIGH
            ic = "NONE"

        # Missing optional domains raise uncertainty; never reduce industrial score.
        miss_parts = set(missing[i].split(";")) if missing[i] else set()
        if "sta" in miss_parts or "environmental" in miss_parts:
            if unc == UNCERTAINTY_LOW:
                unc = UNCERTAINTY_MEDIUM
            elif unc == UNCERTAINTY_MEDIUM:
                unc = UNCERTAINTY_HIGH

        profile_codes[i] = _join(p_codes)
        supporting[i] = _join(s_codes)
        ambiguous_codes[i] = _join(a_codes)
        limiting[i] = _join(lim_codes)
        candidates[i] = candidate
        rationales[i] = rationale
        sufficiency[i] = suff
        uncertainty[i] = unc
        interp_conf[i] = ic

    return pd.DataFrame(
        {
            "event_id": event_ids,
            "infrastructure_evidence_score": infra_scores,
            "temporal_evidence_score": temporal_scores,
            "historical_evidence_score": historical_scores,
            "anomaly_evidence_score": anomaly_scores,
            "sta_evidence_score": sta_scores,
            "environmental_evidence_score": env_scores,
            "industrial_evidence_score": industrial_scores,
            "environmental_support_score": env_support_scores,
            "evidence_fusion_score": fusion_scores,
            "evidence_coverage": coverage,
            "evidence_strength": strength,
            "evidence_profile_codes": profile_codes,
            "supporting_evidence_codes": supporting,
            "ambiguous_evidence_codes": ambiguous_codes,
            "limiting_evidence_codes": limiting,
            "source_intelligence_candidate": candidates,
            "candidate_rationale": rationales,
            "candidate_is_ground_truth": np.full(n, False),
            "evidence_sufficiency": sufficiency,
            "evidence_uncertainty": uncertainty,
            "interpretation_confidence": interp_conf,
        }
    )
