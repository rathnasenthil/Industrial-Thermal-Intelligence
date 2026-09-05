"""Environmental / satellite context summarization for Stage I.7.

Missing environmental datasets → domain unavailable.
Presence flags are only read when the corresponding *_available is True.
Unavailable evidence is never coerced to False/0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evidence_fusion.fusion_schema import ENV_AVAILABILITY_COLUMNS, clean_text


def _boolish_present(series: pd.Series, available: pd.Series) -> list[str | None]:
    """Return PRESENT/ABSENT/None; None when source unavailable."""
    out: list[str | None] = []
    for avail, val in zip(available.tolist(), series.tolist()):
        if not bool(avail):
            out.append(None)
            continue
        if val is None or (isinstance(val, float) and val != val):
            out.append(None)
            continue
        if isinstance(val, (bool, np.bool_)):
            out.append("PRESENT" if bool(val) else "ABSENT")
            continue
        text = str(val).strip().lower()
        if text in ("true", "1", "yes"):
            out.append("PRESENT")
        elif text in ("false", "0", "no"):
            out.append("ABSENT")
        else:
            out.append(None)
    return out


def extract_environmental_evidence(events_df: pd.DataFrame) -> pd.DataFrame:
    """Build environmental domain evidence columns for every event."""
    n = len(events_df)
    event_ids = events_df["event_id"].astype(str).to_numpy()

    present_avail_cols = [c for c in ENV_AVAILABILITY_COLUMNS if c in events_df.columns]
    if not present_avail_cols:
        return pd.DataFrame(
            {
                "event_id": event_ids,
                "environmental_domain_available": np.full(n, False),
                "environmental_landcover_signal": np.full(n, None, dtype=object),
                "environmental_vegetation_signal": np.full(n, None, dtype=object),
                "environmental_agriculture_signal": np.full(n, None, dtype=object),
                "environmental_builtup_signal": np.full(n, None, dtype=object),
                "environmental_water_signal": np.full(n, None, dtype=object),
                "environmental_evidence_summary": np.full(
                    n, "environmental domain unavailable", dtype=object
                ),
            }
        )

    def avail(col: str) -> pd.Series:
        if col not in events_df.columns:
            return pd.Series([False] * n, index=events_df.index)
        return events_df[col].fillna(False).astype(bool)

    land_avail = avail("landcover_available")
    veg_avail = avail("vegetation_context_available")
    ag_avail = avail("agriculture_context_available")
    built_avail = avail("builtup_context_available")
    water_avail = avail("water_context_available")
    sat_avail = avail("satellite_context_available")

    any_available = (
        land_avail | veg_avail | ag_avail | built_avail | water_avail | sat_avail
    ).to_numpy()

    landcover_signal: list[str | None] = []
    for a, cls in zip(
        land_avail.tolist(),
        (
            events_df["dominant_landcover_class"]
            if "dominant_landcover_class" in events_df.columns
            else pd.Series([None] * n)
        ).tolist(),
    ):
        if not a:
            landcover_signal.append(None)
        else:
            landcover_signal.append(clean_text(cls, None))

    veg_signal = _boolish_present(
        events_df["vegetation_present"] if "vegetation_present" in events_df.columns else pd.Series([None] * n),
        veg_avail,
    )
    ag_signal = _boolish_present(
        events_df["agriculture_present"] if "agriculture_present" in events_df.columns else pd.Series([None] * n),
        ag_avail,
    )
    built_signal = _boolish_present(
        events_df["builtup_present"] if "builtup_present" in events_df.columns else pd.Series([None] * n),
        built_avail,
    )
    water_signal = _boolish_present(
        events_df["water_present"] if "water_present" in events_df.columns else pd.Series([None] * n),
        water_avail,
    )

    summaries: list[str] = []
    for i in range(n):
        if not any_available[i]:
            summaries.append("no environmental datasets available for this event")
            continue
        parts: list[str] = []
        if land_avail.iloc[i] and landcover_signal[i] is not None:
            parts.append(f"landcover={landcover_signal[i]}")
        if veg_avail.iloc[i] and veg_signal[i] is not None:
            parts.append(f"vegetation={veg_signal[i]}")
        if ag_avail.iloc[i] and ag_signal[i] is not None:
            parts.append(f"agriculture={ag_signal[i]}")
        if built_avail.iloc[i] and built_signal[i] is not None:
            parts.append(f"builtup={built_signal[i]}")
        if water_avail.iloc[i] and water_signal[i] is not None:
            parts.append(f"water={water_signal[i]}")
        if sat_avail.iloc[i]:
            parts.append("satellite=AVAILABLE")
        summaries.append("; ".join(parts) if parts else "environmental sources marked available but empty")

    return pd.DataFrame(
        {
            "event_id": event_ids,
            "environmental_domain_available": any_available,
            "environmental_landcover_signal": np.asarray(landcover_signal, dtype=object),
            "environmental_vegetation_signal": np.asarray(veg_signal, dtype=object),
            "environmental_agriculture_signal": np.asarray(ag_signal, dtype=object),
            "environmental_builtup_signal": np.asarray(built_signal, dtype=object),
            "environmental_water_signal": np.asarray(water_signal, dtype=object),
            "environmental_evidence_summary": np.asarray(summaries, dtype=object),
        }
    )
