"""Evidence-availability profiling across fusion domains."""

from __future__ import annotations

import numpy as np
import pandas as pd

DOMAIN_ORDER: tuple[str, ...] = (
    "temporal",
    "infrastructure",
    "sta",
    "environmental",
)


def build_availability_profile(
    temporal: pd.DataFrame,
    infrastructure: pd.DataFrame,
    sta: pd.DataFrame,
    environmental: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble per-event availability counts and source lists."""
    event_ids = temporal["event_id"].astype(str).to_numpy()
    flags = {
        "temporal": temporal["temporal_evidence_available"].astype(bool).to_numpy(),
        "infrastructure": infrastructure["infrastructure_evidence_available"].astype(bool).to_numpy(),
        "sta": sta["sta_domain_available"].astype(bool).to_numpy(),
        "environmental": environmental["environmental_domain_available"].astype(bool).to_numpy(),
    }

    present_lists: list[str] = []
    missing_lists: list[str] = []
    counts: list[int] = []
    summaries: list[str] = []

    for i in range(len(event_ids)):
        present = [name for name in DOMAIN_ORDER if flags[name][i]]
        missing = [name for name in DOMAIN_ORDER if not flags[name][i]]
        present_lists.append(";".join(present) if present else "")
        missing_lists.append(";".join(missing) if missing else "")
        counts.append(len(present))
        summaries.append(
            f"present={len(present)}/{len(DOMAIN_ORDER)} [{','.join(present) or 'none'}]; "
            f"missing=[{','.join(missing) or 'none'}]"
        )

    return pd.DataFrame(
        {
            "event_id": event_ids,
            "evidence_sources_present_count": np.asarray(counts, dtype=np.int64),
            "evidence_sources_present": np.asarray(present_lists, dtype=object),
            "evidence_sources_missing": np.asarray(missing_lists, dtype=object),
            "evidence_availability_summary": np.asarray(summaries, dtype=object),
        }
    )
