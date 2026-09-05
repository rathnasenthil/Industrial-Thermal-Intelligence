"""
CLI for GIFT Stage I.7 — Evidence Fusion / Source Intelligence.

    python -m src.evidence_fusion.run_evidence_fusion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.evidence_fusion.config import EvidenceFusionConfig
from src.evidence_fusion.fusion_pipeline import load_events, run_evidence_fusion, save_outputs
from src.evidence_fusion.fusion_report import save_report

_DEFAULT_OUTPUT = Path("data/processed/thermal_events_with_evidence_fusion.csv")
_DEFAULT_REPORT = Path("data/processed/evidence_fusion_report.json")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GIFT Stage I.7: explainable evidence fusion / source-intelligence candidates. "
            "Candidates are not ground truth. No ML / risk scores."
        )
    )
    parser.add_argument("--events", type=Path, default=None, help="Override I.6 events CSV path.")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = EvidenceFusionConfig()
    events_path = Path(args.events) if args.events is not None else Path(config.events_path)
    if not events_path.exists():
        print(f"ERROR: events file not found: {events_path}")
        return 2

    print(f"Loading events from {events_path} ...")
    events_df = load_events(events_path)
    print(f"Loaded {len(events_df):,} events.")
    print(f"Fusion config: {config.to_dict()}")

    result = run_evidence_fusion(
        events_df,
        config,
        events_input_path=str(events_path),
        output_path=str(args.output),
    )
    save_outputs(result, args.output)
    save_report(result.report, args.report)

    print()
    print(f"Wrote {len(result.events_df):,} events -> {args.output}")
    print(f"Wrote report -> {args.report}")
    print(json.dumps(result.report["candidate_distribution"], indent=2))
    print(json.dumps({"processing_time_seconds": result.report["processing_time_seconds"]}, indent=2))
    if result.report.get("warnings"):
        print("Warnings:")
        for w in result.report["warnings"]:
            try:
                print(f"  - {w}")
            except UnicodeEncodeError:
                print(f"  - {w.encode('ascii', 'replace').decode('ascii')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
