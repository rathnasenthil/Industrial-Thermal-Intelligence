"""
CLI for GIFT Stage VI — Decision & Risk Prioritization.

    python -m src.risk_prioritization.run_risk_prioritization
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.risk_prioritization.config import RiskPrioritizationConfig
from src.risk_prioritization.risk_pipeline import load_events, run_risk_prioritization, save_outputs
from src.risk_prioritization.risk_report import save_report

_DEFAULT_OUTPUT = Path("data/processed/thermal_events_with_risk_prioritization.csv")
_DEFAULT_REPORT = Path("data/processed/risk_prioritization_report.json")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GIFT Stage VI: deterministic investigation priority / risk decision support. "
            "Not a probability of industrial fire. Not validated risk accuracy."
        )
    )
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = RiskPrioritizationConfig()
    events_path = Path(args.events) if args.events is not None else Path(config.events_path)
    if not events_path.exists():
        print(f"ERROR: events file not found: {events_path}")
        return 2

    print(f"Loading events from {events_path} ...")
    events_df = load_events(events_path)
    print(f"Loaded {len(events_df):,} events.")

    result = run_risk_prioritization(
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
    print(json.dumps(result.report["priority_distribution"], indent=2))
    print(
        json.dumps(
            {
                "risk_score_min": result.report["risk_score_distribution"]["min"],
                "risk_score_max": result.report["risk_score_distribution"]["max"],
                "processing_time_seconds": result.report["processing_time_seconds"],
            },
            indent=2,
        )
    )
    print("Risk prioritization is not a validated probability of industrial fire.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
