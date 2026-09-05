"""
CLI for GIFT Stage I.6 — Satellite / Environmental Context.

    python -m src.environmental_context.run_environmental_context
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.environmental_context.config import EnvironmentalContextConfig
from src.environmental_context.context_pipeline import (
    load_events,
    resolve_events_input_path,
    run_environmental_context,
    save_outputs,
)
from src.environmental_context.context_report import save_report

_DEFAULT_OUTPUT = Path("data/processed/thermal_events_with_environmental_context.csv")
_DEFAULT_REPORT = Path("data/processed/environmental_context_report.json")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GIFT Stage I.6: environmental/satellite context evidence for thermal events. "
            "Does not classify fire sources. Does not download or fabricate datasets."
        )
    )
    parser.add_argument("--events", type=Path, default=None, help="Override events CSV path.")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    parser.add_argument("--context-buffer-km", type=float, default=1.0)
    parser.add_argument("--broad-context-buffer-km", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = EnvironmentalContextConfig(
        context_buffer_km=args.context_buffer_km,
        broad_context_buffer_km=args.broad_context_buffer_km,
    )
    if args.events is not None:
        events_path = Path(args.events)
        if not events_path.exists():
            print(f"ERROR: events file not found: {events_path}")
            return 2
    else:
        events_path = resolve_events_input_path(config)

    print(f"Loading events from {events_path} ...")
    events_df = load_events(events_path)
    print(f"Loaded {len(events_df):,} events.")
    print(f"Context config: {config.to_dict()}")

    result = run_environmental_context(
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
    print(json.dumps(result.report["per_source_availability"], indent=2))
    print(json.dumps({"processing_time_seconds": result.report["processing_time_seconds"]}, indent=2))
    if result.report.get("warnings"):
        print("Warnings:")
        for w in result.report["warnings"]:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
