#!/usr/bin/env python3
"""Run the portable weather/accident case study (data not included)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from weather_traffic.analysis import (
    fetch_frost, monthly_weather, load_accidents, join_and_features, run_models,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accidents", required=True, type=Path, help="Monthly TRINE export (CSV, semicolon-separated)")
    weather = parser.add_mutually_exclusive_group(required=True)
    weather.add_argument("--weather-daily", type=Path, help="Existing station/day Frost observations CSV")
    weather.add_argument("--fetch-frost", action="store_true", help="Retrieve observations using FROST_CLIENT_ID")
    parser.add_argument("--outdir", default=Path("outputs"), type=Path)
    args = parser.parse_args()
    daily = fetch_frost() if args.fetch_frost else pd.read_csv(args.weather_daily)
    monthly = monthly_weather(daily)
    accidents = load_accidents(args.accidents)
    joined = join_and_features(accidents, monthly)
    args.outdir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(args.outdir / "monthly_weather.csv", index=False)
    joined.to_csv(args.outdir / "final_dataset.csv", index=False)
    if args.fetch_frost:
        daily.to_csv(args.outdir / "weather_daily.csv", index=False)
    results = run_models(joined, outdir=args.outdir)
    print(results.round(3).to_string(index=False))
    print(f"Outputs written to {args.outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
