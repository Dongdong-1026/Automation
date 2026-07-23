#!/usr/bin/env python3
"""Driver: take summary.json, append to history CSV, regenerate accuracy page."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.accuracy import (
    append_to_history,
    build_accuracy_data,
    render_html,
    update_actuals,
)


def _read_csv_rows(csv_path: Path) -> list[dict]:
    """Read all existing rows from predictions_history.csv without raising."""
    try:
        if not csv_path.exists():
            return []
        with csv_path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        print(f"WARN: failed to read {csv_path}: {exc}", file=sys.stderr)
        return []


def main():
    """Run the accuracy update; report failures without raising."""
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--summary", required=True, type=Path)
        parser.add_argument("--csv", required=True, type=Path)
        parser.add_argument("--ticker", required=True)
        parser.add_argument("--data-out", required=True, type=Path)
        parser.add_argument("--html-out", required=True, type=Path)
        args = parser.parse_args()

        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        today = date.today().isoformat()

        # 1) Append today's prediction to history
        row = {
            "prediction_date": today,
            "ticker": args.ticker.lstrip("^"),
            "vol_ann": summary.get("vol_ann"),
            "direction": summary.get("direction"),
        }
        predictions = summary.get("predictions") or {}
        for horizon in [1, 5, 10, 15, 20, 25, 30]:
            row[f"T+{horizon}_pred"] = predictions.get(f"{horizon}d")
            row[f"T+{horizon}_actual"] = ""
            row[f"T+{horizon}_correct"] = ""
        pattern_attention = summary.get("pattern_attention") or {}
        for index, (name, weight) in enumerate(
            sorted(pattern_attention.items(), key=lambda item: item[1], reverse=True)[:10],
            1,
        ):
            row[f"top{index}_pattern"] = name
            row[f"top{index}_weight"] = weight
        append_to_history(args.csv, row)
        print(f"Appended row for {today}")

        # 2) Update actuals for past predictions
        rows = _read_csv_rows(args.csv)
        try:
            update_actuals(args.csv, rows, ticker="^" + args.ticker.lstrip("^"))
        except Exception as exc:
            print(f"WARN: update_actuals failed: {exc}", file=sys.stderr)

        # 3) Rebuild data + HTML
        rows = _read_csv_rows(args.csv)
        data = build_accuracy_data(rows, ticker=args.ticker)
        args.data_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.data_out.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        args.html_out.write_text(render_html(data), encoding="utf-8")
        print(f"Wrote {args.data_out} and {args.html_out}")
        return True
    except Exception as exc:
        print(f"WARN: accuracy run failed: {exc}", file=sys.stderr)
        return False


if __name__ == "__main__":
    main()
