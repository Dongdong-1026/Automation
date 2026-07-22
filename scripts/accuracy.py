#!/usr/bin/env python3
"""Compute model accuracy by comparing historical predictions to actuals.

Reads model_artifacts/HSI/predictions_history.csv (LFS) and:
  - Appends today's prediction row
  - Fetches yfinance actuals up to today (skip holidays + future)
  - Computes direction accuracy per horizon
  - Aggregates monthly + finds best prediction day for same target date
  - Emits JSON + HTML for GitHub Pages
"""

from __future__ import annotations

import csv
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yfinance as yf

CSV_FIELDS = [
    "prediction_date", "ticker",
    "T+1_pred", "T+1_actual", "T+1_correct",
    "T+5_pred", "T+5_actual", "T+5_correct",
    "T+10_pred", "T+10_actual", "T+10_correct",
    "T+15_pred", "T+15_actual", "T+15_correct",
    "T+20_pred", "T+20_actual", "T+20_correct",
    "T+25_pred", "T+25_actual", "T+25_correct",
    "T+30_pred", "T+30_actual", "T+30_correct",
    "vol_ann", "direction",
    "top1_pattern", "top1_weight",
    "top2_pattern", "top2_weight",
    "top3_pattern", "top3_weight",
    "top4_pattern", "top4_weight",
    "top5_pattern", "top5_weight",
    "top6_pattern", "top6_weight",
    "top7_pattern", "top7_weight",
    "top8_pattern", "top8_weight",
    "top9_pattern", "top9_weight",
    "top10_pattern", "top10_weight",
]


def _serialize_value(v: Any) -> str:
    """CSV-safe string conversion."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return str(v)
    if isinstance(v, (date, datetime)):
        return v.strftime("%Y-%m-%d")
    return str(v)


def append_to_history(csv_path: Path, row: dict[str, Any]) -> None:
    """Append a single prediction row to predictions_history.csv. Creates file with header if missing."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow({k: _serialize_value(row.get(k)) for k in CSV_FIELDS})


def compute_direction_accuracy(
    preds: list[float], actuals: list[float]
) -> float | None:
    """Compute fraction of horizons where sign(pred) == sign(actual).

    Excludes:
      - Any horizon where pred == 0 (model was neutral)
      - Any horizon where actual is None or NaN (unrealized / holiday)
    Returns None if no comparable horizons remain.
    """
    if len(preds) != len(actuals):
        raise ValueError("preds and actuals must have the same length")
    correct = 0
    total = 0
    for p, a in zip(preds, actuals):
        if p == 0:
            continue
        if a is None:
            continue
        if isinstance(a, float) and math.isnan(a):
            continue
        total += 1
        # Both positive, or both negative = correct
        if (p > 0 and a > 0) or (p < 0 and a < 0):
            correct += 1
    if total == 0:
        return None
    return correct / total


def fetch_actuals(ticker: str, start: str, end: str) -> dict[str, float]:
    """Fetch close prices from yfinance between start and end dates.

    Returns: {YYYY-MM-DD: close_price} for all available business days.
    Empty dict if yfinance fails or no data.
    """
    try:
        data = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
    except Exception:
        return {}
    if data is None or data.empty:
        return {}
    # Use 'Close' (or 'Adj Close' fallback); drop NaN
    price_col = "Close" if "Close" in data.columns else (
        "Adj Close" if "Adj Close" in data.columns else None
    )
    if price_col is None:
        return {}
    out: dict[str, float] = {}
    for idx, row in data.iterrows():
        price = row[price_col]
        if hasattr(price, "item"):
            price = price.item()
        if price is None or (isinstance(price, float) and math.isnan(price)):
            continue
        out[idx.strftime("%Y-%m-%d")] = float(price)
    return out