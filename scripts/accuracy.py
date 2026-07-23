#!/usr/bin/env python3
"""Compute model accuracy by comparing historical predictions to actuals.

Reads model_artifacts/HSI/predictions_history.csv (LFS) and:
  - Appends today's prediction row
  - Fetches yfinance actuals up to today (skip holidays + future)
  - Computes direction accuracy per horizon
  - Aggregates monthly + finds best prediction day for same target date
  - Emits JSON + HTML for GitHub Pages

Hardening contract (reviewer-driven):
  - No function in this module raises. On error, print a warning and
    return a benign default (None, {}, False).
  - NaN detection uses math.isnan directly (works for float and np.float*).
  - Numeric detection uses numbers.Real (covers int / float / np.integer /
    np.floating) but excludes bool / str.
"""

from __future__ import annotations

import csv
import json
import math
import numbers
import sys
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

CANONICAL_HEADER = "prediction_date"


def _is_nan(v: Any) -> bool:
    """True iff v is a numeric NaN (float / np.float* / etc.).

    Uses math.isnan directly — works for builtin float and all numpy
    floating-point dtypes without isinstance gymnastics. Non-numeric inputs
    (None, str, list, ...) return False instead of raising.
    """
    if v is None:
        return False
    if isinstance(v, bool):
        return False  # bool is Real but should never count as NaN
    if not isinstance(v, numbers.Real):
        return False
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


def _is_numeric(v: Any) -> bool:
    """True iff v is Real but not bool — i.e. a usable direction number."""
    if isinstance(v, bool):
        return False
    return isinstance(v, numbers.Real)


def _serialize_value(v: Any) -> str:
    """CSV-safe string conversion. Math.isnan handles np.float* natively."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if _is_nan(v):
        return ""
    if isinstance(v, (date, datetime)):
        return v.strftime("%Y-%m-%d")
    return str(v)


def _header_matches(path: Path, expected_first_field: str = CANONICAL_HEADER) -> bool:
    """Return True if path exists, is non-empty, and its first CSV column
    matches the canonical header. Read defensively — never raises.
    """
    try:
        if not path.exists():
            return False
        size = path.stat().st_size
        if size == 0:
            return False
        with path.open("r", encoding="utf-8", newline="") as f:
            first_line = f.readline()
        if not first_line.strip():
            return False
        # Parse just the first column of the header line.
        try:
            first_field = next(csv.reader([first_line]))[0]
        except (csv.Error, IndexError, StopIteration):
            return False
        return first_field == expected_first_field
    except (OSError, ValueError):
        return False


def _write_header_and_row(path: Path, row: dict[str, Any]) -> bool:
    """Re-create path with a canonical header then append `row`.

    Returns True on success. Used when an existing file's header is
    missing / malformed. Never raises.
    """
    try:
        # Open in 'w' to truncate anything garbage, then immediately append.
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=CSV_FIELDS, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerow(
                {k: _serialize_value(row.get(k)) for k in CSV_FIELDS}
            )
        return True
    except (OSError, csv.Error, ValueError) as exc:
        print(
            f"[accuracy] WARNING: failed to recreate history at {path}: {exc}",
            file=sys.stderr,
        )
        return False


def append_to_history(csv_path: Path, row: dict[str, Any]) -> bool:
    """Append a single prediction row to predictions_history.csv.

    Behavior:
      - Creates parent dirs.
      - If the CSV is missing / empty / has a malformed header,
        re-creates it with the canonical header before appending.
      - File opens and writes are wrapped in try/except — never raises.
      - Prints a warning and returns False on failure.
    """
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(
            f"[accuracy] WARNING: could not mkdir {csv_path.parent}: {exc}",
            file=sys.stderr,
        )
        return False

    needs_recreate = not _header_matches(csv_path)

    if needs_recreate:
        return _write_header_and_row(csv_path, row)

    try:
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=CSV_FIELDS, extrasaction="ignore"
            )
            writer.writerow(
                {k: _serialize_value(row.get(k)) for k in CSV_FIELDS}
            )
        return True
    except (OSError, csv.Error, ValueError) as exc:
        print(
            f"[accuracy] WARNING: failed to append to {csv_path}: {exc}",
            file=sys.stderr,
        )
        return False


def compute_direction_accuracy(
    preds: Any, actuals: Any
) -> float | None:
    """Compute fraction of horizons where sign(pred) == sign(actual).

    Hardened per reviewer findings — NEVER raises. On invalid input
    (length mismatch, non-list), prints a warning and returns None.

    Skips, without raising:
      - length mismatch between preds and actuals (-> None)
      - preds / actuals that are None
      - preds / actuals that are NaN (math.isnan handles float + np.float*)
      - preds / actuals that are not Real numbers (str, bool, list, ...)
      - pred == 0 (model was neutral)
    """
    if not isinstance(preds, list) or not isinstance(actuals, list):
        print(
            "[accuracy] WARNING: preds/actuals must be lists, got "
            f"{type(preds).__name__}/{type(actuals).__name__}",
            file=sys.stderr,
        )
        return None
    if len(preds) != len(actuals):
        print(
            f"[accuracy] WARNING: length mismatch preds={len(preds)} "
            f"actuals={len(actuals)}; returning None",
            file=sys.stderr,
        )
        return None

    correct = 0
    total = 0
    for p, a in zip(preds, actuals):
        # --- skip invalid predictions ---
        if p is None or not _is_numeric(p):
            continue
        if _is_nan(p):
            continue
        # --- skip invalid actuals ---
        if a is None or not _is_numeric(a):
            continue
        if _is_nan(a):
            continue
        # --- skip neutral (zero) predictions ---
        if p == 0:
            continue
        total += 1
        # Both positive, or both negative = correct
        if (p > 0 and a > 0) or (p < 0 and a < 0):
            correct += 1
    if total == 0:
        return None
    return correct / total


def _parse_float(value: Any) -> float | None:
    """Parse a finite numeric value, returning None for invalid input."""
    try:
        if value is None or isinstance(value, bool):
            return None
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return parsed
    except Exception:
        return None


def _horizon_days(horizon: Any) -> int | None:
    """Normalize horizon forms such as ``1``, ``"1"``, and ``"1d"``."""
    try:
        value = horizon[:-1] if isinstance(horizon, str) and horizon.endswith("d") else horizon
        days = int(value)
        return days if days > 0 else None
    except Exception:
        return None


def _row_pred_for_horizon(row: Any, horizon: Any) -> float | None:
    """Extract a prediction value for a horizon such as ``"1d"``."""
    try:
        if not isinstance(row, dict):
            return None
        days = _horizon_days(horizon)
        if days is None:
            return None
        return _parse_float(row.get(f"T+{days}_pred"))
    except Exception:
        return None


def _row_actual_for_horizon(row: Any, horizon: Any) -> float | None:
    """Extract an actual value for a horizon such as ``"1d"``."""
    try:
        if not isinstance(row, dict):
            return None
        days = _horizon_days(horizon)
        if days is None:
            return None
        return _parse_float(row.get(f"T+{days}_actual"))
    except Exception:
        return None


def compute_monthly_accuracy(
    rows: Any, horizon: str = "1d"
) -> dict[str, float]:
    """Group rows by prediction month and compute direction accuracy.

    Unrealized or malformed rows are skipped. Returns an empty mapping for
    invalid input and never raises.
    """
    if not isinstance(rows, list) or _horizon_days(horizon) is None:
        return {}

    monthly: dict[str, list[float]] = {}
    try:
        for row in rows:
            if not isinstance(row, dict):
                continue
            pred = _row_pred_for_horizon(row, horizon)
            actual = _row_actual_for_horizon(row, horizon)
            if pred is None or pred == 0 or actual is None:
                continue
            pred_date = row.get("prediction_date")
            if isinstance(pred_date, (date, datetime)):
                pred_date = pred_date.isoformat()
            if not isinstance(pred_date, str):
                continue
            try:
                month_key = date.fromisoformat(pred_date).strftime("%Y-%m")
            except (ValueError, TypeError):
                continue
            correct_value = row.get(f"T+{_horizon_days(horizon)}_correct")
            if isinstance(correct_value, str):
                normalized = correct_value.strip().lower()
                if normalized in {"true", "1"}:
                    correct = True
                elif normalized in {"false", "0"}:
                    correct = False
                else:
                    correct = (pred > 0 and actual > 0) or (pred < 0 and actual < 0)
            elif isinstance(correct_value, bool):
                correct = correct_value
            else:
                correct = (pred > 0 and actual > 0) or (pred < 0 and actual < 0)
            monthly.setdefault(month_key, []).append(1.0 if correct else 0.0)
    except Exception as exc:
        print(f"[accuracy] WARNING: monthly accuracy failed: {exc}", file=sys.stderr)
        return {}

    return {month: sum(values) / len(values) for month, values in monthly.items() if values}


def find_best_prediction_for_target(
    rows: Any, target_date: str, horizon: Any = None
) -> dict | None:
    """Return the lowest-error row whose effective target matches a date.

    When ``horizon`` is None, all supported horizons are searched. Ties are
    resolved by the shorter horizon. Invalid input is skipped and never raises.
    """
    if not isinstance(rows, list) or not isinstance(target_date, str):
        return None
    try:
        target = date.fromisoformat(target_date).isoformat()
    except (ValueError, TypeError):
        return None

    requested_horizon = None
    if horizon is not None:
        requested_horizon = _horizon_days(horizon)
        if requested_horizon is None:
            return None

    candidates: list[tuple[float, int, dict]] = []
    try:
        for row in rows:
            if not isinstance(row, dict):
                continue
            pred_date = row.get("prediction_date")
            if isinstance(pred_date, datetime):
                pred_date = pred_date.date()
            elif isinstance(pred_date, str):
                try:
                    pred_date = date.fromisoformat(pred_date)
                except (ValueError, TypeError):
                    continue
            if not isinstance(pred_date, date):
                continue

            for days in (1, 5, 7, 10, 15, 20, 25, 30):
                if requested_horizon is not None and days != requested_horizon:
                    continue
                pred = _row_pred_for_horizon(row, days)
                actual = _row_actual_for_horizon(row, days)
                if pred is None or actual is None:
                    continue
                if (pred_date + timedelta(days=days)).isoformat() != target:
                    continue
                candidates.append((abs(pred - actual), days, row))
    except Exception as exc:
        print(f"[accuracy] WARNING: best-prediction search failed: {exc}", file=sys.stderr)
        return None

    if not candidates:
        return None
    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
    return candidates[0][2]


def fetch_actuals(ticker: str, start: str, end: str) -> dict[str, float]:
    """Fetch close prices from yfinance between start and end dates.

    Returns: {YYYY-MM-DD: close_price} for all available business days.
    Returns {} on any failure — including yfinance errors, malformed
    DataFrames, missing or NaN close prices, and unexpected per-row
    exceptions. Never raises.
    """
    out: dict[str, float] = {}

    try:
        data = yf.Ticker(ticker).history(
            start=start, end=end, auto_adjust=False
        )
    except Exception as exc:
        print(
            f"[accuracy] WARNING: yfinance fetch failed for {ticker}: {exc}",
            file=sys.stderr,
        )
        return {}

    try:
        if data is None:
            return {}
        # data.empty itself can raise on exotic indexers — guard it.
        try:
            empty = data.empty
        except Exception:
            empty = True
        if empty:
            return {}
        try:
            cols = set(data.columns)
        except Exception:
            cols = set()
        if "Close" in cols:
            price_col = "Close"
        elif "Adj Close" in cols:
            price_col = "Adj Close"
        else:
            return {}

        try:
            rows_iter = data.iterrows()
        except Exception as exc:
            print(
                f"[accuracy] WARNING: iterrows failed: {exc}",
                file=sys.stderr,
            )
            return {}

        for idx, row in rows_iter:
            try:
                price = row[price_col]
                if hasattr(price, "item"):
                    price = price.item()
                if price is None or _is_nan(price):
                    continue
                key = idx.strftime("%Y-%m-%d")
                out[key] = float(price)
            except Exception as exc:
                # Per-row failure — skip this row, keep going.
                print(
                    f"[accuracy] WARNING: skipping row at {idx}: {exc}",
                    file=sys.stderr,
                )
                continue
    except Exception as exc:
        # Catch-all for anything we didn't anticipate above.
        print(
            f"[accuracy] WARNING: unexpected fetch_actuals error: {exc}",
            file=sys.stderr,
        )
        return {}

    return out


def build_accuracy_data(rows: list[dict], ticker: str = "^HSI") -> dict:
    """Build the full data dict for the accuracy page.

    Aggregates:
      - ``monthly_accuracy`` by calling :func:`compute_monthly_accuracy`
        for the 1-day horizon.
      - ``summary.overall_accuracy`` across all supported horizons
        (1, 5, 10, 15, 20, 25, 30 days), counting only rows where both
        pred and actual are valid, non-zero, and non-NaN.
      - ``best_predictions`` — for each unique target date in the last
        30 days, the lowest-error prediction across all horizons.

    Per the module's never-raise contract, any unexpected failure during
    aggregation prints a warning and falls back to a benign default.
    """
    try:
        from datetime import date as _date, timedelta as _timedelta

        monthly = compute_monthly_accuracy(rows, horizon="1d")

        # Overall: average across all horizons with non-null actuals
        overall_correct = 0
        overall_total = 0
        try:
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    for h in (1, 5, 10, 15, 20, 25, 30):
                        pred = _row_pred_for_horizon(row, f"{h}d")
                        actual = _row_actual_for_horizon(row, f"{h}d")
                        if pred is None or pred == 0 or actual is None:
                            continue
                        overall_total += 1
                        if (pred > 0 and actual > 0) or (pred < 0 and actual < 0):
                            overall_correct += 1
        except Exception as exc:
            print(
                f"[accuracy] WARNING: overall accuracy failed: {exc}",
                file=sys.stderr,
            )
        overall_acc = overall_correct / overall_total if overall_total else None

        # Best predictions for last 30 days' target dates
        try:
            today = _date.today()
        except Exception:
            today = None

        best_predictions: list[dict] = []
        seen_targets: set[str] = set()
        try:
            if isinstance(rows, list):
                # First, collect all unique target dates in CSV
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    pd_str = row.get("prediction_date", "")
                    if not isinstance(pd_str, str) or not pd_str:
                        continue
                    try:
                        pd_obj = _date.fromisoformat(pd_str)
                    except ValueError:
                        continue
                    for h in (1, 5, 10, 15, 20, 25, 30):
                        td = (pd_obj + _timedelta(days=h)).isoformat()
                        seen_targets.add(td)
                # For each of the last 30 target dates, find best prediction
                for td in sorted(seen_targets)[-30:]:
                    try:
                        best = find_best_prediction_for_target(rows, td)
                    except Exception as exc:
                        print(
                            f"[accuracy] WARNING: best prediction lookup "
                            f"failed for {td}: {exc}",
                            file=sys.stderr,
                        )
                        best = None
                    if not isinstance(best, dict):
                        continue
                    # Get top 10 patterns for that best day
                    top_patterns: list[dict] = []
                    for i in range(1, 11):
                        try:
                            pat = best.get(f"top{i}_pattern")
                            wt = best.get(f"top{i}_weight")
                        except Exception:
                            pat = None
                            wt = None
                        if pat and wt:
                            try:
                                top_patterns.append(
                                    {"name": str(pat), "weight": float(wt)}
                                )
                            except (TypeError, ValueError):
                                pass
                    pred_val = best.get("T+1_pred")
                    actual_val = best.get("T+1_actual")
                    try:
                        pred_f = float(pred_val) if pred_val not in (None, "") else None
                    except (TypeError, ValueError):
                        pred_f = None
                    try:
                        actual_f = float(actual_val) if actual_val not in (None, "") else None
                    except (TypeError, ValueError):
                        actual_f = None
                    best_predictions.append({
                        "target_date": td,
                        "prediction_date": best.get("prediction_date", ""),
                        "horizon": best.get("best_horizon", "?"),
                        "pred": pred_f,
                        "actual": actual_f,
                        "top_patterns": top_patterns,
                    })
        except Exception as exc:
            print(
                f"[accuracy] WARNING: best predictions loop failed: {exc}",
                file=sys.stderr,
            )

        last_updated = today.isoformat() if today else ""

        return {
            "ticker": ticker,
            "last_updated": last_updated,
            "summary": {
                "overall_accuracy": overall_acc,
                "samples_total": overall_total,
                "months_covered": len(monthly) if isinstance(monthly, dict) else 0,
            },
            "monthly_accuracy": monthly if isinstance(monthly, dict) else {},
            "best_predictions": best_predictions,
        }
    except Exception as exc:
        print(
            f"[accuracy] WARNING: build_accuracy_data failed: {exc}",
            file=sys.stderr,
        )
        return {
            "ticker": ticker,
            "last_updated": "",
            "summary": {
                "overall_accuracy": None,
                "samples_total": 0,
                "months_covered": 0,
            },
            "monthly_accuracy": {},
            "best_predictions": [],
        }


def _bar(value: float, max_value: float, width: int = 30) -> str:
    """Render a text bar chart as a string of filled/empty blocks."""
    if max_value == 0:
        return "░" * width
    pct = max(0, min(1, value / max_value))
    filled = round(pct * width)
    return "▓" * filled + "░" * (width - filled)


def _top_patterns_svg(top_patterns: list[dict]) -> str:
    """Render top 10 patterns as horizontal SVG bars."""
    if not top_patterns:
        return "<p class=\"empty\">無資料</p>"
    max_w = max(p["weight"] for p in top_patterns) or 0.001
    rows = []
    for p in top_patterns:
        w_px = (p["weight"] / max_w) * 280
        rows.append(
            f'<div class="pattern-row">'
            f'<span class="pattern-name">{p["name"]}</span>'
            f'<svg class="pattern-bar" width="280" height="14">'
            f'<rect x="0" y="0" width="{w_px:.1f}" height="14" fill="#1f883d"/>'
            f'</svg>'
            f'<span class="pattern-weight">{p["weight"]*100:.1f}%</span>'
            f'</div>'
        )
    return "\n".join(rows)


def _monthly_bars_svg(monthly: dict[str, float]) -> str:
    """Render 6-month accuracy as horizontal bars."""
    if not monthly:
        return "<p class=\"empty\">無資料</p>"
    rows = []
    for month in sorted(monthly.keys())[-6:]:
        acc = monthly[month] * 100
        w_px = acc * 2.5
        rows.append(
            f'<div class="month-row">'
            f'<span class="month-name">{month}</span>'
            f'<svg class="month-bar" width="250" height="18">'
            f'<rect x="0" y="0" width="{w_px:.1f}" height="18" fill="#0969da"/>'
            f'</svg>'
            f'<span class="month-acc">{acc:.0f}%</span>'
            f'</div>'
        )
    return "\n".join(rows)


def render_html(data: dict) -> str:
    """Render the accuracy data dict as a self-contained HTML page."""
    last_updated = data.get("last_updated", "")
    summary = data.get("summary", {})
    monthly = data.get("monthly_accuracy", {})
    best = data.get("best_predictions", [])
    overall = summary.get("overall_accuracy")
    overall_str = f"{overall*100:.1f}%" if overall is not None else "—"
    samples = summary.get("samples_total", 0)
    months = summary.get("months_covered", 0)

    monthly_svg = _monthly_bars_svg(monthly)
    best_table_rows = []
    for b in best[:20]:
        try:
            pred_val = b.get("pred")
            actual_val = b.get("actual")
            pred_num = pred_val if isinstance(pred_val, (int, float)) else 0
            actual_num = actual_val if isinstance(actual_val, (int, float)) else 0
            err = abs(pred_num - actual_num) * 100
            best_table_rows.append(
                f'<tr><td>{b.get("target_date","")}</td>'
                f'<td>{b.get("prediction_date","")}</td>'
                f'<td>{pred_num*100:+.2f}%</td>'
                f'<td>{actual_num*100:+.2f}%</td>'
                f'<td>{err:.2f}%</td></tr>'
            )
        except Exception:
            continue

    # Find the best prediction (lowest error) for top-10 display
    top_pred = best[0] if best else None
    top_patterns_svg = _top_patterns_svg(top_pred["top_patterns"]) if top_pred else "<p class=\"empty\">無資料</p>"

    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<title>HSI 預測準確率分析</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang TC", sans-serif;
         max-width: 960px; margin: 2em auto; padding: 0 1em; color: #1f2328; background: #ffffff; }}
  h1 {{ font-size: 1.8em; border-bottom: 2px solid #d0d7de; padding-bottom: 0.3em; }}
  h2 {{ font-size: 1.3em; margin-top: 2em; color: #0969da; }}
  .meta {{ color: #656d76; font-size: 0.9em; }}
  .summary {{ display: flex; gap: 2em; background: #f6f8fa; padding: 1em 1.5em;
             border-radius: 6px; margin: 1em 0; }}
  .summary .stat {{ text-align: center; }}
  .summary .stat-value {{ font-size: 2em; font-weight: bold; color: #1f883d; }}
  .summary .stat-label {{ color: #656d76; font-size: 0.85em; }}
  .month-row, .pattern-row {{ display: flex; align-items: center; margin: 0.3em 0; gap: 0.5em; }}
  .month-name, .pattern-name {{ width: 80px; font-family: monospace; font-size: 0.9em; }}
  .pattern-name {{ width: 200px; }}
  .month-acc, .pattern-weight {{ width: 50px; text-align: right; font-family: monospace; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }}
  th, td {{ border: 1px solid #d0d7de; padding: 0.4em 0.6em; text-align: left; }}
  th {{ background: #f6f8fa; }}
  .empty {{ color: #656d76; font-style: italic; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #d0d7de; color: #656d76; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>HSI 預測準確率分析</h1>
<p class="meta">最後更新：{last_updated} | 標的：{data.get("ticker","")}</p>

<h2>整體概覽</h2>
<div class="summary">
  <div class="stat">
    <div class="stat-value">{overall_str}</div>
    <div class="stat-label">方向命中率</div>
  </div>
  <div class="stat">
    <div class="stat-value">{samples}</div>
    <div class="stat-label">總樣本數</div>
  </div>
  <div class="stat">
    <div class="stat-value">{months}</div>
    <div class="stat-label">覆蓋月數</div>
  </div>
</div>

<h2>近 6 個月每月準確率</h2>
{monthly_svg}

<h2>最近 30 天同目標日期對比（誤差最小者為最佳預測日）</h2>
<table>
  <thead><tr><th>目標日期</th><th>最佳預測日</th><th>預測值</th><th>實際值</th><th>誤差</th></tr></thead>
  <tbody>
  {chr(10).join(best_table_rows) if best_table_rows else '<tr><td colspan="5" class="empty">無資料</td></tr>'}
  </tbody>
</table>

<h2>最佳預測日的 Top 10 因子 (Pattern Attention)</h2>
{top_patterns_svg}

<footer>由 daily-morning-push 自動生成</footer>
</body>
</html>
"""


if __name__ == "__main__":  # pragma: no cover
    # Module smoke check — useful when run directly for ad-hoc verification.
    print("accuracy module loaded ok")
