# HSI 推送改造 + 准确率分析模块 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the HSI morning push with a redesigned Chat card (vol headline, traditional Chinese, no Volatility_Path.png) and a new accuracy analysis module that compares historical predictions to actual yfinance data, with a GitHub Pages static visualization showing 6-month accuracy + best-prediction-day top-10 factors.

**Architecture:** Single workflow (`daily-morning-push.yml`) does everything sequentially: LSTM inference → parse summary → append prediction to LFS CSV → compute accuracy vs yfinance actuals → emit GitHub Pages HTML → push Chat card. No external dependencies beyond yfinance (already used) and Jinja2-style string templating (we use f-strings).

**Tech Stack:** Python 3.10+, papermill, yfinance, pandas, numpy, pytest, requests. No new dependencies.

## Global Constraints

- Python: 3.10 (matches existing workflows)
- Branch: `main` (push directly, no PR flow)
- All Chinese text in `push_to_google_chat.py` user-facing strings: **Traditional Chinese** (繁體)
- LLM prompt in `summarize_with_llm.py`: **stay Simplified Chinese** (don't change)
- Future volatility: use LSTM vol_head raw output (`summary["vol_ann"]`), do NOT post-process with sqrt(T)
- Top 10 factors: from `pattern_attention_full_report.csv` (per-day, 81 patterns)
- Accuracy metric: **direction accuracy** only (sign(pred) == sign(actual))
- Holidays/future: skip in actuals (NaN → null → not in denominator)
- All accuracy failures must not block Chat push (`continue-on-error: true` everywhere)
- LFS used for `predictions_history.csv` (grows over time)

---

### Task 1: Extend summary.json with vol_ann + pattern_attention

**Files:**
- Modify: `scripts/collect_run_artifacts.py:107-118` (the return dict in `_parse_proportional_report`)
- Modify: `scripts/collect_run_artifacts.py:200-210` (the build_summary vol/prediction extraction)
- Test: `tests/test_collect_run_artifacts.py`

**Interfaces:**
- Consumes: existing `base_results["vol_ann"]` (already in summary), existing `pattern_attention_full_report.csv` (in run dir)
- Produces: `summary["vol_ann"]` (float, e.g., 14.6), `summary["pattern_attention"]` (dict[str, float])

- [ ] **Step 1: Write failing test for vol_ann extraction**

In `tests/test_collect_run_artifacts.py`, add:

```python
def test_parse_report_extracts_vol_ann(fake_run_dir, monkeypatch):
    """vol_ann (annualized) must be extracted from raw_vol field."""
    from scripts.collect_run_artifacts import _parse_proportional_report
    report = fake_run_dir / "Proportional_Inference_Report.txt"
    report.write_text(
        "Current T+0: 25000.00\n"
        "T+1 | 2026-08-01 | 24500 (  -2%) | 25050 ( +0.20%) | 25600 ( +2.40%) | 0.92%\n"
        "T+30 | 2026-08-30 | 22000 (-12%) | 25900 ( +3.60%) | 29800 (+19.20%) | 5.05%\n",
        encoding="utf-8",
    )
    parsed = _parse_proportional_report(report)
    assert parsed is not None
    assert "vol_ann" in parsed, "vol_ann (annualized) must be exposed"
    # vol_ann ≈ T+1 vol × sqrt(252) ≈ 0.92 × 15.87 ≈ 14.6
    assert abs(parsed["vol_ann"] - 14.60) < 0.5, f"expected ~14.6, got {parsed['vol_ann']}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Desktop/slashiee/恒生指数预测/LSTM_ipython-main && pytest tests/test_collect_run_artifacts.py::test_parse_report_extracts_vol_ann -v`
Expected: FAIL with `KeyError: 'vol_ann'` or similar (the field doesn't exist yet)

- [ ] **Step 3: Implement vol_ann extraction**

In `scripts/collect_run_artifacts.py`, find the `_parse_proportional_report` return dict and add the `vol_ann` field. Also add the helper to compute it from the report.

Find this code in `_parse_proportional_report` (around line 80-100):
```python
horizon_to_ret[f"{horizon_n}d"] = base_return
horizon_to_vol[f"{horizon_n}d"] = vol
```

Add below the loop, before the return statement:
```python
# Annualized vol: T+1 vol × sqrt(252) (LSTM vol_head raw × time scaling)
vol_t1_pct = horizon_to_vol.get("1d", 0.0) * 100  # back to percent
vol_ann = vol_t1_pct * math.sqrt(252)  # 0.92 × 15.87 ≈ 14.6
```

Update the return dict to include `vol_ann`:
```python
return {
    "predictions": horizon_to_ret,
    "volatility": horizon_to_vol.get("1d", 0.0),
    "volatility_by_horizon": horizon_to_vol,
    "vol_ann": vol_ann,  # NEW: annualized vol (LSTM raw output, like VIX)
    "direction": direction,
    "source": str(report_path),
}
```

Add `import math` at the top if not already present.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_collect_run_artifacts.py::test_parse_report_extracts_vol_ann -v`
Expected: PASS

- [ ] **Step 5: Write failing test for pattern_attention parsing**

Add a helper that creates a fake `pattern_attention_full_report.csv` in `fake_run_dir`, then test:

```python
def test_build_summary_includes_pattern_attention(fake_run_dir, fake_artifacts_root):
    """pattern_attention dict must be populated from pattern_attention_full_report.csv."""
    import csv
    csv_path = fake_run_dir / "pattern_attention_full_report.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pattern", "avg_attention", "max_attention", "min_attention", "std_attention"])
        w.writeheader()
        w.writerow({"pattern": "MA5_Cross_MA20_Bullish", "avg_attention": 0.082, "max_attention": 0.1, "min_attention": 0.05, "std_attention": 0.02})
        w.writerow({"pattern": "Realized_Vol_GK", "avg_attention": 0.075, "max_attention": 0.09, "min_attention": 0.04, "std_attention": 0.018})
    from scripts.collect_run_artifacts import build_summary
    summary = build_summary(
        run_dir=fake_run_dir,
        root=fake_artifacts_root,
        ticker="^HSI",
        commit_sha=None,
    )
    assert "pattern_attention" in summary
    assert "MA5_Cross_MA20_Bullish" in summary["pattern_attention"]
    assert abs(summary["pattern_attention"]["MA5_Cross_MA20_Bullish"] - 0.082) < 0.001
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_collect_run_artifacts.py::test_build_summary_includes_pattern_attention -v`
Expected: FAIL with `KeyError: 'pattern_attention'`

- [ ] **Step 7: Implement pattern_attention parsing in build_summary**

In `scripts/collect_run_artifacts.py`, find `build_summary` function. After the existing vol/prediction extraction, add:

```python
# Pattern attention (per-day, 81 patterns): read pattern_attention_full_report.csv
pattern_csv = run_dir / "pattern_attention_full_report.csv"
if pattern_csv.exists():
    try:
        import csv as _csv
        with pattern_csv.open(encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            summary["pattern_attention"] = {
                row["pattern"]: float(row["avg_attention"])
                for row in reader
                if row.get("pattern") and row.get("avg_attention")
            }
    except (OSError, KeyError, ValueError) as exc:
        summary["pattern_attention"] = {}
        summary.setdefault("warnings", []).append(f"pattern_attention parse failed: {exc}")
else:
    summary["pattern_attention"] = {}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_collect_run_artifacts.py -v`
Expected: 5+ tests PASS (existing 4 + 2 new = 6)

- [ ] **Step 9: Commit**

```bash
git add scripts/collect_run_artifacts.py tests/test_collect_run_artifacts.py
git commit -m "feat(artifacts): expose vol_ann (LSTM raw output) and pattern_attention"
```

---

### Task 2: Build accuracy.py core (CSV writer + yfinance fetcher + direction accuracy)

**Files:**
- Create: `scripts/accuracy.py`
- Test: `tests/test_accuracy.py`
- Modify: `tests/conftest.py` (add fixture for fake predictions_history.csv)

**Interfaces:**
- Consumes: `summary.json` from collect_run_artifacts, `predictions_history.csv` path, yfinance for actuals
- Produces: pure-Python functions `append_to_history(csv_path, row_dict)`, `fetch_actuals(ticker, start, end) -> dict[date, close]`, `compute_direction_accuracy(pred, actual) -> float | None`

- [ ] **Step 1: Write failing test for CSV append**

In `tests/test_accuracy.py`:

```python
"""Tests for scripts.accuracy."""
from __future__ import annotations

import csv
from pathlib import Path
from datetime import date

import pytest


def test_append_to_history_creates_new_file(tmp_path: Path):
    """First append should create the CSV with header."""
    from scripts.accuracy import append_to_history
    csv_path = tmp_path / "predictions_history.csv"
    row = {
        "prediction_date": date(2026, 7, 22),
        "ticker": "^HSI",
        "T+1_pred": 0.0007,
        "T+1_actual": None,
        "T+1_correct": None,
        "vol_ann": 14.6,
        "direction": "up",
        "top1_pattern": "MA5_Cross_MA20_Bullish",
        "top1_weight": 0.082,
    }
    append_to_history(csv_path, row)
    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["ticker"] == "^HSI"
    assert rows[0]["T+1_pred"] == "0.0007"
    assert rows[0]["vol_ann"] == "14.6"
    assert rows[0]["direction"] == "up"


def test_append_to_history_appends_to_existing(tmp_path: Path):
    """Second append should not overwrite the first row."""
    from scripts.accuracy import append_to_history
    csv_path = tmp_path / "predictions_history.csv"
    row1 = {"prediction_date": date(2026, 7, 22), "ticker": "^HSI", "T+1_pred": 0.001, "T+1_actual": None, "T+1_correct": None, "vol_ann": 14.6, "direction": "up", "top1_pattern": "P1", "top1_weight": 0.1}
    row2 = {"prediction_date": date(2026, 7, 23), "ticker": "^HSI", "T+1_pred": 0.002, "T+1_actual": None, "T+1_correct": None, "vol_ann": 15.0, "direction": "down", "top1_pattern": "P2", "top1_weight": 0.2}
    append_to_history(csv_path, row1)
    append_to_history(csv_path, row2)
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["prediction_date"] == "2026-07-22"
    assert rows[1]["prediction_date"] == "2026-07-23"


def test_compute_direction_accuracy_simple():
    """Direction accuracy: sign(pred) == sign(actual), both non-zero."""
    from scripts.accuracy import compute_direction_accuracy
    # 3/4 correct (the 0.0 cases are excluded)
    preds = [0.001, -0.002, 0.003, 0.0, -0.001]
    actuals = [0.005, -0.001, -0.002, 0.001, 0.001]
    # 1: sign(0.001)==sign(0.005) ✓
    # 2: sign(-0.002)==sign(-0.001) ✓
    # 3: sign(0.003)!=sign(-0.002) ✗
    # 4: pred=0 → excluded
    # 5: sign(-0.001)!=sign(0.001) ✗
    # accuracy = 2/3 (excluding the 0 case from both num and denom)
    result = compute_direction_accuracy(preds, actuals)
    assert result == pytest.approx(2/3)


def test_compute_direction_accuracy_handles_nan():
    """NaN actuals (unrealized) must be excluded from accuracy calculation."""
    from scripts.accuracy import compute_direction_accuracy
    import math
    preds = [0.001, 0.002, -0.003]
    actuals = [0.005, math.nan, -0.001]
    # First: 0.001 vs 0.005 → correct
    # Second: excluded (future/unrealized)
    # Third: -0.003 vs -0.001 → correct
    # accuracy = 2/2
    result = compute_direction_accuracy(preds, actuals)
    assert result == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_accuracy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.accuracy'`

- [ ] **Step 3: Create scripts/accuracy.py with append_to_history and compute_direction_accuracy**

```python
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
        return f"{v:.6f}"
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_accuracy.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Write failing test for fetch_actuals (with mocked yfinance)**

```python
def test_fetch_actuals_returns_close_prices(monkeypatch):
    """fetch_actuals returns {date: close} from yfinance for non-holiday dates."""
    from scripts import accuracy

    # Mock yfinance with a fake ticker
    class FakeTicker:
        def history(self, start, end):
            import pandas as pd
            dates = pd.date_range(start, end, freq="B")  # business days
            return pd.DataFrame({
                "Close": [25000 + i for i in range(len(dates))],
            }, index=dates)

    class FakeYFinance:
        def Ticker(self, symbol):
            return FakeTicker()

    monkeypatch.setattr(accuracy.yf, "Ticker", FakeYFinance().Ticker)

    result = accuracy.fetch_actuals("^HSI", "2026-07-20", "2026-07-25")
    # Should have 4 business days (Mon-Fri = 4)
    assert len(result) >= 3
    assert all(close > 0 for close in result.values())


def test_fetch_actuals_handles_empty(monkeypatch):
    """fetch_actuals returns {} if yfinance returns no data."""
    from scripts import accuracy
    import pandas as pd

    class EmptyTicker:
        def history(self, start, end):
            return pd.DataFrame()

    class FakeYFinance:
        def Ticker(self, symbol):
            return EmptyTicker()

    monkeypatch.setattr(accuracy.yf, "Ticker", FakeYFinance().Ticker)
    result = accuracy.fetch_actuals("^HSI", "2026-07-20", "2026-07-25")
    assert result == {}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_accuracy.py -v -k "fetch_actuals"`
Expected: FAIL with `AttributeError: module 'scripts.accuracy' has no attribute 'yf'`

- [ ] **Step 7: Implement fetch_actuals**

Add to `scripts/accuracy.py`:

```python
import yfinance as yf  # at top


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
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_accuracy.py -v`
Expected: 6 tests PASS

- [ ] **Step 9: Commit**

```bash
git add scripts/accuracy.py tests/test_accuracy.py
git commit -m "feat(accuracy): CSV writer + yfinance actuals + direction accuracy"
```

---

### Task 3: Build accuracy.py analytics (monthly + best-prediction-day)

**Files:**
- Modify: `scripts/accuracy.py` (add analytics functions)
- Test: `tests/test_accuracy.py` (add analytics tests)

**Interfaces:**
- Consumes: list of historical prediction rows (dicts from CSV)
- Produces: `compute_monthly_accuracy(rows) -> {month: accuracy}`, `find_best_prediction_for_target(rows, target_date) -> row | None`

- [ ] **Step 1: Write failing tests for monthly accuracy aggregation**

In `tests/test_accuracy.py`:

```python
def test_compute_monthly_accuracy_groups_by_month():
    """Returns dict keyed by YYYY-MM string with accuracy per month."""
    from scripts.accuracy import compute_monthly_accuracy
    rows = [
        # 2026-06: 2 correct out of 2
        {"prediction_date": "2026-06-15", "T+1_pred": "0.001", "T+1_actual": "0.002", "T+1_correct": "true"},
        {"prediction_date": "2026-06-16", "T+1_pred": "-0.001", "T+1_actual": "-0.003", "T+1_correct": "true"},
        # 2026-07: 1 correct, 1 wrong out of 2
        {"prediction_date": "2026-07-15", "T+1_pred": "0.001", "T+1_actual": "0.002", "T+1_correct": "true"},
        {"prediction_date": "2026-07-16", "T+1_pred": "0.001", "T+1_actual": "-0.002", "T+1_correct": "false"},
    ]
    result = compute_monthly_accuracy(rows, horizon="1d")
    assert result["2026-06"] == 1.0  # 2/2
    assert result["2026-07"] == 0.5  # 1/2


def test_compute_monthly_accuracy_skips_unrealized():
    """Rows with null actuals (NaN/empty) are excluded."""
    from scripts.accuracy import compute_monthly_accuracy
    rows = [
        {"prediction_date": "2026-07-15", "T+1_pred": "0.001", "T+1_actual": "", "T+1_correct": ""},
        {"prediction_date": "2026-07-16", "T+1_pred": "0.002", "T+1_actual": "-0.001", "T+1_correct": "true"},
    ]
    result = compute_monthly_accuracy(rows, horizon="1d")
    # Only 1 valid sample, 100% correct
    assert result["2026-07"] == 1.0


def test_find_best_prediction_for_target_picks_lowest_error():
    """Given multiple predictions for the same target date, return the one with min |pred - actual|."""
    from scripts.accuracy import find_best_prediction_for_target
    rows = [
        # Target 2026-07-20
        # Prediction from 2026-07-19 (T+1): predicted +0.2%, actual +0.5%, error 0.3%
        {"prediction_date": "2026-07-19", "T+1_pred": "0.002", "T+1_actual": "0.005", "T+1_correct": "false"},
        # Prediction from 2026-07-15 (T+5): predicted +0.4%, actual +0.5%, error 0.1% ← BEST
        {"prediction_date": "2026-07-15", "T+5_pred": "0.004", "T+5_actual": "0.005", "T+5_correct": "true"},
        # Prediction from 2026-07-13 (T+7): predicted +0.6%, actual +0.5%, error 0.1%
        {"prediction_date": "2026-07-13", "T+7_pred": "0.006", "T+7_actual": "0.005", "T+7_correct": "true"},
    ]
    best = find_best_prediction_for_target(rows, target_date="2026-07-20")
    assert best is not None
    assert best["prediction_date"] == "2026-07-15"  # T+5 had error 0.1, T+7 also 0.1, picks first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_accuracy.py -v -k "compute_monthly or find_best"`
Expected: FAIL with `ImportError: cannot import name 'compute_monthly_accuracy'`

- [ ] **Step 3: Implement compute_monthly_accuracy**

Add to `scripts/accuracy.py`:

```python
def _parse_float(s: str) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _row_pred_for_horizon(row: dict, horizon: str) -> float | None:
    """Extract prediction value for the given horizon (e.g., '1d' → T+1)."""
    return _parse_float(row.get(f"T+{horizon}_pred"))


def _row_actual_for_horizon(row: dict, horizon: str) -> float | None:
    return _parse_float(row.get(f"T+{horizon}_actual"))


def compute_monthly_accuracy(
    rows: list[dict], horizon: str = "1d"
) -> dict[str, float]:
    """Group rows by prediction month (YYYY-MM) and compute direction accuracy.

    Skips rows with null/empty actuals (unrealized horizons).
    Returns {YYYY-MM: accuracy}, only for months with >= 1 valid sample.
    """
    monthly: dict[str, list[float]] = {}
    for row in rows:
        pred = _row_pred_for_horizon(row, horizon)
        actual = _row_actual_for_horizon(row, horizon)
        if pred is None or pred == 0:
            continue
        if actual is None:
            continue
        pred_date = row.get("prediction_date", "")
        if not pred_date:
            continue
        month_key = pred_date[:7]  # YYYY-MM
        correct = (pred > 0 and actual > 0) or (pred < 0 and actual < 0)
        monthly.setdefault(month_key, []).append(1.0 if correct else 0.0)
    return {m: sum(v) / len(v) for m, v in monthly.items() if v}
```

- [ ] **Step 4: Implement find_best_prediction_for_target**

Add to `scripts/accuracy.py`:

```python
def find_best_prediction_for_target(
    rows: list[dict], target_date: str, horizon: str | None = None
) -> dict | None:
    """Find the prediction row that best predicted `target_date`.

    For each row, the effective target_date = prediction_date + horizon days.
    Among rows whose effective target matches `target_date`, return the one
    with the lowest |pred - actual|. Returns None if no match.

    If horizon is None, search all horizons.
    """
    candidates = []
    for row in rows:
        pred_date = row.get("prediction_date", "")
        if not pred_date:
            continue
        # Check each horizon
        for h in [1, 5, 10, 15, 20, 25, 30]:
            if horizon is not None and h != horizon:
                continue
            pred = _row_pred_for_horizon(row, f"{h}d")
            actual = _row_actual_for_horizon(row, f"{h}d")
            if pred is None or actual is None:
                continue
            # Effective target date
            from datetime import date, timedelta
            try:
                pd = date.fromisoformat(pred_date)
                effective_target = (pd + timedelta(days=h)).isoformat()
            except (ValueError, TypeError):
                continue
            if effective_target == target_date:
                error = abs(pred - actual)
                candidates.append((error, h, row))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))  # sort by error, then horizon
    return candidates[0][2]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_accuracy.py -v`
Expected: 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/accuracy.py tests/test_accuracy.py
git commit -m "feat(accuracy): monthly aggregation + best-prediction-day finder"
```

---

### Task 4: Build accuracy.py output (JSON + HTML)

**Files:**
- Modify: `scripts/accuracy.py` (add build_accuracy_data and render_html functions)
- Test: `tests/test_accuracy.py` (add output tests)
- Create: `docs/accuracy.html` (template starter — will be generated)

**Interfaces:**
- Consumes: rows, monthly dict, best prediction row
- Produces: `docs/accuracy_data.json` (machine-readable) + `docs/accuracy.html` (static page)

- [ ] **Step 1: Write failing test for build_accuracy_data**

```python
def test_build_accuracy_data_structure():
    """Output JSON has all required top-level keys."""
    from scripts.accuracy import build_accuracy_data
    rows = [
        {"prediction_date": "2026-07-15", "ticker": "^HSI", "T+1_pred": "0.001", "T+1_actual": "0.002", "T+1_correct": "true", "vol_ann": "14.6", "direction": "up"},
    ]
    data = build_accuracy_data(rows, ticker="^HSI")
    assert "ticker" in data
    assert "last_updated" in data
    assert "monthly_accuracy" in data
    assert "best_predictions" in data  # list of {target_date, ...}
    assert "summary" in data
    assert "samples_total" in data["summary"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_accuracy.py::test_build_accuracy_data_structure -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement build_accuracy_data**

Add to `scripts/accuracy.py`:

```python
def build_accuracy_data(rows: list[dict], ticker: str = "^HSI") -> dict:
    """Build the full data dict for the accuracy page."""
    from datetime import date, timedelta

    monthly = compute_monthly_accuracy(rows, horizon="1d")

    # Overall: average across all horizons with non-null actuals
    overall_correct = 0
    overall_total = 0
    for row in rows:
        for h in [1, 5, 10, 15, 20, 25, 30]:
            pred = _row_pred_for_horizon(row, f"{h}d")
            actual = _row_actual_for_horizon(row, f"{h}d")
            if pred is None or pred == 0 or actual is None:
                continue
            overall_total += 1
            if (pred > 0 and actual > 0) or (pred < 0 and actual < 0):
                overall_correct += 1
    overall_acc = overall_correct / overall_total if overall_total else None

    # Best predictions for last 30 days' target dates
    from datetime import date, timedelta
    today = date.today()
    best_predictions = []
    seen_targets = set()
    # First, collect all unique target dates in CSV
    for row in rows:
        pd_str = row.get("prediction_date", "")
        if not pd_str:
            continue
        try:
            pd = date.fromisoformat(pd_str)
        except ValueError:
            continue
        for h in [1, 5, 10, 15, 20, 25, 30]:
            td = (pd + timedelta(days=h)).isoformat()
            seen_targets.add(td)
    # For each target, find best prediction
    for td in sorted(seen_targets)[-30:]:  # last 30 days
        best = find_best_prediction_for_target(rows, td)
        if best is None:
            continue
        # Get top 10 patterns for that best day
        top_patterns = []
        for i in range(1, 11):
            pat = best.get(f"top{i}_pattern")
            wt = best.get(f"top{i}_weight")
            if pat and wt:
                try:
                    top_patterns.append({"name": pat, "weight": float(wt)})
                except ValueError:
                    pass
        best_predictions.append({
            "target_date": td,
            "prediction_date": best["prediction_date"],
            "horizon": best.get("best_horizon", "?"),
            "pred": float(best.get("T+1_pred", 0)) if best.get("T+1_pred") else None,
            "actual": float(best.get("T+1_actual", 0)) if best.get("T+1_actual") else None,
            "top_patterns": top_patterns,
        })

    return {
        "ticker": ticker,
        "last_updated": today.isoformat(),
        "summary": {
            "overall_accuracy": overall_acc,
            "samples_total": overall_total,
            "months_covered": len(monthly),
        },
        "monthly_accuracy": monthly,
        "best_predictions": best_predictions,
    }
```

- [ ] **Step 4: Write failing test for render_html**

```python
def test_render_html_contains_key_sections(tmp_path):
    """Generated HTML must include all 4 sections + GitHub Primer styling."""
    from scripts.accuracy import render_html, build_accuracy_data
    rows = [
        {"prediction_date": "2026-07-15", "ticker": "^HSI", "T+1_pred": "0.001", "T+1_actual": "0.002", "T+1_correct": "true", "vol_ann": "14.6", "direction": "up", "top1_pattern": "MA5", "top1_weight": "0.08"},
    ]
    data = build_accuracy_data(rows)
    html = render_html(data)
    assert "整體" in html or "概覽" in html  # header
    assert "近 6 個月" in html or "每月" in html
    assert "Top 10" in html or "因子" in html
    assert "<svg" in html  # charts use SVG
    assert "MA5" in html  # pattern name appears
```

- [ ] **Step 5: Run test to verify it fails**

Run: `pytest tests/test_accuracy.py::test_render_html_contains_key_sections -v`
Expected: FAIL with ImportError

- [ ] **Step 6: Implement render_html**

Add to `scripts/accuracy.py`:

```python
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
        err = abs((b.get("pred") or 0) - (b.get("actual") or 0)) * 100
        best_table_rows.append(
            f'<tr><td>{b.get("target_date","")}</td>'
            f'<td>{b.get("prediction_date","")}</td>'
            f'<td>{(b.get("pred") or 0)*100:+.2f}%</td>'
            f'<td>{(b.get("actual") or 0)*100:+.2f}%</td>'
            f'<td>{err:.2f}%</td></tr>'
        )

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
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_accuracy.py -v`
Expected: 10 tests PASS

- [ ] **Step 8: Commit**

```bash
git add scripts/accuracy.py tests/test_accuracy.py
git commit -m "feat(accuracy): build_accuracy_data + render_html for static page"
```

---

### Task 5: Card redesign — new structure, traditional Chinese, vol headline

**Files:**
- Modify: `scripts/push_to_google_chat.py` (entire file restructure)
- Test: `tests/test_push_to_google_chat.py` (update tests)

**Interfaces:**
- Consumes: existing summary.json with new `vol_ann` field
- Produces: new card with vol as first content section, big text, traditional Chinese

- [ ] **Step 1: Rewrite the file with new structure**

Replace `scripts/push_to_google_chat.py` with the new version. Key changes:
- Vol moved to first content section (after AI summary)
- Vol uses `summary["vol_ann"]` directly, formatted as `14.60%`
- Vol text: font 28px bold, "📊 未來波動率"
- No parentheses, no calculation line
- All other Chinese text → traditional Chinese
- Emoji for sections: 📊 預測, 📈 預測, 🎯 命中, 🖼️ 圖, etc.

The full file is shown below. Save and replace:

```python
#!/usr/bin/env python3
"""Assemble a Google Chat cardsV2 message from a summary JSON and POST it.

Card structure (top to bottom):
  1. AI summary (golden text) - if llm_summary present
  2. Future volatility (LARGE, vol_ann from LSTM) - PRIMARY HEADLINE
  3. Key predictions (7 horizons)
  4. 1D hit summary (if available)
  5. PNG charts (excluding Volatility_Path.png)
  6. View commit button + footer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

DEFAULT_TIMEOUT = 15
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
EXCLUDED_PNGS = {"Proportional_Inference_Report.txt", "run_metrics.json"}
VOL_PNG_NAME = "Step8_Volatility_Path.png"  # excluded from push per user request


def _status_emoji(status: str) -> str:
    return {
        "ok": "✅",
        "bootstrapped": "⚠️",
        "incomplete": "⚠️",
        "failed": "❌",
    }.get(status, "ℹ️")


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def _fmt_pct_signed(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.{digits}f}%"


def _prediction_rows(predictions: dict[str, float] | None) -> list[str]:
    if not predictions:
        return []
    rows = []
    for horizon in ("1d", "5d", "10d", "15d", "20d", "25d", "30d"):
        if horizon in predictions and predictions[horizon] is not None:
            rows.append(f"• {horizon}: {_fmt_pct_signed(predictions[horizon])}")
    return rows


def _png_widgets(png_files: list[dict[str, str]]) -> list[dict[str, Any]]:
    widgets: list[dict[str, Any]] = []
    for p in png_files or []:
        widgets.append({"image": {"imageUrl": p.get("url", "")}})
        widgets.append({
            "textParagraph": {
                "text": f"<a href=\"{p.get('url', '')}\">{p.get('name', 'image')}</a>"
            }
        })
    if not widgets:
        widgets.append({"textParagraph": {"text": "（圖片未生成）"}})
    return widgets


def build_card(summary: dict[str, Any]) -> dict[str, Any]:
    """Build a Google Chat cardsV2 payload from a summary dict."""
    status = summary.get("status", "ok")
    emoji = _status_emoji(status)
    ticker = summary.get("ticker", "")
    generated_at = summary.get("generated_at", "")
    title_date = generated_at[:10] if generated_at else ""
    title = f"{emoji} {ticker} 早晨預測 · {title_date} · {status}"

    sections: list[dict[str, Any]] = []

    # Section 0: AI summary (golden text)
    llm_summary = (summary.get("llm_summary") or "").strip()
    if llm_summary:
        sections.append({
            "widgets": [{
                "textParagraph": {
                    "text": (
                        "<font color=\"#FFD700\"><b>✨ AI 總結</b></font><br>"
                        f"<font color=\"#FFD700\">{_html_escape(llm_summary)}</font>"
                    )
                }
            }]
        })

    # Section 1: 未來波動率 (LARGE, PRIMARY HEADLINE)
    # Use vol_ann directly from LSTM vol_head (like VIX annualized 30-day)
    vol_ann = summary.get("vol_ann")
    if vol_ann is not None:
        sections.append({
            "widgets": [{
                "textParagraph": {
                    "text": (
                        f"<b>📊 未來波動率</b><br>"
                        f"<font size=\"6\"><b>{_fmt_pct(vol_ann / 100, 2)}</b></font>"
                    )
                }
            }]
        })

    # Section 2: 本次關鍵預測
    widgets: list[dict[str, Any]] = []
    if status in ("failed", "incomplete"):
        widgets.append({
            "textParagraph": {
                "text": f"<b>錯誤：</b> {_html_escape(summary.get('error') or '未知')}"
            }
        })
    else:
        rows = _prediction_rows(summary.get("predictions"))
        if rows:
            widgets.append({
                "textParagraph": {"text": "<b>📈 本次關鍵預測</b><br>" + "<br>".join(rows)}
            })
        direction = summary.get("direction")
        if direction:
            widgets.append({
                "textParagraph": {
                    "text": f"<b>方向判斷：</b> {direction}"
                }
            })
    if widgets:
        sections.append({"widgets": widgets})

    # Section 3: 1D 命中摘要
    rev = summary.get("latest_1d_review")
    if rev:
        sections.append({
            "widgets": [{
                "textParagraph": {
                    "text": (
                        "<b>🎯 最近 1D 命中摘要</b><br>"
                        f"日期：{rev.get('date')}<br>"
                        f"樣本：{rev.get('sample_count')}<br>"
                        f"方向命中率：{rev.get('direction_accuracy_pct', 0):.2f}%"
                    )
                }
            }]
        })

    # Section 4: 預測圖 (exclude Volatility_Path.png per user request)
    png_files = [
        p for p in (summary.get("png_files") or [])
        if p.get("name") != VOL_PNG_NAME
    ]
    if png_files:
        sections.append({
            "header": "🖼️ 預測圖",
            "widgets": _png_widgets(png_files),
        })

    # Section 5: Footer with View commit button
    sha = summary.get("commit_sha") or ""
    actions = []
    if sha and sha != "none":
        actions.append({
            "text": "View commit",
            "onClick": {
                "openLink": {
                    "url": (
                        f"https://github.com/${{REPO_OWNER}}/${{REPO_NAME}}/"
                        f"commit/{sha}"
                    ),
                },
            },
        })
    footer_widgets: list[dict[str, Any]] = []
    if actions:
        # cardsV2 widget type is 'buttonList' (not 'buttons' — that's v1 card syntax)
        footer_widgets.append({"buttonList": {"buttons": actions}})
    footer_widgets.append({
        "textParagraph": {"text": "由 daily-morning-push 自動生成"}
    })
    sections.append({"widgets": footer_widgets})

    return {
        "cardsV2": [{
            "cardId": "hsi-morning-push",
            "card": {
                "header": {"title": title},
                "sections": sections,
            },
        }]
    }


def post_card(card: dict[str, Any], webhook_url: str, timeout: int = DEFAULT_TIMEOUT) -> int:
    """POST the card to the webhook. Returns HTTP status code."""
    try:
        resp = requests.post(
            webhook_url,
            json=card,
            timeout=timeout,
        )
        return resp.status_code
    except requests.RequestException:
        return 0


def _post_with_retry(card: dict[str, Any], webhook_url: str) -> int:
    last = 0
    for attempt in range(3):
        last = post_card(card, webhook_url)
        if 200 <= last < 300:
            return last
        if last not in RETRYABLE_STATUSES:
            return last
        time.sleep(2 ** attempt)
    return last


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--payload", required=True, type=Path)
    p.add_argument("--webhook-env", default="GOOGLE_CHAT_WEBHOOK")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    webhook = os.environ.get(args.webhook_env, "")
    if not webhook:
        print(f"ERROR: env var {args.webhook_env} is empty", file=sys.stderr)
        return 1
    summary = json.loads(args.payload.read_text(encoding="utf-8"))
    card = build_card(summary)
    code = _post_with_retry(card, webhook)
    if 200 <= code < 300:
        print(f"OK: chat push status={code}")
        return 0
    print(f"ERROR: chat push failed status={code}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Update tests for new card structure**

Replace `tests/test_push_to_google_chat.py` with new assertions:

```python
"""Tests for push_to_google_chat.build_card."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import responses

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.push_to_google_chat import build_card, post_card  # noqa: E402


def test_build_card_ok_includes_all_sections(fake_summary_json):
    """Card includes all required sections with traditional Chinese."""
    fake_summary_json["vol_ann"] = 14.6  # NEW field
    card = build_card(fake_summary_json)
    assert card["cardsV2"][0]["card"]["header"]["title"].startswith("✅")
    body_text = json.dumps(card)
    # 7 predictions present
    assert "0.42%" in body_text
    # Vol headline uses vol_ann directly (14.6%)
    assert "14.60%" in body_text
    # Traditional Chinese labels
    assert "未來波動率" in body_text
    assert "本次關鍵預測" in body_text
    assert "方向判斷" in body_text
    # PNGs present
    assert "pred_path.png" in body_text
    # Button present
    assert "View commit" in body_text


def test_build_card_failed_status_uses_error_icon(fake_summary_json):
    fake_summary_json["status"] = "failed"
    fake_summary_json["error"] = "inference exploded"
    fake_summary_json["predictions"] = None
    card = build_card(fake_summary_json)
    assert "❌" in card["cardsV2"][0]["card"]["header"]["title"]
    body = json.dumps(card)
    assert "inference exploded" in body


def test_build_card_excludes_volatility_path_png(fake_summary_json):
    """Step8_Volatility_Path.png must NOT appear in the card (per user request)."""
    fake_summary_json["png_files"] = [
        {"name": "Step8_Proportional_Price_Path.png", "url": "https://example.com/price.png"},
        {"name": "Step8_Volatility_Path.png", "url": "https://example.com/vol.png"},
    ]
    card = build_card(fake_summary_json)
    body = json.dumps(card)
    assert "price.png" in body
    assert "vol.png" not in body, "Volatility_Path should be excluded"


def test_build_card_no_vol_no_crash(fake_summary_json):
    """If vol_ann is missing, vol section is skipped (not crashed)."""
    fake_summary_json.pop("vol_ann", None)
    card = build_card(fake_summary_json)
    # Should not raise; vol section simply absent
    body = json.dumps(card)
    assert "未來波動率" not in body


def test_build_card_with_llm_summary_renders_golden_section(fake_summary_json):
    fake_summary_json["llm_summary"] = "短期承壓，中長期向好。"
    card = build_card(fake_summary_json)
    sections = card["cardsV2"][0]["card"]["sections"]
    first = sections[0]["widgets"][0]["textParagraph"]["text"]
    assert "#FFD700" in first
    assert "AI 總結" in first


@responses.activate
def test_post_card_returns_status_code(fake_summary_json):
    responses.add(
        responses.POST,
        "https://chat.googleapis.com/v1/spaces/AAA/messages",
        json={"name": "spaces/AAA/messages/1"},
        status=200,
    )
    card = build_card(fake_summary_json)
    code = post_card(card, "https://chat.googleapis.com/v1/spaces/AAA/messages")
    assert code == 200
    sent = responses.calls[0].request
    body = json.loads(sent.body)
    assert "cardsV2" in body
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest tests/test_push_to_google_chat.py -v`
Expected: 7 tests PASS (some existing ones may need adjustment, but core should pass)

- [ ] **Step 4: Commit**

```bash
git add scripts/push_to_google_chat.py tests/test_push_to_google_chat.py
git commit -m "feat(chat): card redesign - vol headline (T+1 position, big, traditional Chinese)

- Move 未來波動率 to position 1 (after AI summary)
- Use vol_ann from LSTM vol_head directly (no post-processing)
- font 6 (28px) for the vol number
- Remove all (T+30 日級) parentheses labels
- Remove calculation annotation line
- All user-facing Chinese: 繁體中文 (未來/關鍵/預測/方向/判斷/波動率/命中)
- Exclude Step8_Volatility_Path.png from push"
```

---

### Task 6: Initialize predictions_history.csv with backfill

**Files:**
- Create: `model_artifacts/HSI/predictions_history.csv` (LFS-tracked)
- Test: `tests/test_accuracy.py` (verify backfill function)

- [ ] **Step 1: Write failing test for backfill function**

Add to `tests/test_accuracy.py`:

```python
def test_backfill_from_existing_run(tmp_path):
    """If a run dir has all the artifacts, extract one history row from it."""
    from scripts.accuracy import backfill_from_run
    # Create fake run dir
    run = tmp_path / "model_artifacts" / "HSI" / "20260721" / "run_154008"
    run.mkdir(parents=True)
    (run / "Proportional_Inference_Report.txt").write_text(
        "Current T+0: 25000.00\n"
        "T+1 | 2026-08-22 | 24000 ( -4%) | 25100 ( +0.40%) | 26000 ( +4%) | 0.92%\n",
        encoding="utf-8",
    )
    # Pattern attention CSV
    import csv as _csv
    pat_csv = run / "pattern_attention_full_report.csv"
    with pat_csv.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["pattern", "avg_attention"])
        w.writeheader()
        w.writerow({"pattern": "MA5_Cross", "avg_attention": "0.08"})
        w.writerow({"pattern": "Vol_GK", "avg_attention": "0.05"})
    row = backfill_from_run(run, "2026-07-21")
    assert row["prediction_date"] == "2026-07-21"
    assert row["ticker"] == "^HSI"
    assert row["vol_ann"] == "14.6"  # 0.92 × √252
    assert row["top1_pattern"] == "MA5_Cross"
    assert row["top1_weight"] == "0.08"
    assert row["direction"] == "up"  # T+1 base return +0.40% > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_accuracy.py::test_backfill_from_existing_run -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement backfill_from_run**

Add to `scripts/accuracy.py`:

```python
def _sign(x: float) -> str:
    if x > 0:
        return "up"
    if x < 0:
        return "down"
    return "neutral"


def backfill_from_run(run_dir: Path, prediction_date: str) -> dict:
    """Build a single history row from a completed run dir's artifacts.

    Args:
        run_dir: Path like model_artifacts/HSI/20260721/run_154008
        prediction_date: ISO date string for when the prediction was made

    Returns: dict ready to feed to append_to_history.
    """
    from scripts.collect_run_artifacts import build_summary
    summary = build_summary(
        run_dir=run_dir,
        root=run_dir.parents[2],  # model_artifacts/HSI
        ticker="^HSI",
        commit_sha=None,
    )
    row: dict[str, Any] = {
        "prediction_date": prediction_date,
        "ticker": "HSI",  # use clean ticker in CSV
        "vol_ann": summary.get("vol_ann"),
        "direction": summary.get("direction"),
    }
    # Add all 7 horizons
    preds = summary.get("predictions") or {}
    for h in [1, 5, 10, 15, 20, 25, 30]:
        row[f"T+{h}_pred"] = preds.get(f"{h}d")
        # Actual is filled in by update_actuals() later
        row[f"T+{h}_actual"] = ""
        row[f"T+{h}_correct"] = ""
    # Top 10 patterns
    patt = summary.get("pattern_attention") or {}
    sorted_pat = sorted(patt.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (name, weight) in enumerate(sorted_pat, 1):
        row[f"top{i}_pattern"] = name
        row[f"top{i}_weight"] = weight
    return row
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_accuracy.py -v`
Expected: 11 tests PASS

- [ ] **Step 5: Run backfill from existing local run**

```bash
cd ~/Desktop/slashiee/恒生指数预测/LSTM_ipython-main
python3 -c "
from pathlib import Path
from scripts.accuracy import backfill_from_run, append_to_history
run = Path('model_artifacts/HSI/20260721/run_154008')
if run.exists():
    row = backfill_from_run(run, '2026-07-21')
    csv_path = Path('model_artifacts/HSI/predictions_history.csv')
    append_to_history(csv_path, row)
    print(f'Backfilled row: {row[\"prediction_date\"]}, vol_ann={row[\"vol_ann\"]}')
else:
    print(f'Run dir not found: {run}')
"
```

- [ ] **Step 6: Commit (CSV is LFS-tracked)**

```bash
git lfs track "model_artifacts/HSI/predictions_history.csv"
echo "model_artifacts/HSI/predictions_history.csv" >> .gitattributes  # already covered by existing model_artifacts/** rule
git add scripts/accuracy.py tests/test_accuracy.py model_artifacts/HSI/predictions_history.csv .gitattributes
git commit -m "feat(accuracy): backfill from existing run + initial CSV row"
```

---

### Task 7: Workflow - add accuracy step + enable GitHub Pages

**Files:**
- Modify: `.github/workflows/daily-morning-push.yml`
- Test: `tests/test_accuracy.py` (one CLI-level test)

- [ ] **Step 1: Add the accuracy step to the workflow**

Edit `.github/workflows/daily-morning-push.yml` — after the "Build summary JSON" step, add:

```yaml
      - name: Compute accuracy and emit page
        id: accuracy
        env:
          GH_PAGES_REPO: ${{ github.repository }}
        run: |
          set -e
          python scripts/accuracy_run.py \
            --summary summary.json \
            --csv model_artifacts/HSI/predictions_history.csv \
            --ticker "^HSI" \
            --data-out docs/accuracy_data.json \
            --html-out docs/accuracy.html
        continue-on-error: true
```

Also change the existing "Commit latest artifacts" step to include `docs/` and the new CSV:

```yaml
      - name: Commit latest artifacts
        id: commit
        continue-on-error: true
        run: |
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git config user.name "github-actions[bot]"
          git add -f model_artifacts/latest/${TICKER_DIR}/
          git add -f model_artifacts/HSI/predictions_history.csv || true
          git add -f docs/accuracy.html docs/accuracy_data.json || true
          if git diff --staged --quiet; then
            echo "No changes to commit."
            echo "committed=false" >> "$GITHUB_OUTPUT"
          else
            git commit -m "[Auto] Morning push $(date -u +%Y-%m-%d) ${TICKER}"
            git push
            echo "committed=true" >> "$GITHUB_OUTPUT"
          fi
```

- [ ] **Step 2: Create the workflow driver script**

Create `scripts/accuracy_run.py`:

```python
#!/usr/bin/env python3
"""Driver: take summary.json, append to history CSV, regenerate accuracy page."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.accuracy import (
    append_to_history,
    backfill_from_run,
    build_accuracy_data,
    render_html,
    update_actuals,
)


def _read_csv_rows(csv_path: Path) -> list[dict]:
    """Read all existing rows from predictions_history.csv."""
    if not csv_path.exists():
        return []
    import csv
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
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
    for h in [1, 5, 10, 15, 20, 25, 30]:
        preds = summary.get("predictions") or {}
        row[f"T+{h}_pred"] = preds.get(f"{h}d")
        row[f"T+{h}_actual"] = ""
        row[f"T+{h}_correct"] = ""
    patt = summary.get("pattern_attention") or {}
    for i, (name, weight) in enumerate(
        sorted(patt.items(), key=lambda x: x[1], reverse=True)[:10], 1
    ):
        row[f"top{i}_pattern"] = name
        row[f"top{i}_weight"] = weight
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
    args.data_out.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.html_out.write_text(render_html(data), encoding="utf-8")
    print(f"Wrote {args.data_out} and {args.html_out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add update_actuals to accuracy.py**

Add to `scripts/accuracy.py`:

```python
def update_actuals(
    csv_path: Path, rows: list[dict], ticker: str = "^HSI"
) -> int:
    """For each historical row, fill in actuals up to today (skip future).

    Returns: count of rows updated.
    """
    today = date.today()
    # Find min prediction_date and max target date (T+30 from latest)
    if not rows:
        return 0
    pred_dates = []
    for r in rows:
        try:
            pred_dates.append(date.fromisoformat(r["prediction_date"]))
        except (ValueError, KeyError):
            pass
    if not pred_dates:
        return 0
    earliest = min(pred_dates)
    # Pull actuals from earliest prediction_date to today + 30 (covers T+30 lag)
    end = today + timedelta(days=30)
    actuals = fetch_actuals(ticker, earliest.isoformat(), end.isoformat())
    if not actuals:
        return 0

    updated = 0
    for r in rows:
        try:
            pd = date.fromisoformat(r["prediction_date"])
        except (ValueError, KeyError):
            continue
        for h in [1, 5, 10, 15, 20, 25, 30]:
            td = pd + timedelta(days=h)
            if td > today:
                continue
            actual = actuals.get(td.isoformat())
            if actual is None:
                continue
            pred = _row_pred_for_horizon(r, f"{h}d")
            if pred is None or pred == 0:
                continue
            r[f"T+{h}_actual"] = f"{actual:.6f}"
            r[f"T+{h}_correct"] = "true" if (pred > 0) == (actual > 0) else "false"
            updated += 1

    # Rewrite the CSV
    if updated:
        import csv as _csv
        from scripts.accuracy import CSV_FIELDS
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    return updated
```

- [ ] **Step 4: Run accuracy locally to test**

```bash
cd ~/Desktop/slashiee/恒生指数预测/LSTM_ipython-main
python scripts/accuracy_run.py \
  --summary summary.json \
  --csv model_artifacts/HSI/predictions_history.csv \
  --ticker "HSI" \
  --data-out docs/accuracy_data.json \
  --html-out docs/accuracy.html
ls -la docs/accuracy*.html docs/accuracy_data.json
```

- [ ] **Step 5: Verify HTML was generated**

```bash
head -20 docs/accuracy.html
```

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/daily-morning-push.yml scripts/accuracy_run.py scripts/accuracy.py docs/accuracy.html docs/accuracy_data.json
git commit -m "feat(workflow): add accuracy step + generate GitHub Pages page"
```

- [ ] **Step 7: Manual step: enable GitHub Pages in repo settings**

This is a one-time setup the user must do in the GitHub web UI:
1. Go to https://github.com/Dongdong-2026/Automation/settings/pages
2. Source: "Deploy from a branch"
3. Branch: `main`, folder: `/docs`
4. Save

This enables https://dongdong-1026.github.io/Automation/accuracy.html

---

### Task 8: Integration smoke test

**Files:** none modified; verification only

- [ ] **Step 1: Run full pytest suite**

Run: `cd ~/Desktop/slashiee/恒生指数预测/LSTM_ipython-main && pytest -v`
Expected: 20+ tests PASS (all from previous tasks)

- [ ] **Step 2: Trigger workflow once**

In the browser, go to https://github.com/Dongdong-2026/Automation/actions/workflows/daily-morning-push.yml and click "Run workflow".

- [ ] **Step 3: Verify Chat card content**

Check Chat space for:
- ✅ "📊 未來波動率" as the first content (after AI summary)
- ✅ Big "14.60%" (or whatever the new vol_ann is)
- ✅ No calculation line
- ✅ No Volatility_Path.png image
- ✅ Traditional Chinese throughout

- [ ] **Step 4: Verify GitHub Pages renders**

After GitHub Pages deploys (~2 min), visit:
https://dongdong-1026.github.io/Automation/accuracy.html

Should show 4 sections with SVG charts.

- [ ] **Step 5: Final commit (if any tweaks needed)**

If HTML or card content needs polish, fix and commit.
