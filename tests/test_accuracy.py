"""Tests for scripts.accuracy."""
from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path

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
    # accuracy = 2/4 = 0.5 (excluding the 0 case; 4 comparable horizons,
    # 2 sign matches, 2 sign mismatches)
    result = compute_direction_accuracy(preds, actuals)
    assert result == pytest.approx(0.5)


def test_compute_direction_accuracy_handles_nan():
    """NaN actuals (unrealized) must be excluded from accuracy calculation."""
    from scripts.accuracy import compute_direction_accuracy
    preds = [0.001, 0.002, -0.003]
    actuals = [0.005, math.nan, -0.001]
    # First: 0.001 vs 0.005 → correct
    # Second: excluded (future/unrealized)
    # Third: -0.003 vs -0.001 → correct
    # accuracy = 2/2
    result = compute_direction_accuracy(preds, actuals)
    assert result == pytest.approx(1.0)


def test_fetch_actuals_returns_close_prices(monkeypatch):
    """fetch_actuals returns {date: close} from yfinance for non-holiday dates."""
    from scripts import accuracy

    # Mock yfinance with a fake ticker
    class FakeTicker:
        def history(self, start, end, **kwargs):
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


# --------------------------------------------------------------------------
# Hardened direction-accuracy tests (reviewer findings)
# --------------------------------------------------------------------------

def test_compute_direction_accuracy_length_mismatch_returns_none():
    """Length mismatch must NEVER raise — return None per 'never raise' contract."""
    from scripts.accuracy import compute_direction_accuracy
    # No exception expected; returns None
    assert compute_direction_accuracy([0.001, 0.002], [0.005]) is None
    assert compute_direction_accuracy([0.001], [0.005, -0.001]) is None
    assert compute_direction_accuracy([], []) is None


def test_compute_direction_accuracy_handles_invalid_preds():
    """None/NaN/nonnumeric predictions must be skipped, not raise."""
    from scripts.accuracy import compute_direction_accuracy
    preds = [0.001, None, math.nan, "bogus", 0.003]
    actuals = [0.005, 0.006, 0.007, 0.008, -0.005]
    # Index 0: +0.001 vs +0.005 -> correct
    # Index 1: None pred -> skipped
    # Index 2: NaN pred  -> skipped
    # Index 3: nonnumeric pred -> skipped
    # Index 4: +0.003 vs -0.005 -> wrong
    # 1 correct / 2 total = 0.5
    result = compute_direction_accuracy(preds, actuals)
    assert result == pytest.approx(0.5)


def test_compute_direction_accuracy_handles_np_float64_nan():
    """np.float64 NaN must be detected as NaN (not just builtin float NaN)."""
    import numpy as np
    from scripts.accuracy import compute_direction_accuracy
    preds = [0.001, np.float64(0.002), np.float64(0.003)]
    actuals = [0.005, np.float64(math.nan), -0.005]
    # Index 0: 0.001 vs 0.005 -> correct
    # Index 1: actual is np.float64 NaN -> skipped
    # Index 2: 0.003 vs -0.005 -> wrong
    # 1 correct / 2 total = 0.5
    result = compute_direction_accuracy(preds, actuals)
    assert result == pytest.approx(0.5)


def test_append_to_history_missing_header_recreates_file(tmp_path: Path):
    """Existing file with malformed/empty header must be overwritten with a proper header."""
    from scripts.accuracy import append_to_history
    csv_path = tmp_path / "predictions_history.csv"
    # Write a headerless / malformed file (just garbage)
    csv_path.write_text("garbage,row,with,bad,header\n", encoding="utf-8")
    row = {
        "prediction_date": date(2026, 7, 22),
        "ticker": "^HSI",
        "T+1_pred": 0.0007,
        "T+1_actual": None,
        "T+1_correct": None,
        "vol_ann": 14.6,
        "direction": "up",
        "top1_pattern": "P1",
        "top1_weight": 0.1,
    }
    append_to_history(csv_path, row)
    # File should have been re-created with the canonical header
    with csv_path.open(encoding="utf-8") as f:
        first_line = f.readline().strip()
    # First column of the canonical header must be the canonical one
    assert first_line.startswith("prediction_date")
    # And the data row should also be there
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["ticker"] == "^HSI"


def test_append_to_history_handles_write_error(tmp_path: Path, monkeypatch):
    """Filesystem errors during append must be swallowed (never raise)."""
    from scripts.accuracy import append_to_history
    csv_path = tmp_path / "predictions_history.csv"

    from pathlib import Path as _Path

    original_open = _Path.open

    def _patched_open(self, *a, **kw):
        if self == csv_path:
            raise OSError("simulated disk full")
        return original_open(self, *a, **kw)

    monkeypatch.setattr(_Path, "open", _patched_open)
    row = {
        "prediction_date": date(2026, 7, 22),
        "ticker": "^HSI",
        "T+1_pred": 0.0007,
        "vol_ann": 14.6,
        "direction": "up",
        "top1_pattern": "P1",
        "top1_weight": 0.1,
    }
    # Must not raise — even though open() raises OSError
    append_to_history(csv_path, row)


# --------------------------------------------------------------------------
# Task 3: monthly aggregation + best-prediction-day finder
# --------------------------------------------------------------------------

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