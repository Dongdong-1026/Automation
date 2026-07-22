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