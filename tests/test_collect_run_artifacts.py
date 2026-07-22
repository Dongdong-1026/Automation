"""Tests for collect_run_artifacts.build_summary."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts/ importable when pytest is run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect_run_artifacts import build_summary  # noqa: E402


def test_build_summary_happy_path(fake_run_dir, fake_artifacts_root):
    summary = build_summary(
        run_dir=fake_run_dir,
        root=fake_artifacts_root,
        ticker="^HSI",
        commit_sha="abc1234",
    )
    assert summary["status"] == "ok"
    assert summary["ticker"] == "^HSI"
    assert summary["predictions"]["1d"] == pytest.approx(0.0042)
    assert summary["volatility"] == pytest.approx(0.0182)
    assert summary["direction"] == "up"
    names = {p["name"] for p in summary["png_files"]}
    assert names == {"pred_path.png", "quantile_band.png"}
    rev = summary["latest_1d_review"]
    assert rev["date"] == "2026-07-18"
    assert rev["direction_accuracy_pct"] == pytest.approx(58.33)


def test_build_summary_missing_run_dir(tmp_path):
    summary = build_summary(
        run_dir=tmp_path / "does_not_exist",
        root=tmp_path,
        ticker="^HSI",
        commit_sha="deadbeef",
    )
    assert summary["status"] == "failed"
    assert summary["error"] is not None
    assert summary["predictions"] is None
    assert summary["png_files"] == []


def test_build_summary_run_dir_without_prediction_json(fake_run_dir, fake_artifacts_root):
    (fake_run_dir / "prediction_summary.json").unlink()
    summary = build_summary(
        run_dir=fake_run_dir,
        root=fake_artifacts_root,
        ticker="^HSI",
        commit_sha="abc1234",
    )
    assert summary["status"] == "incomplete"
    assert summary["predictions"] is None
    # PNGs still detected
    assert {p["name"] for p in summary["png_files"]} == {"pred_path.png", "quantile_band.png"}


def test_build_summary_no_review_csv(fake_run_dir):
    root = fake_run_dir.parents[2]
    summary = build_summary(
        run_dir=fake_run_dir,
        root=root,
        ticker="^HSI",
        commit_sha=None,
    )
    assert summary["latest_1d_review"] is None
    assert summary["status"] == "ok"
    assert summary["commit_sha"] is None

def test_parse_proportional_report_volatility_correct(fake_run_dir, monkeypatch):
    """Regression: volatility field must use actual vol column, not return ×16."""
    from scripts.collect_run_artifacts import _parse_proportional_report
    import json

    # Add a properly formatted Proportional_Inference_Report to the fake run dir
    report = fake_run_dir / "Proportional_Inference_Report.txt"
    report.write_text("\u2501" * 89 + "\n"
        "📌 Current T+0: 25226.77 | Conf: ±1.5σ\n"
        "T+1      | 2026-07-22   |   24898.42 (  -1.30%) |   25245.44 (  +0.07%) |   25597.31 (  +1.47%) |     0.92%\n"
        "T+5      | 2026-07-28   |   24097.72 (  -4.48%) |   24855.22 (  -1.47%) |   25636.53 (  +1.62%) |     2.06%\n"
        "T+30     | 2026-09-01   |   24237.93 (  -3.92%) |   26146.93 (  +3.65%) |   28206.29 ( +11.81%) |     5.05%\n",
        encoding="utf-8")

    parsed = _parse_proportional_report(report)
    assert parsed is not None
    assert abs(parsed["volatility"] - 0.0092) < 0.0001, f"volatility should be 0.0092 (T+1 vol), got {parsed['volatility']}"
    assert abs(parsed["volatility_by_horizon"]["30d"] - 0.0505) < 0.0001
    assert abs(parsed["predictions"]["1d"] - 0.0007) < 0.0001, f"T+1 return should be 0.0007, got {parsed['predictions']['1d']}"
    # Critical regression: T+30 return (3.65%) must NOT appear as volatility
    assert parsed["volatility"] < 0.1, f"volatility {parsed['volatility']} looks wrong (>10%) — old bug returns"


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
    # vol_ann ~= T+1 vol x sqrt(252) ~= 0.92 x 15.87 ~= 14.6
    assert abs(parsed["vol_ann"] - 14.60) < 0.5, f"expected ~14.6, got {parsed['vol_ann']}"


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
