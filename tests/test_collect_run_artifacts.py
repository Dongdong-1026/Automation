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
