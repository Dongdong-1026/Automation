"""Shared pytest fixtures for HSI Morning Push tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def fake_run_dir(tmp_path: Path) -> Path:
    """Create a fake model_artifacts run directory with representative files."""
    run = tmp_path / "model_artifacts" / "HSI" / "2026-07-21" / "run_0800"
    run.mkdir(parents=True)

    # Fake prediction JSON
    (run / "prediction_summary.json").write_text(
        json.dumps(
            {
                "predictions": {
                    "1d": 0.0042,
                    "5d": 0.0118,
                    "10d": 0.0203,
                    "15d": 0.0276,
                    "20d": 0.0321,
                    "25d": 0.0355,
                    "30d": 0.0389,
                },
                "volatility": 0.0182,
                "direction": "up",
            }
        ),
        encoding="utf-8",
    )

    # Fake PNGs (minimal valid PNG bytes)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4"
        b"\x89\x00\x00\x00\rIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
        b"\x86\x1b\xb6\xee\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    for name in ("pred_path.png", "quantile_band.png"):
        (run / name).write_bytes(png_bytes)

    return run


@pytest.fixture
def fake_artifacts_root(fake_run_dir: Path) -> Path:
    """Return the artifacts root (model_artifacts/) containing a fake 1D review CSV."""
    root = fake_run_dir.parents[2]  # fake_run_dir / ../../..
    csv_path = root / "Step8_Review_Daily_1D.csv"
    csv_path.write_text(
        "actual_date_used,ticker,sample_count,direction_accuracy_pct,"
        "pred_avg_return_pct,actual_avg_return_pct\n"
        "2026-07-18,^HSI,12,58.33,0.21,0.18\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def fake_summary_json() -> dict:
    """A canonical summary JSON used by push_to_google_chat tests."""
    return {
        "schema_version": 1,
        "ticker": "^HSI",
        "run_dir": "model_artifacts/HSI/2026-07-21/run_0800",
        "status": "ok",
        "error": None,
        "predictions": {
            "1d": 0.0042,
            "5d": 0.0118,
            "10d": 0.0203,
            "15d": 0.0276,
            "20d": 0.0321,
            "25d": 0.0355,
            "30d": 0.0389,
        },
        "volatility": 0.0182,
        "direction": "up",
        "png_files": [
            {"name": "pred_path.png", "url": "https://example.com/pred_path.png"},
            {"name": "quantile_band.png", "url": "https://example.com/quantile_band.png"},
        ],
        "latest_1d_review": {
            "date": "2026-07-18",
            "sample_count": 12,
            "direction_accuracy_pct": 58.33,
            "pred_avg_return_pct": 0.21,
            "actual_avg_return_pct": 0.18,
        },
        "commit_sha": "abc1234",
        "generated_at": "2026-07-21T08:01:42Z",
    }
