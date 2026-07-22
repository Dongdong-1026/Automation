#!/usr/bin/env python3
"""Collect summary JSON for one INFERENCE run.

Reads the most recent run directory under model_artifacts/, extracts the
prediction summary, the 1D review snapshot, and the list of PNG figures,
then emits a single JSON object on stdout. Designed to never raise —
failures are encoded in the `status` field so the downstream Chat
push step can still send a status message.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

PNG_SUFFIX = ".png"
PREDICTION_FILE = "prediction_summary.json"
PREDICTION_REPORT_FILE = "Proportional_Inference_Report.txt"
REVIEW_CSV = "Step8_Review_Daily_1D.csv"
PREDICTION_HORIZONS = ["1d", "5d", "10d", "15d", "20d", "25d", "30d"]


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_prediction(run_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read the predictions dict from a run directory.

    Prefer the structured prediction_summary.json (if present, e.g. from TRAIN).
    Otherwise parse the human-readable Proportional_Inference_Report.txt
    that is written by Step 8 in every successful INFERENCE/TRAIN run.
    """
    p = run_dir / PREDICTION_FILE
    if p.exists():
        try:
            with p.open(encoding="utf-8") as f:
                return json.load(f), None
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"failed to parse {PREDICTION_FILE}: {exc}"

    # Fall back: parse Proportional_Inference_Report.txt
    report = run_dir / PREDICTION_REPORT_FILE
    if report.exists():
        parsed = _parse_proportional_report(report)
        if parsed is not None:
            return parsed, None
        return None, f"failed to parse {PREDICTION_REPORT_FILE}"

    return None, f"{PREDICTION_FILE} missing in {run_dir}"


def _parse_proportional_report(report_path: Path) -> dict[str, Any] | None:
    """Parse the Proportional_Inference_Report.txt text file.

    Expected row format:
      T+1      | 2026-07-22   |   24898.42 (  -1.30%) |   25245.44 (  +0.07%) |   25597.31 (  +1.47%) |     0.92%

    Columns: horizon, date, bear(price, ret), base(price, ret), bull(price, ret), vol_pct
    """
    import re
    horizon_pat = re.compile(r"^T\+(\d+)\s*\|")
    # Match signed numbers with optional decimal part (so integer prices like 24500 work too).
    cell_pat = re.compile(r"([+-]?\d+(?:\.\d+)?)")
    try:
        text = report_path.read_text(encoding="utf-8")
    except OSError:
        return None

    horizon_to_ret: dict[str, float] = {}
    horizon_to_vol: dict[str, float] = {}
    for line in text.splitlines():
        m = horizon_pat.match(line)
        if not m:
            continue
        horizon_n = int(m.group(1))
        nums = cell_pat.findall(line)
        # Layout (after the regex change to also match integers):
        #   0=horizon number, 1=year, 2=month, 3=day,
        #   4=bear_price, 5=bear_ret%, 6=base_price, 7=base_ret%,
        #   8=bull_price, 9=bull_ret%, 10=vol%
        if len(nums) < 11:
            continue
        try:
            # nums[7] is the base return in PERCENT, e.g. 0.07 → divide by 100
            base_return = float(nums[7]) / 100.0
            vol = float(nums[10]) / 100.0  # vol also in percent
        except ValueError:
            continue
        horizon_to_ret[f"{horizon_n}d"] = base_return
        horizon_to_vol[f"{horizon_n}d"] = vol

    if not horizon_to_ret:
        return None

    # Annualized vol (T+1 daily vol × sqrt(252) — like VIX). Stored at percent scale
    # so downstream UIs can render "~14.6%" without an extra factor of 100.
    vol_t1_pct = horizon_to_vol.get("1d", 0.0) * 100.0
    vol_ann = vol_t1_pct * math.sqrt(252)  # e.g. 0.92 × 15.87 ≈ 14.6

    # Determine direction by aggregating 1d return sign
    first_ret = next(iter(horizon_to_ret.values()), 0.0)
    if first_ret > 0.005:
        direction = "up"
    elif first_ret < -0.005:
        direction = "down"
    else:
        direction = "neutral"

    return {
        "predictions": horizon_to_ret,
        "volatility": horizon_to_vol.get("1d", 0.0),  # daily vol (T+1 column) is the canonical "future vol"
        "volatility_by_horizon": horizon_to_vol,
        "vol_ann": vol_ann,  # NEW: annualized vol (LSTM vol_head raw, VIX-style)
        "direction": direction,
        "source": str(report_path),
    }


def _list_pngs(run_dir: Path, repo_owner: str | None = None, repo_name: str | None = None, branch: str | None = None) -> list[dict[str, str]]:
    """List PNGs and build LFS-aware URLs.

    URL pattern (when owner/repo/branch all provided):
      https://media.githubusercontent.com/media/{owner}/{repo}/{branch}/model_artifacts/latest/{TICKER_DIR}/{name}

    GitHub LFS objects are NOT served by raw.githubusercontent.com (that URL
    returns the pointer text). media.githubusercontent.com resolves the pointer
    to the actual binary content.
    """
    if not run_dir.exists():
        return []
    files = []
    for p in sorted(run_dir.glob(f"*{PNG_SUFFIX}")):
        entry: dict[str, str] = {"name": p.name}
        if repo_owner and repo_name and branch:
            try:
                ticker_dir = run_dir.parents[1].name
            except IndexError:
                ticker_dir = "HSI"
            entry["url"] = (
                f"https://media.githubusercontent.com/media/{repo_owner}/{repo_name}/"
                f"{branch}/model_artifacts/latest/{ticker_dir}/{p.name}"
            )
        files.append(entry)
    return files


def _read_latest_review(root: Path, ticker: str) -> dict[str, Any] | None:
    csv_path = root / REVIEW_CSV
    if not csv_path.exists():
        return None
    try:
        with csv_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return None
    matching = [r for r in rows if r.get("ticker") == ticker]
    if not matching:
        return None
    last = matching[-1]
    try:
        return {
            "date": last.get("actual_date_used"),
            "sample_count": int(last.get("sample_count") or 0),
            "direction_accuracy_pct": float(last.get("direction_accuracy_pct") or 0.0),
            "pred_avg_return_pct": float(last.get("pred_avg_return_pct") or 0.0),
            "actual_avg_return_pct": float(last.get("actual_avg_return_pct") or 0.0),
        }
    except (TypeError, ValueError):
        return None


def build_summary(
    run_dir: Path,
    root: Path,
    ticker: str,
    commit_sha: str | None,
    repo_owner: str | None = None,
    repo_name: str | None = None,
    branch: str | None = None,
) -> dict[str, Any]:
    """Build the summary JSON dict. Never raises."""
    summary: dict[str, Any] = {
        "schema_version": 1,
        "ticker": ticker,
        "run_dir": str(run_dir),
        "status": "ok",
        "error": None,
        "predictions": None,
        "volatility": None,
        "direction": None,
        "png_files": [],
        "latest_1d_review": None,
        "commit_sha": commit_sha,
        "generated_at": _utc_now_iso(),
    }

    if not run_dir.exists():
        summary["status"] = "failed"
        summary["error"] = f"run_dir does not exist: {run_dir}"
        summary["png_files"] = []
        return summary

    prediction, err = _read_prediction(run_dir)
    if prediction is None:
        summary["status"] = "incomplete"
        summary["error"] = err
    else:
        summary["predictions"] = prediction.get("predictions")
        summary["volatility"] = prediction.get("volatility")
        summary["direction"] = prediction.get("direction")
        # vol_ann may come from prediction_summary.json (TRAIN path) or from
        # _parse_proportional_report (INFERENCE path) — either way, propagate.
        if prediction.get("vol_ann") is not None:
            summary["vol_ann"] = prediction["vol_ann"]

    # Pattern attention (per-day, up to 81 patterns): read pattern_attention_full_report.csv
    # which the notebook writes alongside the other artifacts. Each row has columns
    # pattern, avg_attention, max_attention, min_attention, std_attention.
    pattern_csv = run_dir / "pattern_attention_full_report.csv"
    if pattern_csv.exists():
        try:
            with pattern_csv.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                summary["pattern_attention"] = {
                    (row.get("pattern") or ""): float(row["avg_attention"])
                    for row in reader
                    if row.get("pattern") and row.get("avg_attention") not in (None, "")
                }
        except (OSError, KeyError, ValueError, csv.Error) as exc:
            summary["pattern_attention"] = {}
            summary.setdefault("warnings", []).append(
                f"pattern_attention parse failed: {exc}"
            )
    else:
        summary["pattern_attention"] = {}

    summary["png_files"] = _list_pngs(run_dir, repo_owner, repo_name, branch)

    if summary["status"] == "incomplete" and not summary["png_files"]:
        summary["status"] = "failed"

    summary["latest_1d_review"] = _read_latest_review(root, ticker)
    return summary


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--ticker", required=True)
    p.add_argument("--commit-sha", default=None)
    p.add_argument("--repo-owner", default=os.environ.get("REPO_OWNER", ""))
    p.add_argument("--repo-name", default=os.environ.get("REPO_NAME", ""))
    p.add_argument("--branch", default=os.environ.get("BRANCH", "main"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    summary = build_summary(
        run_dir=args.run_dir,
        root=args.root,
        ticker=args.ticker,
        commit_sha=args.commit_sha,
        repo_owner=args.repo_owner or None,
        repo_name=args.repo_name or None,
        branch=args.branch or None,
    )
    json.dump(summary, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
