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
import sys
from pathlib import Path
from typing import Any

PNG_SUFFIX = ".png"
PREDICTION_FILE = "prediction_summary.json"
REVIEW_CSV = "Step8_Review_Daily_1D.csv"


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_prediction(run_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    p = run_dir / PREDICTION_FILE
    if not p.exists():
        return None, f"{PREDICTION_FILE} missing in {run_dir}"
    try:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"failed to parse {PREDICTION_FILE}: {exc}"
    return data, None


def _list_pngs(run_dir: Path, repo_owner: str | None = None, repo_name: str | None = None, branch: str | None = None) -> list[dict[str, str]]:
    """List PNGs and build raw.githubusercontent.com URLs.

    URL pattern (when owner/repo/branch all provided):
      https://raw.githubusercontent.com/{owner}/{repo}/{branch}/model_artifacts/latest/{TICKER_DIR}/{name}
    This matches the path used by the workflow's "Commit latest artifacts" step.
    """
    if not run_dir.exists():
        return []
    files = []
    for p in sorted(run_dir.glob(f"*{PNG_SUFFIX}")):
        entry: dict[str, str] = {"name": p.name}
        if repo_owner and repo_name and branch:
            # The PNGs get copied into model_artifacts/latest/{TICKER_DIR}/ by the workflow
            # We infer TICKER_DIR from the run path: .../model_artifacts/{TICKER_DIR}/{date}/run_*/
            try:
                ticker_dir = run_dir.parents[1].name
            except IndexError:
                ticker_dir = "HSI"
            entry["url"] = (
                f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/"
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
