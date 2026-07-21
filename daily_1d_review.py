#!/usr/bin/env python3
"""
Run Step8 review and print latest matured 1D direction summary.

Usage:
  python daily_1d_review.py --root model_artifacts --ticker ^HSI
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Print latest 1D review summary")
    parser.add_argument("--root", type=str, default="model_artifacts", help="Root artifacts folder")
    parser.add_argument("--ticker", type=str, default=None, help="Ticker filter, e.g. ^HSI")
    args = parser.parse_args()

    root = Path(args.root)
    step8_script = Path(__file__).with_name("step8_review.py")

    cmd = [sys.executable, str(step8_script), "--root", str(root)]
    if args.ticker:
        cmd += ["--ticker", args.ticker]

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        return proc.returncode

    daily_path = root / "Step8_Review_Daily_1D.csv"
    if not daily_path.exists():
        print("ℹ️ 尚未生成 Step8_Review_Daily_1D.csv（可能暂无已到期1D样本）")
        return 0

    df = pd.read_csv(daily_path)
    if df.empty:
        print("ℹ️ Step8_Review_Daily_1D.csv 为空（暂无已到期1D样本）")
        return 0

    print("\n📌 最近已收盘交易日（1D）方向命中摘要")
    for _, r in df.iterrows():
        print(
            f"- 交易日: {r.get('actual_date_used')} | 标的: {r.get('ticker')} | "
            f"样本: {int(r.get('sample_count', 0))} | "
            f"方向命中率: {float(r.get('direction_accuracy_pct', 0.0)):.2f}% | "
            f"预测均值: {float(r.get('pred_avg_return_pct', 0.0)):.4f}% | "
            f"实际均值: {float(r.get('actual_avg_return_pct', 0.0)):.4f}%"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
