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
