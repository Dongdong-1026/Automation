#!/usr/bin/env python3
"""Assemble a Google Chat cardsV2 message from a summary JSON and POST it.

The webhook URL is read from an env var (default GOOGLE_CHAT_WEBHOOK).
The script never logs the URL.
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

DEFAULT_TIMEOUT = 15  # seconds
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


def _status_emoji(status: str) -> str:
    return {
        "ok": "✅",
        "bootstrapped": "⚠️",
        "incomplete": "⚠️",
        "failed": "❌",
    }.get(status, "ℹ️")


def _html_escape(text: str) -> str:
    """Escape characters that would break the cardsV2 HTML subset."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def _fmt_num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _prediction_rows(predictions: dict[str, float] | None) -> list[str]:
    if not predictions:
        return []
    rows = []
    for horizon in ("1d", "5d", "10d", "15d", "20d", "25d", "30d"):
        if horizon in predictions:
            rows.append(f"• {horizon}: {_fmt_pct(predictions[horizon])}")
    return rows


def _png_widgets(png_files: list[dict[str, str]]) -> list[dict[str, Any]]:
    widgets: list[dict[str, Any]] = []
    for p in png_files or []:
        widgets.append({
            "image": {"imageUrl": p.get("url", "")},
        })
        widgets.append({
            "textParagraph": {
                "text": f"<a href=\"{p.get('url', '')}\">{p.get('name', 'image')}</a>"
            }
        })
    if not widgets:
        widgets.append({"textParagraph": {"text": "（图片未生成）"}})
    return widgets


def build_card(summary: dict[str, Any]) -> dict[str, Any]:
    """Build a Google Chat cardsV2 payload from a summary dict."""
    status = summary.get("status", "ok")
    emoji = _status_emoji(status)
    ticker = summary.get("ticker", "")
    generated_at = summary.get("generated_at", "")
    title_date = generated_at[:10] if generated_at else ""
    title = f"{emoji} {ticker} 早晨预测 · {title_date} · {status}"

    sections: list[dict[str, Any]] = []

    # Section 0: LLM-generated summary (golden text), if available
    llm_summary = (summary.get("llm_summary") or "").strip()
    if llm_summary:
        sections.append({
            "widgets": [{
                "textParagraph": {
                    "text": (
                        "<font color=\"#FFD700\"><b>✨ AI 总结</b></font><br>"
                        f"<font color=\"#FFD700\">{_html_escape(llm_summary)}</font>"
                    )
                }
            }]
        })

    # Section 1: key numerics (or error)
    widgets: list[dict[str, Any]] = []
    if status in ("failed", "incomplete"):
        widgets.append({
            "textParagraph": {
                "text": f"<b>错误：</b> {summary.get('error') or '未知'}"
            }
        })
    else:
        rows = _prediction_rows(summary.get("predictions"))
        if rows:
            widgets.append({
                "textParagraph": {"text": "<b>本次关键预测</b><br>" + "<br>".join(rows)}
            })
        vol = summary.get("volatility")
        vol_by_horizon = summary.get("volatility_by_horizon") or {}
        vol_t1 = vol_by_horizon.get("1d", vol)  # daily (T+1) vol, falls back to top-level field
        vol_t30 = vol_by_horizon.get("30d")     # 30-day cumulative vol
        direction = summary.get("direction")
        # Always label the headline with T+30 to match the user's mental model
        # ("future volatility" implies the 1-month horizon, not 1-day).
        # Falls back through T+30 → T+1 → top-level field if needed.
        headline_horizon = "T+30 日级"
        if vol_t30 is not None:
            headline_vol = vol_t30
        elif vol_t1 is not None:
            headline_vol = vol_t1
        else:
            headline_vol = vol
        vol_lines = [
            f"<b>未来波动率（{headline_horizon}）：</b> {_fmt_pct(headline_vol)}<br>"
            f"<b>方向判断：</b> {direction or '—'}"
        ]
        # Show calculation method as a small annotation
        if vol_t1 is not None and vol_t30 is not None and vol_t30 != vol_t1:
            ratio = vol_t30 / vol_t1 if vol_t1 else 0
            vol_lines.append(
                f"<font color=\"#888888\">└ 计算：LSTM vol_head → 日级 σ × √30 ≈ {_fmt_pct(vol_t1)} × {ratio:.2f} ≈ {_fmt_pct(vol_t30)}</font>"
            )
        elif vol_t1 is not None:
            vol_lines.append(
                f"<font color=\"#888888\">└ 计算：LSTM vol_head 直接输出</font>"
            )
        widgets.append({
            "textParagraph": {"text": "<br>".join(vol_lines)}
        })
    sections.append({"widgets": widgets})

    # Section 2: 1D review
    rev = summary.get("latest_1d_review")
    if rev:
        sections.append({
            "widgets": [{
                "textParagraph": {
                    "text": (
                        "<b>最近 1D 命中摘要</b><br>"
                        f"日期：{rev.get('date')}<br>"
                        f"样本：{rev.get('sample_count')}<br>"
                        f"方向命中率：{rev.get('direction_accuracy_pct', 0):.2f}%<br>"
                        f"预测均值：{_fmt_pct((rev.get('pred_avg_return_pct') or 0) / 100, 4)}<br>"
                        f"实际均值：{_fmt_pct((rev.get('actual_avg_return_pct') or 0) / 100, 4)}"
                    )
                }
            }]
        })

    # Section 3: PNGs
    sections.append({
        "header": "预测图",
        "widgets": _png_widgets(summary.get("png_files") or []),
    })

    # Section 4: Footer / actions
    sha = summary.get("commit_sha") or ""
    actions = []
    if sha and sha != "none":
        # cardsV2 button: 'text' at top level, 'openLink.url' nested inside 'onClick'
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
        "textParagraph": {
            "text": "由 daily-morning-push 自动生成"
        }
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
