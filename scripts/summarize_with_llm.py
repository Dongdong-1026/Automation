#!/usr/bin/env python3
"""Generate a 1-2 sentence Chinese summary of HSI predictions using an LLM.

Reads the summary JSON, calls an Anthropic-Messages-API-compatible LLM
(works with Anthropic, MiniMax-M3, etc.), and writes the summary back
into the same JSON under the field 'llm_summary'.

Failure modes:
  - Missing LLM_API_BASE / LLM_API_KEY: exit 0, no field added
  - Network / API error: exit 0, no field added (workflow continues,
    Chat card just will not have the LLM summary)

The script never raises. It prints an ERROR to stderr but returns 0 so
the workflow pipeline is not broken by an LLM outage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_MAX_TOKENS = 200
DEFAULT_TIMEOUT = 30

SYSTEM_PROMPT = (
    "You are a senior financial analyst. Based on the provided Hang Seng "
    "Index (HSI) multi-horizon prediction data, generate a concise 1-2 "
    "sentence Chinese summary. Requirements: (1) strictly based on data, "
    "no fabricated numbers; (2) include direction judgment and the key "
    "short-term forecast (T+1 or T+5); (3) 30-60 Chinese characters; "
    "(4) declarative sentences, no emoji, no markdown."
)


def build_user_prompt(summary):
    pred = summary.get("predictions") or {}
    vol = summary.get("volatility")
    direction = summary.get("direction")
    parts = ["[HSI multi-horizon prediction snapshot]"]
    parts.append("Direction: " + (direction or "unknown"))
    if pred:
        rows = []
        for k in ("1d", "5d", "10d", "15d", "20d", "25d", "30d"):
            if k in pred and pred[k] is not None:
                rows.append("  " + k + ": " + f"{float(pred[k]) * 100:+.2f}%")
        if rows:
            parts.append("Predicted returns:\n" + "\n".join(rows))
    if vol is not None:
        parts.append("Future volatility: " + f"{float(vol) * 100:.2f}%")
    rev = summary.get("latest_1d_review") or {}
    if rev and rev.get("sample_count"):
        parts.append(
            "Recent 1D hit rate: "
            + f"{float(rev.get('direction_accuracy_pct', 0)):.1f}%"
            + " ("
            + str(int(rev.get("sample_count", 0)))
            + " samples)"
        )
    parts.append("\nPlease generate the 1-2 sentence Chinese summary now.")
    return "\n".join(parts)


def call_anthropic(api_base, api_key, model, system, user, max_tokens):
    url = api_base.rstrip("/") + "/v1/messages"
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data.get("content") or []
    parts = [block.get("text", "") for block in content if block.get("type") == "text"]
    return "".join(parts).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=str,
                        help="Path to summary JSON; will be read AND overwritten in place")
    parser.add_argument("--api-base", default=os.environ.get("LLM_API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "MiniMax-M3"))
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--field", default="llm_summary",
                        help="Field name in JSON to write the summary to")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: overwrite --payload)")
    args = parser.parse_args()

    payload_path = args.payload
    out_path = args.out or payload_path

    with open(payload_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    if not args.api_base or not args.api_key:
        print("WARN: LLM_API_BASE or LLM_API_KEY not set; skipping LLM summary", file=sys.stderr)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False)
        return 0

    user_prompt = build_user_prompt(summary)
    try:
        text = call_anthropic(args.api_base, args.api_key, args.model,
                              SYSTEM_PROMPT, user_prompt, args.max_tokens)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as exc:
        print("WARN: LLM call failed (" + type(exc).__name__ + ": " + str(exc) + "); skipping summary",
              file=sys.stderr)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False)
        return 0

    summary[args.field] = text
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False)
    print("OK: llm summary (" + str(len(text)) + " chars) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
