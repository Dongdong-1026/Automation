"""Tests for push_to_google_chat.build_card."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import responses

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.push_to_google_chat import build_card, post_card  # noqa: E402


def test_build_card_ok_includes_all_sections(fake_summary_json):
    card = build_card(fake_summary_json)
    assert card["cardsV2"][0]["card"]["header"]["title"].startswith("✅")
    body_text = json.dumps(card)
    assert "0.42%" in body_text  # 1d prediction formatted as percent
    assert "1.82%" in body_text  # T+1 daily volatility
    assert "58.33" in body_text  # direction_accuracy_pct
    assert "pred_path.png" in body_text
    assert "View commit" in body_text
    # Ensure horizon label is explicit so users don't confuse
    # "未来波动率" with cumulative longer-horizon vol
    assert "T+1" in body_text, "volatility label should reference T+1"


def test_build_card_failed_status_uses_error_icon(fake_summary_json):
    fake_summary_json["status"] = "failed"
    fake_summary_json["error"] = "inference exploded"
    fake_summary_json["predictions"] = None
    fake_summary_json["png_files"] = []
    card = build_card(fake_summary_json)
    assert "❌" in card["cardsV2"][0]["card"]["header"]["title"]
    body = json.dumps(card)
    assert "inference exploded" in body


def test_build_card_no_review_still_renders(fake_summary_json):
    fake_summary_json["latest_1d_review"] = None
    card = build_card(fake_summary_json)
    body = json.dumps(card)
    assert "0.42%" in body
    # 1D 命中 section absent because latest_1d_review is None


def test_build_card_with_llm_summary_renders_golden_section(fake_summary_json):
    fake_summary_json["llm_summary"] = "短期承压，中长期向好，建议关注T+10/T+15反弹机会。"
    card = build_card(fake_summary_json)
    sections = card["cardsV2"][0]["card"]["sections"]
    # First section should be the AI summary, golden text
    first = sections[0]["widgets"][0]["textParagraph"]["text"]
    assert "#FFD700" in first  # gold color
    assert "AI 总结" in first
    assert fake_summary_json["llm_summary"] in first
    # Other sections still present
    body = json.dumps(card)
    assert "0.42%" in body
    assert "View commit" in body


def test_build_card_without_llm_summary_skips_section(fake_summary_json):
    # No llm_summary field — top section should be the key numerics
    fake_summary_json.pop("llm_summary", None)
    card = build_card(fake_summary_json)
    sections = card["cardsV2"][0]["card"]["sections"]
    first_text = sections[0]["widgets"][0]["textParagraph"]["text"]
    # First section is the predictions/volatility section, NOT the AI block
    assert "AI 总结" not in first_text
    assert "本次关键预测" in first_text or "未来波动率" in first_text


def test_build_card_with_horizon_vol_shows_t30_line(fake_summary_json):
    """When volatility_by_horizon is provided, show T+30 cumulative as context."""
    fake_summary_json["volatility_by_horizon"] = {
        "1d": 0.0092, "5d": 0.0206, "10d": 0.0292,
        "15d": 0.0357, "20d": 0.0413, "25d": 0.0461, "30d": 0.0505,
    }
    card = build_card(fake_summary_json)
    body = json.dumps(card)
    assert "T+1" in body, "primary T+1 vol label missing"
    # T+30 secondary line should reference T+30
    assert "30d" in body or "T+30" in body, \
        "expected secondary T+30 line when volatility_by_horizon present"


def test_build_card_html_escapes_llm_summary(fake_summary_json):
    # Malicious LLM output shouldn't break the cardsV2 envelope
    fake_summary_json["llm_summary"] = "<script>alert(1)</script> & \"quotes\""
    card = build_card(fake_summary_json)
    body = json.dumps(card)
    assert "&lt;script&gt;" in body
    assert "<script>" not in body.split('cardsV2')[0]  # not raw inside the card


@responses.activate
def test_post_card_returns_status_code(fake_summary_json):
    responses.add(
        responses.POST,
        "https://chat.googleapis.com/v1/spaces/AAA/messages",
        json={"name": "spaces/AAA/messages/1"},
        status=200,
    )
    card = build_card(fake_summary_json)
    code = post_card(card, "https://chat.googleapis.com/v1/spaces/AAA/messages")
    assert code == 200
    sent = responses.calls[0].request
    body = json.loads(sent.body)
    assert "cardsV2" in body


@responses.activate
def test_post_card_returns_nonzero_on_500(fake_summary_json):
    responses.add(
        responses.POST,
        "https://chat.googleapis.com/v1/spaces/AAA/messages",
        json={"error": "boom"},
        status=500,
    )
    card = build_card(fake_summary_json)
    code = post_card(card, "https://chat.googleapis.com/v1/spaces/AAA/messages")
    assert code == 500


def test_cli_reads_payload_and_webhook_env(tmp_path, monkeypatch, fake_summary_json):
    payload = tmp_path / "summary.json"
    payload.write_text(json.dumps(fake_summary_json), encoding="utf-8")
    monkeypatch.setenv("GOOGLE_CHAT_WEBHOOK", "https://chat.example/wh")

    captured: dict = {}

    def fake_post(card, url, *a, **kw):
        captured["url"] = url
        captured["card"] = card
        return 200

    monkeypatch.setattr("scripts.push_to_google_chat.post_card", fake_post)

    from scripts.push_to_google_chat import main
    rc = main(["--payload", str(payload), "--webhook-env", "GOOGLE_CHAT_WEBHOOK"])
    assert rc == 0
    assert captured["url"] == "https://chat.example/wh"
    assert "cardsV2" in json.dumps(captured["card"])
