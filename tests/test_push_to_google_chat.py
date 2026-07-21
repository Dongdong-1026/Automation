"""Tests for push_to_google_chat."""

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
    assert "1.82%" in body_text  # volatility
    assert "58.33" in body_text  # direction_accuracy_pct
    assert "pred_path.png" in body_text
    assert "View commit" in body_text


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
