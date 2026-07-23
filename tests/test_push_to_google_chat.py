"""Tests for push_to_google_chat.build_card."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import responses

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect_run_artifacts import build_summary  # noqa: E402
from scripts.push_to_google_chat import build_card, post_card  # noqa: E402


def test_build_card_ok_includes_all_sections(fake_summary_json):
    """Card includes all required sections with traditional Chinese."""
    fake_summary_json["vol_ann"] = 14.6  # NEW field
    card = build_card(fake_summary_json)
    assert card["cardsV2"][0]["card"]["header"]["title"].startswith("✅")
    body_text = json.dumps(card, ensure_ascii=False)
    # 7 predictions present
    assert "0.42%" in body_text
    # Vol headline uses vol_ann directly (14.6%)
    assert "14.60%" in body_text
    # Traditional Chinese labels
    assert "未來波動率" in body_text
    assert "本次關鍵預測" in body_text
    assert "方向判斷" in body_text
    # PNGs present
    assert "pred_path.png" in body_text
    # Button present
    assert "View commit" in body_text


def test_build_card_failed_status_uses_error_icon(fake_summary_json):
    fake_summary_json["status"] = "failed"
    fake_summary_json["error"] = "inference exploded"
    fake_summary_json["predictions"] = None
    card = build_card(fake_summary_json)
    assert "❌" in card["cardsV2"][0]["card"]["header"]["title"]
    body = json.dumps(card)
    assert "inference exploded" in body


def test_build_card_excludes_volatility_path_png(fake_summary_json):
    """Step8_Volatility_Path.png must NOT appear in the card (per user request)."""
    fake_summary_json["png_files"] = [
        {"name": "Step8_Proportional_Price_Path.png", "url": "https://example.com/price.png"},
        {"name": "Step8_Volatility_Path.png", "url": "https://example.com/vol.png"},
    ]
    card = build_card(fake_summary_json)
    body = json.dumps(card)
    assert "price.png" in body
    assert "vol.png" not in body, "Volatility_Path should be excluded"


def test_build_card_no_vol_no_crash(fake_summary_json):
    """If vol_ann is missing, vol section is skipped (not crashed)."""
    fake_summary_json.pop("vol_ann", None)
    card = build_card(fake_summary_json)
    # Should not raise; vol section simply absent
    body = json.dumps(card, ensure_ascii=False)
    assert "未來波動率" not in body


def test_build_card_with_llm_summary_renders_golden_section(fake_summary_json):
    fake_summary_json["llm_summary"] = "短期承壓，中長期向好。"
    card = build_card(fake_summary_json)
    sections = card["cardsV2"][0]["card"]["sections"]
    first = sections[0]["widgets"][0]["textParagraph"]["text"]
    assert "#FFD700" in first
    assert "AI 總結" in first
    assert fake_summary_json["llm_summary"] in first


def test_build_card_no_review_still_renders(fake_summary_json):
    fake_summary_json["latest_1d_review"] = None
    card = build_card(fake_summary_json)
    body = json.dumps(card, ensure_ascii=False)
    assert "0.42%" in body
    assert "最近 1D 命中摘要" not in body


def test_build_card_without_llm_summary_starts_with_vol_headline(fake_summary_json):
    fake_summary_json.pop("llm_summary", None)
    fake_summary_json["vol_ann"] = 14.6
    card = build_card(fake_summary_json)
    sections = card["cardsV2"][0]["card"]["sections"]
    first_text = sections[0]["widgets"][0]["textParagraph"]["text"]
    assert "AI 總結" not in first_text
    assert "未來波動率" in first_text
    assert "14.60%" in first_text


def test_build_card_uses_only_vol_ann_for_headline(fake_summary_json):
    fake_summary_json["vol_ann"] = 14.6
    fake_summary_json["volatility_by_horizon"] = {
        "1d": 0.0092,
        "5d": 0.0206,
        "10d": 0.0292,
        "15d": 0.0357,
        "20d": 0.0413,
        "25d": 0.0461,
        "30d": 0.0505,
    }
    card = build_card(fake_summary_json)
    body = json.dumps(card, ensure_ascii=False)
    assert "14.60%" in body
    assert "T+30" not in body
    assert "LSTM vol_head" not in body


def test_build_card_html_escapes_llm_summary(fake_summary_json):
    fake_summary_json["llm_summary"] = '<script>alert(1)</script> & "quotes"'
    card = build_card(fake_summary_json)
    body = json.dumps(card)
    assert "&lt;script&gt;" in body
    assert "<script>" not in body


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


def test_build_card_handles_vol_ann_as_string(fake_summary_json):
    """Regression: vol_ann may arrive as a JSON string ("14.6").
    build_card must coerce to float — never raise TypeError on /.
    """
    fake_summary_json["vol_ann"] = "14.6"  # string, not float
    card = build_card(fake_summary_json)
    body = json.dumps(card, ensure_ascii=False)
    assert "14.60%" in body, f"expected 14.60%% in card, got: {body[:500]}"


def test_build_summary_coerces_vol_ann_from_prediction_json(fake_run_dir, fake_artifacts_root):
    """Regression: prediction_summary.json may carry vol_ann as a string.
    build_summary must coerce it to float so downstream consumers never trip
    on the str / 100 TypeError.
    """
    import json as _json
    # Rewrite the prediction_summary.json shipped by the fake_run_dir fixture
    # to contain vol_ann as a JSON-encoded string.
    pred_path = fake_run_dir / "prediction_summary.json"
    parsed = _json.loads(pred_path.read_text(encoding="utf-8"))
    parsed["vol_ann"] = "14.6"  # string on disk
    pred_path.write_text(_json.dumps(parsed), encoding="utf-8")

    summary = build_summary(
        run_dir=fake_run_dir,
        root=fake_artifacts_root,
        ticker="^HSI",
        commit_sha="abc1234",
    )
    assert summary["vol_ann"] == pytest.approx(14.6), (
        f"expected summary['vol_ann'] coerced to 14.6 float, got {summary['vol_ann']!r}"
    )
    assert isinstance(summary["vol_ann"], float)
