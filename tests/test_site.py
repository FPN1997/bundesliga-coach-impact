"""
Tests for src/site.py's page assembly. The data it embeds includes scraped
strings (team and coach names), so the one thing that must never happen is
a value breaking out of the <script> block it's embedded in.
"""

from __future__ import annotations

import json
import re

from src import site


def test_template_has_exactly_one_data_placeholder():
    assert site.TEMPLATE.read_text().count("/*__DATA__*/null") == 1


def test_embedded_data_cannot_close_the_script_tag(monkeypatch, tmp_path):
    hostile = "</script><script>alert(1)</script>"
    payload = {"meta": {"data_through": "2026-09-20"}, "next_matchday": {"fixtures": []},
               "coach_effect": {"sackings": [{"coach_in": hostile}]}}
    monkeypatch.setattr(site, "collect", lambda: payload)
    monkeypatch.setattr(site, "SITE_DIR", tmp_path)
    monkeypatch.setattr(site, "preview_text", lambda data: "")
    monkeypatch.setattr(site, "draw_preview_image", lambda data, path: None)

    html = site.build_site().read_text()

    assert hostile not in html
    # exactly one <script> ... </script> pair in the page: the data didn't add any
    assert len(re.findall(r"</script>", html, flags=re.I)) == 1
    # and the escaped JSON still decodes to the original value
    embedded = html.split("const DATA = ", 1)[1].split(";\nconst NS", 1)[0]
    assert json.loads(embedded)["coach_effect"]["sackings"][0]["coach_in"] == hostile
    assert (tmp_path / ".nojekyll").exists()


def _preview_data():
    matches = [{"match": k, "actual": 1.0 + 0.1 * (k >= 0), "expected": 1.0,
                "actual_ci95": [0.8, 1.3]} for k in range(-8, 8)]
    return {"meta": {"seasons": ["2014-2015", "2026-2027"]},
            "coach_effect": {"results": {
                "mid_season": {"ppg": {"raw_change": 0.54, "counterfactual_change": 0.35,
                                       "effect": 0.19, "n_treated": 65}},
                "event_study": {"n_changes": 65, "matches": matches}}}}


def test_preview_text_uses_the_current_numbers_with_real_minus_signs():
    data = _preview_data()
    data["coach_effect"]["results"]["mid_season"]["ppg"]["effect"] = -0.05
    text = site.preview_text(data)
    assert "+0.54" in text and "+0.35" in text and "−0.05" in text


def test_share_card_is_the_size_link_previews_expect(tmp_path):
    from PIL import Image
    site.draw_preview_image(_preview_data(), tmp_path / "og.png")
    assert Image.open(tmp_path / "og.png").size == (1200, 630)


def test_page_has_absolute_preview_urls_and_escaped_description(monkeypatch, tmp_path):
    data = {**_preview_data(), "next_matchday": {"fixtures": []}}
    data["meta"]["data_through"] = "2026-09-20"
    monkeypatch.setattr(site, "collect", lambda: data)
    monkeypatch.setattr(site, "SITE_DIR", tmp_path)
    monkeypatch.setattr(site, "preview_text", lambda d: 'a "quoted" <b>')
    page = site.build_site().read_text()
    assert f'content="{site.SITE_URL}og.png"' in page
    assert 'content="a &quot;quoted&quot; &lt;b&gt;"' in page
    assert "__OG_DESCRIPTION__" not in page and "__SITE_URL__" not in page
    assert (tmp_path / "og.png").exists()


def _meter(n_clubs: int = 12) -> dict:
    clubs = [{"team": f"Club {i}", "coach": f"Coach {i}", "season_matches": 6,
              "sack_risk": 0.3 / (i + 1)} for i in range(n_clubs)]
    clubs.insert(0, {"team": "Changed FC", "coach": "New Coach", "season_matches": 6,
                     "changed_since_last_match": True, "coach_since": "2026-09-22"})
    return {"clubs": clubs, "risk_model": {"base_rate": 0.066}, "risk_horizon": 4,
            "data_through": "2026-09-20"}


def test_meter_share_image_is_16_by_9(tmp_path):
    from PIL import Image
    site.draw_meter_image(_meter(), tmp_path / "m.png")
    assert Image.open(tmp_path / "m.png").size == (1200, 675)


def test_page_build_draws_the_meter_image_only_when_there_is_a_meter(monkeypatch, tmp_path):
    data = {**_preview_data(), "next_matchday": {"fixtures": []}, "sack_o_meter": None}
    data["meta"]["data_through"] = "2026-09-20"
    monkeypatch.setattr(site, "collect", lambda: data)
    monkeypatch.setattr(site, "SITE_DIR", tmp_path)
    site.build_site()
    assert not (tmp_path / site.METER_IMAGE).exists()
    data["sack_o_meter"] = _meter()
    site.build_site()
    assert (tmp_path / site.METER_IMAGE).exists()
