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

    html = site.build_site().read_text()

    assert hostile not in html
    # exactly one <script> ... </script> pair in the page: the data didn't add any
    assert len(re.findall(r"</script>", html, flags=re.I)) == 1
    # and the escaped JSON still decodes to the original value
    embedded = html.split("const DATA = ", 1)[1].split(";\nconst NS", 1)[0]
    assert json.loads(embedded)["coach_effect"]["sackings"][0]["coach_in"] == hostile
    assert (tmp_path / ".nojekyll").exists()
