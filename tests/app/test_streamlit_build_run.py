"""Cycle 17 — Streamlit Mode A headless smoke (streamlit.testing.v1.AppTest).

The PRD excludes Streamlit widgets from unit testing (smoke/manual only), but
the page still has a seam worth verifying without a browser: does driving the
actual widgets (not calling rag_lab functions directly) produce the same
YAML-write + build-and-persist effects the ACs require?
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest

from rag_lab.config import ExperimentConfig

_PAGE = str(Path(__file__).resolve().parents[1] / "app" / "pages" / "1_build_run.py")


def _write_corpus(tmp_path):
    corpus = tmp_path / "corpus" / "2569" / "ครั้งที่ 1"
    corpus.mkdir(parents=True)
    (corpus / "เรื่อง ค่าธรรมเนียม.md").write_text(
        "## Page 1\nค่าธรรมเนียม การศึกษา ภาคเรียน", encoding="utf-8"
    )
    return tmp_path / "corpus"


def test_build_run_page_writes_yaml_and_builds_via_the_ui(tmp_path):
    corpus = _write_corpus(tmp_path)
    output_dir = tmp_path / "out"
    config_path = tmp_path / "exp.yaml"

    at = AppTest.from_file(_PAGE)
    at.run(timeout=30)
    assert not at.exception

    at.sidebar.text_input(key="config_path").set_value(str(config_path))

    # AC1 "+param" and "แก้ YAML": the text area is the only source of truth for
    # config content (corpus dir, output dir, per-strategy params) — edit it
    # directly and prove the edit reaches the built manifest, not just the display.
    seeded = yaml.safe_load(at.text_area(key="yaml_editor").value)
    seeded["corpus"]["input_dir"] = str(corpus)
    seeded["output_dir"] = str(output_dir)
    seeded["chunkers"][0]["params"] = {"chunk_size": 77}
    at.text_area(key="yaml_editor").set_value(yaml.safe_dump(seeded, allow_unicode=True))
    at.run(timeout=30)
    assert not at.exception

    at.button(key="build_button").click().run(timeout=60)
    assert not at.exception

    assert config_path.exists()
    written = ExperimentConfig.from_yaml(config_path)
    assert written.loaders[0].type == "plain"
    assert written.embedders[0].type == "hashing"
    assert written.chunkers[0].params["chunk_size"] == 77

    combo_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    assert len(combo_dirs) == 1
    manifest = json.loads((combo_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["combo"]["chunker"]["params"]["chunk_size"] == 77
