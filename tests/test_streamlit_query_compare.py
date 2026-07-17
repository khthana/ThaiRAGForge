"""Mode B (Query & Compare) headless smoke (streamlit.testing.v1.AppTest).

Covers the two things route_query wiring added: the compare-mode path still
works unchanged, and smart routing degrades gracefully (a clear st.error, not
a crash) when the index dir doesn't have the specific combos routing needs --
real bge-m3/ConGen-PhayaThaiBERT builds are too heavy for a fast smoke test,
so the graceful-failure path is what's actually exercised here."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from rag_lab.config import ExperimentConfig, StrategySpec
from rag_lab.query_service import discover_indices
from rag_lab.runner import run_experiment

_APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _build_index(tmp_path):
    corpus = tmp_path / "corpus" / "2569" / "ครั้งที่ 1"
    corpus.mkdir(parents=True)
    (corpus / "เรื่อง ค่าธรรมเนียม.md").write_text(
        "## Page 1\nค่าธรรมเนียม การศึกษา ภาคเรียน", encoding="utf-8"
    )
    out = tmp_path / "out"
    config = ExperimentConfig(
        experiment_name="e",
        corpus={"input_dir": (tmp_path / "corpus").as_posix()},
        output_dir=out.as_posix(),
        loaders=[StrategySpec(type="plain")],
        chunkers=[StrategySpec(type="fixed_size", params={"chunk_size": 100})],
        embedders=[StrategySpec(type="hashing")],
    )
    run_experiment(config)
    return out


def _point_at_custom_index_dir(at: AppTest, out) -> None:
    """The 'Index output dir' picker is a selectbox over data/index/*
    (whatever's built on disk) plus a 'Custom path...' escape hatch -- tests
    need the escape hatch to point at a throwaway tmp_path index, regardless
    of what real indices happen to exist in this checkout."""
    at.sidebar.selectbox(key="output_dir_choice").set_value("Custom path...")
    at.run(timeout=30)
    at.sidebar.text_input(key="output_dir").set_value(str(out))
    at.run(timeout=30)


def test_compare_mode_still_works_unchanged(tmp_path):
    out = _build_index(tmp_path)

    at = AppTest.from_file(_APP)
    at.run(timeout=30)
    assert not at.exception

    _point_at_custom_index_dir(at, out)
    assert not at.exception
    assert at.sidebar.checkbox(key="smart_routing").value is False

    # multiselect's `default=` isn't reflected in AppTest's `.value` until an
    # explicit selection is made, even though the live app would use it --
    # select explicitly rather than relying on it. `.options` returns the
    # *formatted* display labels (via format_func), not raw combo_ids, so
    # get the raw ids straight from the built index instead.
    combo_ids = [info.combo_id for info in discover_indices(out)]
    at.sidebar.multiselect(key="selected_combos").set_value(combo_ids)
    at.run(timeout=30)

    at.text_input(key="query").set_value("ค่าธรรมเนียม")
    at.button(key="search_button").click().run(timeout=30)
    assert not at.exception
    assert any("ค่าธรรมเนียม" in md.value for md in at.markdown)


def test_smart_routing_fails_gracefully_without_the_required_combos(tmp_path):
    out = _build_index(tmp_path)  # only a fixed_size+hashing combo -- no bge-m3/ConGen combo

    at = AppTest.from_file(_APP)
    at.run(timeout=30)

    _point_at_custom_index_dir(at, out)

    at.sidebar.checkbox(key="smart_routing").set_value(True)
    at.run(timeout=30)
    assert not at.exception

    at.text_input(key="query").set_value("ค่าธรรมเนียม")
    at.button(key="search_button").click().run(timeout=30)

    assert not at.exception  # LookupError must be caught, not crash the page
    assert at.error  # a visible, actionable message instead


def test_hashing_only_index_shows_a_toy_embedder_warning(tmp_path):
    """The hashing embedder is a fast placeholder with no real semantic
    understanding (see config/experiments/dev_smoke.yaml's own comment) --
    an index built entirely from it should warn loudly, since a user pointed
    at one and got confusingly bad results with no idea why."""
    out = _build_index(tmp_path)  # only fixed_size+hashing

    at = AppTest.from_file(_APP)
    at.run(timeout=30)
    _point_at_custom_index_dir(at, out)

    assert any("hashing" in w.value for w in at.warning)


def test_combo_label_disambiguates_same_chunker_type_by_chunk_size(tmp_path):
    """Two fixed_size combos differing only by chunk_size used to render as
    the identical label ('fixed_size + hashing' twice) in the multiselect --
    the label must include chunk_size so they're distinguishable."""
    corpus = tmp_path / "corpus" / "2569" / "ครั้งที่ 1"
    corpus.mkdir(parents=True)
    (corpus / "a.md").write_text("## Page 1\nเนื้อหา ทดสอบ", encoding="utf-8")
    out = tmp_path / "out"
    config = ExperimentConfig(
        experiment_name="e",
        corpus={"input_dir": (tmp_path / "corpus").as_posix()},
        output_dir=out.as_posix(),
        loaders=[StrategySpec(type="plain")],
        chunkers=[
            StrategySpec(type="fixed_size", params={"chunk_size": 100}),
            StrategySpec(type="fixed_size", params={"chunk_size": 50}),
        ],
        embedders=[StrategySpec(type="hashing")],
    )
    run_experiment(config)

    at = AppTest.from_file(_APP)
    at.run(timeout=30)
    _point_at_custom_index_dir(at, out)

    labels = at.sidebar.multiselect(key="selected_combos").options
    assert len(set(labels)) == 2
    assert any("100" in label for label in labels)
    assert any("50" in label for label in labels)
