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

_APP = str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py")


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


def _turn_off_routing(at: AppTest) -> None:
    """Smart routing ships ON, and it replaces the side-by-side comparison with
    a single routed answer -- so every test that reaches for the
    "Combinations to compare" multiselect has to opt out of it first."""
    if at.sidebar.checkbox(key="smart_routing").value:
        at.sidebar.checkbox(key="smart_routing").set_value(False)
        at.run(timeout=30)


def test_compare_mode_still_works_unchanged(tmp_path):
    out = _build_index(tmp_path)

    at = AppTest.from_file(_APP)
    at.run(timeout=30)
    assert not at.exception

    _point_at_custom_index_dir(at, out)
    assert not at.exception

    # Routing ships ON (2026-08-24), so compare mode is now the opt-out and the
    # "Combinations to compare" multiselect does not exist until it is off.
    assert at.sidebar.checkbox(key="smart_routing").value is True
    at.sidebar.checkbox(key="smart_routing").set_value(False)
    at.run(timeout=30)

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


def test_entity_boost_checkbox_narrows_and_labels_the_result(tmp_path):
    """entity_boost is a pass-through to query_indices (query_service.py) --
    this only needs to prove the checkbox is actually wired, not re-test
    EntityFilter/detect_entities themselves (covered by their own unit
    tests). A person-naming query against an entity_tags-loader index must
    come back labeled '(entity-boosted)', and the detected-entity caption
    must render."""
    corpus = tmp_path / "corpus" / "2569" / "ครั้งที่ 1"
    corpus.mkdir(parents=True)
    (corpus / "เรื่อง ประวัติ.md").write_text(
        "## Page 1\nผศ.ดร.สมชาย ใจดี เป็นกรรมการหลักสูตร", encoding="utf-8"
    )
    out = tmp_path / "out"
    config = ExperimentConfig(
        experiment_name="e",
        corpus={"input_dir": (tmp_path / "corpus").as_posix()},
        output_dir=out.as_posix(),
        loaders=[StrategySpec(type="entity_tags")],
        chunkers=[StrategySpec(type="fixed_size", params={"chunk_size": 100})],
        embedders=[StrategySpec(type="hashing")],
    )
    run_experiment(config)

    at = AppTest.from_file(_APP)
    at.run(timeout=30)
    _point_at_custom_index_dir(at, out)

    combo_ids = [info.combo_id for info in discover_indices(out)]
    _turn_off_routing(at)
    at.sidebar.multiselect(key="selected_combos").set_value(combo_ids)
    at.sidebar.checkbox(key="entity_boost").set_value(True)
    at.run(timeout=30)

    at.text_input(key="query").set_value("ผศ.ดร.สมชาย ใจดี มีประวัติอย่างไรบ้าง")
    at.button(key="search_button").click().run(timeout=30)

    assert not at.exception
    assert any("Detected entities" in c.value for c in at.caption)
    assert any("(entity-boosted)" in h.value for h in at.subheader)


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

    _turn_off_routing(at)
    labels = at.sidebar.multiselect(key="selected_combos").options
    assert len(set(labels)) == 2
    assert any("100" in label for label in labels)
    assert any("50" in label for label in labels)


def test_the_warm_up_is_off_until_asked(tmp_path, monkeypatch):
    """The button exists and nothing is warmed by merely opening the page.

    Deliberate: warming holds ~3.2GB RAM and ~3.3GB VRAM, and this repo shares
    one 12GB card with the eval scripts -- an automatic grab at UI start is how
    a GPU run dies. A deployment opts in with RAG_LAB_WARM_ON_START=1."""
    from rag_lab.io.index_cache import clear_index_cache, index_cache_info

    monkeypatch.delenv("RAG_LAB_WARM_ON_START", raising=False)
    out = _build_index(tmp_path)
    clear_index_cache()

    at = AppTest.from_file(_APP, default_timeout=30).run(timeout=60)
    _point_at_custom_index_dir(at, out)

    assert any(b.key == "warm_now" for b in at.sidebar.button), "no warm-up control"
    assert index_cache_info()["size"] == 0, "opening the page warmed the caches"


def test_pressing_the_button_warms_the_routed_indices(tmp_path, monkeypatch):
    """The shipped route map points at real bge-m3/qwen3 combos, which this
    fixture does not have -- so what is exercised is that the press reaches
    `warm_serving_caches` and that an unreachable target is REPORTED rather
    than raised. A warm-up is an optimisation; it must not take the app down."""
    monkeypatch.delenv("RAG_LAB_WARM_ON_START", raising=False)
    out = _build_index(tmp_path)

    at = AppTest.from_file(_APP, default_timeout=60).run(timeout=90)
    _point_at_custom_index_dir(at, out)
    at.sidebar.button(key="warm_now").click().run(timeout=90)

    assert not at.exception, f"warming crashed the app: {at.exception}"
    # Every shipped target is missing from this toy dir, so the report is all
    # failures -- rendered as warnings, with the app still usable.
    assert at.sidebar.warning, "an unreachable target was not reported"


def test_the_shipped_defaults_are_routed_hybrid(tmp_path):
    """The decision of 2026-08-24, pinned where a user would meet it.

    Before it, the page opened on `dense` with routing off -- the weakest
    configuration in every table this project publishes, and it was that only
    because both widgets took their first option. A revert would not fail
    anything else here: compare mode works either way.
    """
    at = AppTest.from_file(_APP)
    at.run(timeout=30)
    assert not at.exception
    assert at.sidebar.selectbox(key="retriever").value == "hybrid"
    assert at.sidebar.checkbox(key="smart_routing").value is True
    # the depth the routed measurement was taken at
    assert at.sidebar.selectbox(key="fetch_depth").value == 200
