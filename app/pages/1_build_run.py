"""Mode A UI — Build/Run experiments. Thin Streamlit shell over rag_lab.config
and rag_lab.runner (ADR-0001: core stays Streamlit-free).

The YAML text area is the single source of truth for config content — Build
parses *that text*, not separate widget state. This page used to also have
per-field sidebar widgets and loaders/chunkers/embedders multiselects, but
Streamlit doesn't re-seed a text_area once its key has a session value, so any
widget duplicating YAML content went stale (visibly interactive, silently
ignored) the moment a user touched the text area — first the sidebar fields,
then the multiselects. Both are gone now: `config_path` (where to save,
genuinely not config content) is the only widget outside the YAML; strategy
names are just listed in a caption, discovered live from the registries. This
is what makes per-strategy params (chunk_size, dim, ...) editable and makes
"แก้ YAML" (edit YAML) real, matching the issue's own framing that YAML is the
source of truth. The written file is exactly what
`python -m rag_lab.cli run --config ...` can run — the UI has no private
format.

A Streamlit multipage page (not a second branch of streamlit_app.py): Mode B's
`st.stop()` on "no indices found" would otherwise kill this page's run too.

Run with:  streamlit run app/streamlit_app.py   (this page appears in the nav)
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import yaml

# make src/ importable when launched via `streamlit run`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import rag_lab.factory  # noqa: E402,F401 (import triggers strategy registration)
from rag_lab.combos import enumerate_build_combos  # noqa: E402
from rag_lab.config import CorpusSpec, ExperimentConfig, StrategySpec  # noqa: E402
from rag_lab.registries import (  # noqa: E402
    chunker_registry,
    embedder_registry,
    loader_registry,
)
from rag_lab.runner import run_experiment  # noqa: E402

st.set_page_config(page_title="RAG Lab — Build/Run", layout="wide")
st.title("RAG Lab — Build/Run Experiments (Mode A)")

config_path = st.sidebar.text_input(
    "Config YAML path (save/build target)", "config/experiments/dev_smoke.yaml", key="config_path"
)


st.caption(
    f"Available — loaders: {loader_registry.names()} · "
    f"chunkers: {chunker_registry.names()} · embedders: {embedder_registry.names()}. "
    "Edit the YAML below to pick strategies, params, corpus dir, subset, and output dir."
)

seed_config = ExperimentConfig(
    experiment_name="dev_smoke",
    corpus=CorpusSpec(input_dir="academic_resolutions/2569", subset="dev", limit=20),
    output_dir="data/index/dev_smoke",
    loaders=[StrategySpec(type="plain")],
    chunkers=[StrategySpec(type="fixed_size")],
    embedders=[StrategySpec(type="hashing")],
)

st.subheader("Config YAML — this is what actually builds")
yaml_text = st.text_area(
    "Edit corpus/output paths and params here before building",
    value=seed_config.to_yaml_string(),
    height=320,
    key="yaml_editor",
)

if st.button("Write YAML + Build", type="primary", key="build_button"):
    try:
        config = ExperimentConfig.model_validate(yaml.safe_load(yaml_text))
    except Exception as exc:
        st.error(f"Invalid config YAML: {exc}")
    else:
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        Path(config_path).write_text(yaml_text, encoding="utf-8")
        st.success(
            f"Wrote `{config_path}` — run it identically via "
            f"`python -m rag_lab.cli run --config {config_path}`"
        )

        total = len(enumerate_build_combos(config))
        progress = st.progress(0.0)
        status = st.empty()
        done: list = []

        def _on_combo_done(result) -> None:
            done.append(result)
            progress.progress(len(done) / total)
            status.write(f"[{result.status}] {result.combo_id}")

        results = run_experiment(config, on_combo_done=_on_combo_done)

        st.subheader("Per-combination metrics")
        for r in results:
            with st.expander(f"{r.combo_id} — {r.status}"):
                if r.status == "ok":
                    st.write(f"chunks: {r.n_chunks}")
                    st.write(f"chunk size (chars): {r.chunk_size_stats}")
                    st.write(f"timing (s): {r.timings}")
                else:
                    st.error(r.error)
