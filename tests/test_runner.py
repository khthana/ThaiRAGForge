"""Cycle 1 (tracer) — run_experiment builds one cached, manifested Index per combo.

Uses the `hashing` baseline embedder so the whole batch runs deterministically
without a GPU model.
"""
from __future__ import annotations

from rag_lab.config import ExperimentConfig
from rag_lab.io.artifact_store import ArtifactStore
from rag_lab.runner import run_experiment


def _write_corpus(tmp_path):
    root = tmp_path / "corpus" / "2569" / "ครั้งที่ 1"
    root.mkdir(parents=True)
    (root / "เรื่อง ค่าธรรมเนียม.md").write_text(
        "## Page 1\nค่าธรรมเนียม การศึกษา ภาคเรียน", encoding="utf-8"
    )
    (root / "เรื่อง หลักสูตร.md").write_text(
        "## Page 1\nหลักสูตร วิศวกรรม คอมพิวเตอร์", encoding="utf-8"
    )
    return tmp_path / "corpus"


def _write_config(tmp_path, corpus, out):
    text = f"""
experiment_name: tracer
corpus:
  input_dir: {corpus.as_posix()}
output_dir: {out.as_posix()}
loaders:
  - type: plain
chunkers:
  - type: fixed_size
    params: {{chunk_size: 100, chunk_overlap: 0}}
  - type: fixed_size
    params: {{chunk_size: 50, chunk_overlap: 0}}
embedders:
  - type: hashing
run_mode: cartesian
"""
    path = tmp_path / "exp.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_run_experiment_builds_artifact_and_manifest_per_combo(tmp_path):
    corpus = _write_corpus(tmp_path)
    out = tmp_path / "out"
    config = ExperimentConfig.from_yaml(_write_config(tmp_path, corpus, out))

    results = run_experiment(config)

    # 1 loader × 2 chunkers × 1 embedder = 2 combos, all built
    assert len(results) == 2
    assert all(r.status == "ok" for r in results)

    combo_dirs = [d for d in out.iterdir() if d.is_dir()]
    assert len(combo_dirs) == 2
    for d in combo_dirs:
        assert (d / "manifest.json").exists()
        index = ArtifactStore().load(d)
        assert len(index.chunks) > 0


def test_failing_combo_is_isolated_and_others_still_build(tmp_path):
    corpus = _write_corpus(tmp_path)
    out = tmp_path / "out"
    text = f"""
experiment_name: iso
corpus:
  input_dir: {corpus.as_posix()}
output_dir: {out.as_posix()}
loaders:
  - type: plain
chunkers:
  - type: fixed_size
    params: {{chunk_size: 100, chunk_overlap: 0}}
embedders:
  - type: hashing
  - type: does_not_exist
run_mode: cartesian
"""
    path = tmp_path / "iso.yaml"
    path.write_text(text, encoding="utf-8")

    results = run_experiment(ExperimentConfig.from_yaml(path))

    assert sorted(r.status for r in results) == ["error", "ok"]
    ok = next(r for r in results if r.status == "ok")
    assert (out / ok.combo_id / "manifest.json").exists()
    err = next(r for r in results if r.status == "error")
    assert err.error  # failure message captured, batch did not crash


def test_keyless_api_embedder_combo_is_isolated_as_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    corpus = _write_corpus(tmp_path)
    out = tmp_path / "out"
    text = f"""
experiment_name: api-iso
corpus:
  input_dir: {corpus.as_posix()}
output_dir: {out.as_posix()}
loaders:
  - type: plain
chunkers:
  - type: fixed_size
    params: {{chunk_size: 100, chunk_overlap: 0}}
embedders:
  - type: hashing
  - type: api
    params: {{provider: openai, model: text-embedding-3-large}}
run_mode: cartesian
"""
    path = tmp_path / "api-iso.yaml"
    path.write_text(text, encoding="utf-8")

    results = run_experiment(ExperimentConfig.from_yaml(path))

    assert sorted(r.status for r in results) == ["error", "ok"]
    ok = next(r for r in results if r.status == "ok")
    assert (out / ok.combo_id / "manifest.json").exists()
    err = next(r for r in results if r.status == "error")
    assert "OPENAI_API_KEY" in err.error  # clear, not a bare crash
