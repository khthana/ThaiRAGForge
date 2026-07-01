"""Thin CLI over the core pipeline (walking skeleton).

    python -m rag_lab.cli build   --input-dir academic_resolutions/2569 --out data/index/dev --limit 50
    python -m rag_lab.cli retrieve --index data/index/dev --query "การลดค่าธรรมเนียม" -k 5

Both entry points are thin wrappers; all real work lives in the importable core.
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer

# The corpus is Thai; make sure stdout can encode it on a Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass

from rag_lab.chunkers import FixedSizeChunker
from rag_lab.config import ExperimentConfig
from rag_lab.embedders.local_st_embedder import LocalSTEmbedder
from rag_lab.io.artifact_store import ArtifactStore
from rag_lab.loaders import PlainLoader
from rag_lab.pipeline import build_index, retrieve
from rag_lab.retrievers import DenseRetriever
from rag_lab.runner import run_experiment

app = typer.Typer(add_completion=False, help="RAG Lab walking-skeleton CLI")


@app.command()
def run(config: str = typer.Option(..., "--config")) -> None:
    """Run a batch experiment from a YAML config: build + cache + manifest per combo."""
    cfg = ExperimentConfig.from_yaml(config)
    results = run_experiment(cfg)
    for r in results:
        typer.echo(f"[{r.status}] {r.combo_id} chunks={r.n_chunks} {r.error or ''}")
    ok = sum(1 for r in results if r.status == "ok")
    typer.echo(f"Done: {ok}/{len(results)} combos built -> {cfg.output_dir}")


@app.command()
def build(
    input_dir: str = typer.Option(..., "--input-dir"),
    out: str = typer.Option(..., "--out"),
    limit: int = typer.Option(0, "--limit", help="dev subset size; 0 = all"),
    chunk_size: int = typer.Option(512, "--chunk-size"),
    chunk_overlap: int = typer.Option(50, "--chunk-overlap"),
) -> None:
    """Load .md resolutions -> chunk -> embed -> save an Index artifact."""
    paths = sorted(Path(input_dir).rglob("*.md"))
    if limit:
        paths = paths[:limit]
    loader = PlainLoader()
    resolutions = [loader.load(str(p)) for p in paths]
    typer.echo(f"Loaded {len(resolutions)} resolutions from {input_dir}")

    index = build_index(
        resolutions,
        FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        LocalSTEmbedder(),
    )
    ArtifactStore().save(index, out)
    typer.echo(f"Built {len(index.chunks)} chunks → {out}")


@app.command(name="retrieve")
def retrieve_cmd(
    index: str = typer.Option(..., "--index"),
    query: str = typer.Option(..., "--query"),
    k: int = typer.Option(5, "-k", "--k"),
) -> None:
    """Query a saved Index artifact and print the top-k chunks."""
    idx = ArtifactStore().load(index)
    result = retrieve(query, idx, LocalSTEmbedder(), DenseRetriever(), k=k)
    for r in result.results:
        preview = r.text.replace("\n", " ")[:80]
        typer.echo(f"[{r.rank}] score={r.score:.3f} p{r.page} {r.resolution_id}\n    {preview}")


if __name__ == "__main__":
    app()
