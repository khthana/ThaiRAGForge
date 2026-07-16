"""Vertical slice proving Qdrant native filtering solves the problem
MetadataFilter can't: "chunks belonging to a resolution about person X."

Ingests one already-built combo's chunks into an embedded local Qdrant
collection (payload: chunk_id/resolution_id/page/text only -- no per-chunk
entity data, since no loader tags all four entity types at once, see
docs/entity-extraction-and-gold-eval-log.md), then demonstrates a
person-scoped query two ways:

1. Unfiltered QdrantRetriever -- same ranking as DenseRetriever would give,
   since it's dense cosine search either way.
2. Filtered with `resolution_id_in` = the person's real resolution_ids,
   resolved in app code via the same alias-index join used by
   build_gold_candidates.py (person -> canonical -> resolution_ids) -- proves
   the resolution-level filtering plan from the Qdrant integration notes
   (see MEMORY.md / project_qdrant_integration).

Does not touch runner.py, query_service.py, or the UI -- this is the
vertical slice the memory notes called for before any of that wiring.

Run with:
    .venv/Scripts/python.exe tools/eval/build_qdrant_person_slice.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO / "academic_resolutions"
TAGS_DIR = CORPUS_ROOT / "entity_tags"
DICT_DIR = REPO / "data" / "entity_dictionaries"
COMBO_DIR = REPO / "data" / "index" / "chunker_compare_full" / "plain__fixed_size__e5__0638916c"
QDRANT_PATH = REPO / "data" / "qdrant" / "fixed_size_e5_person_slice"
COLLECTION = "fixed_size_e5"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "corpus_prep"))
from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.factory import build_embedder  # noqa: E402
from rag_lab.io.artifact_store import ArtifactStore  # noqa: E402
from rag_lab.retrievers import QdrantRetriever  # noqa: E402
from rag_lab.schema import Query  # noqa: E402

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: E402

import build_gold_candidates as bgc  # noqa: E402


def ingest(index_dir: Path, qdrant_path: Path, collection: str) -> int:
    store = ArtifactStore()
    index = store.load(index_dir)

    client = QdrantClient(path=str(qdrant_path))
    if client.collection_exists(collection):
        print(f"collection {collection!r} already exists at {qdrant_path}, skipping ingest")
        client.close()
        return len(index.chunks)

    dim = index.embeddings.shape[1]
    client.create_collection(
        collection_name=collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    batch = 512
    for start in range(0, len(index.chunks), batch):
        end = min(start + batch, len(index.chunks))
        client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=i,
                    vector=index.embeddings[i].tolist(),
                    payload={
                        "chunk_id": index.chunks[i].chunk_id,
                        "resolution_id": index.chunks[i].resolution_id,
                        "page": index.chunks[i].page,
                        "text": index.chunks[i].text,
                    },
                )
                for i in range(start, end)
            ],
        )
        print(f"  upserted {end}/{len(index.chunks)}")
    client.close()
    return len(index.chunks)


def resolve_person_resolution_ids(name_substring: str) -> tuple[str, list[str]]:
    """Reuses build_gold_candidates.py's exact-match alias join (not substring)
    to go from a person name to their real, verified resolution_ids."""
    people = bgc._load_json(DICT_DIR / "people.json")
    # Exact match on given name, not substring: "ธนา" as `in` matches 16 people
    # who share the name-root (ธนากร, ธนาพล, ธนาดล, ...) -- the same substring
    # pitfall build_gold_candidates.py's docstring warns about.
    matches = [e for e in people if e["canonical_given"] == name_substring]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly 1 person match for {name_substring!r}, got {len(matches)}")
    canonical = matches[0]["canonical_full_name"]

    alias_index = bgc._build_person_alias_index(people)
    by_file = bgc._load_json(TAGS_DIR / "people_by_file.json")

    rids: set[str] = set()
    for relpath, mentions in by_file.items():
        if any(alias_index.get((m["given_name"], m["surname"])) == canonical for m in mentions):
            full_path = CORPUS_ROOT / relpath
            try:
                text = full_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = full_path.read_text(encoding="utf-8-sig")
            for m in mentions:
                if alias_index.get((m["given_name"], m["surname"])) != canonical:
                    continue
                if bgc._has_non_secretarial_mention(text, m["given_name"], m["surname"]):
                    rids.add(bgc._resolution_id_for(relpath))
    return canonical, sorted(rids, key=bgc._sort_key)


def main() -> None:
    print(f"ingesting {COMBO_DIR.name} into embedded Qdrant at {QDRANT_PATH} ...")
    n = ingest(COMBO_DIR, QDRANT_PATH, COLLECTION)
    print(f"collection ready: {n} chunks\n")

    person_query = "ธนา"
    canonical, resolution_ids = resolve_person_resolution_ids(person_query)
    print(f"resolved {person_query!r} -> {canonical}")
    print(f"real resolution_ids ({len(resolution_ids)}):")
    for rid in resolution_ids:
        print(" -", rid)

    manifest = json.loads((COMBO_DIR / "manifest.json").read_text(encoding="utf-8"))
    embedder = build_embedder(StrategySpec.model_validate(manifest["combo"]["embedder"]))
    query_text = f"{canonical} มีประวัติเกี่ยวข้องกับหลักสูตรใดบ้าง"
    vector = embedder.embed_query(query_text)

    retriever = QdrantRetriever(path=str(QDRANT_PATH), collection_name=COLLECTION)
    from rag_lab.schema import Index
    import numpy as np
    empty_index = Index(chunks=[], embeddings=np.zeros((0, len(vector))))

    print(f"\n=== unfiltered top-10 for {query_text!r} ===")
    unfiltered = retriever.retrieve(Query(text=query_text, vector=vector), empty_index, k=10)
    hit_rids = set()
    for r in unfiltered:
        in_set = "IN-SET" if r.resolution_id in resolution_ids else "      "
        print(f"  [{in_set}] rank={r.rank} score={r.score:.3f} resolution_id={r.resolution_id[:70]}")
        hit_rids.add(r.resolution_id)
    print(f"  ({len(hit_rids & set(resolution_ids))}/{len(resolution_ids)} of the person's real resolutions in top-10 unfiltered)")

    print(f"\n=== filtered to the person's {len(resolution_ids)} real resolution_ids ===")
    filtered = retriever.retrieve(
        Query(text=query_text, vector=vector, filters={"resolution_id_in": resolution_ids}),
        empty_index,
        k=10,
    )
    for r in filtered:
        print(f"  rank={r.rank} score={r.score:.3f} resolution_id={r.resolution_id[:70]}")
    all_in_set = all(r.resolution_id in resolution_ids for r in filtered)
    print(f"  all results within the person's resolution set: {all_in_set}")

    dump_path = REPO / "data" / "qdrant" / "person_slice_demo_result.json"
    dump_path.write_text(
        json.dumps(
            {
                "canonical": canonical,
                "resolution_ids": resolution_ids,
                "query_text": query_text,
                "unfiltered_top10": [
                    {"rank": r.rank, "score": r.score, "resolution_id": r.resolution_id}
                    for r in unfiltered
                ],
                "filtered_top10": [
                    {"rank": r.rank, "score": r.score, "resolution_id": r.resolution_id}
                    for r in filtered
                ],
                "all_filtered_in_set": all_in_set,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nfull result (readable Thai) written to {dump_path}")


if __name__ == "__main__":
    main()
