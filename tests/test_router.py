"""router.py: query-shape classification (person/program/unmatched) and RRF
merge of already-ranked RetrievalResults."""
from __future__ import annotations

from rag_lab.router import (
    ROUTE_PERSON,
    ROUTE_PROGRAM,
    ROUTE_UNMATCHED,
    classify_query,
    rrf_merge,
)
from rag_lab.schema import RankedChunk, RetrievalResult


def _chunk(resolution_id: str, rank: int, chunk_id: str | None = None) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id or f"{resolution_id}:0",
        resolution_id=resolution_id,
        page=1,
        score=1.0 / rank,
        rank=rank,
        text="",
    )


def test_classify_person_query_via_live_regex_not_dictionary():
    # "ผศ.ดร." + name pattern -- matched even though this exact person is
    # never in any dictionary in this test, demonstrating zero staleness
    # dependency for the person axis.
    query = "ผศ.ดร.สมชาย ใจดี มีประวัติเป็นกรรมการหลักสูตรใดบ้าง"
    assert classify_query(query) == ROUTE_PERSON


def test_classify_program_query_via_fallback_marker():
    # No real canonical program name here, but the "สาขาวิชา" structural
    # marker still routes it to program -- the staleness fallback for a
    # brand-new, not-yet-catalogued curriculum.
    query = "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาการทดสอบระบบขั้นสูง เปลี่ยนแปลงอย่างไรบ้าง"
    assert classify_query(query) == ROUTE_PROGRAM


def test_classify_unmatched_query():
    query = "ในการประชุมครั้งนี้ มีการพิจารณาเรื่องค่าธรรมเนียมการศึกษาในกรณีใดบ้าง"
    assert classify_query(query) == ROUTE_UNMATCHED


def test_rrf_merge_promotes_resolution_ranked_well_in_multiple_lists():
    r1 = RetrievalResult(
        query="q", combination_id="a", top_k=10, retriever="dense",
        results=[_chunk("res-1", 1), _chunk("res-2", 2), _chunk("res-3", 3)],
    )
    r2 = RetrievalResult(
        query="q", combination_id="b", top_k=10, retriever="dense",
        results=[_chunk("res-2", 1), _chunk("res-4", 2), _chunk("res-1", 3)],
    )
    merged = rrf_merge([r1, r2], top_k=10)
    ranked_ids = [rc.resolution_id for rc in merged.results]
    # res-1 and res-2 each appear near the top of both lists; res-3/res-4
    # each only appear once, further down -- fusion should rank the
    # doubly-reinforced resolutions ahead of the singly-seen ones.
    assert ranked_ids.index("res-1") < ranked_ids.index("res-3")
    assert ranked_ids.index("res-2") < ranked_ids.index("res-4")


def test_rrf_merge_dedupes_to_one_entry_per_resolution():
    r1 = RetrievalResult(
        query="q", combination_id="a", top_k=10, retriever="dense",
        results=[_chunk("res-1", 1, "res-1:0"), _chunk("res-1", 2, "res-1:1")],
    )
    merged = rrf_merge([r1], top_k=10)
    assert len(merged.results) == 1
