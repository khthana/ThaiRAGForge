"""router.py: query-shape classification (person/program/unmatched) and RRF
merge of already-ranked RetrievalResults."""
from __future__ import annotations

from rag_lab.router import (
    ROUTE_PERSON,
    ROUTE_PROGRAM,
    ROUTE_UNMATCHED,
    classify_query,
    detect_entities,
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


def test_classify_person_query_tolerates_a_space_after_the_title():
    # match_people (loaders/person_loader.py) requires zero space between
    # title and name -- tuned to this corpus's own table/prose convention.
    # A user typing a query naturally writes a space there ("ผศ. ธนา..."),
    # which must still route to person, not fall through to unmatched.
    query = "ผศ. ธนา หงษ์สุวรรณ มีประวัติเป็นกรรมการหลักสูตรใดบ้าง"
    assert classify_query(query) == ROUTE_PERSON


def test_classify_still_ignores_generic_rank_mention_with_a_space():
    # The pre-existing _NOT_A_NAME guard in match_people must still apply
    # after the title-spacing collapse -- "ศ. นั้น" is prose about the rank
    # itself, not a specific person, and must not be misrouted to person.
    query = "ตำแหน่ง ศ. นั้น จะได้ทรงพระกรุณาโปรดเกล้าฯ แต่งตั้ง"
    assert classify_query(query) == ROUTE_UNMATCHED


def test_classify_program_query_via_fallback_marker():
    # No real canonical program name here, but the "สาขาวิชา" structural
    # marker still routes it to program -- the staleness fallback for a
    # brand-new, not-yet-catalogued curriculum.
    query = "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาการทดสอบระบบขั้นสูง เปลี่ยนแปลงอย่างไรบ้าง"
    assert classify_query(query) == ROUTE_PROGRAM


def test_classify_unmatched_query():
    query = "ในการประชุมครั้งนี้ มีการพิจารณาเรื่องค่าธรรมเนียมการศึกษาในกรณีใดบ้าง"
    assert classify_query(query) == ROUTE_UNMATCHED


def test_detect_entities_person_via_titled_regex():
    detected = detect_entities("ผศ.ดร.สมชาย ใจดี มีประวัติเป็นกรรมการหลักสูตรใดบ้าง")
    assert detected == {"people": ["สมชาย ใจดี"]}


def test_detect_entities_person_via_no_title_dictionary_fallback():
    def fake_dict_matcher(query):
        return [{"given_name": "ธนา", "surname": "หงษ์สุวรรณ", "full_name": "ธนา หงษ์สุวรรณ"}]

    detected = detect_entities(
        "ธนา หงษ์สุวรรณ มีประวัติอย่างไรบ้าง", people_dict_matcher=fake_dict_matcher
    )
    assert detected == {"people": ["ธนา หงษ์สุวรรณ"]}


def test_detect_entities_program():
    # a "(" right after the program name bounds match_programs' fuzzy-match
    # span (see program_loader.py's _bounded_span) -- mirrors
    # test_program_loader.py's test_real_dictionary_loads_and_is_usable
    detected = detect_entities(
        "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมโยธา (การปรับปรุงแก้ไขหลักสูตร) เปลี่ยนแปลงอย่างไรบ้าง"
    )
    assert "programs" in detected
    assert detected["programs"]


def test_detect_entities_course_via_injected_fake_matcher():
    detected = detect_entities("รายวิชานี้คืออะไร", course_matcher=lambda q: ["01276764"])
    assert detected == {"courses": ["01276764"]}


def test_detect_entities_multiple_kinds_at_once():
    detected = detect_entities(
        "ผศ.ดร.สมชาย ใจดี สอนวิชานี้หรือไม่", course_matcher=lambda q: ["01276764"]
    )
    assert detected["people"] == ["สมชาย ใจดี"]
    assert detected["courses"] == ["01276764"]


def test_detect_entities_no_match_returns_empty_dict():
    # Fakes injected for every kind, not just courses -- isolates this "no
    # match anywhere" case from the real people.json/programs.json
    # dictionaries, which a long enough generic sentence could otherwise
    # risk an accidental substring collision against.
    detected = detect_entities(
        "ในการประชุมครั้งนี้ มีการพิจารณาเรื่องค่าธรรมเนียมการศึกษาในกรณีใดบ้าง",
        people_dict_matcher=lambda q: [],
        program_matcher=lambda q: [],
        course_matcher=lambda q: [],
    )
    assert detected == {}


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
