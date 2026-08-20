"""Bare-field program matching, and the programme identity underneath it.

A person searching types the field ("วิศวกรรมคอมพิวเตอร์"), not the
60-character canonical. Before that can be resolved, entries that are ONE
programme under two names have to be recognised, or a caller that counts sees
the same programme twice.

The most important tests here are the NEGATIVE ones. A first version of
`_same_programme` also merged entries sharing a degree when one field name
extended the other; it collapsed 28 entries and among them
`วิศวกรรมไฟฟ้า` with `วิศวกรรมไฟฟ้าสื่อสารและเครือข่าย` -- genuinely different
programmes, and precisely the prefix-group problem `program_loader`'s own module
docstring opens with. That rule is gone, and these tests are what stop it coming
back.
"""
from __future__ import annotations

import pytest

from rag_lab.loaders.program_loader import (
    load_dictionary,
    match_programs_by_field,
    programme_groups,
)
from rag_lab.router import classify_query, detect_entities


def _entry(degree: str, field: str, prefix: str = "หลักสูตร") -> dict:
    return {
        "canonical": f"{prefix}{degree} สาขาวิชา{field}",
        "prefix_type": prefix,
        "degree": degree,
        "field": field,
        "count": 1,
    }


# ---------------------------------------------------------------- grouping
def test_a_renamed_associate_degree_is_one_programme():
    """KOSEN renamed the degree in 2568; 2569/3 amends "ฉบับปี พ.ศ. ๒๕๖๗", the
    curriculum 2567/2 approved under the older name."""
    d = [
        _entry("อนุปริญญา", "วิศวกรรมคอมพิวเตอร์"),
        _entry("อนุปริญญาวิศวกรรมศาสตร์", "วิศวกรรมคอมพิวเตอร์"),
    ]
    assert len(programme_groups(d)) == 1


def test_degree_LEVELS_are_never_merged():
    """บัณฑิต is a suffix of มหาบัณฑิต. An unrestricted "one degree name extends
    the other" rule would merge a bachelor's into a master's -- the degree-swap
    error the 2026-08-11 guard exists to prevent."""
    d = [
        _entry("วิศวกรรมศาสตรบัณฑิต", "วิศวกรรมคอมพิวเตอร์"),
        _entry("วิศวกรรมศาสตรมหาบัณฑิต", "วิศวกรรมคอมพิวเตอร์"),
    ]
    assert len(programme_groups(d)) == 2


def test_a_longer_field_is_a_DIFFERENT_programme():
    """The rejected rule. `program_loader`'s docstring names this exact pair as
    'two different, both real, programmes'."""
    d = [
        _entry("วิศวกรรมศาสตรบัณฑิต", "วิศวกรรมไฟฟ้า"),
        _entry("วิศวกรรมศาสตรบัณฑิต", "วิศวกรรมไฟฟ้าสื่อสารและเครือข่าย"),
    ]
    assert len(programme_groups(d)) == 2, "a longer field name is not a rename"


def test_unrelated_entries_are_never_merged():
    d = [
        _entry("วิศวกรรมศาสตรบัณฑิต", "วิศวกรรมโยธา"),
        _entry("ศิลปศาสตรบัณฑิต", "ภาษาญี่ปุ่น"),
    ]
    assert len(programme_groups(d)) == 2


def test_grouping_keeps_every_surface_form():
    """Collapsing must never delete a spelling: a document titled with the
    retired name has to keep matching."""
    d = [
        _entry("อนุปริญญา", "วิศวกรรมคอมพิวเตอร์"),
        _entry("อนุปริญญาวิศวกรรมศาสตร์", "วิศวกรรมคอมพิวเตอร์"),
    ]
    (group,) = programme_groups(d)
    assert {e["canonical"] for e in group} == {e["canonical"] for e in d}


def test_on_the_real_dictionary_it_collapses_only_the_three_kosen_pairs():
    d = load_dictionary()
    groups = programme_groups(d)
    merged = [g for g in groups if len(g) > 1]
    assert len(d) - len(groups) == 3
    assert len(merged) == 3
    for g in merged:
        assert {e["field"] for e in g} == {g[0]["field"]}
        assert all("อนุปริญญา" in e["degree"] for e in g), (
            "the only collapses on real data are the KOSEN associate-degree renames"
        )


# --------------------------------------------------------- field matching
def test_a_bare_field_returns_every_programme_offering_it():
    got = match_programs_by_field("วิศวกรรมคอมพิวเตอร์")
    assert len(got) == 4, "all four, never a guess at which degree level was meant"
    assert all("วิศวกรรมคอมพิวเตอร์" in c for c in got)


def test_a_field_matches_as_a_standalone_phrase_only():
    d = [_entry("วิศวกรรมศาสตรบัณฑิต", "CALCULUS 2")]
    assert match_programs_by_field("วิชา CALCULUS 21 ปรับปรุง", d) == []
    assert match_programs_by_field("วิชา CALCULUS 2 ปรับปรุง", d) != []


def test_a_shorter_field_inside_a_longer_matched_one_does_not_also_fire():
    d = [
        _entry("อนุปริญญา", "แมคคาทรอนิกส์"),
        _entry("อนุปริญญา", "วิศวกรรมแมคคาทรอนิกส์"),
    ]
    got = match_programs_by_field("สาขาวิชาวิศวกรรมแมคคาทรอนิกส์", d)
    assert got == [d[1]["canonical"]]


def test_no_field_in_the_text_returns_nothing():
    assert match_programs_by_field("อะไรก็ไม่รู้ที่ไม่มีในดิกชันนารี") == []


# ------------------------------------------------------- detect_entities
def test_field_matching_is_off_by_default():
    """`detect_entities` feeds entity_lookup and EntityFilter, whose published
    numbers were measured without it."""
    assert detect_entities("วิศวกรรมคอมพิวเตอร์") == {}


def test_field_matching_when_asked_for():
    got = detect_entities("วิศวกรรมคอมพิวเตอร์", include_field_matches=True)
    assert len(got["programs"]) == 4


def test_field_matching_is_a_fallback_not_an_addition():
    """A query that already named a programme must not have its siblings
    appended -- otherwise asking for one programme returns four."""
    q = "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมคอมพิวเตอร์"
    exact = detect_entities(q)
    widened = detect_entities(q, include_field_matches=True)
    assert exact == widened
    assert len(widened["programs"]) == 1


# ---------------------------------------------------------- classify_query
def test_a_bare_field_now_routes_to_program():
    assert classify_query("วิศวกรรมคอมพิวเตอร์") == "program"


def test_the_field_branch_does_not_steal_faculty_queries():
    """5 of the 13 faculty Gold queries contain a program field inside their
    faculty name, so the branch's LAST position is load-bearing."""
    assert classify_query("ในคณะเทคโนโลยีสารสนเทศ หลักสูตรใดเชิญอาจารย์พิเศษ") == "faculty"


def test_a_query_naming_nothing_is_still_unmatched():
    assert classify_query("อะไรก็ไม่รู้") == "unmatched"


@pytest.mark.parametrize(
    "query,expected",
    [
        ("หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมคอมพิวเตอร์", "program"),
        ("สาขาวิชาอะไรสักอย่าง", "program"),
    ],
)
def test_the_earlier_program_branches_still_win(query, expected):
    assert classify_query(query) == expected


def test_the_fallback_is_last_resort_across_every_kind():
    """Gated on `not programs` alone this fired on 5 of the 106 Gold queries:
    faculty queries whose faculty name CONTAINS a programme field. A query that
    already resolved to *something* must not be widened."""
    q = "หลักสูตรไหนของคณะบริหารธุรกิจที่ใช้อาจารย์พิเศษสอนเกินร้อยละ 50"
    exact = detect_entities(q)
    assert exact.get("faculties"), "fixture must resolve to a faculty"
    assert detect_entities(q, include_field_matches=True) == exact


def test_the_fallback_still_fires_for_the_case_it_exists_for():
    """The converse of the test above: narrowing it must not disable it."""
    assert len(detect_entities("วิศวกรรมคอมพิวเตอร์", include_field_matches=True)["programs"]) == 4


def test_five_of_the_thirteen_faculty_gold_queries_contain_a_programme_field():
    """The figure CLAUDE.md cites for why the `classify_query` branch must come
    LAST. No report emits it, so this test is its source -- re-derive it by
    running this file, never by trusting the prose."""
    import yaml
    from pathlib import Path

    gold = yaml.safe_load(
        Path("config/eval/gold_query_set_73det.yaml").read_text(encoding="utf-8")
    )
    qs = gold["queries"] if isinstance(gold, dict) else gold
    faculty = [q for q in qs if str(q.get("entity_type", "")).startswith("faculty")]
    assert len(faculty) == 13
    with_field = [q for q in faculty if match_programs_by_field(q["query"])]
    assert len(with_field) == 5
