"""PersonLoader tags each Resolution with metadata['people']: the titled
academic person(s) it mentions, matched via a rank-prefix pattern (not a
dictionary -- person names are open-vocabulary)."""
from __future__ import annotations

from rag_lab.config import StrategySpec
from rag_lab.factory import build_loader
from rag_lab.loaders.person_loader import (
    find_people,
    match_people,
    match_people_by_dictionary,
)

_PEOPLE_DICT = [
    {
        "canonical_title": "ผศ.",
        "canonical_given": "ธนา",
        "canonical_surname": "หงษ์สุวรรณ",
        "canonical_full_name": "ผศ.ธนา หงษ์สุวรรณ",
        "count": 3,
        "aliases": [{"given": "ธนา", "surname": "หงสุวรรณ", "count": 1}],
    },
    {
        "canonical_title": "รศ.",
        "canonical_given": "สุวัฒน์",
        "canonical_surname": "ถิรเศรษฐ์",
        "canonical_full_name": "รศ.สุวัฒน์ ถิรเศรษฐ์",
        "count": 5,
        "aliases": [],
    },
]


class TestMatchPeople:
    def test_spelled_out_rank_plus_ดร_in_parentheses(self):
        text = "(ผู้ช่วยศาสตราจารย์ ดร.อรัญญา วลัยรัชต์)"
        assert match_people(text) == [
            {
                "title": "ผศ.ดร.",
                "given_name": "อรัญญา",
                "surname": "วลัยรัชต์",
                "full_name": "ผศ.ดร.อรัญญา วลัยรัชต์",
            }
        ]

    def test_abbreviated_rank_plus_ดร_no_space(self):
        text = "ทั้งนี้ ผศ.ดร.พิชชา ประสิทธิ์มีบุญ ผู้ช่วยอธิการบดี"
        result = match_people(text)
        assert result == [
            {
                "title": "ผศ.ดร.",
                "given_name": "พิชชา",
                "surname": "ประสิทธิ์มีบุญ",
                "full_name": "ผศ.ดร.พิชชา ประสิทธิ์มีบุญ",
            }
        ]

    def test_trailing_job_title_is_not_captured_as_part_of_the_name(self):
        text = "รองศาสตราจารย์ ดร.คมสัน มาลีสี อธิการบดี ประธานในที่ประชุม"
        result = match_people(text)
        assert len(result) == 1
        assert result[0]["surname"] == "มาลีสี"
        assert "อธิการบดี" not in result[0]["full_name"]

    def test_bare_abbreviated_rank_with_no_ดร(self):
        text = "๒.รศ.สุวัฒน์ ถิรเศรษฐ์<br/>(สาขาวิชาวิศวกรรมโยธา)"
        assert match_people(text) == [
            {
                "title": "รศ.",
                "given_name": "สุวัฒน์",
                "surname": "ถิรเศรษฐ์",
                "full_name": "รศ.สุวัฒน์ ถิรเศรษฐ์",
            }
        ]

    def test_bare_ดร_alone(self):
        text = "๓. ดร.จารุวิสข์ ปราบณศักดิ์</td><td>"
        assert match_people(text) == [
            {
                "title": "ดร.",
                "given_name": "จารุวิสข์",
                "surname": "ปราบณศักดิ์",
                "full_name": "ดร.จารุวิสข์ ปราบณศักดิ์",
            }
        ]

    def test_br_tag_separates_given_name_and_surname_in_table_cells(self):
        text = "๒.ผศ.ดร.วุฒิชัย<br/>ชาติพัฒนาบันท์<br/>(วิศวกรรม...)"
        assert match_people(text) == [
            {
                "title": "ผศ.ดร.",
                "given_name": "วุฒิชัย",
                "surname": "ชาติพัฒนาบันท์",
                "full_name": "ผศ.ดร.วุฒิชัย ชาติพัฒนาบันท์",
            }
        ]

    def test_spelled_out_rank_with_no_ดร(self):
        text = "ศาสตราจารย์วิภาวี สงกลิ่น เป็นผู้เสนอ"
        # spelled-out ranks always have a space before ดร. in this corpus's
        # convention, but must also work with no ดร. at all
        assert match_people(text)[0]["title"] == "ศ."

    def test_no_title_at_all_returns_empty_list(self):
        assert match_people("ไม่มีคำนำหน้าตรงนี้เลย") == []

    def test_bare_อ_title_matches_a_plain_instructor(self):
        # a real corpus gap: special/part-time instructors are routinely
        # cited with the bare "อ." rank (below ผศ./รศ./ศ.), not a doctorate
        # or professorship -- e.g. an "อาจารย์พิเศษ" (special instructor)
        # table listing "อ.อภิชา เชื้อประศิลป์" was previously invisible to
        # tagging entirely
        text = "อ.อภิชา เชื้อประศิลป์ คุณวุฒิ M.A. (Hotel and Tourism Management)"
        assert match_people(text) == [
            {
                "title": "อ.",
                "given_name": "อภิชา",
                "surname": "เชื้อประศิลป์",
                "full_name": "อ.อภิชา เชื้อประศิลป์",
            }
        ]

    def test_spelled_out_อาจารย์_is_not_a_title_trigger(self):
        # unlike the bare "อ." abbreviation, the spelled-out word is a
        # generic category noun throughout this corpus's own procedural
        # prose ("อาจารย์พิเศษ", "อาจารย์ผู้สอนในรายวิชา", "อาจารย์ประจำ
        # หลักสูตร") -- every one of those reads as a syntactically valid
        # "title + 2-token name" and would false-positive if included
        assert match_people("อาจารย์พิเศษ เพื่อบรรยายในรายวิชา") == []
        assert match_people("อาจารย์ผู้สอนในรายวิชาต่อไป") == []

    def test_อ_ดร_combo_still_matches_via_the_existing_ดร_alternative(self):
        # "อ." isn't combined with "ดร." in _TITLE (unlike ผศ./รศ./ศ.), but
        # this doesn't regress the ดร.-alone case: the match just starts at
        # "ดร." instead, same as any other leading non-title text
        result = match_people("อ.ดร.กฤช จรินโท")
        assert len(result) == 1
        assert result[0]["title"] == "ดร."
        assert result[0]["given_name"] == "กฤช"

    def test_rejects_a_generic_rank_mention_followed_by_a_pronoun(self):
        # "ศ. นั้น จะได้..." talks about the professorship rank generically
        # (an appointment procedure), not a specific named person -- a real
        # false positive found via a corpus spot-check
        text = "ตำแหน่งศ.นั้นจะได้ทรงพระกรุณาโปรดเกล้าฯ แต่งตั้ง"
        assert match_people(text) == []

    def test_rejects_อ_followed_by_a_generic_role_noun(self):
        # real false positives found in the corpus-wide check that preceded
        # adding "อ." as a title: a role-noun run right after the bare rank
        # ("the instructor who teaches...", "the instructor who receives...")
        # reads as a plausible given name without this exclusion
        assert match_people("อ.ผู้สอนตรวจสอบความถูกต้องของสื่อการสอน") == []
        assert match_people("อ.ผู้รับ เหตุผลของการเปลี่ยนแปลง") == []

    def test_caps_a_run_on_surname_from_a_dropped_ocr_space(self):
        # a real corpus case: no space between the surname and the next
        # clause ("...พันธุ์เป็นผู้เชี่ยวชาญ") -- the 18-char cap limits how
        # much of the following clause leaks into the surname
        text = "รศ.ดร.ประภาษ อุคคกิมาพันธุ์เป็นผู้เชี่ยวชาญด้านนี้"
        result = match_people(text)
        assert len(result) == 1
        assert len(result[0]["surname"]) <= 18

    def test_dedupes_the_same_person_mentioned_twice(self):
        text = (
            "ผศ.ดร.พิชชา ประสิทธิ์มีบุญ เสนอวาระ "
            "ทั้งนี้ ผศ.ดร.พิชชา ประสิทธิ์มีบุญ เป็นผู้ชี้แจง"
        )
        assert len(match_people(text)) == 1

    def test_multiple_distinct_people_sorted(self):
        text = "รศ.สุวัฒน์ ถิรเศรษฐ์ และ ดร.จารุวิสข์ ปราบณศักดิ์ ร่วมกันเสนอ"
        result = match_people(text)
        assert [p["given_name"] for p in result] == sorted(
            p["given_name"] for p in result
        )
        assert {p["surname"] for p in result} == {"ถิรเศรษฐ์", "ปราบณศักดิ์"}

    def test_full_rank_plus_ดร_normalizes_the_same_as_abbreviated(self):
        spelled = match_people("รองศาสตราจารย์ ดร.คมสัน มาลีสี")[0]
        abbreviated = match_people("รศ.ดร.คมสัน มาลีสี")[0]
        assert spelled["title"] == abbreviated["title"] == "รศ.ดร."

    def test_bridges_a_name_split_across_adjacent_table_cells(self):
        # a real corpus gap, confirmed 0/4 matched before this was added:
        # academic_resolutions/2564/ครั้งที่ 1/รับรองรายงานการประชุม.md --
        # title+given in one <td>, surname alone in the next, joined by
        # literal </td><td> markup with no whitespace/<br/> between (unlike
        # test_br_tag_separates_given_name_and_surname_in_table_cells above,
        # which _SEP already handles within one cell)
        text = (
            "<table><tr><td>ลำดับที่ ๗</td><td>เดิม</td>"
            "<td>รศ.สุขสันต์</td><td>พาณิชพาพิบูล</td></tr>"
            "<tr><td></td><td>แก้ไขเป็น</td>"
            "<td>รศ.ดร.สุขสันต์</td><td>พาณิชพาพิบูล</td></tr></table>"
        )
        result = match_people(text)
        assert {p["full_name"] for p in result} == {
            "รศ.สุขสันต์ พาณิชพาพิบูล",
            "รศ.ดร.สุขสันต์ พาณิชพาพิบูล",
        }

    def test_does_not_bridge_when_the_next_cell_has_more_than_the_surname(self):
        # regression: a naive "bridge any adjacent cell" version false-
        # positived on a real OCR-corrupted table (an "อาจารย์พิเศษสอนเกิน
        # ร้อยละ 50" teaching-load report) -- the cell after the name held
        # garbled trailing prose, not a clean surname, and would otherwise
        # have been tagged as the fake surname "อาจารย์ผู้สอน"
        text = "<td>ผศ.ดร.อำภาพรรณ</td><td>อาจารย์ผู้สอน ภายในไม่เพียงพอ</td>"
        assert match_people(text) == []

    def test_does_not_bridge_a_short_ocr_fragment_as_a_surname(self):
        # regression: a different OCR-corrupted table from the same
        # "อาจารย์พิเศษสอนเกินร้อยละ 50" document family left a bare 5-char
        # garbage fragment ("มเชี่") alone in the next cell -- below the
        # 6-char minimum (the shortest genuine surname found in a
        # full-corpus scan, "มิตะถา", is exactly 6), so it's rejected even
        # though it satisfies the "nothing else in the cell" guard alone
        text = "<td>ดร.วราสินกิจสุนทร</td><td>มเชี่</td>"
        assert match_people(text) == []

    def test_bridges_the_shortest_known_genuine_cross_cell_surname(self):
        # the boundary case the 6-char minimum is tuned against: the
        # shortest real surname found via a full-corpus scan
        text = "<td>รศ.ดร.สมศักดิ์</td><td>มิตะถา</td>"
        assert match_people(text) == [
            {
                "title": "รศ.ดร.",
                "given_name": "สมศักดิ์",
                "surname": "มิตะถา",
                "full_name": "รศ.ดร.สมศักดิ์ มิตะถา",
            }
        ]


class TestMatchPeopleByDictionary:
    def test_matches_canonical_name_with_no_title(self):
        # a user typing a search query usually won't include an academic
        # rank the way this corpus's own documents do
        assert match_people_by_dictionary("ธนา หงษ์สุวรรณ มีประวัติอย่างไรบ้าง", _PEOPLE_DICT) == [
            {"given_name": "ธนา", "surname": "หงษ์สุวรรณ", "full_name": "ผศ.ธนา หงษ์สุวรรณ"}
        ]

    def test_matches_via_a_known_alias_spelling(self):
        assert match_people_by_dictionary("ประวัติ ธนา หงสุวรรณ", _PEOPLE_DICT) == [
            {"given_name": "ธนา", "surname": "หงษ์สุวรรณ", "full_name": "ผศ.ธนา หงษ์สุวรรณ"}
        ]

    def test_no_match_when_name_absent(self):
        assert match_people_by_dictionary("ค่าธรรมเนียมการศึกษา", _PEOPLE_DICT) == []

    def test_titled_text_is_still_matched_by_the_no_title_dictionary_too(self):
        # match_people_by_dictionary itself doesn't require the absence of a
        # title -- it's detect_entities' job (router.py) to try match_people
        # first and only fall back to this; this function alone just does
        # substring lookup regardless of what surrounds the name
        result = match_people_by_dictionary("ผศ.ดร.ธนา หงษ์สุวรรณ เสนอวาระ", _PEOPLE_DICT)
        assert result == [
            {"given_name": "ธนา", "surname": "หงษ์สุวรรณ", "full_name": "ผศ.ธนา หงษ์สุวรรณ"}
        ]


class TestPersonLoaderIntegration:
    def test_tags_metadata_with_people(self, tmp_path):
        d = tmp_path / "2569" / "ครั้งที่ 1"
        d.mkdir(parents=True)
        doc = d / "a.md"
        doc.write_text(
            "## Page 1\nรองศาสตราจารย์ ดร.คมสัน มาลีสี อธิการบดี",
            encoding="utf-8",
        )

        res = build_loader(StrategySpec(type="person")).load(str(doc))

        assert res.metadata["people"] == [
            {
                "title": "รศ.ดร.",
                "given_name": "คมสัน",
                "surname": "มาลีสี",
                "full_name": "รศ.ดร.คมสัน มาลีสี",
            }
        ]

    def test_no_mention_gives_empty_list(self, tmp_path):
        d = tmp_path / "2569" / "ครั้งที่ 1"
        d.mkdir(parents=True)
        doc = d / "a.md"
        doc.write_text("## Page 1\nสภาสถาบันมีมติเห็นชอบตามที่เสนอ", encoding="utf-8")

        res = build_loader(StrategySpec(type="person")).load(str(doc))

        assert res.metadata["people"] == []


class TestFindPeopleSpans:
    """find_people is what match_people is now built on: the same matches,
    with position, before the dedupe/sort. The relation graph needs position
    (person->faculty is 'the name immediately before this สังกัด marker'),
    and a second copy of the rank patterns would drift away from this one."""

    _TEXT = (
        "ที่ประชุมรับทราบ ผศ.ดร.สมชาย ใจดี (สังกัดคณะวิทยาศาสตร์) และ "
        "อ.สมหญิง รักเรียน (สังกัดคณะครุศาสตร์อุตสาหกรรมและเทคโนโลยี) "
        "โดย ผศ.ดร.สมชาย ใจดี เป็นประธาน"
    )

    def test_reports_every_occurrence_in_text_order(self):
        hits = find_people(self._TEXT)
        assert [p["full_name"] for _, _, p in hits] == [
            "ผศ.ดร.สมชาย ใจดี",
            "อ.สมหญิง รักเรียน",
            "ผศ.ดร.สมชาย ใจดี",
        ]
        # strictly increasing starts -- callers pick "the nearest one before
        # position X", which is only meaningful if the order is positional
        starts = [s for s, _, _ in hits]
        assert starts == sorted(starts)

    def test_spans_point_at_the_name_itself(self):
        start, end, person = find_people(self._TEXT)[0]
        assert self._TEXT[start:end] == "ผศ.ดร.สมชาย ใจดี"
        assert person["full_name"] == "ผศ.ดร.สมชาย ใจดี"

    def test_match_people_is_the_deduped_sorted_view_of_it(self):
        # Pins the refactor: match_people's contract (deduped, sorted, no
        # positions) is unchanged, and the two can never disagree about who
        # is in a document because one is derived from the other.
        assert match_people(self._TEXT) == sorted(
            {p["full_name"]: p for _, _, p in find_people(self._TEXT)}.values(),
            key=lambda p: (p["title"], p["given_name"], p["surname"]),
        )
