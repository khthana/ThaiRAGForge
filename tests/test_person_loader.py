"""PersonLoader tags each Resolution with metadata['people']: the titled
academic person(s) it mentions, matched via a rank-prefix pattern (not a
dictionary -- person names are open-vocabulary)."""
from __future__ import annotations

from rag_lab.config import StrategySpec
from rag_lab.factory import build_loader
from rag_lab.loaders.person_loader import match_people, match_people_by_dictionary

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

    def test_rejects_a_generic_rank_mention_followed_by_a_pronoun(self):
        # "ศ. นั้น จะได้..." talks about the professorship rank generically
        # (an appointment procedure), not a specific named person -- a real
        # false positive found via a corpus spot-check
        text = "ตำแหน่งศ.นั้นจะได้ทรงพระกรุณาโปรดเกล้าฯ แต่งตั้ง"
        assert match_people(text) == []

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
