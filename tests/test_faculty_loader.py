"""FacultyLoader tags each Resolution with metadata['faculties']: the
canonical faculty/unit names it mentions, matched fuzzily against
data/entity_dictionaries/faculties.json to tolerate OCR corruption."""
from __future__ import annotations

from rag_lab.config import StrategySpec
from rag_lab.factory import build_loader
from rag_lab.loaders.faculty_loader import match_faculties

_DICT = [
    {"canonical": "คณะวิศวกรรมศาสตร์", "prefix_type": "คณะ"},
    {"canonical": "คณะวิทยาศาสตร์", "prefix_type": "คณะ"},
    {"canonical": "วิทยาลัยวิศวกรรมสังคีต", "prefix_type": "วิทยาลัย"},
    {
        "canonical": "คณะสถาปัตยกรรม ศิลปะและการออกแบบ",
        "prefix_type": "คณะ",
        "aliases": ["คณะสถาปัตยกรรมศาสตร์"],
    },
]


class TestMatchFaculties:
    def test_exact_match(self):
        text = "คณะวิศวกรรมศาสตร์ มีความประสงค์เสนอ"
        assert match_faculties(text, _DICT) == ["คณะวิศวกรรมศาสตร์"]

    def test_no_match_for_unrelated_text(self):
        text = "สภาสถาบันมีมติเห็นชอบตามที่เสนอ"
        assert match_faculties(text, _DICT) == []

    def test_tolerates_a_single_ocr_misread_character(self):
        # "ศ" misread as "ส" -- one character corrupted mid-name
        text = "คณะวิศวกรรมสาสตร์ เสนอหลักสูตรปรับปรุง"
        assert match_faculties(text, _DICT) == ["คณะวิศวกรรมศาสตร์"]

    def test_does_not_match_the_tail_of_มหาวิทยาลัย(self):
        text = "มหาวิทยาลัยมหิดล เสนอความร่วมมือทางวิชาการ"
        assert match_faculties(text, _DICT) == []

    def test_finds_multiple_distinct_faculties_deduped_and_sorted(self):
        text = (
            "คณะวิศวกรรมศาสตร์ และ คณะวิทยาศาสตร์ ร่วมกันเสนอ "
            "โดยคณะวิศวกรรมศาสตร์ เป็นเจ้าภาพหลัก"
        )
        result = match_faculties(text, _DICT)
        assert result == sorted({"คณะวิศวกรรมศาสตร์", "คณะวิทยาศาสตร์"})

    def test_a_คณะ_occurrence_never_matches_a_วิทยาลัย_candidate(self):
        # garbled "คณะ..." text that happens to resemble a วิทยาลัย name
        # should not cross prefix groups even if the fuzzy ratio would pass
        text = "คณะวิศวกรรมสังคีต เสนอโครงการ"
        result = match_faculties(text, _DICT)
        assert "วิทยาลัยวิศวกรรมสังคีต" not in result

    def test_alias_reports_under_the_canonical_name(self):
        # a prior official name (pre-rename), not an OCR typo -- still needs
        # to resolve to the current canonical name
        text = "เรื่อง ขอความเห็นชอบการปรับปรุงหลักสูตร คณะสถาปัตยกรรมศาสตร์"
        result = match_faculties(text, _DICT)
        assert result == ["คณะสถาปัตยกรรม ศิลปะและการออกแบบ"]

    def test_current_name_still_matches_directly(self):
        text = "คณะสถาปัตยกรรม ศิลปะและการออกแบบ เสนอหลักสูตรใหม่"
        result = match_faculties(text, _DICT)
        assert result == ["คณะสถาปัตยกรรม ศิลปะและการออกแบบ"]

    def test_real_dictionary_loads_and_is_usable(self):
        # sanity check against the actual hand-confirmed dictionary, not the
        # small fixture above
        text = "คณะเทคโนโลยีสารสนเทศ ขอความเห็นชอบหลักสูตรปรับปรุง"
        assert match_faculties(text) == ["คณะเทคโนโลยีสารสนเทศ"]


class TestFacultyLoaderIntegration:
    def test_tags_metadata_with_faculties(self, tmp_path):
        d = tmp_path / "2569" / "ครั้งที่ 1"
        d.mkdir(parents=True)
        doc = d / "a.md"
        doc.write_text(
            "## Page 1\nคณะวิศวกรรมศาสตร์ เสนอขอความเห็นชอบหลักสูตรปรับปรุง",
            encoding="utf-8",
        )

        res = build_loader(StrategySpec(type="faculty")).load(str(doc))

        assert res.metadata["faculties"] == ["คณะวิศวกรรมศาสตร์"]

    def test_no_mention_gives_empty_list(self, tmp_path):
        d = tmp_path / "2569" / "ครั้งที่ 1"
        d.mkdir(parents=True)
        doc = d / "a.md"
        doc.write_text("## Page 1\nสภาสถาบันมีมติเห็นชอบตามที่เสนอ", encoding="utf-8")

        res = build_loader(StrategySpec(type="faculty")).load(str(doc))

        assert res.metadata["faculties"] == []
