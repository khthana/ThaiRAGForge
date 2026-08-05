"""ProgramLoader tags each Resolution with metadata['programs']: the
canonical program (หลักสูตร) names it mentions, matched against
data/entity_dictionaries/programs.json."""
from __future__ import annotations

from rag_lab.config import StrategySpec
from rag_lab.factory import build_loader
from rag_lab.loaders.program_loader import match_programs

_DICT = [
    {
        "canonical": "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้า",
        "prefix_type": "หลักสูตร",
        "degree": "วิศวกรรมศาสตรบัณฑิต",
        "field": "วิศวกรรมไฟฟ้า",
    },
    {
        "canonical": "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้าและคอมพิวเตอร์",
        "prefix_type": "หลักสูตร",
        "degree": "วิศวกรรมศาสตรบัณฑิต",
        "field": "วิศวกรรมไฟฟ้าและคอมพิวเตอร์",
    },
    {
        "canonical": "หลักสูตรวิทยาศาสตรบัณฑิต สาขาวิชาฟิสิกส์",
        "prefix_type": "หลักสูตร",
        "degree": "วิทยาศาสตรบัณฑิต",
        "field": "ฟิสิกส์",
    },
    {
        "canonical": "หลักสูตรบริหารธุรกิจบัณฑิต",
        "prefix_type": "หลักสูตร",
        "degree": "บริหารธุรกิจบัณฑิต",
        "field": None,
    },
]


class TestMatchPrograms:
    def test_exact_match_with_field(self):
        text = "หลักสูตรวิทยาศาสตรบัณฑิต สาขาวิชาฟิสิกส์ (การปรับปรุงแก้ไขหลักสูตร)"
        assert match_programs(text, _DICT) == ["หลักสูตรวิทยาศาสตรบัณฑิต สาขาวิชาฟิสิกส์"]

    def test_match_with_no_field_in_dictionary(self):
        text = "หลักสูตรบริหารธุรกิจบัณฑิต (หลักสูตรนานาชาติ)"
        assert match_programs(text, _DICT) == ["หลักสูตรบริหารธุรกิจบัณฑิต"]

    def test_no_match_for_unrelated_text(self):
        text = "สภาสถาบันมีมติเห็นชอบตามที่เสนอ"
        assert match_programs(text, _DICT) == []

    def test_longer_program_name_is_not_shadowed_by_a_shorter_prefix_match(self):
        # "วิศวกรรมไฟฟ้า" is a literal prefix of "วิศวกรรมไฟฟ้าและคอมพิวเตอร์"
        # -- both are real, distinct programs in the fixture dictionary. A
        # mention of the longer one must resolve to the longer canonical,
        # not the shorter one that happens to share a prefix.
        text = "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้าและคอมพิวเตอร์ (การปรับปรุงแก้ไขหลักสูตร)"
        result = match_programs(text, _DICT)
        assert result == ["หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้าและคอมพิวเตอร์"]

    def test_shorter_program_name_still_matches_on_its_own(self):
        text = "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้า (การปรับปรุงแก้ไขหลักสูตร)"
        result = match_programs(text, _DICT)
        assert result == ["หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้า"]

    def test_span_stops_at_newline_when_no_paren_follows(self):
        text = "หลักสูตรวิทยาศาสตรบัณฑิต สาขาวิชาฟิสิกส์\nข้อความถัดไปที่ไม่เกี่ยวข้อง เนื้อหาอื่น ๆ อีกมากมาย"
        assert match_programs(text, _DICT) == ["หลักสูตรวิทยาศาสตรบัณฑิต สาขาวิชาฟิสิกส์"]

    def test_finds_multiple_distinct_programs_deduped_and_sorted(self):
        text = (
            "หลักสูตรวิทยาศาสตรบัณฑิต สาขาวิชาฟิสิกส์ (หลักสูตรปรับปรุง) และ "
            "หลักสูตรบริหารธุรกิจบัณฑิต (หลักสูตรนานาชาติ)"
        )
        result = match_programs(text, _DICT)
        assert result == sorted(
            [
                "หลักสูตรวิทยาศาสตรบัณฑิต สาขาวิชาฟิสิกส์",
                "หลักสูตรบริหารธุรกิจบัณฑิต",
            ]
        )

    def test_real_dictionary_loads_and_is_usable(self):
        text = "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมโยธา (การปรับปรุงแก้ไขหลักสูตร)"
        assert match_programs(text) == ["หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมโยธา"]


class TestProgramLoaderIntegration:
    def test_tags_metadata_with_programs(self, tmp_path):
        d = tmp_path / "2569" / "ครั้งที่ 1"
        d.mkdir(parents=True)
        doc = d / "a.md"
        doc.write_text(
            "## Page 1\nหลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมโยธา (การปรับปรุงแก้ไขหลักสูตร)",
            encoding="utf-8",
        )

        res = build_loader(StrategySpec(type="program")).load(str(doc))

        assert res.metadata["programs"] == ["หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมโยธา"]

    def test_no_mention_gives_empty_list(self, tmp_path):
        d = tmp_path / "2569" / "ครั้งที่ 1"
        d.mkdir(parents=True)
        doc = d / "a.md"
        doc.write_text("## Page 1\nสภาสถาบันมีมติเห็นชอบตามที่เสนอ", encoding="utf-8")

        res = build_loader(StrategySpec(type="program")).load(str(doc))

        assert res.metadata["programs"] == []
