"""ProgramLoader tags each Resolution with metadata['programs']: the
canonical program (หลักสูตร) names it mentions, matched against
data/entity_dictionaries/programs.json."""
from __future__ import annotations

from rag_lab.config import StrategySpec
from rag_lab.factory import build_loader
from rag_lab.loaders.program_loader import (
    _field_agrees,
    _head_contradicted,
    _head_of,
    _subject_contradicted,
    degree_level,
    match_programs,
)

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


class TestDegreeLevel:
    """The ladder itself, before anything reads it."""

    def test_reads_the_longest_degree_first(self):
        # ดุษฎีบัณฑิต and มหาบัณฑิต both END in บัณฑิต, so a shortest-first scan
        # would silently call every doctorate a bachelor's -- and the guard below
        # would then agree with the wrong canonical instead of rejecting it.
        assert degree_level("หลักสูตรวิศวกรรมศาสตรดุษฎีบัณฑิต") == "ดุษฎีบัณฑิต"
        assert degree_level("หลักสูตรวิทยาศาสตรมหาบัณฑิต") == "มหาบัณฑิต"
        assert degree_level("หลักสูตรวิทยาศาสตรบัณฑิต") == "บัณฑิต"
        assert degree_level("หลักสูตรอนุปริญญาสาขาวิชาใดสาขาวิชาหนึ่ง") == "อนุปริญญา"

    def test_no_degree_token_is_undecidable_not_a_level(self):
        # None means "this string shows no level", never "no degree" -- callers
        # must not read it as a mismatch ([[feedback_undefined_is_not_zero]]).
        assert degree_level("หลักสูตรฝึกอบรมเพื่อสะสมหน่วยกิต") is None

    def test_ocr_wrapping_inside_the_degree_token_still_reads(self):
        # The corpus wraps long programme names mid-token, and a guard that read
        # these as "no degree" would go blind exactly where OCR is messiest.
        assert degree_level("หลักสูตรครุศาสตร์อุตสาหกรรม บัณฑิต") == "บัณฑิต"
        assert degree_level("หลักสูตรวิทยาศาสตร<br/>มหาบัณฑิต") == "มหาบัณฑิต"


class TestDegreeFiltersTheCandidateSet:
    """A contradicted degree re-selects among the candidates the text admits, and
    tags nothing when none of them fits.

    The "matches nothing" exit docs/program-matcher-absorption.md measured to be
    missing: 23.1% of accepted matches absorb a different name, and 35.7% of
    those swap only the degree level. A plain reject was written first and walked
    against the corpus: of the 752 mentions where the guard fires, 354 had a
    same-degree, same-subject candidate already in the dictionary -- so rejecting
    would throw away a tag that a degree-aware *selection* keeps. Both halves of
    that split are pinned below.
    """

    # A candidate that AGREES with the span's degree but names a different
    # subject -- present so the no-fallthrough test below has something a
    # fallthrough could wrongly land on.
    _WITH_MASTERS = _DICT + [
        {
            "canonical": "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาวิศวกรรมโยธา",
            "prefix_type": "หลักสูตร",
            "degree": "วิศวกรรมศาสตรมหาบัณฑิต",
            "field": "วิศวกรรมโยธา",
        },
    ]

    # The same dictionary plus the master's the text below actually names, i.e.
    # the 354-mention `RESCUABLE` case: the right tag was there all along and the
    # shipped matcher ranked the bachelor's above it.
    _WITH_THE_RIGHT_MASTERS = _WITH_MASTERS + [
        {
            "canonical": "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้า",
            "prefix_type": "หลักสูตร",
            "degree": "วิศวกรรมศาสตรมหาบัณฑิต",
            "field": "วิศวกรรมไฟฟ้า",
        },
    ]

    # A same-degree candidate with no สาขาวิชา at all, so its subject is
    # *undecidable* rather than wrong ([[feedback_undefined_is_not_zero]]).
    _WITH_FIELDLESS_MASTERS = _DICT + [
        {
            "canonical": "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต",
            "prefix_type": "หลักสูตร",
            "degree": "วิศวกรรมศาสตรมหาบัณฑิต",
            "field": None,
        },
    ]

    def test_a_masters_mention_is_not_tagged_as_the_bachelors(self):
        # The ratio cannot do this alone: บัณฑิต -> มหาบัณฑิต is one token in a
        # ~50-char name, so the bachelor's scores ~0.96 here and wins outright.
        text = "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้า (ปรับปรุง)"
        assert match_programs(text, _DICT) == []

    def test_the_right_degree_is_selected_when_the_dictionary_holds_it(self):
        # 354 of the 752 firings are this: nothing was missing from the
        # dictionary, the ranking was wrong. Same text as the two tests either
        # side of it -- only the dictionary differs.
        text = "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้า (ปรับปรุง)"
        assert match_programs(text, self._WITH_THE_RIGHT_MASTERS) == [
            "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้า"
        ]

    def test_a_same_degree_candidate_with_a_different_subject_is_not_selected(self):
        # Why the subject test exists, decided by measurement rather than taste.
        # A degree-blind fallthrough was walked against the corpus and re-tagged
        # 11 mentions, only 3 correctly: ...ดุษฎีบัณฑิต ...วิศวกรรมไฟฟ้า became
        # ...ดุษฎีบัณฑิต ...วิศวกรรมโยธา, which is this fixture. 213 of the 752
        # firings still land here, so filtering on degree alone would re-import
        # the failure it was meant to remove: it trades a wrong degree for a
        # wrong subject, and a query names the subject.
        text = "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้า (ปรับปรุง)"
        assert match_programs(text, self._WITH_MASTERS) == []

    def test_a_fieldless_candidate_does_not_absorb_a_named_subject(self):
        # 129 firings are undecidable this way. No subject to compare is not
        # agreement -- this gates a re-selection, so no evidence must mean no
        # rescue, or the guard would hand every mismatched mention to whichever
        # candidate is vaguest.
        text = "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้า (ปรับปรุง)"
        assert match_programs(text, self._WITH_FIELDLESS_MASTERS) == []

    def test_the_matching_level_is_still_matched(self):
        # The guard must not become a ban on master's programmes.
        text = "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาวิศวกรรมโยธา (ปรับปรุง)"
        assert match_programs(text, self._WITH_MASTERS) == [
            "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาวิศวกรรมโยธา"
        ]

    def test_a_garbled_degree_in_the_text_changes_nothing(self):
        # Positive evidence from both sides or nothing happens: OCR that eats a
        # vowel out of บัณฑิต leaves the span undecidable, and an undecidable
        # span must not start rejecting matches the matcher used to make.
        text = "หลักสูตรวิทยาศาสตรบณฑิต สาขาวิชาฟิสิกส์ (การปรับปรุงแก้ไขหลักสูตร)"
        assert match_programs(text, _DICT) == ["หลักสูตรวิทยาศาสตรบัณฑิต สาขาวิชาฟิสิกส์"]


class TestHeadNounAndSubjectAreTestedSeparately:
    """The half the degree filter is structurally unable to see.

    `หลักสูตรทันตแพทยศาสตรบัณฑิต` is absorbed by `หลักสูตรแพทยศาสตรบัณฑิต` at the
    *same* degree, so the guard above never fires on it -- it is the instance
    this whole audit was found through. The mechanism is dilution by
    concatenation: the ratio runs over head noun + subject joined, so a
    disagreement confined to one half is averaged away by the other agreeing
    (`ทันต-` scores 0.88). Testing the halves separately removes the dilution.

    The two tests are deliberately DIFFERENT, and the asymmetry is structural:
    `_bounded_span_for` sizes the window from the match position, so the head
    noun is always covered in full while the subject sits at the tail and is
    routinely cut. Truncation is therefore forgiven in the subject and must not
    be forgiven in the head.
    """

    _WITH_DENTAL = _DICT + [
        {
            "canonical": "หลักสูตรแพทยศาสตรบัณฑิต",
            "prefix_type": "หลักสูตร",
            "degree": "แพทยศาสตรบัณฑิต",
            "field": None,
        },
    ]
    _AND_THE_RIGHT_ONE = _WITH_DENTAL + [
        {
            "canonical": "หลักสูตรทันตแพทยศาสตรบัณฑิต",
            "prefix_type": "หลักสูตร",
            "degree": "ทันตแพทยศาสตรบัณฑิต",
            "field": None,
        },
    ]

    def test_head_is_delimited_at_both_ends(self):
        # Left edge: the LAST anchor, because the scanner re-anchors on a heading
        # (`๑) หลักสูตร...`) and reading from the first one reported 149 mentions
        # as contradictions. Right edge: the end of the degree token, because a
        # canonical with no สาขาวิชา lets the window overrun into the next word.
        assert _head_of("๑) หลักสูตรแพทยศาสตรบัณฑิต สาขาวิชาก") == "แพทยศาสตรบัณฑิต"
        assert _head_of("หลักสูตรบริหารธุรกิจบัณฑิตหล") == "บริหารธุรกิจบัณฑิต"

    def test_markup_and_thai_numerals_are_invisible_to_the_head(self):
        # `<br/>`, `</td><td>`, `**` and `๑)` were every false positive the head
        # test had. Dropping everything but Thai letters removes them in one
        # pass -- `br`/`td` are Latin, and Thai digits sit at U+0E50+.
        assert (_head_of("</td><td>หลักสูตรวิศวกรรมศาสตรบัณฑิต<br/>สาขาวิชาโยธา")
                == "วิศวกรรมศาสตรบัณฑิต")

    def test_a_longer_degree_name_contradicts_the_head(self):
        assert _head_contradicted("หลักสูตรแพทยศาสตรบัณฑิต", "หลักสูตรทันตแพทยศาสตรบัณฑิต")
        assert _head_contradicted("หลักสูตรแพทยศาสตรบัณฑิต", "หลักสูตรพยาบาลศาสตรบัณฑิต")

    def test_the_same_head_re_anchored_or_overrun_does_not_contradict(self):
        # Both directions matter: a head test that fires on the scanner finding
        # the same name twice would drop tags the matcher was right about.
        assert not _head_contradicted(
            "หลักสูตรแพทยศาสตรบัณฑิต", "๑) หลักสูตรแพทยศาสตรบัณฑิต สาขาวิชาก")
        assert not _head_contradicted(
            "หลักสูตรแพทยศาสตรบัณฑิต", "หลักสูตรแพทยศาสตรบัณฑิตหล")

    def test_a_truncated_subject_is_the_same_subject(self):
        # 120 of the first rule's 604 firings were this: the window cut the name
        # short and a plain ratio read the cut as disagreement (0.78). Contiguous
        # coverage of the shorter side reads it as 1.00, which is what it is.
        assert not _subject_contradicted(
            "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมดนตรีและสื่อประสม",
            "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมดนตรีและ",
        )

    def test_a_different_subject_contradicts(self):
        assert _subject_contradicted(
            "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมโยธา",
            "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้า",
        )

    def test_a_missing_subject_on_either_side_is_not_a_contradiction(self):
        # [[feedback_undefined_is_not_zero]], pinned in both directions because
        # this gates a drop: no สาขาวิชา must not start rejecting matches.
        assert not _subject_contradicted(
            "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมโยธา", "หลักสูตรวิศวกรรมศาสตรบัณฑิต")
        assert not _subject_contradicted(
            "หลักสูตรวิศวกรรมศาสตรบัณฑิต", "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมโยธา")

    def test_the_motivating_absorption_no_longer_happens(self):
        text = "หลักสูตรทันตแพทยศาสตรบัณฑิต (หลักสูตรปรับปรุง พ.ศ. ๒๕๖๕)"
        assert match_programs(text, self._WITH_DENTAL) == []

    def test_the_right_programme_is_selected_when_the_dictionary_holds_it(self):
        # Same text, one more dictionary entry -- the drop is a consequence of
        # the corpus name being absent, not of the guard banning the shape.
        text = "หลักสูตรทันตแพทยศาสตรบัณฑิต (หลักสูตรปรับปรุง พ.ศ. ๒๕๖๕)"
        assert match_programs(text, self._AND_THE_RIGHT_ONE) == [
            "หลักสูตรทันตแพทยศาสตรบัณฑิต"
        ]

    def test_the_absorbing_canonical_still_matches_its_own_text(self):
        # The guard must not become a ban on แพทยศาสตรบัณฑิต.
        text = "หลักสูตรแพทยศาสตรบัณฑิต (หลักสูตรปรับปรุง พ.ศ. ๒๕๖๕)"
        assert match_programs(text, self._AND_THE_RIGHT_ONE) == ["หลักสูตรแพทยศาสตรบัณฑิต"]

    def test_a_one_character_variant_is_not_a_contradiction(self):
        # Coverage is blind to a difference spread through the string rather than
        # sitting at one end: this real dictionary/OCR pair covers 0.55 while
        # `_field_agrees` -- the project's own settled test for the same relation
        # -- puts it at 0.95. One pair must not be agreeing and contradicting at
        # once, so a drop needs both tests against it (133 of 569 subject drops
        # sat in that band).
        assert not _subject_contradicted(
            "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาวิศวกรรมเล็กทรอนิกส์",
            "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาวิศวกรรมอิเล็กทรอนิกส์",
        )
        assert _field_agrees(
            "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาวิศวกรรมเล็กทรอนิกส์",
            "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาวิศวกรรมอิเล็กทรอนิกส์",
        )


class TestACrossSubjectRescueMayNotCrossTheDegree:
    """The two guards must not undo each other.

    The cross-subject branch is reached precisely when the *winner's* degree is
    uncontradicted -- which says nothing about the runner-up's, so without an
    explicit condition the re-selection can hand the mention to a candidate at a
    degree the text contradicts, i.e. exactly what the settled degree rule
    exists to prevent. It is not hypothetical: the first full walk rescued 6
    mentions across บัณฑิต/มหาบัณฑิต, and the audit's S9 failed on them.

    The text is a real corpus mention (`2564/ครั้งที่ 11`), kept verbatim
    because no clean synthetic name reproduces it -- a candidate that agrees on
    the subject is normally *closer* to the text than the contradicted winner
    and would simply win instead. Here `<br/>` inflates every candidate's window
    differently, which is what lets the wrong one lead.
    """

    _CIVIL = {
        "canonical": "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมโยธา",
        "prefix_type": "หลักสูตร",
        "degree": "วิศวกรรมศาสตรบัณฑิต",
        "field": "วิศวกรรมโยธา",
    }
    _FOOD_MASTERS = {
        "canonical": "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาวิศวกรรมอาหาร",
        "prefix_type": "หลักสูตร",
        "degree": "วิศวกรรมศาสตรมหาบัณฑิต",
        "field": "วิศวกรรมอาหาร",
    }
    _DICTIONARY = [_CIVIL, _FOOD_MASTERS]

    def test_the_rescue_refuses_a_candidate_at_another_degree(self):
        # The document names the BACHELOR's food-engineering programme, broken by
        # markup. The winner (civil) is rightly contradicted on subject; the only
        # candidate whose subject the truncated span supports is the MASTER's,
        # and taking it would be a wrong tag, not a rescue.
        text = ("หลักสูตรวิศวกรรม<br/>ศาสตรบัณฑิต สาขาวิชาวิศวกรรม<br/>อาหาร "
                "ไปยังสำนักงานปลัดกระทรวง")
        assert match_programs(text, self._DICTIONARY) == []

    def test_the_same_candidate_is_still_selected_at_its_own_degree(self):
        # The condition is a filter on the text's evidence, not a ban on the
        # master's programme -- pinned in the positive direction too.
        text = "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาวิศวกรรมอาหาร (หลักสูตรปรับปรุง)"
        assert match_programs(text, self._DICTIONARY) == [
            "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาวิศวกรรมอาหาร"
        ]


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
