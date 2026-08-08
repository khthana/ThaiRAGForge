"""Pin the manifest-entry lookup used by the title-mispairing repair.

The risk is specific and was hit for real while building this script. Manifest
filenames in this corpus mix NBSP with ordinary spaces, so an exact copy of a
stem cannot be written by hand and the lookup has to collapse whitespace. But
the corpus also contains pairs of files whose names differ *only* by a double
space (`2569/ครั้งที่ 2` has two such pairs), and a collapsed lookup returns
whichever it meets first -- which would write one document's title onto another
document's entry. That is precisely the defect the script exists to repair, so
a lookup that can cause one is worse than no script at all.

The real loader (`rag_lab.loaders.common`) keys the manifest by *exact*
filename, so exact-first is also what keeps this script's view of a manifest
identical to the loader's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "corpus_prep"))
import fix_manifest_title_mispairings as fx  # noqa: E402

# the shape that caused the false collision: identical but for one space
A = {"file": "เรื่อง ขออนุมัติ.md", "title": "A"}
B = {"file": "เรื่อง  ขออนุมัติ.md", "title": "B"}


class TestFindEntry:
    def test_exact_match_wins_over_a_collapsed_twin(self):
        # both entries collapse to the same string; only the exact match is right
        assert fx.find_entry([A, B], B["file"])["title"] == "B"
        assert fx.find_entry([A, B], A["file"])["title"] == "A"

    def test_falls_back_to_collapsed_when_unique(self):
        # NBSP vs space: the caller cannot reproduce the stem byte-for-byte
        entry = {"file": "เรื่อง\xa0ขอความเห็นชอบ.md", "title": "T"}
        assert fx.find_entry([entry], "เรื่อง ขอความเห็นชอบ.md")["title"] == "T"

    def test_ambiguous_collapsed_match_is_an_error_not_a_coin_flip(self):
        with pytest.raises(SystemExit) as e:
            fx.find_entry([A, B], "เรื่อง   ขออนุมัติ.md")  # matches neither exactly
        assert "ambiguous" in str(e.value)

    def test_missing_file_fails_loudly(self):
        with pytest.raises(SystemExit) as e:
            fx.find_entry([A], "ไม่มีไฟล์นี้.md")
        assert "not in manifest" in str(e.value)


class TestFixTable:
    def test_every_fix_is_a_real_change(self):
        # a Fix whose expected title equals its replacement is a no-op that would
        # silently report success
        for fix in fx.FIXES:
            assert fix.expected != fix.title, fix.file

    def test_the_2565_8_pair_is_a_mutual_swap(self):
        # the property that forces atomic application downstream: each half's
        # replacement is the other half's current title
        pair = [f for f in fx.FIXES if f.folder == "2565/ครั้งที่ 8"]
        assert len(pair) == 2
        assert pair[0].title == pair[1].expected
        assert pair[1].title == pair[0].expected
