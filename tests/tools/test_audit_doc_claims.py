"""Pure-logic tests for audit_doc_claims.py's exemption rules.

The risk this file exists to guard is specific: D2 only stays useful while its
two exemptions (a figure cited as superseded, a figure in a dated snapshot) are
narrower than the check itself. Every exemption added to make a real doc pass
moves the script one step toward a vacuous PASS -- the same failure C4 hit in
`audit_pipeline_invariants.py` when its subject matter moved off-repo. So the
tests below pin both directions: an exempted phrasing is exempted, and a plain
wrong number next to one is still caught.

No file I/O against the real docs -- those change every session; the regexes
are the stable surface.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "eval"))
import audit_doc_claims as adc  # noqa: E402


class TestNumberExtraction:
    def test_finds_four_decimal_figures(self):
        assert adc.NUM.findall("recall@10 0.6831 vs 0.6631") == ["0.6831", "0.6631"]

    def test_ignores_dates_and_short_decimals(self):
        # 2026-08-08 and "alpha=0.50" must not be read as reportable figures:
        # both are everywhere in these docs and neither appears in a report.
        assert adc.NUM.findall("on 2026-08-08 at alpha=0.50 and k=1.5") == []

    def test_ignores_longer_precision(self):
        assert adc.NUM.findall("0.68312") == []


class TestSupersessionExemption:
    def test_recognises_a_retired_figure(self):
        for phrase in [
            "the long-quoted 0.9291 predates the rebuild",
            "do not cite the 0.6935 figure",
            "recall@10 rose 0.3908 to 0.4930",
            "dropped from 0.7048 to 0.6152",
            "the old all-73 mean of 0.8922",
            "0.6364 -> 0.5688",
        ]:
            assert adc.SUPERSEDED.search(phrase), phrase

    def test_does_not_exempt_a_plain_claim(self):
        # The whole point: an ordinary sentence stating a current number must
        # not read as a supersession note.
        assert not adc.SUPERSEDED.search("hard routing scores 0.6831 recall@10")


class TestDatedSnapshotExemption:
    def test_recognises_a_labelled_snapshot(self):
        assert adc.DATED.search("**Aggregate — refreshed 2026-07-25 against the rebuilt indices**")
        assert adc.DATED.search("caveat added 2026-07-29")

    def test_a_bare_date_is_not_a_snapshot_label(self):
        # Otherwise every block in a doc that timestamps its edits would be
        # exempt, which is most of them -- that is the vacuity failure.
        assert not adc.DATED.search("closed 2026-08-08: 5 routes, 0/106 unrouted")


class TestSignificanceWording:
    def test_verdict_words(self):
        assert adc.NOT_SIG.search("both ns")
        assert adc.NOT_SIG.search("no pair significant")
        assert adc.NOT_SIG.search("no longer significant")
        assert adc.IS_SIG.search("significantly beats")

    def test_p_value_forms_used_in_these_docs(self):
        for text, want in [
            ("p=0.0462", "0.0462"),
            ("Holm-adj p=0.0252", "0.0252"),
            ("Holm 0.0144", "0.0144"),
        ]:
            assert adc.P_VALUE.search(text).group(1) == want, text
