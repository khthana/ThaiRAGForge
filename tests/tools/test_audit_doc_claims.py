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


class TestCountExtraction:
    """D5's regexes. The shape is this project's other universal figure form,
    and D2 is structurally blind to all of it (it matches only 4 decimals)."""

    def test_finds_both_languages(self):
        assert adc.COUNT_PROSE.findall("0 of 23,156 live files") == [("0", "23,156")]
        assert adc.COUNT_PROSE.findall("70 จาก 84 คู่") == [("70", "84")]

    def test_commas_are_not_part_of_the_value(self):
        # The prose and the reports differ on the separator freely, so the
        # comparison has to be on values -- "23,156" must match "23156".
        assert adc._int("23,156") == adc._int("23156") == 23156

    def test_a_decimal_is_not_a_count(self):
        # Otherwise "0.6831 of 1.0" and version strings would enter D5's
        # denominator, which is D2's job and a different matching rule.
        assert adc.COUNT_PROSE.findall("0.6831 of 1.0000") == []
        assert adc.COUNT_SLASH.findall("v1.2/3.4") == []

    def test_the_slash_form_reads_a_table_cell(self):
        assert adc.COUNT_SLASH.findall("| 17/106 |") == [("17", "106")]

    def test_a_path_or_a_date_is_not_a_pair(self):
        assert adc.COUNT_SLASH.findall("2026-08-12 and 33/13/30/30") == []


class TestCountAllowlist:
    """The measured discrimination is 64% real vs 4-13% perturbed, so ~36% of
    *correct* figures land in the residue by construction -- which makes this
    allowlist load-bearing, and therefore the easiest place for D5 to go
    vacuous. Both directions are pinned, as for RETIRED_REPORTS."""

    def test_every_entry_is_a_provenance_record(self):
        entries = adc._allowlist("counts")
        assert entries, "D5's allowlist section is empty -- did it get renamed?"
        for e in entries:
            assert e.get("reason", "").strip(), e
            assert e.get("checked"), e
            assert (adc.REPO / e["doc"]).exists(), e["doc"]
            # Keyed on the exact figure string, so editing "0 of 239" to
            # "0 of 241" stops matching and re-flags rather than staying cleared.
            assert adc.COUNT_PROSE.fullmatch(str(e["figure"])), e["figure"]

    def test_no_entry_exempts_a_figures_own_perturbation(self):
        # An allowlist holding both "83 of 84" and "84 of 84" would clear the
        # wrong one too, which is the vacuity this check exists to prevent.
        keyed = {(e["doc"], str(e["figure"])) for e in adc._allowlist("counts")}
        for doc, fig in keyed:
            n, m = (adc._int(x) for x in adc.COUNT_PROSE.fullmatch(fig).groups())
            for pn, pm in [(n + 1, m), (n, m + 1), (n + 7, m)]:
                assert (doc, f"{pn} of {pm}") not in keyed, f"{doc}: {fig}"


class TestDatedIsNotInheritedByD5:
    def test_the_count_check_does_not_apply_the_dated_exemption(self):
        # Measured before shipping: DATED cleared 18 of the 26 D5 flags
        # *including the one genuine defect* ("0 of 240" where the report says
        # 239), because CLAUDE.md dates nearly every bullet. It is calibrated
        # for D2's denominator (1,298 figures), not D5's (72) -- an exemption
        # is only ever right for the check it was measured on.
        import inspect

        # The docstring explains the exclusion at length, so read the body only
        # -- everything after the closing triple quote.
        body = inspect.getsource(adc.audit_counts).split('"""')[2]
        assert "DATED" not in body
        assert "SUPERSEDED" in body


class TestGeneratorResolution:
    def test_a_declared_generator_wins_over_the_name_heuristic(self, tmp_path):
        # The retired snapshots are named after scripts that still exist, so the
        # declaration is what makes their provenance honest rather than guessed.
        r = tmp_path / "gold_chunker_compare_report.md"
        r.write_text("# x\n\nGenerated by `tools/eval/run_gold_chunker_eval.py`.\n", encoding="utf-8")
        assert adc._generator_for(r).name == "run_gold_chunker_eval.py"

    def test_a_report_naming_no_script_has_no_generator(self, tmp_path):
        r = tmp_path / "person_cross_cell_fix_review.md"
        r.write_text("# x\n\n> **Not an eval report.** commit `e1523b3`.\n", encoding="utf-8")
        assert adc._generator_for(r) is None


class TestRetiredReportExemption:
    """Both directions, because a retirement list is the easiest way to make
    D1a vacuous: every entry added is a report the check stops examining."""

    def test_every_entry_carries_a_written_reason(self):
        for name, reason in adc.RETIRED_REPORTS.items():
            assert reason.strip(), name

    def test_the_list_stays_small_relative_to_the_corpus_of_reports(self):
        # Not a style rule: D1a's denominator is "live reports", so a list that
        # grew to cover most of them would report 0-of-a-handful and mean nothing.
        live = [p for p in adc.RESULTS.glob("*.md") if p.name not in adc.RETIRED_REPORTS]
        assert len(adc.RETIRED_REPORTS) < len(live) / 2

    def test_no_current_report_is_exempted(self):
        # The reports every 9-way script and the paper summary actually read
        # must stay inside D1a, whatever else is retired.
        for name in [
            "hybrid_significance_test_9way.md", "bm25_vs_embedder_significance_test_9way.md",
            "embedder_significance_test_9way.md", "routing_eval.md", "rq4_score.md",
            "oracle_union_ceiling.md", "power_analysis.md",
        ]:
            assert name not in adc.RETIRED_REPORTS, name


class TestEvalInputs:
    """D4's edges are hand-curated, so a typo'd or moved path degrades the check
    to a permanent D4b WARN that nobody reads -- the same shape as a retirement
    list going vacuous."""

    def test_every_declared_input_and_report_exists(self):
        for src, reports in adc.EVAL_INPUTS.items():
            assert (adc.REPO / src).exists(), src
            for rname in reports:
                # Mirrors D4's own resolution rule: a separator means
                # repo-relative (the `docs/` artifacts), otherwise data/results.
                r = (adc.REPO / rname) if "/" in rname else (adc.RESULTS / rname)
                assert r.exists(), f"{rname} (declared under {src})"

    def test_both_report_locations_are_declared(self):
        # If every entry lived in data/results the "/" branch would be dead code
        # and a docs artifact could be added back in the broken form unnoticed.
        names = [r for v in adc.EVAL_INPUTS.values() for r in v]
        assert any("/" in n for n in names) and any("/" not in n for n in names)


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
