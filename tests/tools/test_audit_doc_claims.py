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

import pytest

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


class TestCountExtraction:
    """D5's extractor, widened twice on 2026-08-20 after D6 caught the holes.

    Both widenings were measured against the 12 real docs before being applied
    (+4 and +1 figure respectively), because a rule that admits far more is a
    different check, not a fixed one. The decimal guard is what must NOT loosen:
    a real decimal on either side has to stay rejected, or D5 starts inventing
    pairs out of "0.8856" and "1,046".
    """

    def test_a_plain_count_is_extracted(self):
        assert adc.COUNT_PROSE.findall("71 of 91 pairs") == [("71", "91")]

    def test_the_thai_form_is_extracted(self):
        assert adc.COUNT_PROSE.findall("70 จาก 84") == [("70", "84")]

    def test_a_count_ending_a_sentence_is_extracted(self):
        r"""`(?![\d.])` also rejected a trailing full stop, which hid a live
        claim ("agrees on 105 of 115.") and left its allowlist entry exempting
        nothing -- invisible until D6 asked whether entries still match."""
        assert adc.COUNT_PROSE.findall("agrees on 105 of 115.") == [("105", "115")]

    def test_a_decimal_denominator_is_still_refused(self):
        assert adc.COUNT_PROSE.findall("5 of 115.3") == []

    def test_a_decimal_numerator_is_still_refused(self):
        assert adc.COUNT_PROSE.findall("115.3 of 200") == []

    def test_a_longer_number_is_still_refused_on_the_right(self):
        assert adc.COUNT_PROSE.findall("5 of 1153") == [("5", "1153")]
        assert adc.COUNT_PROSE.findall("5 of 115 3") == [("5", "115")]

    def test_emphasis_between_the_halves_no_longer_hides_a_count(self):
        r"""`union **873** of 1,046` broke the `\s+of\s+` join."""
        assert adc.COUNT_PROSE.findall("union **873** of 1,046") == []
        assert adc.COUNT_PROSE.findall(
            adc._unemph("union **873** of 1,046")) == [("873", "1,046")]

    def test_html_bold_is_stripped_too(self):
        assert adc.COUNT_PROSE.findall(
            adc._unemph("<b>206</b> of 435")) == [("206", "435")]

    def test_unemph_leaves_the_numbers_alone(self):
        assert adc._unemph("0.8856 and 1,046") == "0.8856 and 1,046"


class TestAllowlistLiveness:
    """D6: an exemption that matches nothing is how an allowlist goes vacuous.

    The check must ask what D2/D5 EXTRACT, not whether the raw text contains the
    string -- its first version compared against file text and called
    `873 of 1,046` dead while the doc plainly contained it (as `**873** of
    1,046`, which the extractor could not see). Both answers are "dead"; only
    the extraction-based one says why it matters.
    """

    def test_every_numbers_entry_names_an_audited_doc(self):
        docs = {str(d).replace("\\", "/") for d in adc.DOCS}
        for e in adc._allowlist("numbers"):
            assert e["doc"] in docs, e

    def test_every_counts_entry_names_an_audited_doc(self):
        docs = {str(d).replace("\\", "/") for d in adc.DOCS}
        for e in adc._allowlist("counts"):
            assert e["doc"] in docs, e

    def test_every_entry_carries_a_reason_and_a_date(self):
        for section in ("numbers", "counts"):
            for e in adc._allowlist(section):
                assert e.get("reason", "").strip(), e
                assert e.get("checked"), e

    def test_d6_is_a_warn_not_a_fail(self):
        """An entry may legitimately outlive one edit of a sentence; a gate
        nobody can clear is a gate nobody reads (the D1b rule)."""
        before = len(adc.findings)
        adc.audit_allowlist_liveness()
        check, status, _ = adc.findings[before]
        assert check.startswith("D6")
        assert status in {"PASS", "WARN"}


class TestInputsAllowlistIsKeyedOnContent:
    """D4's `inputs` exemptions, and why they are not keyed on the pair alone.

    `EVAL_INPUTS` holds edges like `program_loader.py -> relation-graph.md`
    that exist *because* a matcher repair moved a report without touching its
    generator. A pair-keyed exemption clears such an edge forever, which is the
    "an exemption list is the easiest way to make a check vacuous" failure D1c
    was written for, one section down. So an entry may carry `src_sha` and is
    honoured only while the source still hashes to it.

    Exercised in BOTH directions: the passing direction alone would be
    satisfied by a `_inputs_cleared` that ignored `src_sha` entirely.
    """

    def test_every_inputs_entry_names_a_declared_edge(self):
        for e in adc._allowlist("inputs"):
            assert e["src"] in adc.EVAL_INPUTS, e
            assert e["report"] in adc.EVAL_INPUTS[e["src"]], e

    def test_every_inputs_entry_carries_a_reason_and_a_date(self):
        for e in adc._allowlist("inputs"):
            assert e.get("reason", "").strip(), e
            assert e.get("checked"), e

    def test_a_matching_sha_clears_the_pair(self, monkeypatch):
        entry = {"src": "a.py", "report": "r.md", "src_sha": "cafe", "checked": "x",
                 "reason": "y"}
        monkeypatch.setattr(adc, "_allowlist", lambda s: [entry] if s == "inputs" else [])
        monkeypatch.setattr(adc, "_sha", lambda p: "cafe")
        monkeypatch.setattr(Path, "exists", lambda self: True)
        cleared, dead = adc._inputs_cleared()
        assert cleared == {("a.py", "r.md")} and dead == []

    def test_a_STALE_sha_does_not_clear_and_is_reported_dead(self, monkeypatch):
        entry = {"src": "a.py", "report": "r.md", "src_sha": "cafe", "checked": "x",
                 "reason": "y"}
        monkeypatch.setattr(adc, "_allowlist", lambda s: [entry] if s == "inputs" else [])
        monkeypatch.setattr(adc, "_sha", lambda p: "f00d")
        monkeypatch.setattr(Path, "exists", lambda self: True)
        cleared, dead = adc._inputs_cleared()
        assert cleared == set(), "a moved source must re-flag, not inherit its clearance"
        assert len(dead) == 1 and "f00d" in dead[0] and "cafe" in dead[0]

    def test_a_legacy_entry_without_a_sha_still_clears(self, monkeypatch):
        """`src_sha` is optional on purpose: backfilling one today would bless a
        source state nobody verified, which is the opposite of its point."""
        entry = {"src": "a.py", "report": "r.md", "checked": "x", "reason": "y"}
        monkeypatch.setattr(adc, "_allowlist", lambda s: [entry] if s == "inputs" else [])
        cleared, dead = adc._inputs_cleared()
        assert cleared == {("a.py", "r.md")} and dead == []

    def test_the_program_loader_edges_are_not_exempted_at_all(self):
        """The strongest state for these three is no exemption, and that is the
        state they are in.

        They were cleared by `src_sha` on 2026-08-20 (a purely additive edit),
        and on 2026-08-21 a comment-only edit re-flagged all three -- exactly as
        intended. The discharge was to re-render each report and show the output
        unchanged (`program-matcher-absorption.md` and `program-tag-regeneration.md`
        byte-identical, `relation-graph.md` identical but for its timestamp
        line), which leaves the pairs passing on their own merit, so the
        exemptions were deleted rather than re-stamped. Re-adding one would
        pre-emptively disarm the very edge that exists because a matcher repair
        twice moved these reports without touching their generators."""
        exempted = {(e["src"], e["report"]) for e in adc._allowlist("inputs")}
        for report in ("docs/relation-graph.md", "docs/program-matcher-absorption.md",
                       "docs/program-tag-regeneration.md"):
            assert ("src/rag_lab/loaders/program_loader.py", report) not in exempted

    def test_any_program_loader_exemption_must_be_content_keyed(self):
        """If one is ever added back, it may not be a standing one. Vacuous
        today by construction -- the rule above is what makes it so -- and it
        bites the moment someone writes a pair-keyed entry for this source."""
        for e in adc._allowlist("inputs"):
            if e["src"] == "src/rag_lab/loaders/program_loader.py":
                assert e.get("src_sha"), (
                    f"{e['report']} is exempted without a src_sha, which disarms "
                    "the edge permanently"
                )


class TestUnitFigureExtraction:
    """D7's trigger pattern. The unit set is evidence, not taste."""

    def test_finds_latency_and_throughput(self):
        got = adc.UNIT_FIGURE.findall("p50 2,058.9 ms and 9.81 q/s at C=5")
        assert got == [("2,058.9", "ms"), ("9.81", "q/s")]

    def test_matches_with_or_without_a_space(self):
        assert adc.UNIT_FIGURE.findall("p50 1167.4ms") == [("1167.4", "ms")]

    def test_excludes_the_units_that_measured_badly(self):
        # Per-unit discrimination over the audited docs decided this, and the
        # table is in the source. `MB` (n+7 clears 52%), `%` (30%) and `x` (53%)
        # clear a wrong number too often to be checks; `s` (real 49%) and `GB`
        # (real 0%, because prose rounds to GB while reports state MB) would go
        # red on correct writing, which is equally disqualifying.
        text = "3.2 GB and 136.3 % and 3.87 x and 7.85 s and 244 MB"
        assert adc.UNIT_FIGURE.findall(text) == []

    def test_does_not_match_a_unit_glued_to_a_word(self):
        # "300 msec", "12 q/second" are not this project's notation.
        assert adc.UNIT_FIGURE.findall("300 msec") == []
        assert adc.UNIT_FIGURE.findall("12 q/second") == []

    def test_significant_digits_ignores_separators_and_leading_zeros(self):
        assert adc._significant("1,167.4") == 5
        assert adc._significant("0.6") == 1
        assert adc._significant("475") == 3

    def test_the_significance_floor_is_three(self):
        # "50 ms" and "2 s" collide with almost any report by coincidence; the
        # floor is what keeps the check from being satisfied by chance.
        assert adc._MIN_SIGNIFICANT == 3
        assert adc._significant("50") < adc._MIN_SIGNIFICANT
        assert adc._significant("475.6") >= adc._MIN_SIGNIFICANT


class TestD7InheritsNeitherOfD2sExemptions:
    """The decisive design fact, and it was measured rather than assumed.

    Over the audited docs, `SUPERSEDED` clears 44% of D7's residue and `DATED`
    42% -- and `SUPERSEDED` clears the one true positive D7 was built on, the
    reranker-latency line in `paper-results-summary.md` that still quoted a
    73-query run. Identical to the trap D5 documents: an exemption is only ever
    right for the check it was calibrated on.
    """

    def test_the_unit_check_applies_neither_regex(self):
        import inspect

        body = inspect.getsource(adc.audit_unit_figures).split('"""')[2]
        assert "SUPERSEDED" not in body
        assert "DATED" not in body

    def test_a_superseded_marker_does_not_clear_a_unit_figure(self, tmp_path, monkeypatch):
        doc = tmp_path / "prose.md"
        doc.write_text("The old p50 was 9999.9 ms, now superseded.\n", encoding="utf-8")
        monkeypatch.setattr(adc, "DOCS", [Path("prose.md")])
        monkeypatch.setattr(adc, "REPO", tmp_path)
        monkeypatch.setattr(adc, "_artifact_numbers", lambda: {"1.0"})
        monkeypatch.setattr(adc, "_artifacts", lambda: [])
        monkeypatch.setattr(adc, "_allowlist", lambda section: [])
        adc.findings.clear()
        adc.audit_unit_figures(show_all=True)
        assert adc.findings[-1][1] == "WARN", "an untraceable ms figure must be reported"
        assert "1 untraceable" in adc.findings[-1][2]


class TestD7MatchingIsExactNotRounded:
    """A rounding tolerance was built, measured and REJECTED.

    Accepting any report value equal to the prose figure when rounded or
    truncated to the prose's own precision moves real traceability 70% -> 76%
    and perturbation clearance (n+1) 8% -> 23%. Three times the clearance for
    six points of coverage is the trade D5 refused when it rejected V2/V3.
    """

    def _run(self, tmp_path, monkeypatch, prose, hay):
        doc = tmp_path / "prose.md"
        doc.write_text(prose, encoding="utf-8")
        monkeypatch.setattr(adc, "DOCS", [Path("prose.md")])
        monkeypatch.setattr(adc, "REPO", tmp_path)
        monkeypatch.setattr(adc, "_artifact_numbers", lambda: hay)
        monkeypatch.setattr(adc, "_artifacts", lambda: [])
        monkeypatch.setattr(adc, "_allowlist", lambda section: [])
        adc.findings.clear()
        adc.audit_unit_figures(show_all=True)
        return adc.findings[-1]

    def test_a_rounded_figure_is_still_flagged(self, tmp_path, monkeypatch):
        f = self._run(tmp_path, monkeypatch, "a 475 ms routed query\n", {"475.6"})
        assert f[1] == "WARN" and "1 untraceable" in f[2]

    def test_an_exact_figure_clears(self, tmp_path, monkeypatch):
        f = self._run(tmp_path, monkeypatch, "a 475.6 ms routed query\n", {"475.6"})
        assert f[1] == "PASS" and "0 untraceable" in f[2]

    def test_separators_are_normalised_on_both_sides(self, tmp_path, monkeypatch):
        f = self._run(tmp_path, monkeypatch, "p50 2,058.9 ms\n", {"2058.9"})
        assert f[1] == "PASS"


class TestUnitsAllowlist:
    def test_an_entry_clears_only_its_exact_figure(self, tmp_path, monkeypatch):
        doc = tmp_path / "prose.md"
        doc.write_text("p50 1191 ms and p95 1522 ms\n", encoding="utf-8")
        monkeypatch.setattr(adc, "DOCS", [Path("prose.md")])
        monkeypatch.setattr(adc, "REPO", tmp_path)
        monkeypatch.setattr(adc, "_artifact_numbers", lambda: set())
        monkeypatch.setattr(adc, "_artifacts", lambda: [])
        monkeypatch.setattr(
            adc, "_allowlist",
            lambda section: [{"doc": "prose.md", "figure": "1191 ms"}] if section == "units" else [],
        )
        adc.findings.clear()
        adc.audit_unit_figures(show_all=True)
        # The allowlisted one clears; its neighbour on the same line does not.
        assert "1 untraceable" in adc.findings[-1][2]
        assert "1 allowlisted" in adc.findings[-1][2]

    def test_the_live_units_section_is_audited_by_d6(self):
        # D6 must see `units` or a dead entry there is invisible -- the same
        # hole D1c closes for RETIRED_REPORTS.
        import inspect

        src = inspect.getsource(adc.audit_allowlist_liveness)
        assert '"units"' in src

    def test_no_live_units_entry_names_a_file_outside_what_D7_reads(self):
        """D7 reads DOCS *and* Python docstrings since 2026-08-23, so a units
        entry may name either -- but nothing else. An entry naming a file the
        check never opens exempts nothing and is the easiest way to make an
        allowlist look thorough while covering a file no check visits."""
        docs = {str(d).replace("\\", "/") for d in adc.DOCS}
        sources = {r for r, _ in adc._source_docstrings()}
        for e in adc._allowlist("units"):
            assert e["doc"] in docs or e["doc"] in sources, (
                f"{e['doc']} is exempted but not audited")

    def test_every_live_units_entry_carries_a_reason_and_a_date(self):
        for e in adc._allowlist("units"):
            assert e.get("reason", "").strip(), e
            assert e.get("checked"), e


# --------------------------------------------------------------------- D8 ---
class TestD8BlockSplitting:
    """The block is the unit, and the split is what makes D8 non-vacuous.

    CLAUDE.md writes its bullets with no blank line between them. Split on blank
    lines alone and the whole Conventions list is ONE block -- every figure in
    the file in one bag, so a superseded value always has a current one somewhere
    beside it and the check passes on everything. That is the haystack-too-big
    failure D2's own design rule warns about, one level down.
    """

    def test_a_top_level_bullet_starts_a_new_block(self):
        text = "- first bullet 0.1111\n  continued\n- second bullet 0.2222\n"
        got = adc._prose_blocks(text)
        assert len(got) == 2
        assert "0.1111" in got[0][1] and "0.1111" not in got[1][1]

    def test_claude_md_does_not_collapse_to_one_block(self):
        blocks = adc._prose_blocks((adc.REPO / "CLAUDE.md").read_text(encoding="utf-8"))
        assert len(blocks) > 30, "CLAUDE.md collapsed to a handful of blocks"

    def test_html_block_tags_split_too(self):
        text = '<div class="note">\nfoo 0.1111\n</div>\n<div class="win">\nbar 0.2222\n'
        assert len(adc._prose_blocks(text)) >= 2

    def test_line_numbers_point_at_the_block_start(self):
        text = "intro\n\n- bullet 0.1111\n  more\n"
        blocks = dict((b, s) for s, b in adc._prose_blocks(text))
        assert blocks["- bullet 0.1111\n  more"] == 3


class TestD8SupersededIsDerivedNotTyped:
    def test_current_and_superseded_are_disjoint(self):
        for name, report, section, rows, _ in adc.WATCHED_QUANTITIES:
            cur, old = adc._quantity_values(report, section, rows)
            assert not (cur & old), f"{name}: a value cannot be both"

    def test_every_watched_quantity_actually_resolves(self):
        # A renamed row label would silently empty a quantity, and an empty
        # quantity flags nothing -- the vacuous-PASS shape this whole file
        # exists to prevent. D8 reports it as UNRESOLVED; here it is a failure.
        for name, report, section, rows, _ in adc.WATCHED_QUANTITIES:
            cur, _ = adc._quantity_values(report, section, rows)
            assert cur, f"{name} resolved to nothing ({report})"

    def test_a_missing_section_yields_nothing_rather_than_the_whole_file(self):
        got = adc._table_values("| recall@10 | routed (shipped) | 0.6811 |",
                              "## no such heading", [r"routed"])
        assert got == set()

    def test_the_section_stops_at_the_next_heading(self):
        text = ("## A\n| recall@10 | routed (shipped) | 0.1111 |\n"
                "## B\n| recall@10 | routed (shipped) | 0.2222 |\n")
        assert adc._table_values(text, "## A", [r"routed \(shipped\)"]) == {0.1111}


class TestD8AgreesWithTheTestedParsers:
    """One authority, checked from the side.

    D8 reads report tables generically rather than importing each script's own
    parser, because importing one costs ~6 s and pulls torch into an audit that
    has to stay fast. That is only safe if the two agree, so the comparison lives
    here, where the 6 s can be afforded.
    """

    def test_routed_shipped_matches_parse_routing_eval_routed(self):
        pytest.importorskip("numpy")
        sys.path.insert(0, str(adc.REPO / "tools" / "eval"))
        from reranker_rrf_routed_test import parse_routing_eval_routed

        text = (adc.RESULTS / "routing_eval.md").read_text(encoding="utf-8")
        for retriever in ("hybrid", "dense"):
            spec = next(w for w in adc.WATCHED_QUANTITIES
                        if w[0] == f"routed arms ({retriever})")
            cur, _ = adc._quantity_values(spec[1], spec[2], spec[3])
            assert parse_routing_eval_routed(text, retriever) in cur

    def test_rrf4_arms_match_parse_routed_arms(self):
        pytest.importorskip("numpy")
        sys.path.insert(0, str(adc.REPO / "tools" / "eval"))
        from reranker_rrf_routed_test import parse_routed_arms

        text = (adc.RESULTS / "reranker_rrf_routed_test.md").read_text(encoding="utf-8")
        spec = next(w for w in adc.WATCHED_QUANTITIES if w[0] == "rrf4 2x2 arms")
        cur, _ = adc._quantity_values(spec[1], spec[2], spec[3])
        assert set(parse_routed_arms(text).values()) <= cur


# ------------------------------------------------- D7 over Python docstrings --
class TestD7ReadsDocstrings:
    """A docstring is prose and was outside every check here until 2026-08-23.

    Only D7 was extended, and that was a measurement: over these docstrings the
    ms/q-s rule scores 61% real / 15% at n+1 (a check) while the 4-decimal rule
    scores 96% / 71% -- as weak as D2 itself, which had never been scored this
    way. The registry below is what stops the source set silently shrinking.
    """

    def test_the_source_set_is_not_empty_and_covers_the_shipped_package(self):
        found = adc._source_docstrings()
        assert len(found) > 200
        assert any(r.startswith("src/rag_lab/") for r, _ in found)
        assert any(r.startswith("tests/") for r, _ in found)

    def test_a_module_docstring_is_read(self):
        rels = {r for r, _ in adc._source_docstrings()}
        assert "src/rag_lab/query_service.py" in rels

    def test_a_syntax_error_is_skipped_rather_than_fatal(self, tmp_path, monkeypatch):
        bad = tmp_path / "src" / "rag_lab"
        bad.mkdir(parents=True)
        (bad / "broken.py").write_text("def (:\n", encoding="utf-8")
        monkeypatch.setattr(adc, "REPO", tmp_path)
        assert adc._source_docstrings() == []
