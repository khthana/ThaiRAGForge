"""Tests for `tools/eval/serving_warmup_profile.py`.

Two things are pinned, and both are places where a *convenient* answer would be
a wrong one.

**`cold` has no steady state.** Its four queries are four first callers, one per
route, each loading an index and possibly an embedder of its own -- so the
"steady (best of 2nd-4th)" column is meaningless there. Printing the minimum
anyway would put a plausible small number in the table and hide the actual shape
of the problem: a deployment without a warm-up does not get one slow query and
then fast ones, it gets one slow query per route.

**The probe's cost is a difference of two medians**, so it is quotable only when
it is larger than the noise its two arms carry. At n=3 on this rig it often is
not, and the report must then say so rather than print the subtraction --
section 2 measures the same retrieval directly and does not depend on it.

Both are verified in the failing direction: the noise test asserts the delta is
absent from the rendered text, so an implementation that always printed it would
fail here.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "eval"))
W = importlib.import_module("serving_warmup_profile")

ROUTES = ["person", "program", "unmatched", "course"]


def _run(arm: str, qms: list[float], warm: float) -> dict:
    return {
        "arm": arm,
        "warm_ms": warm,
        "queries": [{"query": q, "route": r, "ms": m, "ids": ["a", "b"]}
                    for q, r, m in zip(W.QUERIES, ROUTES, qms)],
        "first_ms": qms[0],
        "four_ms": round(sum(qms), 1),
        "steady_ms": W.steady_ms(arm, [{"ms": m} for m in qms]),
        "index_cache_entries": 4,
        "embedder_cache_entries": 2,
    }


def _data(probe_warm: tuple[float, float, float] = (29000.0, 29010.0, 29020.0),
          nop_warm: tuple[float, float, float] = (28000.0, 28010.0, 28020.0)) -> dict:
    runs = []
    for i in range(3):
        runs.append(_run("cold", [12000 + i, 9000 + i, 7000 + i, 5000 + i], 0.0))
        runs.append(_run("warm_no_probe", [1100 + i, 380 + i, 390 + i, 400 + i], nop_warm[i]))
        runs.append(_run("warm_probe", [450 + i, 330 + i, 340 + i, 350 + i], probe_warm[i]))
    data = {"runs": runs, "arms": [W.median_arm(a, runs) for a in W.ARMS],
            "probe": {"F=200": 342.6, "F=None (class default)": 856.3}}
    data["checks"] = W.checks(data)
    return data


class TestColdHasNoSteadyState:
    def test_the_rule_itself_refuses_cold(self):
        rows = [{"ms": 12000.0}, {"ms": 9000.0}, {"ms": 7000.0}, {"ms": 5000.0}]
        assert W.steady_ms("cold", rows) is None
        assert W.steady_ms("warm_probe", rows) == 5000.0

    def test_median_arm_reports_none_for_cold(self):
        arms = {a["arm"]: a for a in _data()["arms"]}
        assert arms["cold"]["steady_ms"] is None
        assert arms["warm_probe"]["steady_ms"] is not None

    def test_the_table_prints_n_a_rather_than_a_number(self):
        md = W.render(_data())
        cold_row = next(l for l in md.splitlines() if l.startswith("| `cold`"))
        assert "n/a" in cold_row
        # the four-query total is still a real figure, so the row is not blank
        assert "33,00" in cold_row or "33,0" in cold_row

    def test_the_report_says_why_rather_than_leaving_a_hole(self):
        assert "no steady state" in W.render(_data())


class TestProbeCostIsOnlyQuotedWhenItBeatsNoise:
    def test_a_clean_separation_is_quoted(self):
        # warm arms 1,000 ms apart, each spreading 20 ms
        md = W.render(_data())
        assert "that probe costs" in md
        assert "not separable from noise" not in md

    def test_a_delta_inside_the_spread_is_refused(self):
        # the two arms now overlap: delta 10 ms against a 2,000 ms spread
        md = W.render(_data(probe_warm=(28000.0, 29000.0, 30000.0),
                            nop_warm=(28000.0, 28990.0, 30000.0)))
        assert "not separable from noise" in md
        assert "that probe costs" not in md


class TestSelfChecksDiscriminate:
    def test_s1_sees_an_arm_that_changed_an_answer(self):
        data = _data()
        data["runs"][4]["queries"][0]["ids"] = ["a", "z"]
        s1 = next(c for c in W.checks(data) if c[0].startswith("S1"))
        assert s1[1] == "FAIL"

    def test_s1_reads_every_pass_not_just_the_medians(self):
        # one arm disagreeing with ITSELF between passes must be caught; the
        # medians carry pass 1's ids, so a median-only check would miss this.
        data = _data()
        data["runs"][7]["queries"][2]["ids"] = ["a", "z"]   # warm_probe, pass 3
        s1 = next(c for c in W.checks(data) if c[0].startswith("S1"))
        assert s1[1] == "FAIL"

    def test_s2_sees_a_repeated_route(self):
        data = _data()
        for r in data["runs"]:
            r["queries"][2]["route"] = "person"
        data["arms"] = [W.median_arm(a, data["runs"]) for a in W.ARMS]
        s2 = next(c for c in W.checks(data) if c[0].startswith("S2"))
        assert s2[1] == "FAIL"

    def test_s4_sees_an_inert_fetch_depth(self):
        data = _data()
        data["probe"]["F=None (class default)"] = 342.6
        s4 = next(c for c in W.checks(data) if c[0].startswith("S4"))
        assert s4[1] == "FAIL"

    def test_s5_ignores_cold_but_catches_a_warm_arm_that_did_not_warm(self):
        data = _data()
        # cold ends with full caches too -- the queries fill them -- and that
        # must not be read as a failure.
        for r in data["runs"]:
            if r["arm"] == "cold":
                r["index_cache_entries"] = 4
        data["arms"] = [W.median_arm(a, data["runs"]) for a in W.ARMS]
        assert next(c for c in W.checks(data) if c[0].startswith("S5"))[1] == "PASS"

        for r in data["runs"]:
            if r["arm"] == "warm_probe":
                r["index_cache_entries"] = 1
        data["arms"] = [W.median_arm(a, data["runs"]) for a in W.ARMS]
        assert next(c for c in W.checks(data) if c[0].startswith("S5"))[1] == "FAIL"


class TestTheAnchorIsParsedNotFrozen:
    def test_it_reads_the_published_steady_state(self):
        # If this ever returns None the S3 row degrades to "UNPARSED -- the
        # cross-check could not be made" rather than passing silently, which is
        # the behaviour a frozen literal cannot have.
        got = W.parse_cost_steady()
        assert got is None or 100.0 < got < 5000.0

    def test_an_unparseable_report_degrades_to_a_warning(self, monkeypatch):
        monkeypatch.setattr(W, "COST_REPORT", REPO / "does-not-exist.md")
        s3 = next(c for c in W.checks(_data()) if c[0].startswith("S3"))
        assert s3[1] == "WARN" and "UNPARSED" in s3[2]
