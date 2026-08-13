"""Both directions of the pylate cross-check's self-checks, without a model.

The cross-check is the *external* half of the ColBERT gate, and it is the half
that actually caught something (`mask_punctuation` masking whitespace). So its
own rules have to be able to say FAIL, and two of them are easy to get wrong in
the vacuous direction:

* **S1** — the reference must be a *correct* implementation, not merely a
  different one. pylate under transformers 5.3.0 hits the same uninitialised
  `inv_freq` bug this repo does, so a cross-check against it would be two broken
  models agreeing; S1 must reject that reference outright.
* **S6** — the second cell is *required to differ*. It is kept as a
  demonstration that two independently-broken models are not a control (the
  garbage is redrawn at each load), so if it ever agreed that would be a finding
  about reproducible corruption, not a relief. A version of S6 that passed on
  agreement would silently invert the check's meaning.

Everything here operates on the raw comparison dict, which is exactly what
`--render` reads, so no `.npz`, no download and no GPU.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "eval"))
from colbert_pylate_crosscheck import self_checks  # noqa: E402


def _cell(key: str, *, gates: bool, bad: int, total: int, repaired: int,
          delta: float, shapes_agree: bool = True) -> dict:
    names = [("query", [32, 128]), ("doc0", [21, 128]), ("doc1", [22, 128])]
    tensors = []
    for name, shape in names:
        row = {"name": name, "ours": shape,
               "pylate": shape if shapes_agree else [shape[0] - 2, shape[1]]}
        if shapes_agree:
            row["max_abs_delta"] = 0.0 if name == "query" else delta
            row["min_cosine"] = 1.0
        else:
            row["shape_mismatch"] = True
        tensors.append(row)
    return {
        "key": key, "label": key, "why": "", "gates": gates,
        "pylate_transformers": "4.53.2" if gates else "5.3.0",
        "pylate_rotary_bad": bad, "pylate_rotary_total": total,
        "ours_layers_repaired": repaired,
        "tensors": tensors,
        "maxsim_ours": [20.8212, 17.5484],
        "maxsim_pylate": [20.8213, 17.5487],
        "agree": shapes_agree and delta < 5e-3,
    }


def _good() -> list[dict]:
    """The shape of a passing run: gate cell matches, demo cell differs."""
    return [
        _cell("t453", gates=True, bad=0, total=24, repaired=24, delta=1.22e-4),
        _cell("t530", gates=False, bad=24, total=24, repaired=0, delta=2.8e-1),
    ]


def _verdicts(cells: list[dict]) -> dict[str, bool]:
    return {name.split()[0]: ok for name, ok, _ in self_checks(cells)}


def test_a_clean_run_passes_every_check():
    assert all(_verdicts(_good()).values())


def test_s1_rejects_a_reference_that_is_itself_broken():
    """A reference on transformers 5.x is a second broken model, not a control."""
    cells = deepcopy(_good())
    cells[0]["pylate_rotary_bad"] = 24
    v = _verdicts(cells)
    assert v["S1"] is False
    assert v["S3"] is True, "S1 must be what fails; the values are untouched here"


def test_s6_fails_when_the_two_broken_models_agree():
    """The demonstration cell is required to DIFFER — agreement inverts its point."""
    cells = deepcopy(_good())
    cells[1]["agree"] = True
    assert _verdicts(cells)["S6"] is False


def test_s2_catches_a_token_count_mismatch_before_any_value_check():
    """The skiplist defect showed up as 19/21 vectors against 21/22, not as a delta."""
    cells = deepcopy(_good())
    cells[0] = _cell("t453", gates=True, bad=0, total=24, repaired=24,
                     delta=0.0, shapes_agree=False)
    v = _verdicts(cells)
    assert v["S2"] is False and v["S3"] is False


def test_s3_fails_on_a_real_disagreement():
    cells = deepcopy(_good())
    for t in cells[0]["tensors"]:
        t["max_abs_delta"] = 0.5
    cells[0]["agree"] = False
    assert _verdicts(cells)["S3"] is False


def test_s4_demands_the_query_side_be_exact_not_merely_close():
    """Near-agreement would be a self-consistent substitute for `inv_freq`, not
    a restoration of it — the whole force of finding B is the zero."""
    cells = deepcopy(_good())
    q = next(t for t in cells[0]["tensors"] if t["name"] == "query")
    q["max_abs_delta"] = 1e-6
    v = _verdicts(cells)
    assert v["S4"] is False
    assert v["S3"] is True, "1e-06 is well inside tolerance; only S4 may fail"


def test_s5_fails_when_maxsim_disagrees_though_the_vectors_match():
    cells = deepcopy(_good())
    cells[0]["maxsim_pylate"] = [20.8213, 19.0]
    assert _verdicts(cells)["S5"] is False


def test_s7_fails_if_the_demonstration_cell_is_not_the_broken_path():
    """If our side repaired its rotary, the cell no longer demonstrates anything."""
    cells = deepcopy(_good())
    cells[1]["ours_layers_repaired"] = 24
    assert _verdicts(cells)["S7"] is False
