"""Decompose the 126 human-judged residual-relevance verdicts: for each one
the qrels missed, was the entity string actually present in the candidate's
body text (a fixable tagging/qrels-construction gap) or genuinely absent (a
semantic-only match, the real pooling-bias signature)?

The user's suggestion this answers: the query names a person/course/faculty/
program, so just extract that name and search for it in the candidate's text.
That is exactly how the Gold qrels themselves were built
(`tools/corpus_prep/build_gold_candidates.py`) -- program on TITLE-only
substring, person on an exact given+surname regex excluding secretarial
signatures, faculty on a filing-title-gated dict tag, course on a code tag.
Reapplying those same per-type rules to the 126 candidates' full text (not
just their title) separates two very different explanations for the same
"y" verdict:

  present  -> the entity IS there in the body, but the construction rule
              (title-only for program; filing-title-gate for faculty;
              a prior tagging miss for person/course) didn't count it --
              a fixable gap in qrels construction/tagging, not evidence
              retrieval found something lexically invisible.
  absent   -> the entity string is nowhere in the body, yet the human still
              judged it relevant -- this is the genuine semantic-match
              signature the whole residual-relevance study exists to detect.

Caveat specific to `program`: the module docstring in build_gold_candidates.py
documents that body-containment was tried and *rejected* for program
(curriculum-bundle siblings co-mention the program in a shared summary table
without being *about* it) -- so "present" for a program item is expected to
be common and does NOT by itself mean a construction bug; it just narrows
down what to inspect. Person/faculty/course have no such documented
over-inclusion risk, so "present" there is a more direct bug signal.

Run with:
    .venv/Scripts/python.exe tools/eval/residual_relevance_decompose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "tools" / "corpus_prep"))

import residual_relevance_sample as rrs  # noqa: E402
from build_gold_candidates import (  # noqa: E402
    _build_person_alias_index,
    _has_non_secretarial_mention,
    _load_json,
    DICT_DIR,
)

_GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_SHEET = REPO / "data" / "results" / "residual_relevance" / "review_sheet.yaml"


def _person_present(text: str, entity: str, alias_index: dict[tuple[str, str], str]) -> bool | None:
    """None when `entity` can't be resolved to any (given, surname) alias --
    shouldn't happen since `entity` comes straight from people.json, but fails
    loud rather than silently mis-scoring if the dictionary ever changes.
    Multiple (given, surname) spellings can share one canonical (OCR-variant
    aliases) -- must check every one of them, not just the first found, or a
    document using a variant spelling other than whichever alias iteration
    happens to hit first is wrongly scored "absent"."""
    aliases = [(g, s) for (g, s), canonical in alias_index.items() if canonical == entity]
    if not aliases:
        return None
    return any(_has_non_secretarial_mention(text, given, surname) for given, surname in aliases)


def main() -> None:
    qrels_entries = yaml.safe_load(_GOLD.read_text(encoding="utf-8"))
    entity_of = {e["query"]: (e["entity_type"], e["entity"]) for e in qrels_entries}

    people = _load_json(DICT_DIR / "people.json")
    alias_index = _build_person_alias_index(people)

    sheet = yaml.safe_load(_SHEET.read_text(encoding="utf-8"))
    doc_index = rrs.build_full_document_index()

    counts: dict[str, dict[str, int]] = {}
    unresolved: list[int] = []
    detail: list[tuple] = []

    for item in sheet:
        verdict = str(item.get("verdict", "")).strip().lower()
        if verdict != "y":
            continue  # decomposing only the "relevant" verdicts -- that's the surprising 125
        etype, entity = entity_of[item["query"]]
        text = doc_index.get(item["candidate"])
        if text is None:
            unresolved.append(item["id"])
            continue

        if etype == "person":
            present = _person_present(text, entity, alias_index)
            if present is None:
                unresolved.append(item["id"])
                continue
        else:
            present = entity in text

        bucket = counts.setdefault(etype, {"present": 0, "absent": 0})
        bucket["present" if present else "absent"] += 1
        detail.append((item["id"], etype, entity, item["candidate"], present))

    print(f"{sum(len(v.values()) and (v['present']+v['absent']) for v in counts.values())} "
          f"'y' verdicts decomposed, {len(unresolved)} unresolved (ids {unresolved})\n")
    print(f"{'entity_type':28s} {'present (body has entity)':28s} {'absent (semantic-only)':24s} rate absent")
    total_present = total_absent = 0
    for etype, c in sorted(counts.items()):
        n = c["present"] + c["absent"]
        rate = c["absent"] / n if n else 0.0
        print(f"{etype:28s} {c['present']:<28d} {c['absent']:<24d} {rate:.1%}")
        total_present += c["present"]
        total_absent += c["absent"]
    n = total_present + total_absent
    print(f"{'TOTAL':28s} {total_present:<28d} {total_absent:<24d} "
          f"{(total_absent/n if n else 0):.1%}")

    print("\nabsent (entity string not found anywhere in candidate body -- genuine "
          "semantic-only relevance) items, by entity_type:")
    for id_, etype, entity, cand, present in detail:
        if not present:
            print(f"  id {id_:3d} [{etype}] entity={entity!r}")


if __name__ == "__main__":
    main()
