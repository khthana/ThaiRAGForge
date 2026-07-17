"""Generate entity-anchored gold-query-set candidates for retrieval eval.

Ground truth for these two query shapes is derivable deterministically from
data already on disk -- no LLM judgment needed, which avoids the hallucinated-
relevance risk of asking a model to read resolutions and guess what's relevant:

- Program history ("หลักสูตร X มีการเปลี่ยนแปลงอะไรบ้าง"): relevant_resolution_ids
  = every resolution whose *title* (not body) names the program. Body-tag
  membership (programs_by_file.json) was tried first and rejected: for a
  curriculum-bundle program (e.g. "วิศวกรรมคอมพิวเตอร์"), it returned 120 hits vs
  19 that actually match on title -- the other 101 are bundle-sibling files
  that merely co-mention the program name in a shared summary table (see
  ADR-0004 curriculum-bundle splitting). Title match is precise because each
  split piece's manifest-derived title names exactly the program it's about.

- Person history ("อาจารย์ X มีประวัติเป็นกรรมการหลักสูตรใดบ้าง"): relevant_resolution_ids
  = every resolution where the person's canonical identity (via
  canonicalize_people.py's alias index, not raw spelling) appears in
  people_by_file.json. Requires EXACT given+surname match, not substring --
  substring "ธนา" over-matches 16 different canonical people (ธนากร, ธนาพล,
  ธนาดล, ...) who happen to share a name-root.

Output is a CANDIDATE pool, not a finished gold set: it still needs human
review before use (person hits may include incidental mentions -- e.g. an
attendee list -- not just committee-membership; program windows are not
capped to any particular year range). Curate a ~30-50 entry subset from this
pool into config/eval/gold_query_set.yaml (see CONTEXT.md's Gold query set
definition, ADR-0002 for why relevance is judged at the Resolution level).

Read-only: never writes into the corpus itself. Output goes to
academic_resolutions/entity_tags/ (gitignored, same convention as
tag_people.py / tag_programs.py / canonicalize_people.py).

Run with:
    .venv/Scripts/python.exe tools/corpus_prep/build_gold_candidates.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO / "academic_resolutions"
TAGS_DIR = CORPUS_ROOT / "entity_tags"
DICT_DIR = REPO / "data" / "entity_dictionaries"

# tools/corpus_prep is run directly (not via pytest's configured pythonpath),
# so src/ needs to be on sys.path explicitly to reuse the already-tested
# path-parsing helpers instead of duplicating them here.
sys.path.insert(0, str(REPO / "src"))
from rag_lab.loaders.common import make_resolution_id, parse_path  # noqa: E402

_RID_YEAR_SESSION = re.compile(r"^(\d+)/(\d+)s?/")

# ADR-0004 curriculum-bundle splits: one bundled meeting item becomes N
# physical `<original title> — <curriculum title>` pieces, each its own
# resolution_id. Confirmed corpus-wide (707/707 split files, 0 counter-
# examples): every `__N` piece's manifest title contains this exact
# " — " separator, and `__N` numbering is always a dense 1..N run (never
# reused for title-collision disambiguation), so splitting a resolution_id
# on the first " — " is a safe, lossless way to recover the shared original-
# event identity without re-reading meeting_manifest.json.
_EVENT_SEP = " — "


def _event_key(resolution_id: str) -> str:
    """Collapse a split-bundle piece's resolution_id back to its shared
    original meeting-item identity, for counting "how many distinct
    decisions/resolutions" rather than "how many retrievable files" --
    ADR-0004 splitting duplicates the shared preamble (committee roster,
    reviewer bios, ...) verbatim into every piece, so a person/program
    named there shows up once per piece even though it reflects one
    underlying event. Non-split resolutions have no separator and are
    trivially their own singleton group."""
    return resolution_id.split(_EVENT_SEP, 1)[0]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolution_id_for(relpath: str) -> str:
    full_path = str(CORPUS_ROOT / relpath)
    year, session, title = parse_path(full_path)
    return make_resolution_id(full_path, year, session, title)


def _sort_key(resolution_id: str) -> tuple[int, int]:
    m = _RID_YEAR_SESSION.match(resolution_id)
    return (int(m.group(1)), int(m.group(2))) if m else (9999, 9999)


def _build_person_alias_index(people: list[dict]) -> dict[tuple[str, str], str]:
    """(given, surname) -> canonical_full_name, covering both the canonical
    spelling itself and every OCR-variant alias from canonicalize_people.py."""
    index: dict[tuple[str, str], str] = {}
    for entry in people:
        index[(entry["canonical_given"], entry["canonical_surname"])] = entry["canonical_full_name"]
        for alias in entry["aliases"]:
            index[(alias["given"], alias["surname"])] = entry["canonical_full_name"]
    return index


def program_candidates(min_hits: int) -> list[dict]:
    programs = _load_json(DICT_DIR / "programs.json")
    resolution_ids = sorted(
        {
            _resolution_id_for(relpath)
            for relpath in _load_json(TAGS_DIR / "programs_by_file.json")
        }
    )

    candidates = []
    for program in programs:
        canonical = program["canonical"]
        field = program["field"] or ""
        if not canonical or not field:
            continue
        # Match on the full canonical string ("หลักสูตร{degree} สาขาวิชา{field}"),
        # not the bare field: field alone collides with faculty names that
        # share the same word (e.g. field "ครุศาสตร์อุตสาหกรรม" also appears in
        # the faculty name "คณะครุศาสตร์อุตสาหกรรมและเทคโนโลยี", which pulled in
        # every resolution from that faculty regardless of program).
        hits = sorted(
            (rid for rid in resolution_ids if canonical in rid),
            key=_sort_key,
        )
        # Threshold on event_count (distinct original resolutions), not raw
        # file count -- a program split into 3 ADR-0004 pieces from one
        # bundle is one change event, not three, and shouldn't qualify a
        # min_hits=2 candidate on file-count alone.
        event_count = len({_event_key(rid) for rid in hits})
        if event_count < min_hits:
            continue
        candidates.append(
            {
                "entity_type": "program",
                "entity": canonical,
                "query": f"หลักสูตร{field}มีการเปลี่ยนแปลงกี่ครั้งและแต่ละครั้งมีรายละเอียดอะไรบ้าง",
                "relevant_resolution_ids": hits,
                "hit_count": len(hits),
                "event_count": event_count,
            }
        )
    return candidates


_SEP = r"(?:\s+|<br\s*/?>\s*)"
_SECRETARIAL_MARKER = "เลขานุการ"
_SECRETARIAL_WINDOW = 80


def _has_non_secretarial_mention(text: str, given: str, surname: str) -> bool:
    """True if `given surname` appears anywhere in `text` NOT immediately
    followed by a secretary/certifier role marker. Meeting minutes end with
    a signature line -- "(ชื่อผู้ลงนาม) ผู้ช่วยเลขานุการ ทำหน้าที่แทนกรรมการ
    และเลขานุการ" -- printed on nearly every resolution the acting secretary
    certified. That's an administrative signature, not evidence the person
    was substantively involved in the resolution's content, and inflates a
    handful of people (e.g. one saw 696/2853 corpus-wide "hits", almost all
    of them this pattern) into implausible committee-history candidates."""
    pattern = re.compile(re.escape(given) + _SEP + re.escape(surname))
    for m in pattern.finditer(text):
        window = text[m.end() : m.end() + _SECRETARIAL_WINDOW]
        if _SECRETARIAL_MARKER not in window:
            return True
    return False


def person_candidates(min_hits: int) -> list[dict]:
    people = _load_json(DICT_DIR / "people.json")
    alias_index = _build_person_alias_index(people)
    by_file = _load_json(TAGS_DIR / "people_by_file.json")

    hits_by_canonical: dict[str, set[str]] = {}
    for relpath, mentions in by_file.items():
        canonicals_in_file = {
            alias_index[(m["given_name"], m["surname"])]
            for m in mentions
            if (m["given_name"], m["surname"]) in alias_index
        }
        if not canonicals_in_file:
            continue
        full_path = CORPUS_ROOT / relpath
        try:
            text = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = full_path.read_text(encoding="utf-8-sig")
        rid = _resolution_id_for(relpath)
        for mention in mentions:
            canonical = alias_index.get((mention["given_name"], mention["surname"]))
            if canonical and _has_non_secretarial_mention(
                text, mention["given_name"], mention["surname"]
            ):
                hits_by_canonical.setdefault(canonical, set()).add(rid)

    candidates = []
    for canonical_name, rids in hits_by_canonical.items():
        event_count = len({_event_key(rid) for rid in rids})
        if event_count < min_hits:
            continue
        candidates.append(
            {
                "entity_type": "person",
                "entity": canonical_name,
                "query": f"{canonical_name} มีประวัติเกี่ยวข้องกับหลักสูตรใดบ้าง ในช่วงใด ให้แสดงรายละเอียดทั้งหมด",
                "relevant_resolution_ids": sorted(rids, key=_sort_key),
                "hit_count": len(rids),
                "event_count": event_count,
            }
        )
    return candidates


def render_report(program_hits: list[dict], person_hits: list[dict]) -> str:
    lines = [
        "# Gold query-set candidates",
        "",
        f"- Program-anchored candidates: {len(program_hits)}",
        f"- Person-anchored candidates: {len(person_hits)}",
        f"- Total: {len(program_hits) + len(person_hits)}",
        "",
        "These are CANDIDATES, not a finished gold set -- pick a ~30-50 entry",
        "subset (mixing program and person entries, varied hit_count) into",
        "config/eval/gold_query_set.yaml. See the module docstring in",
        "build_gold_candidates.py for why title/exact-match beats body-tag/",
        "substring here, and what still needs human review before trusting a",
        "candidate as real ground truth.",
        "",
        "hit_count is the raw retrievable-file count; event_count collapses",
        "ADR-0004 curriculum-bundle split pieces that share one original",
        "meeting item back to a single count -- use event_count for \"กี่ครั้ง\"-",
        "style stratification/expected answers, hit_count for recall grading.",
        "",
        "## Top program candidates by event_count",
        "",
    ]
    for c in sorted(program_hits, key=lambda c: -c["event_count"])[:15]:
        lines.append(f"- (event_count={c['event_count']}, hit_count={c['hit_count']}) {c['entity']}")
    lines += ["", "## Top person candidates by event_count", ""]
    for c in sorted(person_hits, key=lambda c: -c["event_count"])[:15]:
        lines.append(f"- (event_count={c['event_count']}, hit_count={c['hit_count']}) {c['entity']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-program-hits", type=int, default=2)
    parser.add_argument("--min-person-hits", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(TAGS_DIR),
        help="Directory to save candidate pool + report",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    programs = program_candidates(args.min_program_hits)
    people = person_candidates(args.min_person_hits)

    (out_dir / "gold_candidates.json").write_text(
        json.dumps({"programs": programs, "people": people}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    (out_dir / "gold_candidates_report.md").write_text(
        render_report(programs, people), encoding="utf-8"
    )
    print(f"program candidates: {len(programs)}")
    print(f"person candidates: {len(people)}")
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
