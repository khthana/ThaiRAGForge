# -*- coding: utf-8 -*-
"""Split curriculum-bundle resolutions into per-curriculum files.

Some resolutions (มติ) bundle several curricula into one agenda item, e.g.
"ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีไม่กระทบกระเทือนโครงสร้าง) ... จำนวน ๓
หลักสูตร ดังนี้ ๑) ... ๒) ... ๓) ...". Because `resolution_id` is the
citation/retrieval unit, one id per bundle is too coarse — this script splits
each bundle into one physical `.md` file per curriculum, so each curriculum
gets its own resolution_id and title. See docs/adr/0004 for the rationale
(physical split, not a loader-interface change).

Detection is content-based: any `.md` whose body declares
"จำนวน N หลักสูตร" with N >= 2 is a candidate. The declaration's position
anchors a search window for the top-level enumerators ("๑)", "๒)", ...);
`<table>...</table>` spans are blanked out (not removed — offsets must stay
aligned) before that search, because nested numbering inside OCR'd tables
(faculty/ผลงานวิชาการ rows) is otherwise miscounted as top-level items.

A split only happens when the enumerator count matches the declared N exactly
and the numbers are sequential. Anything else is written to
`academic_resolutions/curriculum_split_review.md` for manual follow-up,
never guessed at silently.

Writes, per split file `<dir>/<stem>.md`:
- `<dir>/<stem>__1.md` .. `<stem>__N.md` — one per curriculum, each carrying
  the shared preamble (meeting/committee framing) plus its own body. Filenames
  stay opaque per ADR-0003; the real title goes into meeting_manifest.json.
- `<dir>/<stem>__k_LINK.txt` — copy of the original `_LINK.txt` (same source
  PDF covers every curriculum in the bundle).
- `meeting_manifest.json` in that meeting folder is patched in place (old
  `stem.md` entry removed, N new entries added) so the next
  `rebuild_manifests.py --apply` picks up the per-curriculum titles via its
  existing `_prev_titles()` fallback — no changes needed there.
- The original `<stem>.md` / `<stem>_LINK.txt` are archived as `*.dup`
  (existing convention: recoverable, excluded from `rglob("*.md")`).

Idempotent: a stem already archived (`<stem>.md.dup` exists) or already split
(`<stem>__1.md` exists) is skipped on re-run.

Run from the repo root:

    python tools/corpus_prep/split_curriculum_bundles.py            # dry-run
    python tools/corpus_prep/split_curriculum_bundles.py --apply    # write

Re-run `rebuild_manifests.py --apply` afterwards to regenerate master_list.csv.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
APPLY = "--apply" in sys.argv

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "academic_resolutions"
REVIEW_FILE = ROOT / "curriculum_split_review.md"

THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
DECL = re.compile(r"จำนวน\s*([0-9๐-๙]+)\s*หลักสูตร")
TABLE = re.compile(r"<table.*?</table>", re.S | re.I)
# The enumerator must be immediately followed by "หลักสูตร" -- otherwise it
# matches a curriculum's own inner subsection headings ("๑) เหตุผลการปรับปรุง",
# "๒) สาระในการปรับปรุงแก้ไข" or the period-style "๑. เหตุผล..."), which reuse
# the identical "๑)/๑." numbering as the top-level curriculum list and would
# otherwise be mistaken for the next curriculum in the list. Both "๑)" and
# "๑." top-level styles appear in the corpus, so both delimiters are accepted.
# Some OCR passes wrap the enumerator line in a markdown heading ("## ๑) ...")
# instead of plain text -- an optional leading "#" run is tolerated so those
# aren't silently skipped (which previously let a later, unrelated numbered
# line -- e.g. a "มติที่ประชุม" recap restating curriculum 1's name -- be
# mistaken for the real next boundary instead).
ENUM = re.compile(
    r"(?:^|\n)[ \t]*(?:#{1,6}[ \t]*)?([๐-๙]{1,2})[.)][ \t]*\**[ \t]*"
    r"(?=หลักสูตร)(?!หลักสูตร[ \t]*(?:เป็น|เน้น|ควร|ต้อง|จะ|ให้|มี|อยู่|คือ))")
MIN_PIECE_LEN = 60          # floor against a literally-empty piece
# A curriculum bundle's declaration sentence often carries its own inline
# name-list ("จำนวน ๔ หลักสูตร ดังนี้ ๑) ชื่อ๑ ๒) ชื่อ๒ ๓) ชื่อ๓ ๔) ชื่อ๔") --
# structurally identical to the real per-curriculum list (same "๑) หลักสูตร.."
# markers) but each item is just a bare name+year line. The real content for
# each curriculum, when present, appears later and re-numbers "๑) ๒) ..."
# from scratch. Instead of guessing from piece-size ratios, require every
# piece in a candidate run to show a concrete sign of real content; a
# name-only run fails this and the search retries further into the document.
REAL_CONTENT = re.compile(
    r"โดยผ่านการพิจารณาจาก"    # "considered by [committee]" framing line
    r"|เหตุผลการปรับปรุง"       # "rationale for revision" subsection heading
    r"|สาระในการปรับปรุง"       # "substance of the revision" subsection heading
    r"|^##\s*Page\s*\d+\s*$"    # a page marker falls inside the piece
    r"|<table",                 # a table is embedded in the piece
    re.M)
PAGE_MARKER = re.compile(r"^##\s*Page\s*\d+\s*$", re.M)
DOC_HEADER = re.compile(r"^\s*#\s*Document:.*$", re.M)
SPLIT_PIECE = re.compile(r"__\d+$")
# User explicitly declined to split these categories even though some of them
# happen to use the same "จำนวน N หลักสูตร ... ๑) ๒)" structure (e.g. an
# อาจารย์พิเศษ course-load doc broken out per curriculum) -- exclude by title
# regardless of structural match. Also exclude topics that are structurally
# never a "here are N curricula being individually decided" bundle even
# though they mention curriculum counts: status/count reports, MOUs,
# tuition-registration paperwork.
EXCLUDE_TITLE = re.compile(
    r"อาจารย์พิเศษ|อาจารย์บัณฑิต|มาตรฐาน|ภาระงาน"
    r"|รายงานการส่ง.*หลักสูตร|ความสอดคล้องของหลักสูตร"  # CHECO submission reports
    r"|บันทึกข้อตกลง|ความร่วมมือทางวิชาการ"              # MOUs
    r"|รายงานความคืบหน้า"                                # progress reports (e.g. OBE)
    r"|ลงทะเบียนเรียนเกิน"                                # credit-overload requests
    r"|ฝึกอบรมเพื่อสะสมหน่วยกิต"    # credit-bank training modules -- short
                                    # workshops/CEU courses, not degree curricula;
                                    # same "จำนวน N หลักสูตร ๑) ๒)..." structure
                                    # but each item is inherently a few lines
    r"|อบรมเชิงปฏิบัติการ"          # workshop/training projects (e.g. OBE)
    r"|ประเมินการพัฒนาคุณภาพ"       # internal quality-assessment reports (DPBP etc.)
    r"|ประธานแจ้งให้ที่ประชุมทราบ"  # chair's general announcements agenda item
)
REPORT_ITEM = re.compile(r"จำนวน\s*[0-9๐-๙]+\s*หลักสูตร")
# A curriculum list is often split into degree-level sections (ปริญญาตรี,
# บัณฑิตศึกษา, ...), each restarting its own "๑) ๒)..." numbering from 1 --
# the level header is the signal that a "๑)" is the start of a new section,
# not a stray/unrelated match, even though it breaks strict global sequence.
LEVEL_HEADER = re.compile(
    r"(?:^|\n)[ \t]*(?:#{1,6}[ \t]*)?\**[ \t]*ระดับ(?:ปริญญาตรี|บัณฑิตศึกษา|ปริญญาโท|ปริญญาเอก|อนุปริญญา)")


def thai_int(s: str) -> int:
    return int(s.translate(THAI_DIGITS))


def blank_tables(text: str) -> str:
    """Replace <table>...</table> spans with equal-length spaces so absolute
    offsets in the searched copy still line up with the original text."""
    return TABLE.sub(lambda m: " " * len(m.group(0)), text)


def validate_candidate(text: str, n: int, offsets: list[int]) -> dict:
    """Check one candidate run of N sequential enumerator offsets. Returns
    {"action": "split", ...} if it holds up, else {"action": "mismatch", ...}
    with a reason."""
    bodies = [text[start:offsets[i + 1] if i + 1 < len(offsets) else len(text)]
              for i, start in enumerate(offsets)]
    piece_lens = [len(b) for b in bodies]

    for body in bodies:
        # Guard against a status/count REPORT (e.g. a CHECO or OBE-progress
        # summary) whose top-level "๑) ... จำนวน M หลักสูตร" items are
        # category subtotals, not curriculum names -- never a real bundle.
        if REPORT_ITEM.search(body[:80]):
            return {"action": "mismatch", "declared": n, "found": len(offsets),
                    "reason": "looks like a count/subtotal report, not curriculum names"}

    if min(piece_lens) < MIN_PIECE_LEN:
        return {"action": "mismatch", "declared": n, "found": len(offsets),
                "reason": f"piece too short (min {min(piece_lens)} chars)"}

    # Guard against latching onto the declaration sentence's own inline
    # name-list ("๑) ชื่อ๑ ๒) ชื่อ๒ ...") instead of the real per-curriculum
    # body: a bare name+year line carries no sign of actual content (no
    # committee framing, no subsection heading, no page marker, no table).
    # Reject the whole run if any piece looks like a name-only stub -- the
    # caller retries further into the document for the real content run.
    not_real = [i + 1 for i, b in enumerate(bodies) if not REAL_CONTENT.search(b)]
    if not_real:
        return {"action": "mismatch", "declared": n, "found": len(offsets),
                "reason": f"piece(s) {not_real} look like name-only list entries, not real content"}

    # Guard against two "different" curricula actually being the same text
    # repeated (e.g. a table-of-contents echo of the same entry, or an OCR
    # double-scan) -- a real bundle never lists the identical passage twice.
    if len({b.strip() for b in bodies}) != len(bodies):
        return {"action": "mismatch", "declared": n, "found": len(offsets),
                "reason": "two pieces have identical content -- likely a repeated passage, not distinct curricula"}

    # Same idea, weaker signal: a closing recap can echo curriculum 1's name
    # as the lead-in to the last piece's own closing clause (e.g. "๒) <same
    # curriculum name as piece 1> \n ๒. ให้เสนอ...พิจารณาตามลำดับ"), giving a
    # piece with the same *title* as another piece even though the trailing
    # boilerplate text differs. Titles are the retrieval-facing identity, so
    # a collision there is just as unsafe to apply silently.
    titles = [extract_title(b) for b in bodies]
    if len(set(titles)) != len(titles):
        return {"action": "mismatch", "declared": n, "found": len(offsets),
                "reason": "two pieces extracted the same title -- likely a recap echo, not a distinct curriculum"}

    return {"action": "split", "declared": n, "offsets": offsets}


def find_boundaries(text: str) -> dict:
    """Classify a file and, if splittable, return the enumerator offsets.

    Returns a dict with at least {"action": ...}; "split" results also carry
    {"declared": N, "offsets": [...]}.
    """
    m = DECL.search(text)
    if not m:
        return {"action": "no-declaration"}
    n = thai_int(m.group(1))
    if n < 2:
        return {"action": "single"}

    # N comes from the declaration sentence, but the declaration itself isn't
    # always where the enumerator list opens -- a "มติที่ประชุม ... จำนวน N
    # หลักสูตร ดังนี้" recap near the end of the file matches DECL just as
    # well as the real intro (some intros phrase it as "ระดับปริญญาตรี N
    # หลักสูตร" with no "จำนวน" at all, so DECL's *first* match can be that
    # trailing recap). Search the whole document for boundaries rather than
    # anchoring to this match's position -- the real-content/duplicate-title
    # guards below are what actually keep a wrong list from being accepted.
    blanked = blank_tables(text)
    window = blanked
    offsets: list[int] = []
    local_next = 1
    gap_start = 0
    last_attempt: dict | None = None
    for em in ENUM.finditer(window):
        num = thai_int(em.group(1))
        if num == local_next:
            offsets.append(em.start(1))
            local_next += 1
            gap_start = em.end()
        elif num == 1 and offsets and LEVEL_HEADER.search(window[gap_start:em.start()]):
            # a degree-level header intervened (e.g. "ระดับบัณฑิตศึกษา") --
            # this "๑)" restarts local numbering for the new level, but it's
            # still the next curriculum in the overall list
            offsets.append(em.start(1))
            local_next = 2
            gap_start = em.end()
        elif num == 1 and offsets:
            # a fresh "๑)" appeared mid-run with no level-header restart --
            # the current partial run is stale (most likely the declaration's
            # own inline name-list); abandon it and start a new candidate
            # here. The real per-curriculum content, when it exists, opens
            # its own "๑) ๒) ..." run later in the document.
            offsets = [em.start(1)]
            local_next = 2
            gap_start = em.end()
        # else: out-of-sequence match (stray parenthesis) -- ignore it

        if len(offsets) == n:
            last_attempt = validate_candidate(text, n, offsets)
            if last_attempt["action"] == "split":
                return last_attempt
            offsets = []
            local_next = 1
            gap_start = em.end()

    return last_attempt or {"action": "mismatch", "declared": n, "found": 0}


def last_page_marker(text: str, pos: int) -> str | None:
    marks = [pm for pm in PAGE_MARKER.finditer(text) if pm.start() < pos]
    return marks[-1].group(0).strip() if marks else None


TITLE_STOP = re.compile(r"\n\s*\n|(?:^|\n)\s*[๐-๙]{1,2}\.\s")


def extract_title(body: str) -> str:
    """Text between the enumerator and the first blank line or period-style
    subsection marker ("๑. เหตุผลการปรับปรุง"). Not just the first line -- the
    distinguishing part (e.g. an edition year "(ฉบับปี พ.ศ. ๒๕๖๖)") is often a
    wrapped continuation line rather than on the same line as the curriculum
    name, and dropping it can make two genuinely-different items look like
    duplicates. Stopping at the next "๑." subsection marker also keeps the
    last piece's title from swallowing the document's closing boilerplate
    clause when there's no blank line before it."""
    rest = re.sub(r"^[๐-๙]{1,2}[.)]\s*", "", body, count=1)
    m = TITLE_STOP.search(rest)
    para = rest[: m.start()] if m else rest
    para = re.sub(r"\s*\n\s*", " ", para).strip()
    if len(para) > 200:
        para = para[:200].rstrip() + "…"
    return para or "(ไม่พบชื่อหลักสูตรในย่อหน้าแรก — ตรวจมือ)"


def tag_header(preamble: str, part_no: int, total: int) -> str:
    def _sub(mo: re.Match) -> str:
        return mo.group(0) + f" (ส่วนที่ {part_no}/{total})"

    tagged, count = DOC_HEADER.subn(_sub, preamble, count=1)
    return tagged if count else preamble


def build_pieces(text: str, offsets: list[int]) -> list[tuple[str, str]]:
    """Return [(title, piece_text), ...] for each curriculum, in order."""
    preamble = text[: offsets[0]]
    n = len(offsets)
    pieces = []
    for i, start in enumerate(offsets):
        end = offsets[i + 1] if i + 1 < n else len(text)
        body = text[start:end]
        marker = last_page_marker(text, start)
        page_prefix = ""
        if marker and not body.lstrip().startswith(marker):
            page_prefix = marker + "\n\n"
        piece_text = tag_header(preamble, i + 1, n) + "\n\n" + page_prefix + body
        pieces.append((extract_title(body), piece_text))
    return pieces


def read_manifest(mdir: Path) -> list[dict]:
    mf = mdir / "meeting_manifest.json"
    try:
        entries = json.loads(mf.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return []
    return [e for e in entries if isinstance(e, dict)]


def parent_title(mdir: Path, stem: str, fallback: str) -> str:
    for e in read_manifest(mdir):
        if e.get("file") == f"{stem}.md" and e.get("title"):
            return e["title"]
    return fallback


def already_done(mdir: Path, stem: str) -> bool:
    return (mdir / f"{stem}.md.dup").exists() or (mdir / f"{stem}__1.md").exists()


def perform_split(md_path: Path, text: str, offsets: list[int]) -> dict:
    mdir = md_path.parent
    stem = md_path.stem
    link_path = mdir / f"{stem}_LINK.txt"
    url = link_path.read_text(encoding="utf-8-sig").strip() if link_path.exists() else ""

    base_title = parent_title(mdir, stem, fallback=stem)
    pieces = build_pieces(text, offsets)

    new_entries = []
    piece_titles = []
    for k, (curriculum_title, piece_text) in enumerate(pieces, start=1):
        new_stem = f"{stem}__{k}"
        full_title = f"{base_title} — {curriculum_title}"
        piece_titles.append(full_title)
        new_entries.append({
            "file": f"{new_stem}.md",
            "title": full_title,
            "url": url,
            "title_source": "curriculum-split",
        })
        if APPLY:
            (mdir / f"{new_stem}.md").write_text(piece_text, encoding="utf-8")
            (mdir / f"{new_stem}_LINK.txt").write_text(url, encoding="utf-8")

    if APPLY:
        entries = [e for e in read_manifest(mdir) if e.get("file") != f"{stem}.md"]
        entries.extend(new_entries)
        (mdir / "meeting_manifest.json").write_text(
            json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
        md_path.rename(mdir / f"{stem}.md.dup")
        if link_path.exists():
            link_path.rename(mdir / f"{stem}_LINK.txt.dup")

    return {"path": md_path, "titles": piece_titles}


def main() -> None:
    stats: dict[str, int] = {}
    review_lines: list[str] = []
    split_reports: list[dict] = []

    for md_path in sorted(ROOT.rglob("*.md")):
        if md_path.name.endswith(".dup"):
            continue
        mdir = md_path.parent
        stem = md_path.stem
        if SPLIT_PIECE.search(stem):
            continue  # this file IS a split-piece output, never a fresh candidate
        if already_done(mdir, stem):
            stats["already-split"] = stats.get("already-split", 0) + 1
            continue
        if EXCLUDE_TITLE.search(stem):
            stats["excluded-category"] = stats.get("excluded-category", 0) + 1
            continue

        text = md_path.read_text(encoding="utf-8-sig")
        res = find_boundaries(text)
        action = res["action"]
        stats[action] = stats.get(action, 0) + 1
        rel = md_path.relative_to(ROOT)

        if action == "mismatch":
            reason = res.get("reason", f"found {res['found']} boundary marker(s)")
            review_lines.append(
                f"- `{rel}` — declared {res['declared']} หลักสูตร, {reason}. "
                f"Needs manual check.")
        elif action == "split":
            split_reports.append(perform_split(md_path, text, res["offsets"]))

    print("=== counts ===")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {v:5d}  {k}")
    print(f"  {len(split_reports):5d}  split (in stats as 'split' above)")

    print("\n=== split previews ===")
    for r in split_reports:
        print(f"\n{r['path'].relative_to(ROOT)}  ({len(r['titles'])} pieces)")
        for t in r["titles"]:
            print(f"  - {t[:110]}")

    if review_lines:
        print(f"\n=== review queue: {len(review_lines)} (see {REVIEW_FILE.name}) ===")
        for line in review_lines:
            print(line)
    if APPLY:
        REVIEW_FILE.write_text(
            "# Curriculum-split review queue\n\n"
            "Files that declare (or look like they declare) multiple curricula "
            "but could not be split automatically -- the detector refuses to "
            "guess rather than risk a silently wrong split. Many of these are "
            "*correctly* left alone, not bugs to chase:\n\n"
            "- **piece too short / wildly unbalanced** -- often means the real "
            "per-curriculum content is shared/joint (one combined rationale for "
            "2+ closely related programs) rather than separately elaborated; "
            "splitting would duplicate-guess or produce a misleadingly thin "
            "piece. Usually fine to leave as one resolution.\n"
            "- **found N boundary marker(s) != declared** -- format variance the "
            "detector doesn't handle: curricula listed as table rows instead of "
            "\"๑) ๒)\" text, English-only program names with no \"หลักสูตร\" "
            "prefix, or genuinely malformed/inconsistent numbering in the "
            "source OCR.\n\n"
            + ("\n".join(review_lines) if review_lines else "(none)") + "\n",
            encoding="utf-8")
    print(f"\n{'APPLIED' if APPLY else 'DRY RUN (pass --apply to write)'}")


if __name__ == "__main__":
    main()
