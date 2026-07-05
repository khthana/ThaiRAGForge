# -*- coding: utf-8 -*-
"""Rebuild per-meeting manifests + the reconciled master list (ADR-0003).

Scans the corpus tree `academic_resolutions/<ปี>/ครั้งที่ N[s]/` and, if the
hand-captured agenda `1.docx` is still present at the repo root, reconciles the
two by link identity (Google Drive file id / open?id / folder id / full URL) —
joining first on the id, then on a year+meeting+title fallback for stale docx
URLs. `1.docx` is now optional: once it has been retired (master_list.csv
supersedes it) the tool runs corpus-only. It writes:

- `academic_resolutions/<ปี>/ครั้งที่ N[s]/meeting_manifest.json`
    the metadata source of truth: {file, title, url, title_source} per .md —
    titles are recovered from the docx because filenames are truncated ~100 chars
- `academic_resolutions/master_list.csv`
    one row per resolution, with a reconciliation status (+ dup-link health count)

Re-run this after adding/renaming corpus files (or editing 1.docx, if kept):

    python tools/corpus_prep/rebuild_manifests.py            # dry-run (counts only)
    python tools/corpus_prep/rebuild_manifests.py --apply    # write everything

Expects the normalized folder convention (ครั้งที่ N / ครั้งที่ Ns) and ignores
`*.dup` files (duplicates parked by the 2569-07 reconciliation).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

sys.stdout.reconfigure(encoding="utf-8")
APPLY = "--apply" in sys.argv

REPO = Path(__file__).resolve().parents[2]
DOCX = REPO / "1.docx"
ROOT = REPO / "academic_resolutions"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
DRIVE = re.compile(r"/d/([A-Za-z0-9_-]{20,})|[?&]id=([A-Za-z0-9_-]{20,})|/folders/([A-Za-z0-9_-]{20,})")
NEW_ITEM = re.compile(r"^\s*(\d+\s*\.)?\s*เรื่อง")
HEADER = re.compile(r"^\s*(\d{1,2})\s*/\s*(25\d{2})\s*(s?)")
DOC_HEADER = re.compile(r"^\s*#\s*Document:.*$", re.MULTILINE)


def link_key(url: str | None) -> str | None:
    if not url:
        return None
    m = DRIVE.search(url)
    return next(g for g in m.groups() if g) if m else url.strip()


def clean_title(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub("[\xa0​]", " ", s)  # capture artifacts: bullet, nbsp, zwsp
    return re.sub(r"\s+", " ", s).strip()


def norm_for_match(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", "", s).replace("_", "/")


def body_hash(p: Path) -> str:
    """Content hash ignoring the OCR `# Document:` header (differs per filename)."""
    text = DOC_HEADER.sub("", p.read_text(encoding="utf-8-sig"), count=1).strip()
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ---------- 1. parse 1.docx: paragraphs -> agenda items ----------
def parse_docx() -> list[dict]:
    if not DOCX.exists():
        # 1.docx has been retired (master_list.csv supersedes it). Run corpus-only:
        # every .md is kept, titles/urls come from _LINK.txt + filenames.
        print("note: 1.docx not found — running corpus-only (no docx reconciliation)")
        return []
    z = zipfile.ZipFile(DOCX)
    dom = ET.fromstring(z.read("word/document.xml"))
    rels = ET.fromstring(z.read("word/_rels/document.xml.rels"))
    rid2url = {rel.get("Id"): rel.get("Target") for rel in rels}

    items: list[dict] = []
    cur = None
    order = 0
    for p in dom.iter(f"{{{W_NS}}}p"):
        links = []
        for hl in p.iter(f"{{{W_NS}}}hyperlink"):
            url = rid2url.get(hl.get(f"{{{R_NS}}}id"))
            t = "".join(t.text or "" for t in hl.iter(f"{{{W_NS}}}t"))
            links.append((t, url))
        text = clean_title("".join(t.text or "" for t in p.iter(f"{{{W_NS}}}t")))

        hm = HEADER.match(text)
        if hm:
            cur = (hm.group(2), f"{int(hm.group(1))}{hm.group(3)}")
        if not links:
            # text but no hyperlink: a probable lost-link agenda item
            if text and not hm and NEW_ITEM.match(text):
                order += 1
                items.append({"meet": cur, "title": text, "url": None,
                              "key": None, "order": order, "lost_link": True})
            continue
        for t, u in links:
            t = clean_title(t)
            if not t:
                continue
            k = link_key(u)
            prev = items[-1] if items else None
            if prev and prev["meet"] == cur and prev["key"] and k == prev["key"]:
                if t == prev["title"]:
                    continue                      # exact duplicate line
                if not NEW_ITEM.match(t):
                    prev["title"] += " " + t      # wrapped-title continuation
                    continue
                # else: a distinct item sharing the same link -> its own row
            order += 1
            items.append({"meet": cur, "title": t, "url": u, "key": k,
                          "order": order, "lost_link": False})
    return items


# ---------- 2. scan the corpus tree ----------
def _prev_titles(mdir: Path) -> dict[str, str]:
    """Existing manifest titles keyed by .md filename, so a corpus-only re-run
    (after 1.docx is retired) keeps the full titles once recovered from the docx
    instead of falling back to the ~100-char-truncated filename."""
    try:
        entries = json.loads((mdir / "meeting_manifest.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return {e["file"]: e["title"] for e in entries
            if isinstance(e, dict) and e.get("file") and e.get("title")}


def scan_corpus() -> list[dict]:
    files: list[dict] = []
    for year_dir in sorted(d for d in ROOT.iterdir() if d.is_dir()):
        for mdir in sorted(d for d in year_dir.iterdir() if d.is_dir()):
            sm = re.fullmatch(r"ครั้งที่ (\d+s?)", mdir.name)
            meet = (year_dir.name, sm.group(1)) if sm else (year_dir.name, mdir.name)
            prev = _prev_titles(mdir)
            stems_md = {f.name[:-3] for f in mdir.glob("*.md")}
            seen = set()
            for lf in sorted(mdir.glob("*_LINK.txt")):
                stem = lf.name[: -len("_LINK.txt")]
                url = lf.read_text(encoding="utf-8-sig").strip()
                md = mdir / f"{stem}.md"
                files.append({"year": year_dir.name, "sdir": mdir.name, "meet": meet,
                              "stem": stem, "md": md if md.exists() else None,
                              "url": url, "key": link_key(url), "docx": None,
                              "prev_title": prev.get(f"{stem}.md")})
                seen.add(stem)
            for stem in stems_md - seen:
                files.append({"year": year_dir.name, "sdir": mdir.name, "meet": meet,
                              "stem": stem, "md": mdir / f"{stem}.md",
                              "url": None, "key": None, "docx": None,
                              "prev_title": prev.get(f"{stem}.md")})
    return files


docx_items = parse_docx()
files = scan_corpus()
print(f"docx: {len(docx_items)} items "
      f"({sum(1 for i in docx_items if i['lost_link'])} lost-link)")
print(f"folder: {len(files)} entries, {sum(1 for f in files if f['md'])} with .md")

# ---------- 3. join by link identity ----------
by_key_files = defaultdict(list)
for f in files:
    if f["key"]:
        by_key_files[f["key"]].append(f)
by_key_docx = defaultdict(list)
for it in docx_items:
    if it["key"]:
        by_key_docx[it["key"]].append(it)

for key, its in by_key_docx.items():
    cands = by_key_files.get(key, [])
    for it in its:
        pool = [f for f in cands if f["meet"] == it["meet"] and f["docx"] is None] or \
               [f for f in cands if f["docx"] is None]
        if not pool:
            continue
        tn = norm_for_match(it["title"])

        def score(f):
            sn = norm_for_match(f["stem"])
            if tn.startswith(sn) or sn.startswith(tn):
                return (2, min(len(sn), len(tn)))
            i = 0
            while i < min(len(sn), len(tn)) and sn[i] == tn[i]:
                i += 1
            return (1, i)

        best = max(pool, key=score)
        best["docx"] = it
        it.setdefault("files", []).append(best)

# lost-link docx items: match by title prefix within the meeting
for it in docx_items:
    if it["key"] is None and not it.get("files"):
        tn = norm_for_match(it["title"])
        pool = [f for f in files if f["meet"] == it["meet"] and f["docx"] is None]
        hits = [f for f in pool if tn.startswith(norm_for_match(f["stem"])) or
                norm_for_match(f["stem"]).startswith(tn)]
        if len(hits) == 1:
            hits[0]["docx"] = it
            it["files"] = hits

# secondary pass: docx item HAS a key but it matched no file (the file's _LINK
# was corrected to a precise per-file URL, so keys diverged) OR its key collides
# with a sibling that already claimed the file. Recover the pair by matching on
# meeting + title against a still-unclaimed file, and flag the docx URL as stale.
for it in docx_items:
    if it.get("files"):
        continue
    tn = norm_for_match(it["title"])
    pool = [f for f in files if f["meet"] == it["meet"] and f["docx"] is None]
    hits = [f for f in pool if tn.startswith(norm_for_match(f["stem"])) or
            norm_for_match(f["stem"]).startswith(tn)]
    if len(hits) == 1:
        hits[0]["docx"] = it
        it["files"] = hits
        it["url_stale"] = True

print(f"matched file<->docx: {sum(1 for f in files if f['docx'])}")

# ---------- 4. suspect duplicate-link groups (same key, same folder) ----------
dup_flagged = []   # {dir, url, files: [folder entries], stems, why}
by_dir_key = defaultdict(list)
for f in files:
    if f["key"]:
        by_dir_key[(f["year"], f["sdir"], f["key"])].append(f)
for (year, sdir, key), group in sorted(by_dir_key.items()):
    if len(group) < 2:
        continue
    mds = [f["md"] for f in group if f["md"]]
    identical = len(mds) >= 2 and len({body_hash(m) for m in mds}) == 1
    dup_flagged.append({"dir": f"{year}/{sdir}", "url": group[0]["url"],
                        "files": group, "stems": [f["stem"] for f in group],
                        "why": "identical" if identical else "content-differs-or-missing"})
flagged_stems = {(d["dir"], s) for d in dup_flagged for s in d["stems"]}
print(f"suspect duplicate-link groups: {len(dup_flagged)}")


def is_flagged(f) -> bool:
    return (f"{f['year']}/{f['sdir']}", f["stem"]) in flagged_stems


# ---------- 5. meeting manifests ----------
dirs = defaultdict(list)
for f in files:
    dirs[(f["year"], f["sdir"])].append(f)

for (year, sdir), group in sorted(dirs.items()):
    entries = []
    for f in group:
        if not f["md"]:
            continue  # link-only rows go to the missing report, not the manifest
        it = f["docx"]
        if it:
            title, source = it["title"], "docx"
        elif f["prev_title"]:
            title, source = f["prev_title"], "manifest"
        else:
            title, source = clean_title(f["stem"]), "filename"
        entries.append({
            "file": f["md"].name,
            "title": title,
            "url": f["url"],
            "title_source": source,
            "order": it["order"] if it else 10_000,
        })
    entries.sort(key=lambda e: (e["order"], e["file"]))
    for e in entries:
        del e["order"]
    if APPLY:
        (ROOT / year / sdir / "meeting_manifest.json").write_text(
            json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"manifests {'written' if APPLY else 'planned'}: {len(dirs)}")

# ---------- 6. master list ----------
rows = []
for it in docx_items:
    fs = it.get("files", [])
    y, m = it["meet"] if it["meet"] else ("?", "?")
    if fs:
        f = fs[0]
        status = "ครบ" if f["md"] else "มีลิงก์ แต่ไม่มีไฟล์ .md (ดาวน์โหลด/OCR ไม่สำเร็จ)"
        note = "ลิงก์ใน docx หลุด (ใช้ URL จาก _LINK.txt)" if it["lost_link"] else ""
        if it.get("url_stale"):
            note = (note + "; " if note else "") + \
                "URL ใน docx ล้าสมัย — จับคู่ด้วยชื่อเรื่อง (ใช้ URL จาก _LINK.txt)"
        if is_flagged(f):
            note = (note + "; " if note else "") + \
                "ลิงก์ซ้ำกับเรื่องอื่น — ตรวจสอบว่าไฟล์/URL ถูกเรื่องหรือไม่"
        rows.append([y, m, it["order"], it["title"], f["url"] or (it["url"] or ""),
                     f"{f['year']}/{f['sdir']}/{f['md'].name if f['md'] else ''}",
                     status, note])
    elif it["lost_link"]:
        rows.append([y, m, it["order"], it["title"], "",
                     "", "ไม่พบไฟล์ และลิงก์ใน docx หลุด", "หา URL และดาวน์โหลดใหม่"])
    elif it["key"] and any(f["docx"] for f in by_key_files.get(it["key"], [])):
        rows.append([y, m, it["order"], it["title"], it["url"] or "",
                     "", "ลิงก์ซ้ำกับรายการอื่นที่มีไฟล์แล้ว",
                     "URL เดียวกันถูกใช้กับอีกรายการ — อาจเป็นบรรทัดซ้ำใน docx หรือลิงก์ผิดเรื่อง (ตรวจมือ)"])
    else:
        rows.append([y, m, it["order"], it["title"], it["url"] or "",
                     "", "ไม่มีไฟล์เลย", "ต้องดาวน์โหลด + OCR"])

for f in files:
    if f["docx"] is None:
        y, m = f["meet"]
        # a file that exists on disk is already in the manifest + index, so it is
        # "ครบ" regardless of 1.docx (master_list supersedes docx); provenance
        # goes to the note.
        status = "ครบ" if f["md"] else "มีลิงก์ แต่ไม่มีไฟล์ .md และไม่มีใน docx"
        note = "ไฟล์มีจริง จับเข้าคลังแล้ว (ไม่มีรายการใน 1.docx)"
        if is_flagged(f):
            note += "; ลิงก์ซ้ำกับเรื่องอื่น — ตรวจมือ"
        rows.append([y, m, "", f["prev_title"] or clean_title(f["stem"]), f["url"] or "",
                     f"{f['year']}/{f['sdir']}/{f['md'].name if f['md'] else ''}",
                     status, note])


def sort_key(r):
    y = r[0] if r[0] != "?" else "9999"
    m = str(r[1])
    return (y, m.endswith("s"), int(re.sub(r"\D", "", m) or 99),
            r[2] if r[2] != "" else 10_000)


rows.sort(key=sort_key)
if APPLY:
    with open(ROOT / "master_list.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ปี", "การประชุม", "ลำดับใน docx", "ชื่อเรื่อง", "URL",
                    "ไฟล์", "สถานะ", "หมายเหตุ"])
        w.writerows(rows)
print(f"master rows: {len(rows)}")
summary = defaultdict(int)
for r in rows:
    summary[r[6]] += 1
for k, v in sorted(summary.items(), key=lambda x: -x[1]):
    print(f"  {v:5d}  {k}")
