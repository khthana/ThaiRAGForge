"""Phase 2 of the consensus-flagged re-OCR pipeline: have phi4:latest and
gemma4:e4b independently judge, for each page staged by
reocr_consensus_pages.py (Phase 1), whether the fresh re-OCR is actually
better than the current corpus text -- the same consensus-of-two-models
methodology already used to flag these pages in the first place
(llm_ocr_scan.py). Still staging-only: nothing is written back to the real
corpus here; that's a separate, later phase gated on these results.

Run with:
    .venv/Scripts/python.exe tools/corpus_prep/reocr_adjudicate.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "consensus_review"))
import logic  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO / "academic_resolutions"
STAGING_FILE = CORPUS_ROOT / "llm_ocr_scan" / "reocr_pages_staging.jsonl"
ADJUDICATION_FILE = CORPUS_ROOT / "llm_ocr_scan" / "reocr_adjudication.jsonl"

MODELS = ["phi4:latest", "gemma4:e4b"]
VERDICTS = ("old", "new", "both_bad", "both_ok")

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}

PROMPT_TEMPLATE = (
    "You are comparing two OCR transcriptions of the SAME page from a Thai "
    "academic-committee meeting resolution (มติสภาวิชาการ), to judge which "
    "one is a more accurate, coherent transcription of that page.\n\n"
    "OLD was flagged by an earlier automated scan as containing garbled/"
    "incoherent text. NEW is a fresh, independent re-OCR of the same page "
    "image, produced afterwards.\n\n"
    "Judge purely on coherence and internal consistency of the Thai/English "
    "prose and table structure -- broken words, nonsense insertions, "
    "scrambled table rows/columns, and repeated or missing content are all "
    "signs of a bad transcription. Do not penalize markdown artifacts (##, "
    "---, <table> tags) or terse/abbreviated academic language -- those are "
    "not errors.\n\n"
    "Respond with JSON only: {\"verdict\": \"old\"|\"new\"|\"both_bad\"|"
    "\"both_ok\", \"reason\": \"one short sentence\"}.\n"
    "- \"new\": NEW is clearly better -- OLD has real garbling NEW does not.\n"
    "- \"old\": OLD is clearly better -- NEW introduced new garbling.\n"
    "- \"both_bad\": both still look garbled/incoherent in some way.\n"
    "- \"both_ok\": both look fine, no meaningful difference in quality.\n\n"
    "--- OLD ---\n%%OLD%%\n--- END OLD ---\n\n"
    "--- NEW ---\n%%NEW%%\n--- END NEW ---"
)


def build_prompt(old_text: str, new_text: str) -> str:
    return PROMPT_TEMPLATE.replace("%%OLD%%", old_text).replace("%%NEW%%", new_text)


def call_compare_model(model: str, old_text: str, new_text: str, retries: int = 2) -> dict:
    """Same retry shape as llm_ocr_scan.call_model: temperature=0 first (so a
    malformed/truncated response is worth retrying with a different sample),
    0.3 after. A verdict outside VERDICTS is treated as a parse failure, not
    trusted through -- Phase 3's replace/keep decision depends on this being
    one of the four known values."""
    last_err = ""
    start = time.monotonic()
    for attempt in range(1, retries + 1):
        try:
            temperature = 0.0 if attempt == 1 else 0.3
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": build_prompt(old_text, new_text)}],
                format=SCHEMA,
                think=False,
                options={"temperature": temperature, "num_ctx": 8192, "num_predict": 300},
            )
            content = response["message"]["content"]
            parsed = json.loads(content)
            verdict = str(parsed.get("verdict", ""))
            if verdict not in VERDICTS:
                raise ValueError(f"unexpected verdict: {verdict!r}")
            return {
                "verdict": verdict,
                "reason": str(parsed.get("reason", ""))[:300],
                "error": None,
                "elapsed_s": round(time.monotonic() - start, 2),
            }
        except Exception as e:  # malformed JSON, bad verdict, model error, timeout
            last_err = str(e)
            time.sleep(1)
    return {"verdict": None, "reason": "", "error": last_err, "elapsed_s": round(time.monotonic() - start, 2)}


@dataclass(frozen=True)
class StagedPage:
    pdf: str
    page: int
    files: tuple[str, ...]
    new_text: str
    timestamp: str


def load_staged_pages(staging_file: Path) -> list[StagedPage]:
    pages = []
    for line in staging_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        pages.append(StagedPage(
            pdf=d["pdf"], page=d["page"], files=tuple(d["files"]),
            new_text=d["new_text"], timestamp=d["timestamp"],
        ))
    return pages


def resolve_old_text(corpus_root: Path, staged: StagedPage) -> tuple[str | None, list[str]]:
    """The current corpus text for a staged page (from the first file in
    `staged.files`), plus any sibling files whose stored text for the same
    page differs from it. Split-document siblings normally carry
    byte-identical copies of shared pages (ADR-0004), so a divergence is
    worth surfacing rather than silently picking one arbitrarily -- Phase 3
    still has to decide what to do about it, this just refuses to hide it."""
    page_label = f"Page {staged.page}"
    texts = {f: logic.load_page_markdown(corpus_root, f, page_label) for f in staged.files}
    primary = texts[staged.files[0]]
    diverging = [f for f, t in texts.items() if f != staged.files[0] and t != primary]
    return primary, diverging


def load_done_keys(adjudication_file: Path) -> set[tuple[str, int]]:
    if not adjudication_file.exists():
        return set()
    done = set()
    for line in adjudication_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        done.add((d["pdf"], d["page"]))
    return done


@dataclass(frozen=True)
class AdjudicationResult:
    pdf: str
    page: int
    files: tuple[str, ...]
    old_text_source: str
    diverging_siblings: tuple[str, ...]
    verdicts: dict
    timestamp: str


def append_result(adjudication_file: Path, result: AdjudicationResult) -> None:
    adjudication_file.parent.mkdir(parents=True, exist_ok=True)
    with adjudication_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def main() -> None:
    staged_pages = load_staged_pages(STAGING_FILE)
    done = load_done_keys(ADJUDICATION_FILE)
    todo = [p for p in staged_pages if (p.pdf, p.page) not in done]
    print(f"[INFO] {len(staged_pages)} staged pages, {len(done)} already adjudicated, {len(todo)} remaining")

    for i, staged in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {Path(staged.pdf).name} -- page {staged.page}")
        old_text, diverging = resolve_old_text(CORPUS_ROOT, staged)
        if diverging:
            print(f"   [WARN] sibling text diverges for: {diverging}")
        if old_text is None:
            print("   [SKIP] old text not found in corpus")
            continue

        verdicts = {model: call_compare_model(model, old_text, staged.new_text) for model in MODELS}

        result = AdjudicationResult(
            pdf=staged.pdf,
            page=staged.page,
            files=staged.files,
            old_text_source=staged.files[0],
            diverging_siblings=tuple(diverging),
            verdicts=verdicts,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        append_result(ADJUDICATION_FILE, result)

    print("[FINISH]")


if __name__ == "__main__":
    main()
