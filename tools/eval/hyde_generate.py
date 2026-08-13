"""Generate the HyDE hypothetical documents once, cache them, and describe them.

HyDE is a pure **query transform**: an LLM writes the passage that *would* answer
the question, and that passage is embedded instead of the question. Nothing about
it depends on the index, so the generation happens once here and every combo in
`hyde_retrieval_test.py` reads the same cache.

Three reasons this is a separate script from the measurement, not a phase of it:

1. **`temperature=0` is not reproducible on this stack.** The pricing probe's
   first three runs reproduced output-token counts exactly and the fourth
   disagreed on every query ([[feedback_a_streak_of_identical_runs_is_not_determinism]]).
   If each arm regenerated its own documents, the arms would differ by generator
   noise as well as by treatment and nothing would be paired. One cache, read by
   everything, makes the comparison paired by construction -- the same rule
   `rq4_generate.py` follows by skipping answers that already exist.
2. **It is the only GPU-bound half.** 285 queries at ~7.85 s is ~37 min; the
   retrieval sweep that consumes it is numpy over persisted embeddings. Splitting
   them means a crash in the cheap half never re-spends the expensive half, and
   the cache is written after **every** query for the same reason.
3. **The documents are evidence in their own right.** `docs/hyde-axis-notes.md`
   predicts failure largely on the claim that `phi4` has never seen these people,
   programmes or meeting numbers and will therefore embed fabricated ones. That
   is a property of the generation, not of the retrieval, so it is measured here
   (SS3) rather than left as an argument.

The prompt is **imported** from `probe_hyde_generation_cost.py` rather than
copied, so the price that report published is the price of this exact prompt, and
the axis cannot be quietly tuned by editing a second copy. It is deliberately
unengineered: a HyDE run with a prompt tuned against the gold set would be
measuring the tuning.

Guards, in the shape this project has learned to prefer over prose
([[feedback_an_asserted_invariant_is_not_a_check]]):

* The cache must stay **homogeneous** -- one model, one `num_predict`, one
  `num_ctx`. A cache half-written by a second generator would silently make the
  treatment arm a mixture, and no downstream check could see it. Overriding needs
  `--allow-mixed-cache`.
* Every generation is checked for ollama's truncation signature
  (`prompt_eval_count == num_ctx // 2 + 2`, see `docs/rq4-prompt-truncation.md`).
  A HyDE prompt is ~300 tokens so this cannot fire today, which is exactly why it
  is cheap to leave wired for the day someone lengthens the prompt.
* An existing entry is **never** overwritten. To regenerate, move the cache file;
  that keeps the old documents as the only record of what was measured.

Read-only apart from its own cache and report. Writes no index.

Usage:
    python tools/eval/hyde_generate.py            # generate what is missing
    python tools/eval/hyde_generate.py --render   # rebuild the report, no GPU
    python tools/eval/hyde_generate.py --set 73det
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "eval"))

from probe_hyde_generation_cost import CAP, MODEL, PROMPT  # noqa: E402

GOLD_73DET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
GOLD_FULL = REPO / "config" / "eval" / "gold_query_set.yaml"
CACHE = REPO / "data" / "results" / "hyde_documents.json"
OUT = REPO / "data" / "results" / "hyde_generation.md"

NUM_CTX = 8192

_DIGITS = re.compile(r"[0-9๐-๙]+")
_WS = re.compile(r"\s+")


def load_set(name: str) -> list[dict]:
    """The queries of one set, in file order, with their gold metadata."""
    if name == "73det":
        rows = yaml.safe_load(GOLD_73DET.read_text(encoding="utf-8"))
    elif name == "thematic":
        rows = [
            r
            for r in yaml.safe_load(GOLD_FULL.read_text(encoding="utf-8"))
            if r.get("entity_type") == "thematic"
        ]
    else:  # pragma: no cover - argparse restricts this
        raise ValueError(name)
    return rows


def load_cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    return json.loads(CACHE.read_text(encoding="utf-8"))


def save_cache(cache: dict[str, dict]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )


def cache_settings(cache: dict[str, dict]) -> set[tuple]:
    """The distinct (model, num_predict, num_ctx) triples the cache was built with."""
    return {(e["model"], e["num_predict"], e["num_ctx"]) for e in cache.values()}


def generate(query: str) -> dict:
    import ollama

    t0 = time.perf_counter()
    resp = ollama.generate(
        model=MODEL,
        prompt=PROMPT.format(q=query),
        options={"temperature": 0.0, "num_ctx": NUM_CTX, "num_predict": CAP},
    )
    wall = time.perf_counter() - t0
    prompt_tok = resp.get("prompt_eval_count")
    out_tok = resp.get("eval_count")
    if prompt_tok == NUM_CTX // 2 + 2:
        raise SystemExit(
            f"truncation signature: prompt_eval_count == num_ctx//2+2 == {prompt_tok}. "
            "The prompt exceeded num_ctx and ollama kept only its tail. Raise --num-ctx."
        )
    return {
        "doc": (resp.get("response") or "").strip(),
        "model": MODEL,
        "num_predict": CAP,
        "num_ctx": NUM_CTX,
        "prompt_tok": prompt_tok,
        "out_tok": out_tok,
        "hit_cap": out_tok is not None and out_tok >= CAP,
        "wall": wall,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def collapse(s: str) -> str:
    return _WS.sub(" ", s).strip()


def contains_entity(doc: str, entity: str) -> bool:
    """Does the generated document literally contain the query's own anchor?

    Whitespace-collapsed on both sides and case-folded, because the corpus's own
    matchers do the same (OCR'd minutes wrap long names across lines) and a
    difference in spacing is not a difference in content.
    """
    if not entity:
        return False
    return collapse(entity).casefold() in collapse(doc).casefold()


def novel_number_runs(doc: str, query: str) -> int:
    """Digit runs in the document that do not appear in the query.

    A **proxy** for fabricated numbers, not a measurement of them: a generated
    year that happens to match one in the query is not counted, and a correct
    number the model could not have known is counted as novel. It is reported as
    a proxy and nothing is gated on it.
    """
    in_q = set(_DIGITS.findall(query))
    return sum(1 for run in _DIGITS.findall(doc) if run not in in_q)


def describe(sets: dict[str, list[dict]], cache: dict[str, dict]) -> str:
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    L: list[str] = []
    L.append("# HyDE hypothetical documents - what the generator actually wrote")
    L.append("")
    L.append(f"Generated by `tools/eval/hyde_generate.py`; rendered {now}.")
    L.append("")
    L.append(
        "Cache: `data/results/hyde_documents.json`. One document per query, "
        "generated once and read by every arm of `hyde_retrieval_test.py` -- "
        "`temperature=0` is not reproducible on this stack, so regenerating per "
        "arm would unpair the comparison."
    )
    L.append("")
    settings = sorted(cache_settings(cache))
    L.append(
        "Settings: "
        + "; ".join(f"model `{m}`, num_predict {np_}, num_ctx {nc}" for m, np_, nc in settings)
        + f" ({len(settings)} distinct - a cache with more than one is a mixture)."
    )
    L.append("")

    L.append("## 1. Coverage and cost")
    L.append("")
    L.append("| set | queries | cached | out tokens median | hit the cap | wall total |")
    L.append("|---|---|---|---|---|---|")
    for name, rows in sets.items():
        qs = [r["query"] for r in rows]
        have = [cache[q] for q in qs if q in cache]
        if not have:
            L.append(f"| {name} | {len(qs)} | 0 | - | - | - |")
            continue
        toks = [e["out_tok"] for e in have if e["out_tok"] is not None]
        capped = sum(1 for e in have if e["hit_cap"])
        wall = sum(e["wall"] for e in have)
        L.append(
            f"| {name} | {len(qs)} | {len(have)} | "
            f"{statistics.median(toks):.0f} | {capped} of {len(have)} "
            f"({capped / len(have):.1%}) | {wall / 60:.1f} min |"
        )
    L.append("")
    L.append(
        f"**Hitting the cap means the paragraph ends mid-sentence.** The cap "
        f"(`num_predict={CAP}`) was adopted because the price is output-bound and "
        "the prompt's own \"no more than 5 sentences\" instruction enforces nothing "
        "-- see `data/results/hyde_generation_cost.md`. A high cap-hit rate is not "
        "a defect to fix here, it is a fact about the treatment being measured, and "
        "it is reported so a null result cannot later be blamed on it unexamined."
    )
    L.append("")

    L.append("## 2. Does the document keep the query's own anchor?")
    L.append("")
    L.append(
        "`docs/hyde-axis-notes.md` predicts failure mainly on dilution: on 73det "
        "the discriminative signal is an exact token (**BM25 alone scores 0.8147 "
        "on `person`**), and generated filler can only crowd it out. That is "
        "checkable directly -- does the generated paragraph still contain the "
        "entity the query names?"
    )
    L.append("")
    L.append("| set | entity_type | n | document contains the entity |")
    L.append("|---|---|---|---|")
    for name, rows in sets.items():
        by_type: dict[str, list[dict]] = {}
        for r in rows:
            if r["query"] in cache:
                by_type.setdefault(r.get("entity_type", "?"), []).append(r)
        for et in sorted(by_type):
            grp = by_type[et]
            hits = sum(
                1 for r in grp if contains_entity(cache[r["query"]]["doc"], r.get("entity", ""))
            )
            L.append(f"| {name} | {et} | {len(grp)} | {hits} ({hits / len(grp):.1%}) |")
    L.append("")

    L.append("## 3. Numbers the generator supplied by itself (proxy)")
    L.append("")
    L.append(
        "Digit runs present in the document but absent from the query. A **proxy** "
        "for the fabricated meeting numbers, years and codes the notes predict, not "
        "a measurement of them: a generated year matching one in the query is not "
        "counted, and nothing is gated on this."
    )
    L.append("")
    L.append("| set | n | mean novel digit runs | documents with >= 1 |")
    L.append("|---|---|---|---|")
    for name, rows in sets.items():
        have = [r for r in rows if r["query"] in cache]
        if not have:
            continue
        counts = [novel_number_runs(cache[r["query"]]["doc"], r["query"]) for r in have]
        nz = sum(1 for c in counts if c)
        L.append(
            f"| {name} | {len(have)} | {statistics.mean(counts):.2f} | "
            f"{nz} ({nz / len(have):.1%}) |"
        )
    L.append("")

    L.append("## 4. Two documents in full")
    L.append("")
    L.append(
        "Read them before reading any score. The whole axis rests on what these "
        "paragraphs are, and a table of token counts cannot show that."
    )
    for name, rows in sets.items():
        for r in rows:
            if r["query"] in cache:
                e = cache[r["query"]]
                L.append("")
                L.append(f"**{name}** / `{r.get('entity_type', '?')}` / "
                         f"entity `{r.get('entity', '')}`")
                L.append("")
                L.append("> query: " + collapse(r["query"]))
                L.append("")
                L.append("```")
                L.append(e["doc"])
                L.append("```")
                break
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=["73det", "thematic", "both"], default="both")
    ap.add_argument("--render", action="store_true", help="rebuild the report only, no GPU")
    ap.add_argument("--limit", type=int, default=0, help="generate at most N (smoke)")
    ap.add_argument("--allow-mixed-cache", action="store_true")
    ap.add_argument("--keep-loaded", action="store_true")
    args = ap.parse_args()

    names = ["73det", "thematic"] if args.set == "both" else [args.set]
    sets = {n: load_set(n) for n in names}
    cache = load_cache()

    settings = cache_settings(cache)
    if len(settings) > 1 and not args.allow_mixed_cache:
        raise SystemExit(
            f"cache is already a mixture of {len(settings)} settings: {sorted(settings)}. "
            "Every arm reads this file, so a mixture makes the treatment a blend of "
            "generators. Pass --allow-mixed-cache only if that is intended."
        )
    if settings and (MODEL, CAP, NUM_CTX) not in settings and not args.allow_mixed_cache:
        raise SystemExit(
            f"cache was built with {sorted(settings)} but this run would add "
            f"('{MODEL}', {CAP}, {NUM_CTX}). Move the cache file to regenerate, or pass "
            "--allow-mixed-cache."
        )

    if not args.render:
        todo = [r["query"] for rows in sets.values() for r in rows if r["query"] not in cache]
        if args.limit:
            todo = todo[: args.limit]
        total = sum(len(rows) for rows in sets.values())
        have = sum(1 for rows in sets.values() for r in rows if r["query"] in cache)
        print(f"sets: {', '.join(f'{n}={len(sets[n])}' for n in names)}  "
              f"cached {have}/{total}  to generate: {len(todo)}", flush=True)
        t0 = time.time()
        for i, q in enumerate(todo, 1):
            cache[q] = generate(q)
            save_cache(cache)  # after every query: a crash must not re-spend the GPU
            done, el = i, time.time() - t0
            eta = (el / done) * (len(todo) - done)
            print(f"  [{i}/{len(todo)}] out={cache[q]['out_tok']} "
                  f"{cache[q]['wall']:.1f}s  elapsed {el / 60:.1f}m  eta {eta / 60:.1f}m",
                  flush=True)
        if todo and not args.keep_loaded:
            import ollama

            ollama.generate(model=MODEL, prompt="", keep_alive=0)
            print("unloaded")

    missing = {n: sum(1 for r in rows if r["query"] not in cache) for n, rows in sets.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(describe(sets, cache), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}  (missing: {missing})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
