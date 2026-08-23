# `fetch_depth` — the over-fetch axis, full derivation

**This is the research record. The decision, bounds and traps live in
`CLAUDE.md`; this file is why they hold.** Assembled 2026-08-23 from three places
that had grown apart: the unrouted `rrf` sweep (which had been written inside the
chunker/embedder bullet), the `weighted` arm, and the routed ship decision. All
three are **verbatim**, and this file is in `audit_doc_claims.DOCS` so their
figures stay under D2/D5/D7.

Reports: `data/results/hybrid_fetch_depth_sweep.md`,
`hybrid_weighted_fetch_depth.md`, `routed_fetch_depth_test.md` — the first two
were re-run against rebuild #4 on 2026-08-23 **as a pair**, because the weighted
run's `S7` cross-anchors the published sweep and refreshing either alone breaks
that anchor.

---

## 1. The unrouted `rrf` sweep

**The over-fetch is now MEASURED, and it is a quality/latency trade rather than an
optimisation (2026-08-09, `tools/eval/hybrid_fetch_depth_sweep.py` →
`data/results/hybrid_fetch_depth_sweep.md`).** `HybridRetriever` gained a `fetch_depth`
knob whose default `None` computes `depth = len(index.chunks)` — literally the old k=n
expression, so every published hybrid number is reproduced *by construction* and
`tests/retrievers/test_hybrid_retriever.py` pins it; the sweep runs F ∈ {10 … 10,000, n}
over 36 combos × 106 queries = 3,816 pairs. **The two questions have opposite answers, and
that is the finding.** *Is the ranking the same?* — only at F=n: **F=10,000 reproduces just
87.84%** of top-10s in order (96.59% as a set) and F=1,000 only 70.13%. The pre-registered
guess "F=1000 will be identical" was **wrong**, recorded as such. *Does it matter?* —
barely: macro recall@10 across the 36 combos is 0.5197 at k=n, **0.5162 at F=100
(−0.0035)** and **0.5170 at F=200 (−0.0027)**, and it is **non-monotonic** (F=500's −0.0015
is better than F=1,000's −0.0025) because truncation lifts different chunks' scores at
different rates as F grows. Mechanism worth keeping: a chunk inside dense's top-F but past
BM25's cut loses its BM25 term **outright**, not by a little — that is why this is not an
approximation that merely loses precision. Damage concentrates exactly where this project's
RRF rule predicts (worst combo at F=50 `semantic × e5_small` −0.0579, at F=200
`recursive × bge_m3` −0.0224, at F=1,000 `sentence × sct` −0.0145 — **the last two
identical to the pre-rebuild run, combo and value both**), and **`person` queries
*gain* at F=50 (+0.0217)** — the only entity_type that does, consistent with BM25 carrying
`person` (0.8147) while the cut deletes a weak dense arm's tail. **Timing (paired, one
process, one loaded index, arms alternated per query, BM25 scorer pre-warmed so its one-off
build lands in neither arm, `plain__sentence__qwen3__ff8f6c49`): k=n p50 1089.5 ms → F=200
**417.9 ms** (−0.672 s, 2.6x), F=1,000 421.0 ms.** So the over-fetch is **~62% of hybrid
query time**, and the ~0.42 s left is real scoring work (dense encode + gemv + `get_scores`)
that no depth cut can touch — **do not read the earlier "the remaining ~1.36s is the k=n
over-fetch" as all removable**; that sentence bundled the residual in. **Re-run 2026-08-23 against rebuild #4 (the figures above are that run's): every
finding survived and only the levels moved** — the two questions still answer
oppositely, non-monotonicity still holds, `person` is still the only type that
gains, and the F=200/F=1,000 worst combos are unchanged in both combo and value.
All 6 self-checks pass at full scale (S2 and S4 both 3,816 reproduce / 0 differ
against the *current* persisted results, which is what confirms the refresh is
aligned with the rebuild).
The trade on the
table was ~0.67 s/query for −0.0027 macro
recall@10 at F=200 — a *cost* decision of the same shape as soft-vs-hard routing, and it
needed re-measuring against the hard router (which now ships) before adoption, since
that macro figure is an average over a whole combo family, not a system result.
**That re-measurement is DONE and the decision is made — see the next bullet.** Two method
notes: the sweep replicates the **truncated** tie-break (the fusion dict is filled
`dense[:F]` first, then the BM25-only remainder in BM25 rank order, so equal RRF scores stay
dense-first — the same trap `miss_depth_profile.py` documents at full depth), and **S5
checks the numpy fusion against a real `HybridRetriever(fetch_depth=F)`**, added because the
first version anchored only F=n, where the mechanism under test is inert and the check would
have passed identically had `fuse_at_depth` ignored F. Everything the report renders is
cached in `hybrid_fetch_depth_raw.json`, so `--render` reproduces it without a GPU.

---

## 2. The `weighted` arm, and the routed ship decision

- **`weighted` × `fetch_depth`: MEASURED 2026-08-12 and the guard is LIFTED
(`tools/eval/hybrid_weighted_fetch_depth.py` → `data/results/hybrid_weighted_fetch_depth.md`,
36 combos × 106 queries = 3,816 pairs, 16 min).** From 08-11 to 08-12 that pair *raised*
in `HybridRetriever.__init__`, and the entry here said so — **containment, not a verdict**,
with its own exit condition written into it ("measure it and lift the guard"). The
measurement was run and the pre-registered rule (frozen in the script as `DECISION_RULE`,
committed before the run) came out **LIFT**: the raise and its `allow_unmeasured_truncation`
hatch are gone, and `tests/retrievers/test_hybrid_retriever.py` now pins that permitting the
pair did not quietly make it a **no-op** — truncation under `weighted` must still really
truncate, since that is the whole cost. **LIFT is not a recommendation and the number is the
point**: at F=200 `weighted` loses **−0.0605** macro recall@10 against its own F=n, about
**22x** `rrf`'s −0.0027 at the same depth (it read 18x against −0.0033 before the
2026-08-23 refresh — **the multiple grew because the DENOMINATOR shrank**, not because
`weighted` got worse; state both terms, never the ratio alone), and it does **not** recover
with depth the way
`rrf` does — at F=10,000 of ~75,000 chunks it is still −0.0112 against `rrf`'s −0.0004, so
for `weighted` "deep enough" is essentially n and the knob buys nothing. What licenses
permitting it anyway is that this codebase bans an **unmeasured** configuration from passing
as measured, not a measured-but-worse one (nothing bans `m2v`); the docstring now carries
the cost. Four things worth more than the verdict. (1) **The smoke slice reversed the sign
of the headline** — on 2 combos × 8 queries `weighted` *gained* from truncation, peaking
0.7708 at F=100 against 0.5938 at F=n, which is the KEEP branch; on the full set every Δ is
negative. A smoke run checks that the code runs, it is **not a small version of the answer**
([[feedback_a_smoke_slice_is_not_a_small_answer]]). (2) **P3 refuted, and the plausible
reasoning behind it is the trap**: `BM25Okapi` floors negative IDF, so BM25 scores are ≥ 0
and the *last-ranked* chunk really does score 0 — but only **0.1%** of the terms a cut
zeroes were already 0, because a chunk scores exactly 0 only when it matches **no** query
term and a ~20-token Thai query has common tokens reaching nearly every chunk. BM25 carries
73% of dense's zeroed mass at F=50 (88,313 vs 121,449). The promotion half is real and
negligible (2 of 157,731 dense terms at F=50), so the perturbation is one-sided after all —
for the opposite reason to `rrf`'s. (3) **The mechanism, corrected by the same data**: the
hypothesis was that truncation *creates* the intersection signal `weighted` structurally
lacks (at F=n "also in the other arm's list" is true of every chunk). Truncation does not
add that signal mildly — it makes intersection membership **nearly decisive**, because a cut
arm's normalized term is worth 0.5 × 0.27–0.95 (max-normalized cosine is *flat*: 0.9491 at
rank 10, 0.2699 at rank n) where `rrf` at rank 1,000 forfeits only 0.5/1060 ≈ 0.0005. So
`weighted`'s top-10 goes 8.25/10 in-both-arms at F=200 and 9.99/10 at F=1,000 (`rrf` 7.41 /
8.30): it becomes an intersection-only ranker and evicts what one arm alone found. That
lands exactly where a single arm carries a type — **`person` −0.1957** at F=200 (BM25 carries
person at 0.8147) against `program` **+0.0216**. (4) **P4 refuted in the interesting
direction and it is a hypothesis, never a result**: at F=n `weighted` scores **above** `rrf`
(0.5439 vs 0.5197, **+0.0241** macro recall@10). Descriptive only — no significance test,
macro over 36 combos, **unrouted**, and nothing ships `weighted`; the wrong-pair trap that
killed per-`entity_type` alpha and rrf4 applies here too, so it would need re-measuring
against the hard router before it means anything. **Re-run 2026-08-23 against rebuild #4, as a PAIR with the unrouted sweep (`docs/chunker-embedder-notes.md`) and never alone:
the LIFT verdict, every refutation and every mechanism survived, only levels moved.** The
pairing is forced by S7 — this run's `rrf` columns must reproduce the *published* sweep, so
refreshing either report on its own breaks the anchor rather than merely dating it; the
sweep was re-run first and S7 then reproduced it at all 11 depths. The fusion is
**imported** from
`hybrid_fetch_depth_sweep.py` rather than reimplemented, which makes this run's `rrf` columns
a cross-artifact anchor (S7 reproduces that sweep at all 11 depths); S5/S6 check against the
real `HybridRetriever` at F=n **and** at F ∈ {5, 50, 200, 1000}, since S5 alone would pass
unchanged if the fusion ignored F ([[feedback_anchor_a_check_where_the_mechanism_is_live]]).
The F-invariance of the two normalizers (`_normalize` runs over the already-truncated,
descending-sorted arm list, so `max(top-F) == max(all n)` for any F ≥ 1) is reported as a
lemma and deliberately **not** a self-check — it is true by construction, and a check that
cannot fail is a vacuous PASS dressed up as evidence. Raw cache
`hybrid_weighted_fetch_depth_raw.json` is written before `render()`, so `--render` is free
and a render crash after a GPU run loses nothing.
- **`fetch_depth` against the shipped router, and the ship decision (2026-08-09,
`tools/eval/routed_fetch_depth_test.py` → `data/results/routed_fetch_depth_test.md`,
~2.5 min quality + ~3 min latency).** The unrouted sweep
(`docs/chunker-embedder-notes.md`) left one blocker: its
−0.0027 is a macro over 36 combos retrieving with **no router**, and hard routing has
shipped since 2026-08-08 — the exact wrong-pair trap that killed per-`entity_type` alpha
and rrf4. Re-measured on the 106 queries routed by `classify_query` to their 4 shipped
indices, **the trade gets better on both sides**: pre-registered F=200 vs k=n (3 metrics,
Holm m=3) is recall@10 **+0.0005**, MRR −0.0024, nDCG@10 −0.0022, **all Holm-adj 1.0000**,
and latency **1193.9 → 475.6 ms p50 (−0.718 s, 2.51x)**, paired on the index each query is
actually routed to. **Note the null points the other way here than in those two cases** —
they had to *win* and a null killed them; a depth cut only has to not *lose*, so the null
is what licenses shipping — which is exactly why it must be **cited as a bound**: the CI
rules out a loss worse than **0.0078** on the worst of the three metrics. **The
pre-registered prediction was confirmed and it is the part that carries the mechanism**:
unrouted, `person` is the one entity type that *gains* from a shallow cut (+0.0202 at F=50)
because BM25 carries it while the cut deletes a weak dense arm's tail; routing already
hands `person` its dense specialist, so the gain should shrink — it **reverses** to
−0.0207. Also worth knowing: routed damage at *shallow* F is **worse** than unrouted
(F=10 −0.0705 vs −0.0480) because routing raised the baseline there is more to lose from,
yet routed rankings are *more* stable (84.0% of top-10s identical at F=200 vs 66.0%) —
don't assume the unrouted damage curve transfers in either direction.
**DECISION: wired at the query-time layer, NOT as the class default.**
`app/streamlit_app.py` sets `fetch_depth` per query through `StrategySpec` params
(default 200, with 1000 and "whole corpus" selectable); `HybridRetriever.__init__` keeps
`fetch_depth=None`, pinned by `tests/retrievers/test_hybrid_retriever.py`. The split is
load-bearing, not a hedge: F=200 changes the top-10 on **17 of 106** Gold queries, so as a
constructor default it would silently re-rank every future eval run while ~24k persisted
results and every published table still said k=n — this project's signature
silent-corruption shape. The UI is where 0.72 s is felt; the eval harness is where
reproducibility is. Containment is checked, not assumed: `audit_pipeline_invariants.py`
already classifies `mode_b`/`mode_b_routed` as write-only UI dirs, so nothing an eval
reads can pick up an F=200 result. Anchors: S2 reproduces `routing_eval.md`'s
`routed (shipped)` **0.6811** and S3 the unrouted **0.6229**, both exactly, from an
independent code path; S4 is the live-mechanism check against a real
`HybridRetriever(fetch_depth=F)` on a *routed* index, since S1-S3 exercise only F=n where
truncation is inert ([[feedback_anchor_a_check_where_the_mechanism_is_live]]). The fusion
itself is **imported** from `hybrid_fetch_depth_sweep.py` rather than reimplemented — two
copies of that tie-break would eventually disagree.
