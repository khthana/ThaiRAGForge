# Chunker / embedder / BM25 / hybrid comparison — full derivation

**This is the research record. The verdicts, bounds and operational rules live in
`CLAUDE.md`; this file is why they hold.** Folded out 2026-08-23 from an 18.6 KB
bullet, **verbatim**, so every figure stays under `audit_doc_claims.py`'s
D2/D5/D7 — this file is in `DOCS`.

**Why not `docs/chunker-embedder-comparison-log.md`:** that file is
**append-only** — a stale number in a log *is* the record — and it is
deliberately **outside `DOCS`** for exactly that reason. Folding narrative into
it would have moved these figures out of D2/D5 coverage silently, which is the
one rule the size-reduction work has. Citation-ready headline numbers live in
`docs/paper-results-summary.md`.

---

- Chunker/embedder/BM25/hybrid comparison eval lives in `tools/eval/`. Current (9-embedder)
scripts: `run_gold_chunker_eval.py`, `run_gold_bm25_eval.py`, `run_gold_hybrid_eval.py` +
`run_gold_hybrid_eval_9way_new.py`, `embedder_matrix_9way.py` (retrieval + breakdown +
aggregate significance in one script — defines the `(type, model_name)`-keyed embedder
labels and superseded-combo exclusions every other 9-way script imports),
`embedder_significance_test_by_entity_type_9way.py`, `bm25_vs_embedder_significance_test_9way.py`,
`hybrid_significance_test_9way.py`. (Originals without the `_9way` suffix cover only the
first 6 embedders and are superseded but kept for reference.) Scores against the Gold query
set `config/eval/gold_query_set_73det.yaml` (73 deterministic queries — use this one, not the
252-entry `gold_query_set.yaml`, which dilutes results with low-discrimination thematic
queries — and the reason they don't discriminate is now known: all 179 were
meeting-scoped but never named the meeting ("ในการประชุมครั้งนี้"), so they were
unanswerable as posed. `tools/eval/qualify_thematic_queries.py` rewrote all 179 to name
their meeting 2026-07-30 and `run_thematic_eval.py` re-evaluated them, which changed the
reason to keep them apart: they carry signal that points **the opposite way** on the
chunker axis (fixed_size − semantic is **+0.0258** thematic vs **−0.0363** entity-anchored;
`semantic` is the *worst* chunker on thematic and the best on entity-anchored), so pooling
the sets cancels two real effects instead of diluting one. Still low-powered per pair
(2/27 significant, 62% ties) — cite them as a separate query shape, never averaged in.
Side-by-side: `tools/eval/thematic_vs_deterministic.py`. **The BM25/hybrid arms
(`hybrid_significance_test_9way.py --thematic`) reverse this project's most
robust finding and are the bigger result**: BM25 is weak on thematic (0.2990 vs 0.4930
entity-anchored — no name to match exactly), so "hybrid beats dense-alone for every
embedder" is **entity-anchored-specific**. On thematic recall@10 it is 3 significant for
hybrid / 4 ties / **2 significant against** (`e5` −0.0449, `qwen3` −0.0516), and the
hybrid−dense delta is monotone in dense strength (**r = −0.921**). What sits at BM25's own
score is the **significance boundary** (every embedder below it is significantly helped, none
above it is), *not* the sign flip — the point estimate crosses zero near 0.40, between
`bge_m3` and `jina_v5`. (An earlier "flipping sign right at BM25's own score" phrasing here
and in `paper-results-summary.md` was never supported by its own table; corrected
2026-08-07.) General rule, now measured in both directions and subsuming the old
m2v/sct "RRF failure case": **RRF helps the weaker arm and taxes the stronger one — fuse
only when the two arms are comparable, whichever one is weak.** Report:
`data/results/thematic_hybrid_significance_test.md`. **Whole thematic arm re-run
2026-08-07 against rebuild #3** (dense+BM25+hybrid, 5h12m, exit=0): **0 verdict flips**
across all 81 significance cells, every effect size moved <0.02 — the numbers above are
current, and the old "indices predate the 2026-07-30 corpus fixes" caveat is discharged). Full process narrative: `docs/chunker-embedder-comparison-log.md`; clean
citation-ready numbers for paper-writing: `docs/paper-results-summary.md` (update this one
whenever a headline number changes — the log stays append-only). **Refreshed 2026-07-25**
for the corpus-discovery contamination fix (0% contamination, see
[[project_corpus_discovery_contamination_bug]] / `docs/chunker-embedder-comparison-log.md`,
fix `8c86b63`/`b36f96f`/`dd0c0ae`, rebuild commit `2d36663`) — every conclusion held through
that refresh. **Refreshed again 2026-07-29**, this time for a full second index rebuild
(2026-07-28, unrelated OCR-remediation fix, [[project_reocr_remediation_pipeline]]) — and
this refresh genuinely changed several conclusions below, not just numbers. A stale-cache
incident is worth recording as its own lesson: `embedder_matrix_9way.py` recomputes
dense-alone retrieval fresh every run, so dense-alone numbers refreshed automatically and
correctly — but the BM25/hybrid persisted-results scripts
(`run_gold_bm25_eval.py`/`run_gold_hybrid_eval.py`) were not re-invoked for 3 days, so
every BM25/hybrid significance test kept silently comparing fresh dense numbers against
stale (pre-rebuild) BM25/hybrid numbers until an implausible-looking batch of "flipped"
findings prompted an mtime check. **Lesson: after any index rebuild, refresh every
retrieval path with persisted results (dense, BM25, hybrid) — not only whichever one an
eval script happens to recompute automatically.** Full incident:
`docs/chunker-embedder-comparison-log.md`, "Re-eval หลัง OCR-remediation rebuild" entry.
Current bottom line (2026-07-29, bootstrap + Holm-corrected, all 9 embedders,
OCR-remediation-rebuilt indices):
**the "`semantic` chunking wins" headline this project has repeated since the first
comparison round does not survive being significance-tested, and should be retired.**
`semantic × qwen3_0.6b` recall@10 dropped from a stale 0.7048 to a fresh **0.6152**, no longer
even the top chunker for that embedder numerically (`sentence` 0.6265, `fixed_size` 0.6154) —
which prompted building the missing test
(`tools/eval/hybrid_chunker_significance_test.py`, chunker-vs-chunker at a fixed
embedder+retriever, one family per embedder + an aggregate family across all 9). **Result:
`semantic` never significantly beats any other chunker, anywhere** — not for `qwen3_0.6b`
(all 4 chunkers fully tied, Holm-adj p≥0.44) nor in the aggregate. **Post-rebuild-#4
(2026-08-18) the aggregate order is `recursive` 0.5318 > `sentence` 0.5212 >
`semantic` 0.5186 > `fixed_size` 0.5073 recall@10** — so `semantic` has now slipped
to *third* numerically (it was second at 0.5206 behind `recursive`'s 0.5291), which
only sharpens the retirement. The
*only* significant chunker-pairwise result in the whole aggregate is still
`fixed_size` losing to `recursive` on **nDCG@10** (−0.0298, Holm **0.0216**);
aggregate recall@10 has **no** significant pair at all (that same cell is −0.0204 at
Holm 0.8256), so do not state the laggard finding on recall@10 — several individual
embedders do carry significant recall@10 cells, the aggregate does not. **Revised framing:
`recursive`/`semantic`/`sentence` are a statistically tied top cluster with no provable
winner; `fixed_size` is the one demonstrated laggard.** `semantic` is still a perfectly
reasonable default (never proven worse than anything, and still the one chunker where a
strong dense embedder demonstrably beats BM25, see below) — just no longer citable as "the
best chunker." Cross-chunker-averaged
hybrid recall@10, **2026-08-18** (`qwen3_0.6b` 0.5999, `qwen3` 0.5792,
`jina_v5` 0.5696, `bge-m3` 0.5664, `e5` 0.5644, `e5_small` 0.5598, `congen` 0.4804,
`sct` 0.4065, `m2v` 0.3140 — ordering unchanged except `bge-m3` and `e5` swapping
two adjacent places by 0.0020). **The
dedicated semantic-only top-5 pairwise tie test
(`tools/eval/hybrid_significance_test_semantic_top5.py`) — the tie
**partially broke** in 2026-07-29 and **rebuild #4 partially UN-broke it, which is
the movement to know before citing `bge-m3`.** By 2026-08-06 `bge-m3` was outside
the cluster on *every* metric; as of the 2026-08-18 re-run it is outside on
**recall@10 and nDCG@10 only, and only against two rivals** — it loses to
`qwen3_0.6b` (0.0020 / 0.0060) and `qwen3` (0.0342 / 0.0060), while `jina_v5`
(0.1216 / 0.2992) and `e5_small` (0.1904 / ns) have gone back to ties, and **on MRR
it now ties everything again** (its closest cell, vs `qwen3`, is Holm **0.0940**).
So: **`bge-m3` is separated from the two `qwen3` models on 2 metrics of 3 and tied
with the rest** — not "clearly outside the cluster on every metric", which is
withdrawn. The
remaining four (`qwen3_0.6b` 0.6153, `qwen3` 0.6014, `jina_v5` 0.5941, `e5_small`
0.5854 recall@10, semantic-only; `bge-m3` 0.5436) are still fully, mutually tied on
every metric. Don't cite a single
"best combo" among those four — the tie there is confirmed, not provisional. Hybrid still
significantly beats dense-alone for essentially every one of the 9 embedders on every metric
(**24/27** as of 2026-08-18, down from 26/27 — `qwen3_0.6b` on MRR stays the standing
exception (Holm 0.3674) and **`congen` newly joins it on recall@10 (0.3424) and
nDCG@10 (0.2062)**, which is the RRF rule doing exactly what it says: BM25 got
relatively stronger, `congen` did not, and fusing an arm that is no longer comparable
stops paying) — still the most robust finding of the comparison. Hybrid vs. BM25-alone shifted more: `qwen3_0.6b`,
`qwen3`, `jina_v5`, `e5`, `bge-m3`, `e5_small` all significantly beat BM25 on recall@10 now
(`jina_v5` newly clearly significant, was borderline before); `congen` dropped **out** of that
group (BM25 itself got stronger post-OCR-fix, closing the gap); `sct`/`m2v` remain the
cautionary cases where hybrid is significantly worse than BM25-alone, and **`sct`'s recall@10
deficit is significant again** (reversing the 2026-07-25 "no longer significant, p=0.08"
finding — that finding was itself measured against the since-superseded index). The
cross-chunker **dense-alone 3-way tie at the top is broken, but only half of it stayed
broken.** `qwen3_0.6b` still significantly beats `bge-m3` on every metric
(**+0.1161** recall@10, Holm 0.0000) — but **against `Qwen3-Embedding-4B` its
recall@10 margin went ns at rebuild #4** — Holm **0.0592** on a +0.0475 margin,
where the pre-rebuild +0.0486 had cleared the bar — while MRR (+0.0915) and
nDCG@10 (+0.0694) both still clear it. **Cite it
as: `qwen3_0.6b` beats the 4B model on the ranking metrics and ties it on
recall@10** — a 0.0592 is a near-miss, not a reversal
([[feedback_a_replication_disagrees_by_sign_not_verdict]]). The per-entity_type specialist/weak-spot pattern underneath the old
aggregate tie is **unaffected** by this (separate, already-fresh dense-alone test): `bge-m3` =
person-query specialist, `Qwen3-4B` = only embedder with no provable weak spot across both
main categories (ties bge-m3 on person AND ties ConGen/qwen3_0.6b on program), `Qwen3-0.6B` =
now aggregate-leading but still has a real person-query weak spot `Qwen3-4B` doesn't,
`ConGen-PhayaThaiBERT` = program-query specialist. BM25 alone (`retrievers/bm25.py`) no
longer ties the top dense tier the way it used to — it still **significantly beats
`bge-m3`** on recall@10 (**+0.0829**, Holm 0.0216; aggregate BM25 recall@10 rose
0.3908→0.4930 post-OCR-fix and reads **0.4863** after rebuild #4, more than any
embedder's own gain, because lexical matching is far more sensitive to OCR token
corruption than dense embeddings)
and still significantly beats every weaker embedder; it statistically ties `qwen3` and
`qwen3_0.6b` on recall@10, and `jina_v5` has slipped **from borderline-significant to a
clear tie** (+0.0784, Holm 0.0192 raw → **0.0576** adjusted, was 0.053).
**The bigger 2026-08-18 movement is on MRR, and it is the first AGGREGATE cell in this
whole comparison where a dense embedder significantly beats BM25 outright**:
`bm25 − qwen3_0.6b` MRR **−0.1249**, Holm **0.0072**. Until rebuild #4 that had only
ever happened in one per-chunker cell (below); it is now true of the headline
cross-chunker table. On the same table `bge-m3` and `e5` **lost** their significant
MRR margins over BM25 (0.0664 each), so BM25's MRR standing polarised rather than
moved. The per-chunker breakdown
(`tools/eval/bm25_vs_embedder_significance_test_per_chunker.py`) still
shows `bge-m3` losing to BM25 significantly under `sentence` chunking specifically, and
the `qwen3_0.6b`-under-`semantic` cell **strengthened from one metric to all three**
(recall@10 −0.1044 Holm 0.0070, MRR −0.1809 and nDCG@10 −0.1604 both 0.0000; it was
recall@10 alone at 0.006) — reinforcing `semantic` as the
one chunker where a strong dense embedder demonstrably earns its cost over free BM25. Don't
naively RRF a weak embedder with BM25: `m2v` and `sct` both still significantly *hurt* vs. BM25
alone on recall@10/MRR/nDCG@10 (all three, both models, post-refresh) — a real RRF failure
mode whenever the fused dense signal is weak enough. Cost/latency:
`tools/eval/cost_latency_pareto.py` (vector dim, index size, query latency p50/p95) found
`HybridRetriever.retrieve()` and `BM25Retriever.retrieve()`'s current implementation
(full-corpus `k=n` fetch before fusing, `BM25Okapi` rebuilt from scratch every query) adds a
roughly **fixed ~1.9-2.0s of overhead to every hybrid query, nearly independent of embedder**
(it scales with corpus size, not embedding dim) — the ~2.1-2.7s measured figure is mostly this
avoidable per-query overhead on top of a ~116-668ms intrinsic cost, not RRF fusion itself.
**Half of that overhead is now gone, and the measurement that removed it split the two causes
the sentence above had bundled (2026-08-09).** `BM25Retriever` memoises its `BM25Okapi` on the
`Index` (`Index.lexical_scorer`) instead of rebuilding it per query. **Quote that saving in
seconds, not as a multiple of `get_scores`** — `rank_bm25` loops over query *terms* in Python,
so scoring is linear in query length (~12 ms/token over 74,816 chunks) while the build is not.
The **26.2x / 1.073s vs 0.041s** first published here was measured on a **3-token synthetic**
query and is withdrawn as a headline; re-measured 2026-08-09 on the **real 20-token-median
Gold queries** the project evaluates (n=106, min 13 max 30): build **1035.89 ms** vs
`get_scores` **253.50 ms** = **4.1x**, and **BM25-alone `retrieve()` p50 is 234.45 ms**
(p95 332.78). The ~**1.0s** removed from every query is the part that does *not* depend on
query shape, and that is the number that transfers. **Hybrid goes 2.269s → 1.361s
(1.7x, −0.907s)** — measured paired, both arms in one process against one loaded index, because
[[feedback_check_benchmark_position_drift]]. That gap is the finding: the **remaining ~1.36s is
the `k=n` over-fetch**, i.e. materialising ~75k `RankedChunk` objects per arm and fusing them in
Python, which is a *separate* and still-open cost. Do not read "the rebuild is fixed" as "the
hybrid overhead is fixed". The over-fetch is **not** free to remove either: `HybridRetriever`
fetches k=n so RRF sees complete rankings, so truncating it would change results, unlike this
change which cannot. **The stale-latency consequence is DISCHARGED (2026-08-09): the whole
script was re-run on an idle machine** and `cost_latency_pareto.md`'s BM25/hybrid columns are
current (dense p50 120-840 ms, hybrid p50 1.21-1.86s). `docs/paper-results-summary.md` was
updated with it, so its old split provenance (07-29 latency / 08-07 quality) no longer applies
to the latency half.
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
