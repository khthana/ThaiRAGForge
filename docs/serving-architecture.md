# Serving architecture — what a query costs, and what had to be fixed to serve one

Every measurement in this project until 2026-08-13 was an **eval** measurement:
one process, one index, one query at a time, artifacts built once outside the
timing loop. That is the right shape for a paper. It is the wrong shape for the
thing this project is actually headed for — a faculty deployment with real users
— and the gap between the two is not a matter of degree. An eval harness hoists
construction out of its loop by design; a server pays it on every request.

This document is the serving layer's record: what a served query costs, what the
three caches remove, how the process is warmed, how a rebuild landing underneath
a running server is prevented from being served, which topology to deploy, and
what is still not established. The engine side (Qdrant ingestion, exactness,
`exact=True` vs HNSW) has its own document, `qdrant-serving-pilot.md`; this one
is about the path in front of it.

Reports, all under `data/results/`:
`serving_cost_profile.md`, `serving_cache_memory.md`, `serving_concurrency.md`,
`qdrant_routed_check.md`, `qdrant_concurrency.md`, `qdrant_pilot.md`.
Code: `src/rag_lab/query_service.py`, `factory.py`, `io/index_cache.py`,
`io/artifact_store.py`, `retrievers/qdrant_hybrid.py`, `tools/seal_index_dirs.py`.

---

## 1. The request path

`route_query(text)` does five things:

1. `classify_query` picks a route (`person`, `program`, `course`, `faculty`,
   `unmatched`) from the query's shape.
2. `route_targets(retriever_type)` maps the route to a combo id, and
   `resolve_index` maps that to a directory.
3. the index is loaded — chunks, embeddings, metadata, and (for `hybrid`) a
   `BM25Okapi` memoised on the `Index` object;
4. an embedder and a retriever are built for the resolved spec;
5. the retriever runs, fusing a dense and a lexical arm by RRF at
   `fetch_depth=200`.

Steps 3 and 4 are the ones an eval harness does once and a server was doing
**per request**. That is the whole story of section 2.

Two topologies exist for step 5. **In-process** (`hybrid`) scores the corpus in
numpy and rebuilds/reuses `BM25Okapi` in Python. **Engine** (`qdrant_hybrid`)
sends both arms to a Qdrant server and fuses the returned ranks with the same
`fuse_rrf` — literally the same function, lifted to module level in `hybrid.py`
so the project has one copy of RRF and its tie-break rather than two that would
eventually disagree.

---

## 2. Where the time goes

`serving_cost_profile.md`, shipped `person` route, bge-m3, 57,172 chunks:

| stage | mean ms | cacheable? |
| --- | ---: | --- |
| `build_embedder` (constructor) | 0.0 | lazy — the cost is not here |
| first `embed()` — loads the weights | 9,063.0 | yes, embedder cache |
| warm `embed()` | 14.3 | no — real GPU work |
| `ArtifactStore.load` | 1,185.0 | yes, index cache |
| first `retrieve` (incl. the `BM25Okapi` rebuild) | 1,341.0 | the delta, yes |
| warm `retrieve` | 345.5 | no — scoring + fusion |

One served query with nothing cached is **11,589 ms**, of which **9,049 ms (78%)
is loading weights the previous query already loaded**. The irreducible remainder
is **360 ms**.

**The first measurement of this was wrong by two orders of magnitude, and the
reason generalises.** `LocalSTEmbedder._load()` is lazy — the constructor stores
a model name and `SentenceTransformer(...)` runs inside the first `embed()` — so
timing `build_embedder` returned **0.0 ms**, the 9 s landed in the *encode*
column, and the conclusion was that a cache could win about 10%. A 9-second
encode against a published 13–83 ms is an instrument fault, not a finding. `S1`
in that script now gates warm encode against the published range so the same
mistake cannot pass silently again.

---

## 3. The three caches

All three are **serving-path only**. `build_embedder` and `ArtifactStore.load`
stay uncached for every eval script, and that exclusion is pinned by tests in
both directions. The reason is concrete: a global embedder cache would hold
Qwen3-Embedding-4B resident beside its neighbours during a nine-embedder sweep,
which is the OOM this project already lost five `semantic × 4B` runs to. The
consequence worth stating plainly is that **no published eval number can move
because of anything in this document**.

| cache | env var | default size | why that size |
| --- | --- | ---: | --- |
| `build_embedder_cached` | `RAG_LAB_EMBEDDER_CACHE` | 2 | the five routes resolve to exactly two distinct embedders |
| `load_index_cached` | `RAG_LAB_INDEX_CACHE` | 4 | the five routes resolve to four distinct index dirs (`faculty` and `unmatched` share one) |
| `build_retriever_cached` | `RAG_LAB_RETRIEVER_CACHE` | 4 | one per resolved retriever spec |

End to end on the shipped `route_query`, arms alternated per query in one process
so machine drift cannot be read as the effect, and **routes alternated too** —
consecutive same-route queries would be served by one resident model and a
size-1 cache would look identical to the shipped size:

| arm | p50 (ms) | steady state (ms) | vs none |
| --- | ---: | ---: | ---: |
| none | 12,465 | 12,465 | 1.0x |
| embedder only | 3,389 | 3,069 | 3.7x |
| both | 1,462 | **463** | 8.5x |

**Quote the steady state, not the p50**: the p50 carries the cold fill, and
across three runs of the unchanged script the steady state read 422 / 447 / 446
ms against p50s of 1,476 / 1,548 / 1,462 — runs of an unchanged measurement
differ by about 6%, which is this rig's resolution.

Three rules the caches must obey.

**Sharing an `Index` is safe only because nothing mutates one**, and that was
grepped rather than assumed: across `src/`, `tools/` and `app/` there is exactly
one write to an `Index` attribute — `bm25.py`'s `index.lexical_scorer = (...)`,
the memo the cache exists to preserve. `MetadataFilter` and `EntityFilter` both
go through `Index.select()`, which builds a new object. A test pins the
no-mutation property directly, because if it ever stops holding this becomes a
correctness bug rather than a slow path.

**`with_embeddings` is part of the index cache key.** The engine-served path
loads without the matrix; handing *it* the full variant merely wastes the 234 MB
the flag exists to avoid, but the reverse would hand a row-reading retriever an
empty matrix — a silent wrong answer.

**A retriever is a pure function of its spec**, which is the whole licence for
sharing one. What it must not hold is per-query or per-`Index` state, which is
why `QdrantHybridRetriever._arms` is keyed by collection rather than being a
single slot.

---

## 4. Warm-up

The caches are worth 8.5x — *to the second caller on each route*. A fresh process
has four first callers (four index dirs) and two more (two embedders), so
`query_service.warm_serving_caches` front-loads them. It is **off by default**
(`RAG_LAB_WARM_ON_START=1` to enable): it holds ~3.1 GB of RAM and ~3.3 GB of
VRAM on a card the eval scripts share, so an automatic grab at UI start is how a
GPU run dies.

What it buys, per process state — `serving_warmup_profile.md`, three passes per
arm, each arm in its own process because a warm-up state is a property of a fresh
process and cannot be measured in a loop:

| arm | warm-up | 1st query | 4 queries | steady |
| --- | ---: | ---: | ---: | ---: |
| `cold` | — | 12,923.1 | 31,719.7 | **none** |
| `warm_no_probe` | 29,623.6 | 1,131.7 | 2,541.2 | 380.0 |
| `warm_probe` | 30,642.0 | **454.2** | **1,613.0** | 331.9 |

All in ms. **`cold` has no steady state, and that is the shape of the problem
rather than a gap in the table**: its four queries are four *first* callers, one
per route, each loading an index and possibly an embedder of its own. A
deployment does not get one slow query and then fast ones — it gets one slow
query per route, which is exactly what a warm-up removes.

Three things it had to be taught, each a measurement rather than a design choice.

1. **Building an embedder is not loading one** (section 2), so the warm-up embeds
   one string per route rather than merely constructing.
2. **Everything resident is still not warm.** With all four indices and both
   embedders loaded, the first real query cost **1,131.7 ms** against **380.0 ms**
   for the ones after it; one throwaway retrieval takes it to **454.2 ms**. The
   residue is process-global CUDA/BLAS initialisation, **not per-index** — one
   probe fixes all four routes — so the warm-up does exactly one and reports its
   cost.
3. **The probe must be given the params the deployment serves.** Left at the class
   defaults a `hybrid` probe fuses at `fetch_depth=None`, i.e. over the whole
   corpus — **856.3 ms against the shipped F=200's 342.6 ms**, both measured fully
   warm so the depth is the only difference between them — warming a slower code
   path than the one a user's query takes and charging the difference to startup.

**Read a knob's cost off an arm that is warm in every other respect.** The
2026-08-21 in-session reading of that pair was `2,052 / 1,093 ms` and is
superseded: it timed the probe *during* the warm-up, so both arms carried
process-global initialisation the depth knob has nothing to do with. The ratio
survives (1.9x → 2.5x); the levels fall by ~2.7x.

Whether to warm the BM25 scorer is **derived from `retriever_type`, never a
flag**: a `hybrid` caller who could also say "no scorer" would be asking for a
state that cannot serve.

Measured per topology (`serving_concurrency.md` §3):

| topology | warm-up ms | first query after it, ms |
| --- | ---: | ---: |
| `engine` | 29,736.9 | 197.5 |
| `inproc` | 11,944.2 | 685.1 |

---

## 5. Footprint

`serving_cache_memory.md`. The index cache holds **3,135 MB** for its four routed
indices (9.6% of this 32 GB machine; 616–869 MB each), of which **1,019 MB is the
embedding matrices** — an exact `ndarray.nbytes` figure rather than a measured one
— and **331 MB is the BM25 scorers**, which is what the rebuild buys back. The
embedder cache holds **3,310 MB of VRAM** for its two models on a 12 GB card
(bge-m3 2,174, qwen3-0.6B 1,136), and a clear returns 3,302 of the 3,310. Process
peak working set during that run was **4,126 MB** — a deployment sizes for the
peak, not the steady state.

**The number worth acting on was not the total but where it went.** 940 MB held
for a 223 MB matrix is not self-explanatory, so the load was walked step by step:
**307 MB of the 609 MB held was the transient parquet read** (`pq.read_table` plus
`.to_pydict()`), and deleting both returns only **2 MB** — the rest stays in the
allocator's arenas. So roughly half the per-index footprint was not live data at
all, and the lever was `ArtifactStore.load` rather than the cache.

`ArtifactStore.load` now streams (`pq.ParquetFile.iter_batches`, 1,024 rows at a
time): **379 MB → 244 MB held per index**, at no cost in time (563 ms against 596,
inside the run-to-run spread — building 57,172 pydantic `Chunk` objects dominates
either way). **This is a memory result, not a speed one**, and an isolated probe
that *did* show a large time gain was building plain tuples, i.e. describing a
loader nothing uses.

Four things to keep from it. The batch size is on a **measured knee**, not
pyarrow's default — 65,536 rows is one batch for every shipped index and gives
back almost none of the saving. The sweep was **run twice in opposite orders**,
because the first pass ran the whole-table arm first with a cold page cache and
would otherwise have charged its own I/O to the reader under test. "Smaller" is
only reported beside "identical": `C6` hashes all 57,172 chunks field by field in
both arms and requires agreement, since a reader that silently dropped a batch
would be the best-looking row in the file. And **reordering, not loss, is the
silent failure mode** — `Index.embeddings` is row-aligned to `Index.chunks`, so a
reordered read mispairs every vector and raises nothing; the test pins file order
against a fixture numbered backwards, so a loader that sorted could not pass.

Two method points. **"Is the memory returned?" and "is the object freed?" are
different questions**, and only the second has an exact answer: RSS is an
allocator question, so the leak test is a **weakref** to every cached Index, all
dead after a clear, while the returned-MB figure is reported as operational.
And the instrument is calibrated in-process before it is trusted — a 200 MB array
must register within 10%.

---

## 6. A rebuild landing underneath a running server

A cache that holds an `Index` while its directory is rebuilt will serve the
previous build's rows while every artifact on disk says otherwise. That is this
project's signature two-artifacts-from-different-days failure, made invisible by
living in RAM. It took **two** fixes, and the first one was not enough.

**Fix 1 — stamp the read at both ends.** Every cache *hit* re-stats
`(mtime_ns, size)` of all four artifacts (about four stat calls against a 1,185 ms
reload, so it is not optional), and the load stamps before *and* after, caching
only if the two agree. The original stamped only *after*, which is exactly wrong:
a rebuild overlapping the read leaves the post-load stamp equal to what is now on
disk, so the stale object is cached **under the current stamp** and every later
hit re-stats, agrees, and serves it — pinned permanently, the one failure the
cache exists to prevent.

**Fix 2 — the writer declares.** Both-ends stamping detects a write that
*overlaps* the read. It cannot detect a directory that is **stably inconsistent**:
`save` writes `chunks.parquet` and then `embeddings.npy`, and in between —
**seconds**, for a 234 MB matrix — the directory is new chunks beside the previous
build's vectors with **nothing moving**. A read falling entirely inside that
window stamps the same thing twice. So `save` now writes `_complete.json`
**last**, recording the four artifacts' stamps, and `index_cache._settle` refuses
a directory whose artifacts do not match that declaration.

Measured under load (`serving_concurrency.md` §6), 8 reader threads against a
writer alternating two builds of the same shape, each build stamping its identity
into the chunk text *and* into every vector so a mixed pairing is detectable:

| writer | inter-file gap | seal | reads | mixed | refused |
| --- | ---: | --- | ---: | ---: | ---: |
| `back_to_back` | 0.00s | yes | 5,739 | **0** | 0 |
| `gap_150ms` | 0.15s | yes | 587 | **0** | 232 |
| `gap_150ms_unsealed` | 0.15s | **no (control)** | 43,505 | **36,865** | 0 |

**Read the third row first.** It is the negative control — the same rig with the
reader made to ignore the seal, i.e. what shipped before fix 2 — and without it
"0 mixed" might only mean the rig never exercised the race. `refused` is the
designed behaviour, not a failure.

Four rules carry the seal.

- **The stale seal stays standing during a rewrite.** Clearing it first would make
  a half-written directory look merely *unsealed*, which is the one classification
  that does not refuse.
- **A mismatch is never downgraded to "probably an out-of-band edit, read it
  anyway."** During the inter-file window the directory is stable too, so
  stability cannot tell the two apart. An in-place writer owes a re-seal instead;
  `relabel_index_resolution_ids.py` is the repo's one such writer and calls
  `seal(d)`.
- **Unsealed is a reported gap, not a pass.** Every index built before this
  convention is unsealed and gets the older, narrower guarantee.
  `index_cache_info()` says so per entry, `tools/seal_index_dirs.py --apply`
  sealed all 55 (refusing any directory touched within `--min-age` seconds, since
  sealing something still being written would bless the very pairing this
  catches), and `I7` in `audit_pipeline_invariants.py` watches the fleet.
- **The tests were verified to FAIL on the pre-fix implementation** before being
  trusted, and one existing test changed contract with them: a vectors-only
  rewrite is now *refused* rather than merely invalidating the entry.

---

### 6b. The same race at real size, through the shipped `route_query`

§6 answers the *mechanism* on a 200x8 synthetic whose readers call
`load_index_cached` directly. §6b answers the *deployment* question: 3 reader
threads issuing real `person` queries through the shipped `route_query`
(hybrid, `fetch_depth`=200, all three caches live) against a **scratch copy** of
the `person` route's own index — 57,172 chunks, dim 1024, 305 MB — while a
writer alternates two builds through the same four files.

| writer | seal | served | checked | **mixed** | refused | writes |
|---|---|---:|---:|---:|---:|---:|
| `real_rebuild` | yes | 150 | 138 | **0** | 1,062 | 5 |
| `real_rebuild_unsealed` | **no (control)** | 160 | 148 | **42** | 1 | 4 |

**The scratch copy is the safety rule, not a convenience.** A writer runs in this
loop, so pointing it at `data/index/` would destroy an index costing ~2 h of GPU.
`route_query` takes its `indices` list as an argument, so the redirect is a
swapped `IndexInfo.dir` — no monkeypatching, and nothing can leak into the real
tree.

**The result that matters is not the 0. It is how much work it took to make the
control fire, because that is what bounds the seal's job.** Three measurements,
in order:

1. **No gap at all → 0 mixed in *both* arms.** `ArtifactStore.save` goes from
   `pq.write_table` straight into `np.save`, so at this size the exposure is
   dominated by a **truncated** file, not by two complete mismatched ones. The
   formats catch it themselves (`ValueError: Failed to read all data for array`,
   `JSONDecodeError`) — the failure is **loud**, not silent.
2. **At §6's 0.15 s → still 0, and that is structural rather than unlucky.** A
   read of this index takes ~1.5 s, so **a window shorter than a load cannot
   contain one**: the reader's own before/after stamps straddle the writer's
   transition, and the 2026-08-21 stamping fix catches it *without* the seal.
3. **It fires only once the window exceeds a *contended* load** (~13 s with
   several readers reloading 305 MB against a writer copying the same). At 15 s
   the control mixes **42 of 148** checked reads and the sealed arm mixes **0 of
   138**.

**So the seal's unique job is real but NARROW, and §6b is what narrows it.**
Against a rebuild that *overlaps* a read, the stamp comparison alone is
sufficient. What only the seal sees is a directory left stably inconsistent for
**longer than a read** — which `save` does *not* leave at this size, but which an
**in-place rewrite** does (`relabel_index_resolution_ids.py` rewrites for
minutes), and which a small index does under §6's conditions, where loads are
fast enough to fit inside a short window. None of that argues for removing it;
it says which writer it is protecting against.

**Two vacuity traps, both hit and both fixed at the mechanism.** The control's
first three configurations produced 0 mixed, which would have made the sealed
arm's 0 meaningless — hence the three-step calibration above and `S11`. Then
`S10` itself passed at **"0 mixed of 0 checked"**: with the pause between
rebuilds shorter than a contended load, every read straddled the next cycle and
the sealed arm served *nothing*. An arm that served no query cannot evidence
that serving is safe, so `S10` now requires a non-zero denominator and the pause
must exceed one contended load.

**The cost side, stated because the 0 is not free:** holding a 15 s inconsistent
window open cost **1,062 refusals** over 5 rebuilds. The seal converts an
inconsistency into unavailability, which is the right trade for a wrong answer,
but it is a trade.

### 6c. The defect §6b found in the cache itself

Running §6b at real size surfaced a bug the synthetic could not: `store.load(...)`
sat **unwrapped** inside `load_index_cached`'s retry loop. A racing write has two
outcomes, not one — it can hand back rows from two builds (caught by the stamp
comparison) **or** truncate a file under the reader, in which case `load` *raises*
from inside pyarrow/numpy/json. That exception propagated straight past the check
on the very next line that already knew how to handle it. Two consequences, and
the second is the dangerous one: the caller saw an exception the cache could have
retried, and it was **not** the `RuntimeError` a serving layer retries on — so a
torn read read as a corrupt index.

Fixed 2026-08-23: a load that raises is retried **iff the directory moved under
it**; a *stable* directory that still fails to load is genuinely corrupt and its
exception is re-raised **unchanged**. That second half is the guard against
"retry everything", which would report a genuinely unreadable index as "being
rebuilt" — a wrong diagnosis rather than a slow one. Both halves are pinned in
`tests/io/test_index_cache.py`, and the retry test was verified to **fail** on
the previous implementation before being trusted.

---

## 7. Under load: which topology

`serving_concurrency.md`, every real arm driven through the shipped `route_query`
over the 106 Gold queries at their real route mix, both caches warm, decision rule
frozen in the module before the run.

| arm | plateau q/s | at C | C=1 q/s | scaling |
| --- | ---: | ---: | ---: | --- |
| `encode` (GPU alone) | 30.40 | 1 | 30.40 | SERIAL |
| `inproc` (`hybrid`) | 2.53 | 25 | 1.58 | SCALES |
| `engine` (`qdrant_hybrid`) | **9.81** | 5 | 8.48 | SERIAL |

**TOPOLOGY = ENGINE**, at 3.87x the in-process plateau. Answered by inversion,
since the arrival rate is unknown: 50 users need one query every **5.10 s** on the
engine against **19.74 s** in-process.

Read the scaling labels *with* the levels. `inproc` SCALES (numpy releases the
GIL) but from a base so low that scaling does not rescue it; `engine` is SERIAL in
the sense that C=1 is already 86% of its plateau, so it must be sized by making
one query cheaper, not by adding users.

**NOT ENCODE-DOMINATED**, which reverses the earlier `qdrant_concurrency.md`
headline *for the shipped path*: the winning arm reaches only **32.3%** of the
`encode` ceiling it contains. That earlier run was not wrong — it measured a
hand-assembled pipeline whose embedder and `Index` were built once outside the
loop, which was an idealisation then and is the shipped path now. The remainder
is app-layer work that harness had hoisted out.

Two facts about the GPU arm that transfer. It **loses** throughput as concurrency
rises (a GPU is one device, so concurrent requests queue rather than overlap),
where the engine *gains* — so compare the two at matched C, not at their own best
levels. And encode cost on this card is a function of how long the GPU sat idle
beforehand, which is why `cost_latency_pareto.md`'s 82.94 ms and this run's ~14 ms
are both right: one is the low-load regime, the other the busy one.

---

## 7b. Open loop: what a user waits when nobody throttles the arrivals

`tools/eval/serving_open_loop.py` -> `data/results/serving_open_loop.md`
(2026-08-23). Section 7 measures a **closed** loop: C workers, each issuing its
next query only after the last returns. That is the right shape for finding a
plateau and the wrong one for sizing a deployment, for a structural reason --
**a closed loop throttles itself.** When the system slows, its own clients slow
with it, so a queue can never build. Here a dispatcher emits at a fixed rate
lambda **independent of completions**, and each request is timed from when it
*arrived*, not from when a worker picked it up:

    response = queue wait + service

A closed loop can only ever report the second term.

**Where the knee is.** The engine topology is stable to **8 q/s** and already
unstable at **10 q/s** -- consistent with section 7's 9.81 q/s plateau, reached
by a different harness. Above the knee there is no latency to quote: the queue
grows for as long as the load lasts, and a percentile over whatever finished is
a property of the run length.

| arrivals | lambda | response p50 | p95 | service p50 | stable |
|---|---:|---:|---:|---:|:---:|
| poisson | 1 q/s | 184 ms | n/a | 184 ms | yes |
| poisson | 4 q/s | 283 ms | 1,337 ms | 275 ms | yes |
| poisson | 6 q/s | 362 ms | 1,512 ms | 335 ms | yes |
| poisson | 8 q/s | 1,153 ms | 3,538 ms | 797 ms | yes |
| poisson | 10 q/s | 3,753 ms | 7,263 ms | 932 ms | **no** |

**The result worth carrying: burstiness costs the tail first, and it costs it
well inside the plateau.** The deterministic arms offer the *identical* rate
with even spacing, so any difference is clumping and nothing else. At
**lambda = 2 q/s** the medians match to a millisecond (181 vs 182 ms) while p95
differs **1.7x**; by **lambda = 6 q/s** -- comfortably inside the published
plateau -- even spacing sees **284 ms** at p95 and Poisson at the same rate sees
**1,512 ms**, a **5.3x** gap with nothing about the system changed. **So a
capacity figure taken from a closed loop or from even spacing is optimistic
about what people actually feel**, and the optimism lands on the unlucky user
before it lands on the typical one.

**Sizing, stated in the form the question arrives in.** 50 users issuing one
query every 10 s is 5 q/s: inside the stable range, with p95 already over a
second. The constraint at this scale is still latency rather than capacity --
the same conclusion section 7 reached, now with a tail attached to it.

**A harness confound worth knowing about, because it is a real deployment
note.** `with_embeddings` is part of the index-cache key, so warming *both*
topologies in one process asks for 4 index directories x 2 variants = **8 keys
against a cache sized 4**, and every query evicts an entry the other topology
needs. The first run did this and reported an in-process service p50 of
**4,484 ms** against section 7's published **626.2 ms** at C=1; preparing the
topologies separately took it to **757 ms**. **A process serving both retriever
types needs `RAG_LAB_INDEX_CACHE=8`** -- nothing else here says so, because
nothing else here serves both at once.

**Two method notes.** The stability verdict is **not** a fitted slope: fitting a
line to a sawtooth backlog produced verdicts that contradicted the latencies
beside them (lambda=6 "unstable" at a 342 ms response, lambda=8 "stable" at
1,356 ms), so it compares the **mean depth of the queue early against late**,
with the threshold set by the worker count rather than a round number. And
`dropped` is 0 on every row **including the unstable one** -- a 75 s arm just
past capacity builds a queue the drain window still absorbs, so read the queue
depth, not the drop count, as the sign of divergence.

**Not established here**: no network hop; one work shape (every request is the
same routed hybrid query, so nothing says how a mix of cheap and expensive
requests queues); and the knee is a property of this box, since the GPU is the
serialising layer.

---

## 8. Three defects this work found, none of them in Qdrant

1. **`localhost` cost 2,058.9 ms p50 per request against 15.1 / 14.0 ms for
   `127.0.0.1` — 136.3x on a name.** `docker run -p 6333:6333` publishes on IPv4
   only and `getaddrinfo` returns `::1` first, so the client spends ~2 s on an
   address the server is not on, even though that address refuses instantly. It
   hid because **every eval script already passed `127.0.0.1` while
   `QdrantHybridRetriever` and the Streamlit UI defaulted to `localhost`** — the
   published Qdrant latencies were measured on a path the shipped default did not
   take. Diagnostic worth reusing: the cost was identical for `exact=True` and
   HNSW, identical for `limit=10` and `limit=200`, and did not warm — *a constant
   that survives every knob is the transport, not the work*.
2. **The retriever was the third construction nobody had priced.** `query_indices`
   built a fresh one per query, and a `QdrantHybridRetriever` owns the client plus
   a per-collection arm cache that parses a 78k-term vocabulary sidecar off disk:
   **251.8 ms of a 340.5 ms query**. `route_query` now reads 176.4 ms.
3. **The warm-up gated its probe on `with_rows`**, so an engine-only process got
   none at all — and the probe's job is process-global CUDA/BLAS init, which has
   nothing to do with which rows are resident.

Two harness defects were caught by the run's own self-checks, and both had
reversed a number before they were fixed. `S5` went red because **only the repeat
control was warmed at its level**, so the engine's C=1 cell measured cold against
a warmed control — *a control warmed differently from the treatment is not a
control*; every cell is now warmed at its own level and the drift reads 0.6%.
`S4` went red on `inproc@C=50`, which is **arithmetic, not physics**: the dispatch
counter is shared, so with `r` requests per worker the ratio is bounded above by
`r/(r+1)`. Its domain is now derived by inverting the tolerance itself
(`r/(r+1) ≥ 0.85` ⟹ `r ≥ 6`), with the excluded cells printed rather than dropped.

---

## 8b. Failure modes: what the served path does when it cannot answer

`tools/eval/serving_failure_modes.py` -> `data/results/serving_failure_modes.md`
(2026-08-23). Nine modes, every one driven through the **shipped** `route_query`.
A mode is `ACTIONABLE` only if its message names the artifact that is wrong
**and** a remedy -- checked against the message text, not judged by eye.

**The finding, and it is the worst class this project has.** A collection is a
copy of an `Index`'s rows, so **any** index rebuild stales it -- and
`_to_ranked` builds every result from the engine's stored **payload**, with the
Index supplying only the collection name. So a collection nobody re-ingested
does not fail, it *answers*. Measured on a scratch index and a scratch
collection: after a rebuild without a re-ingest, one `IndexInfo` and one query
returned

| path | build served |
|---|---|
| in-process `hybrid` | **B** (current) |
| served `qdrant_hybrid` | **A** (previous) |

with **no error on either side**. The file path has had a seal against exactly
this since 2026-08-21 (`index_cache._settle`); the engine path had nothing.

**The guard, added the same day, is the engine-side counterpart of the seal.**
`QdrantHybridRetriever._verify` refuses a collection that is a copy of a
different build, on two signals: the row **count** (one call, does most of the
work) and a **sample of rows compared by identity** -- point id == row index at
ingest, so row *i*'s `chunk_id` in the payload must be row *i*'s here. The
second exists because the first cannot see a rebuild that preserves the count,
which is exactly what a re-OCR that moves text without moving chunk boundaries
produces. It runs **once per collection per retriever instance**, and the
serving layer caches instances, so it is once per process, not per query.

**What it costs, measured rather than argued.** Warm, one `count(exact=True)`
(**6.2 ms**) plus one `retrieve` of 8 ids (**2.6 ms**), once per collection per
process; every later call is a set lookup (0.3 us). `count(exact=True)` is kept
over the 2.3 ms `get_collection().points_count` deliberately -- the cheap
counter can lag during indexing, and a guard a lagging counter can fool is worth
less than 4 ms. The warm-up's probe retrieval pays it for the one collection it
probes; the other three pay it on their first query.

**Two things it is not.** It is not a check that you are serving the index you
*meant* to -- that is `route_targets` / `resolve_index`, gated in
`qdrant_routed_check.py`. And it is **not a per-query check**: a collection
re-ingested, or dropped, *after* a process verified it is not re-checked until
that process restarts. That is the cost trade, stated rather than hidden.

**Nothing falls back.** It is tempting to drop to the in-process retrievers when
the engine is unreachable. It must not: the two paths are different retrievers
over different copies of the rows, so a silent switch is a **different answer,
not a degraded one** -- the same reason `resolve_index` refuses an ambiguous
route rather than picking one.

**Three messages were opaque and are not any more**, found by the check rather
than by reading: `[WinError 10061] No connection could be made ...` named
neither Qdrant nor the url nor the collection (and took **14.3 s**; it now
raises in ~2-3 s naming all three), a missing index directory raised a bare
`FileNotFoundError` on `manifest.json`, and an unbuilt route target named the
target but no remedy. A 404 is now separated from a refused connection, because
reporting the first as "the engine is unreachable" sends an operator to restart
a server that is already running.

**The healthy control is what licenses the guard.** Both controls -- in-process
and served, against the four collections actually deployed -- still return 10
results. A guard that refused everything would pass every other check here.

---

## 9. What is NOT established

- **No network hop.** App, embedder and engine are one process on one box. That
  makes this box look *worse* at the app layer than a real deployment will (GIL
  contention bundled with request handling) and hides serialization cost a real
  hop would add.
- ~~**A closed loop, not a bursty arrival process.**~~ **CLOSED 2026-08-23 —
  section 7b.** What remains is narrower: Poisson is a memoryless approximation
  of many independent users, and a real class hitting one deadline is burstier
  than Poisson, so the arrival model is still a model.
- **The `encode` curve does not transfer** — an RTX 3060 in-process is not a
  separate faculty GPU server. It is measured alone precisely so another GPU's
  plateau can be substituted without re-running anything.
- ~~**The staleness race under a *real* rebuild with a query fleet in flight.**~~
  **CLOSED 2026-08-23 — §6b.** What remains open underneath it is narrower: the
  *stable* window at real size had to be opened deliberately, so nothing here
  measures how long a real `build_index` leaves one.
- **No timeout or GPU failure mode.** A refused connection is not a slow one,
  and an embedder that OOMs mid-serve is not probed -- forcing that safely on
  the card the eval scripts share is not worth leaving it wedged.
- **Nothing here says anything about ANN.** The engine recommendation is
  `exact=True`; see `qdrant-serving-pilot.md`.
- **Exactness between topologies is gated elsewhere.** The §5 ranking comparison
  in `serving_concurrency.md` is descriptive by design: RRF consumes *ranks*, so a
  tie settled differently inside either arm comes out of the fusion as a genuinely
  different fused score. The correctness gate lives in `qdrant_routed_check.py`
  (C4/C4b/C5), where the arms are compared directly — served **0.6827** against a
  reference **0.6835** across the four routed collections.

---

## 10. Operating it

```
# one Qdrant server, client and server pinned to the same version
docker run -p 6333:6333 ... qdrant/qdrant:1.18.0

# ingest one collection per routed index (re-run after ANY index rebuild)
python tools/eval/qdrant_pilot_ingest.py --index <dir>
python tools/eval/qdrant_routed_check.py        # 8/8 before trusting it

# seal every index directory the writer did not seal itself
python tools/seal_index_dirs.py --apply

# serve
RAG_LAB_WARM_ON_START=1 streamlit run app/streamlit_app.py --server.fileWatcherType none
```

Use `http://127.0.0.1:6333`, never `localhost`. Re-ingest all four collections and
re-run `qdrant_routed_check.py` **once** after a rebuild, not per combo — a
collection is a copy of an `Index`'s rows, so any rebuild stales it.

Nothing defaults to the engine: `dense` and `hybrid` still ship in-process and
`qdrant_hybrid` is opt-in by name, the same rule `lexical_containment` follows.
