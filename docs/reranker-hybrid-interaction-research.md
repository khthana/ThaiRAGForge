# Why did reranking hurt hybrid but not dense retrieval? A literature check

Status: research/decision-support doc. No code changed. Investigates our own bge-reranker-v2-m3
cross-encoder result against primary IR sources — peer-reviewed papers, the original RRF paper,
and first-party model documentation. Companion to the eval scripts in `tools/eval/` and the
bottom-line summary in `CLAUDE.md`.

## 1. Empirical situation

We built a cross-encoder reranker stage (`BAAI/bge-reranker-v2-m3`) on top of our hybrid retriever
(BM25 + bge-m3 dense, fused via Reciprocal Rank Fusion, `k_rrf=60`, full-corpus fetch then top-k
slice) and evaluated it on the 73-query Gold set with paired bootstrap + Holm-Bonferroni
correction. Protocol: retriever fetches k=50 candidates, reranker re-scores and truncates to k=10;
baseline is the same retriever's own top-10 (deterministic), isolating re-ordering as the only
variable.

| Retriever reranked | Metric | Baseline → Reranked | Holm-adj. p | Direction |
|---|---|---|---|---|
| Hybrid (BM25+dense, RRF) | MRR | 0.848 → 0.760 | 0.006 | **significantly worse** |
| Hybrid (BM25+dense, RRF) | nDCG@10 | 0.675 → 0.617 | 0.030 | **significantly worse** |
| Hybrid (BM25+dense, RRF) | recall@10 | 0.607 → 0.584 | 1.0 | worse, not significant |
| Dense-alone (bge-m3) | recall@10 / MRR / nDCG@10 | — | n.s. both directions | no effect |

Confirmed not an implementation bug (reranker smoke-tested in isolation, scores semantically
sensibly). The question: is "reranking hurts an already-fused hybrid ranking, but not a
dense-alone ranking" a known phenomenon, and what does the literature say about *why*?

---

## 2. Is "reranking hurts hybrid/fused retrieval" documented in the IR literature?

**What we found.** No paper we could locate studies reranking-on-top-of-RRF-hybrid as its
specific subject (a targeted search for "Cornacchia" — suggested as a possible author — turned up
nothing relevant; the name does not appear to attach to a hybrid-reranking paper we could find).
But the closely related question — *does a cross-encoder reranker degrade an already-strong
first-stage ranking, as opposed to always beating it?* — is documented, including for
`bge-reranker-v2-m3` itself:

- **Jacob, Lindgren, Zaharia, Carbin, Khattab, Drozdov, "Drowning in Documents: Consequences of
  Scaling Reranker Inference"**, ReNeuIR 2025 workshop @ SIGIR 2025, arXiv:2411.11767
  (https://arxiv.org/pdf/2411.11767) — read directly. This paper reranks BM25, and two strong
  dense retrievers (`voyage-2`, `text-embedding-3-large`) with several open and closed rerankers
  **including `bge-reranker-v2-m3` by name**, across 8 academic/enterprise datasets. Verbatim
  findings:
  - "reranking with large K decreases recall precipitously... often dropping beneath the quality
    of standalone retrievers. As a consequence, modern rerankers frequently perform worse than
    retrievers when both rank the full dataset."
  - Table 1 (their numbering): filtering to experiments where reranking helped at *some* K, it
    still hurt at the *largest* K tested 53.3% of the time on academic datasets and 44.4% on
    enterprise datasets ("Helps & Scaling Hurts").
  - "BM25 plus reranking might seem promising, but the reality is that this can be outperformed
    by retrieval-only with strong dense embeddings. For example... `text-embedding-3-large` is
    roughly 1.5x as effective as BM25 on the enterprise data."
  - They name the specific failure mode "**phantom hits**": the reranker "assigns a high score to
    documents that have no lexical or semantic overlap with the query" — i.e., the cross-encoder
    is not merely *less* additive over a strong baseline, it can be actively, confidently wrong.
  - In their Figure 4 "fair comparison" (rerankers applied to the *full* corpus, not just a
    retriever's top-k), `bge-reranker-v2-m3` is consistently one of the weaker curves, at points
    tracking below plain BM25.
  - This paper studies reranking BM25-alone and dense-alone, **not RRF-fused hybrid** — a real gap
    between what it tested and our setup. But it directly establishes, with our exact reranker
    model, that "reranker beats a strong/already-good first-stage ranking" is not a safe default
    assumption — the more relevant prior belief it overturns.

- **"Beyond the Reranker: Do RAG Retrieval Enhancements Help Once a Strong Reranker Is Present?"**,
  arXiv:2606.28367 (https://arxiv.org/html/2606.28367v1) — read via full-text fetch (not
  independently cross-checked against the raw PDF, so treat with slightly lower confidence than
  the directly-read sources above; the fetch tool paraphrases through a smaller model, so what
  follows is our best-effort paraphrase of its reported findings, not verbatim quotation). This is
  the mirror-image question to ours (holding the reranker fixed and varying retrieval-side
  enhancements, including rank fusion, rather than holding retrieval fixed and adding a reranker),
  but its finding is symmetric and directly relevant to a mechanism: it reports that reranking the
  same pool has little room to help, because per-query routing, rank fusion, graph rescoring, and
  corrective re-grading all end up reordering or re-weighting candidates the reranker has already
  ordered well — pool-expansion and fusion methods showed no statistically significant gain once a
  strong reranker was already present. Read in reverse: two
  stages that are each independently good at the same job (ordering candidates well) have limited
  *additive* headroom over each other, and by implication, disturbing one stage's output with the
  other risks a net loss rather than a further gain when there's little true headroom left.

- **Lu, Hall, Ma, Ni (Google Research), "HYRR: Hybrid Infused Reranking for Passage Retrieval"**,
  arXiv:2212.10528 (https://arxiv.org/pdf/2212.10528) — read directly. Not itself a "reranking
  hurts hybrid" result, but directly relevant background: HYRR's whole motivation is that
  off-the-shelf rerankers, trained on the candidate *distribution* of one retriever (commonly
  BM25, or the pretraining/fine-tuning mixture's own negatives), do not reliably transfer to a
  *different* retriever's candidate distribution — "the traditional wisdom is that you should
  train the reranker on data that is similar to the distribution that you will observe at
  inference time... However, recent work shows that this does not always guarantee an effective
  reranker (Gao et al., 2021)." HYRR's fix is to specifically train the reranker on candidates
  drawn from a **hybrid retriever** so it's exposed to that distribution during training. This
  matters for us: `bge-reranker-v2-m3`'s public training data (bge-m3-data, Quora, FEVER, per its
  model card, §4 below) has no documented hybrid-RRF-candidate-distribution component — it was not
  trained the way HYRR argues a hybrid-facing reranker should be. This is circumstantial, not
  causal, evidence, but it is a concrete, named mechanism (distribution mismatch between what the
  reranker learned to discriminate and what an RRF-fused list actually contains) that the
  literature already treats as a real risk, not a hypothetical one.

**Verdict for this project.** The literature does not contain a source that studied "cross-encoder
reranking on top of BM25+dense RRF fusion" as its specific subject — that precise gap is real, and
we should not claim otherwise. What it does contain, directly and with our own reranker model, is
strong evidence that (a) cross-encoder reranking degrading an already-strong or already-good
ranking is a documented, reproducible pattern rather than a one-off bug, (b) `bge-reranker-v2-m3`
specifically has been shown elsewhere to sometimes underperform a strong non-reranked baseline,
and (c) there is a named, plausible mechanism (retriever/reranker candidate-distribution mismatch)
for why an off-the-shelf reranker not trained on hybrid-fused candidates might mishandle them. Our
result is consistent with, not contradicted by, the primary literature — but it is not a case the
literature has already run and reported; treat our finding as a genuinely new (if narrow) data
point rather than a replication.

---

## 3. What does the original RRF paper claim, and what can't a cross-encoder see that RRF can?

**Source, read directly.** Cormack, Clarke, Büttcher, **"Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods"**, SIGIR 2009, DOI 10.1145/1571941.1572114,
author's PDF: http://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf.

Exact quotes:

- The formula: `RRFscore(d ∈ D) = Σ_{r∈R} 1/(k + r(d))`, with `k = 60` "fixed during a pilot
  investigation and not altered during subsequent validation."
- On what RRF operates on: "RRF is simpler and more effective than Condorcet Fuse, while sharing
  the valuable property that **it combines ranks without regard to the arbitrary scores returned
  by particular ranking methods**... RRF requires no special voting algorithm or global
  information; ranks may be computed and summed one system at a time."
- Their conjectured mechanism for RRF's strength: "RRF outperforms Condorcet because it is better
  able to harness diversity within individual rankings. One or two systems that rank a document
  highly can substantially improve its rank relative to the more popular documents."
- Scope of their experiments: combining 30 configurations of the Wumpus search engine over TREC
  collections, TREC participant submissions (TREC Robust, TREC 3/5/9), and a LETOR 3
  meta-learner combining ListNet, RankSVM, RankBoost, AdaRank, and their own logistic-gradient
  method.

**Important scope caveat, stated plainly.** This 2009 paper predates dense/neural retrieval
entirely — the "individual rankings" it fuses are all lexical/statistical IR systems and
learning-to-rank baselines over hand-engineered features, not a lexical-vs-dense-embedding split.
**The paper makes no claim about combining a sparse lexical retriever with a dense embedding
retriever specifically** — that pairing is a later community convention (RRF was simply general
enough to be repurposed for it once dense retrieval existed) — and it makes no claim at all about
what a downstream cross-encoder reranker can or cannot see relative to what RRF preserves. Both of
those points are true and important to our situation, but they are *our* inferences, not the
paper's claims:

- BM25's exact-term-match / IDF weighting is well-established classical IR content (Robertson &
  Zaragoza's BM25, not re-derived here), and RRF's rank-only fusion formula, by construction,
  preserves whichever document BM25 ranked #1 as a first-class signal *regardless of what a dense
  or cross-encoder score would have said about it* — this is a direct, correct reading of the
  formula above, not requiring an additional citation.
- Whether a cross-encoder reranker (`bge-reranker-v2-m3`), scoring only the raw `(query, chunk)`
  text pair with no visibility into which retrieval arm surfaced a candidate or why, reliably
  recovers that same "exact lexical match matters here" signal is genuinely uncertain from any
  source we found — cross-encoders are trained on relevance labels, and exact term overlap is
  *one* signal among many they can learn to use, but there is no guarantee (and no paper we found
  measuring) that it dominates their score the way it deterministically dominates a document's
  RRF rank contribution from the BM25 arm.

**Verdict for this project.** The original RRF paper is a real, verified source for what RRF *is*
(rank-based, score-agnostic, robust-by-diversity fusion) but not for the specific hybrid
lexical+dense use case it's typically deployed for today, and not at all for cross-encoder reranker
behavior — that connection is architecturally plausible (a rank-fused list encodes "this doc won on
exact lexical match" as a first-class, un-overridable signal; a cross-encoder sees none of that
provenance, only raw text) but is our own extrapolation, not a documented claim from any primary
source located. State it as a hypothesis, not an established fact, if it appears in the paper.

---

## 4. Cross-encoder calibration: does a reranker's score scale explain "confidently wrong" reordering?

**Sources, read directly or via first-party model card.**

- **The "phantom hits" finding in Drowning in Documents (§2 above) is the model-specific,
  empirical leg of this argument**, and it is the stronger of the two pieces of evidence here
  because it's about our exact reranker, not the model family in the abstract: `bge-reranker-v2-m3`
  itself was observed assigning **high scores to documents with no lexical or semantic overlap
  with the query at all** — the failure isn't "close calls scored inconsistently," it's scores that
  are locally confident and globally indefensible. This is the concrete version of "reranking
  reorders things in ways that look locally plausible per-pair but are globally worse" that our
  own MRR/nDCG drop is consistent with (MRR and nDCG are both sensitive to top-of-list reordering
  in exactly the way a single high-confidence phantom hit at rank 1-3 would damage; recall@10, less
  sensitive to *order* and only to *set membership* within the top 10, is the metric that dropped
  the least and non-significantly in our result — consistent with the reranker mostly *reordering*
  a similar set rather than *replacing* it with worse candidates). This holds regardless of what
  `bge-reranker-v2-m3`'s exact training loss turns out to be — it's an observed behavior, not a
  theoretical prediction.

- **Yu, Cohen, Lamba, Tetreault, Jaimes (Snowflake/Dataminr), "Explain then Rank: Scale
  Calibration of Neural Rankers Using Natural Language Explanations from LLMs"**, Findings of ACL
  2025, https://aclanthology.org/2025.findings-acl.1167.pdf — read directly, supplies a candidate
  *mechanism* for the phantom-hits observation above, with an important scope caveat. Quote: "it
  has been observed that popular pairwise and listwise ranking losses **are not scale calibrated
  due to their translation-invariant property** (Yan et al., 2022)[...] adding a constant to all
  outputs of φ does not alter the loss value." Their own footnote is explicit that this is
  *not* a blanket claim about every cross-encoder: "the cross entropy loss used in monoBERT is
  scale-calibrated, [but] it assumes only binary labels" — i.e., a **pointwise**, binary-label
  cross-entropy objective (as monoBERT uses) is not subject to the translation-invariance problem;
  it's specifically **pairwise and listwise** objectives that are. `bge-reranker-v2-m3`'s model
  card describes mapping its output to `[0,1]` via a sigmoid, which is consistent with a pointwise
  objective, but the card does not state its training loss explicitly, so **we cannot confirm from
  primary sources whether `bge-reranker-v2-m3` is actually a case this specific theoretical
  critique applies to.** Treat the translation-invariance argument as the plausible *family-level*
  mechanism for why cross-encoder scores in general shouldn't be assumed comparable to a
  fusion-tuned ranking's implicit scale — not as a proven property of our specific model.

**Verdict for this project.** The empirically-grounded leg of this argument is solid: our exact
reranker model has been caught, by an independent paper, confidently misscoring irrelevant
documents, and that failure mode's fingerprint (hurts ordering-sensitive metrics, spares
set-membership-sensitive recall@10) matches our own result. The theoretical leg (translation-invariant,
provably-uncalibrated training losses) is real in the literature but its applicability to
`bge-reranker-v2-m3` specifically is unconfirmed — don't present it as a proof for this model, only
as a plausible reason the family of models it belongs to is prone to exactly the kind of
locally-plausible/globally-wrong reordering we observed. It is still not a proof of *our*
mechanism — no source ran our exact hybrid-RRF-then-rerank pipeline — but it is the strongest,
most directly evidenced piece of the puzzle.

---

## 5. Thai-language / low-resource multilingual reranking: a known weak spot?

**What we found — thin, and not reranker-specific.**

- `BAAI/bge-reranker-v2-m3` model card (https://huggingface.co/BAAI/bge-reranker-v2-m3) confirms:
  0.6B parameters, LoRA-tuned on top of the `bge-m3` backbone, trained on "a mixture of
  multilingual datasets" named as `bge-m3-data`, Quora training data, and FEVER training data.
  Evaluated on BEIR, C-MTEB/Retrieval, and MIRACL — but **the model card does not publish a
  per-language results table**, so there is no first-party number for Thai specifically, positive
  or negative. This is a real gap, not an omission on our part — we checked the raw README source
  and it presents benchmark results as images/summary charts without a language-by-language
  breakdown we could extract as text.
- The base embedding model's own paper — Chen et al. (BAAI), **"BGE M3-Embedding: Multi-Lingual,
  Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation"**,
  arXiv:2402.03216 (https://arxiv.org/html/2402.03216v3) — reports MIRACL dev-set nDCG@10 numbers
  that, per our fetch of the paper (not independently cross-checked against the raw PDF, so treat
  with reduced confidence), show **Thai scoring 82.6, noticeably above English's 56.9**, against an
  18-language average of 67.8. If accurate, this argues *against* "Thai is a known dense-retrieval
  weak spot for BGE's multilingual training," not for it — but this is the **embedding** model's
  MIRACL score, evaluated on Wikipedia-derived, clean, well-segmented text, not the **reranker's**
  score on our messy, OCR'd Thai academic-resolution corpus, and MIRACL's Thai split is described
  in Zhang et al.'s MIRACL paper (arXiv:2210.09984, https://arxiv.org/abs/2210.09984 — checked via
  search summary only, not independently read) as one of the lower-resource languages in that
  benchmark alongside Bengali, Hindi, Swahili, Telugu, and Yoruba, so "low-resource" and "scores
  well" are not mutually exclusive on this particular benchmark.
- We could not find any primary source publishing `bge-reranker-v2-m3`-specific (not
  `bge-m3`-embedding-specific) per-language numbers, for Thai or otherwise. Secondary/blog-style
  search results surfaced a claim about a *different* model (`jina-reranker-v3`) scoring 81.06 on
  Thai in a multilingual reranking benchmark, but we could not verify this against jina's own paper
  text directly, so we are not citing it as established.

**Verdict for this project.** This sub-question came up thin, and we should say so rather than
force a conclusion. The one directly relevant number we found (BGE-M3 dense embedding Thai MIRACL
score) points *away* from "Thai is a known BGE weak spot," but it's the wrong model layer (embedder,
not reranker) and the wrong domain (clean Wikipedia text, not OCR'd Thai institutional prose with
its own vocabulary and formatting quirks) to settle the question either way. **We found no evidence
either confirming or ruling out a Thai-specific reranker weakness as a contributor to our result** —
this is a genuine gap in the literature (or in what we could locate of it), not a settled "no
effect" finding. If this explanation matters to the paper's argument, it would need to be tested
directly on our corpus (e.g., compare reranker behavior on Thai vs. English-heavy chunks within our
own eval set) rather than inferred from published multilingual benchmarks.

---

## 6. Does reranking help more when the first-stage retriever is weaker?

**Sources, read directly.**

- **Rosa, Bonifacio, Jeronymo, Abonizio, Fadaee, Lotufo, Nogueira, "In Defense of Cross-Encoders
  for Zero-Shot Retrieval"**, arXiv:2212.06121 (https://arxiv.org/pdf/2212.06121) — read directly.
  Their Table 2 is a direct, controlled test of exactly this question: they rerank BM25 (first-stage
  BEIR avg nDCG@10 = 0.441) and GTR-335M, a stronger dense retriever (first-stage avg = 0.451),
  both with the same monoT5-220M reranker over the same 1000-document candidate pool, across 7 BEIR
  datasets. Result: reranking BM25 lifts the average to **0.496 (+0.055)**; reranking GTR lifts it
  to the same **0.496 (+0.045)**. Both first-stage retrievers converge to an almost identical
  post-rerank score, and the *absolute gain* is smaller over the stronger baseline (+0.045 vs.
  +0.055) — direct, quantified evidence for shrinking headroom as the first-stage retriever gets
  stronger. Notably, in their data the effect **never reverses sign** — reranking still helped both,
  just by a smaller margin over the better retriever. That is a meaningfully different pattern from
  ours, where reranking hybrid didn't just help *less*, it hurt on two of three metrics.
- **Drowning in Documents (§2 above)** adds a second, independent data point in the same direction:
  on their enterprise datasets, reranking BM25 "always seems to lead to an improvement... as the
  number of documents increases," while reranking their strongest first-stage retriever
  (`text-embedding-3-large`, already ~1.5x BM25's own Recall@10) is the setting where reranker
  curves most often dip below the no-reranker baseline (their Figure 3, "text-embedding-3-lg"
  panel; also the source of the sign-test-style "Helps & Scaling Hurts" statistics in §2).
- **Beyond the Reranker (§2 above)**, from the opposite direction, adds a third, consistent
  data point: once a strong reranker is already in place, further retrieval-side improvements
  (including rank fusion) show no measurable additional gain — the same "shrinking/vanishing
  headroom" story, just observed by holding the other variable fixed.

**Verdict for this project.** This is the best-corroborated sub-question in the whole
investigation — three independent primary sources, using different reranker models, different
retrievers, and different benchmarks, all point the same way: **reranking's absolute benefit
shrinks as the first-stage ranking gets stronger, and headroom can shrink all the way to zero or
negative.** Rosa et al.'s controlled BM25-vs-GTR comparison is the cleanest evidence that the
*mechanism* ("less room to improve an already-good ranking") is real and quantifiable; it just
doesn't, on its own, show the mechanism tipping over into active harm — that tip-over is what
Drowning in Documents adds, and what our hybrid result adds a further, more extreme data point for
(hybrid retrieval fusing BM25 with a strong dense embedding is a plausible candidate for an
unusually strong first-stage ranking, if RRF's own beat-dense-alone significance across all 9 of
our embedders, documented elsewhere in this repo, is any guide — see `CLAUDE.md`'s "hybrid
significantly beats dense-alone" bottom line).

---

## 7. Synthesis: best-supported explanation, and what to do about it

**What the literature best supports, stated as an inference chain, not a proven mechanism:**

1. Hybrid RRF fusion in our pipeline is, by our own prior eval work (see `CLAUDE.md`, the hybrid
   significance findings), already a strong ranking — it significantly beats dense-alone for every
   one of 9 embedders on every metric. §6 above shows, across three independent primary sources,
   that reranking gains shrink as the first-stage ranking strengthens, and can flip to a net loss
   once the first-stage ranking is strong enough (Drowning in Documents, using our exact reranker
   model, documents this flip directly, albeit not for RRF hybrid specifically).
2. §4 gives a plausible mechanism for *why* a flip to net loss looks like ours specifically (MRR
   and nDCG hurt, recall@10 not): our exact reranker model has been caught elsewhere assigning
   confidently high scores to phantom, irrelevant hits — a failure mode that would show up exactly
   as "reorders the top of the list badly" (hurting MRR/nDCG) without necessarily "dropping
   relevant docs from the top-10 entirely" (recall@10 more robust). A candidate theoretical reason
   pairwise/listwise cross-encoder training can produce exactly this kind of miscalibration
   (translation-invariant loss, per Yan et al. 2022 as cited in Explain then Rank) exists in the
   literature, but whether it applies to `bge-reranker-v2-m3`'s specific (undocumented) training
   objective is unconfirmed — the phantom-hits observation, not the theory, is what's doing the
   work for this model.
3. §2's HYRR discussion supplies a candidate root cause for *why the reranker itself* might not be
   well-suited to an RRF-fused candidate list specifically: `bge-reranker-v2-m3`'s public training
   mixture (bge-m3-data, Quora, FEVER) has no documented hybrid-retrieval-candidate component, and
   the literature (Gao et al. 2021, per HYRR's citation) already flags retriever/reranker
   distribution mismatch as a real, named failure mode independent of our specific setup.
4. §3's re-reading of the original RRF paper adds a structural argument consistent with the above:
   RRF explicitly discards score magnitude in favor of rank position specifically so it can fuse
   heterogeneous signals "without regard to the arbitrary scores returned by particular ranking
   methods" — which means a document ranked #1 by BM25's exact lexical match is protected in the
   fused list by construction, in a way nothing protects it once a cross-encoder, blind to which
   arm surfaced it, is allowed to re-score and displace it based on its own (uncalibrated) semantic
   judgment.

**Where the literature does not settle it.** We found no paper that ran our specific pipeline
(BM25+dense RRF hybrid, then bge-reranker-v2-m3), so points 1-4 above are a *plausible, multiply
corroborated* explanation assembled from adjacent evidence, not a documented finding we're
reproducing. §5 (Thai-specific reranker weakness) came up genuinely thin — we neither confirmed nor
ruled it out, and it should not be cited as part of the explanation until tested directly on our
own corpus. Whether our RRF hybrid specifically counts as "strong enough first-stage ranking to tip
reranking into net harm" (as opposed to merely "less benefit but still positive," Rosa et al.'s
pattern) is also not something any source measured for our exact combination — we're inferring it
from the fact that it did, empirically, tip negative in our own eval.

**Practical recommendation, following from the above rather than overriding it:**

- **Don't blanket-disable reranking** — the same literature (§2, §6) that explains why hybrid got
  hurt also shows reranking reliably helps weaker rankings (BM25-alone, dense-alone-when-weak). If
  routing logic ever falls back to a single retriever (dense-alone or BM25-alone) for some query
  class, reranking that fallback path remains literature-supported and matches our own dense-alone
  null result (no harm, no proven benefit either, but not the significant-harm pattern hybrid
  showed).
- **Don't rerank the hybrid path with this reranker as currently wired**, given the significant,
  reproduced-on-our-data harm to MRR/nDCG — that's the one piece of evidence with no literature
  support running the other way.
- **Before concluding "hybrid should never be reranked" as a general claim**, the two candidate
  interventions the literature suggests as worth testing, in order of how directly each is
  supported by a source above, are: (a) a reranker trained or at least validated on hybrid-fused
  candidate distributions specifically (HYRR's own approach, §2) rather than assuming an
  off-the-shelf single-retriever-trained cross-encoder transfers; and (b) blending the reranker's
  score back into the RRF fusion (e.g., as a fourth ranked "system" input to RRF, rather than a
  hard truncate-and-replace step) so the fused list keeps its rank-based protection against a
  single miscalibrated signal overriding it, rather than letting the reranker's absolute (and
  provably uncalibrated, per §4) score have the final, unchecked word. Neither (a) nor (b) is
  something we found directly tested in the literature for this exact situation — both are
  extrapolations from the mechanisms above, not citations of a paper that already ran this
  experiment — so treat them as testable hypotheses for a follow-up ablation, not settled fixes.
