# Publication landscape: what is already out there, and where this study fits

Researched 2026-08-24 against primary sources — the papers themselves (arXiv,
ACL Anthology, ACM DL, Springer, CEUR), official dataset pages and repositories,
and official calls-for-papers. Blog posts, leaderboards and secondary summaries
were used only to *find* a source, never to support a claim.

**How to read this.** Every factual claim carries an inline URL to the source
that owns it. Each section ends with an explicit **UNCERTAIN / COULD NOT
VERIFY** block listing what was checked and what came back empty — an absence
recorded is worth more than a plausible guess. The last section,
*What would make our novelty claim weaker than we think*, is the honest
counter-brief; read it before writing any "first to" sentence.

**The one-line summary.** The *language* is covered — Thai has MIRACL, Mr. TyDi,
MLDR, XQuAD-th and iApp-WikiQA, so "first Thai retrieval evaluation" is false
and would be caught. The *domain* is not: every Thai retrieval resource found is
Wikipedia, news, social media or law, and no Thai administrative /
meeting-minutes IR test collection was found anywhere. Scope the novelty claim
to **domain + query shape + methodology**, never to the language.

---

## 1. Thai retrieval resources — what IS and IS NOT already covered

### 1.1 Verdict table

| Resource | Thai? | What it actually is | Primary source |
|---|---|---|---|
| **MIRACL** | **YES** | Thai Wikipedia ad-hoc retrieval, natural queries by native speakers | [aclanthology.org/2023.tacl-1.63](https://aclanthology.org/2023.tacl-1.63/) |
| **Mr. TyDi** | **YES** | Thai Wikipedia, ~1 positive passage/query | [aclanthology.org/2021.mrl-1.12](https://aclanthology.org/2021.mrl-1.12/) |
| **mMARCO** | **NO** | 13 languages, machine-translated; Thai absent | [arxiv.org/abs/2108.13897](https://arxiv.org/abs/2108.13897) |
| **XOR-TyDi** | **NO** | 7 languages (ar, bn, fi, ja, ko, ru, te); Thai absent | [aclanthology.org/2021.naacl-main.46](https://aclanthology.org/2021.naacl-main.46/) |
| **BEIR** | **NO** | **All 18–19 datasets are English** | [arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663) |
| **XQuAD-th** | **YES** | 1,190 QA pairs over 240 paragraphs, *translated* from SQuAD | [github.com/google-deepmind/xquad](https://github.com/google-deepmind/xquad) |
| **iApp-WikiQA** | **YES** | Thai Wikipedia extractive QA, SQuAD format | [huggingface.co/datasets/iapp/iapp_wiki_qa_squad](https://huggingface.co/datasets/iapp/iapp_wiki_qa_squad) |
| **MLDR-th** | **YES** | Long-document retrieval, LLM-generated queries | [arxiv.org/abs/2402.03216](https://arxiv.org/abs/2402.03216) |
| **ThaiSum** | n/a | **Summarization**, not retrieval — no qrels | [github.com/nakhunchumpolsathien/ThaiSum](https://github.com/nakhunchumpolsathien/ThaiSum) |
| **Wisesight** | n/a | **Sentiment classification**, not retrieval | [github.com/PyThaiNLP/wisesight-sentiment](https://github.com/PyThaiNLP/wisesight-sentiment) |
| **NitiBench** | **YES** | Thai **legal** QA/RAG, EMNLP 2025 main | [aclanthology.org/2025.emnlp-main.1739](https://aclanthology.org/2025.emnlp-main.1739/) |
| **SEA-BED** | **YES** | SEA embedding benchmark; its 4 new Thai sets are **not** retrieval | [arxiv.org/abs/2508.12243](https://arxiv.org/abs/2508.12243) |

### 1.2 MIRACL — yes, Thai is in it, and here are the exact numbers

WELL ESTABLISHED. MIRACL ([TACL 2023](https://aclanthology.org/2023.tacl-1.63/),
[arXiv 2210.09984](https://arxiv.org/abs/2210.09984),
[project-miracl.github.io](https://project-miracl.github.io/)) covers 18
languages and **Thai (`th`) is one of them** — its Figure 1 is literally a Thai
example. From the paper's Table 2:

| Split | Queries | Labels (`#J`) | Labels/query |
|---|---|---|---|
| Train | 2,972 | 21,293 | 7.17 |
| Dev | 733 | 7,573 | 10.33 |
| Test-A | 992 | 10,432 | 10.52 |
| Test-B | 650 | 6,493 | 9.99 |

Thai corpus: **542,166 passages** drawn from **128,179** Thai Wikipedia
articles. Overall MIRACL is "over 726k high-quality relevance judgments for 78k
queries over Wikipedia".

**The critical caveat, and it is the paper's own definition.** Table 2 defines
`#J` as "number of labels (**relevant and non-relevant**)", and Table 1 lists
MIRACL's "Avg # Labels/Q" as **9.23**. Annotators judged the **top-10** candidates
from an ensemble of BM25 + mDPR + mColBERT, binary relevance. So **MIRACL's ~9–10
per query is JUDGED depth, not RELEVANT depth** — it is not the same measure as
our 9.87 relevant/query. See §2.4.

Two further MIRACL facts worth having in the related-work section:

- MIRACL's own Thai baselines (Table 5, nDCG@10): **BM25 0.484**, mDPR 0.358,
  **mColBERT 0.481**, mContriever 0.517, **BM25+mDPR hybrid 0.599**. BM25 beats
  mColBERT on Thai, and the hybrid beats everything — independent corroboration
  of two of our own findings (see §4c, §4e).
- MIRACL explicitly contrasts itself with Mr. TyDi, "where each query has on
  average only a single positive (relevant) passage".

### 1.3 Mr. TyDi — yes, Thai, but shallow

WELL ESTABLISHED.
[Mr. TyDi](https://aclanthology.org/2021.mrl-1.12/) ([arXiv
2108.08787](https://arxiv.org/abs/2108.08787),
[github.com/castorini/mr.tydi](https://github.com/castorini/mr.tydi)) covers 11
typologically diverse languages **including Thai**. MIRACL's Table 1 records
Mr. TyDi at **Avg # Labels/Q = 1.02** over ~6.3k queries per language and 71k
total labels — i.e. essentially one known relevant passage per query.

### 1.4 mMARCO and XOR-TyDi — Thai is NOT covered

WELL ESTABLISHED. [mMARCO](https://arxiv.org/abs/2108.13897) is "a multilingual
version of the MS MARCO passage ranking dataset comprising 13 languages …
created using machine translation": English, Spanish, French, Italian,
Portuguese, Indonesian, German, Russian, Chinese, Japanese, Dutch, Vietnamese,
Hindi, Arabic. **No Thai.** The paper's own Mr. TyDi comparison states: "the base
model for our mColBERT finetuning (bert-multilingual-uncased) was not pretrained
on Thai. Hence, Thai results are not shown." (Quoted as the paper's statement;
we are not endorsing the claim about mBERT's pretraining coverage.)

[XOR-TyDi QA](https://aclanthology.org/2021.naacl-main.46/) covers 7 languages —
Arabic, Bengali, Finnish, Japanese, Korean, Russian, Telugu. **No Thai.**

### 1.5 BEIR — English only, on the record

WELL ESTABLISHED. [BEIR](https://arxiv.org/abs/2104.08663) (NeurIPS 2021
Datasets & Benchmarks) states in its appendix: *"Although we aim for a diverse
retrieval evaluation benchmark, due to the limited availability of multilingual
retrieval datasets, all datasets covered in the beir benchmark are currently
English."* So "BEIR has no Thai" is a quotable fact, not an inference.

### 1.6 XQuAD-th, iApp-WikiQA, ThaiSum, Wisesight

WELL ESTABLISHED.

- **XQuAD** ([github.com/google-deepmind/xquad](https://github.com/google-deepmind/xquad),
  [huggingface.co/datasets/google/xquad](https://huggingface.co/datasets/google/xquad)):
  240 paragraphs and **1,190 question–answer pairs** from SQuAD v1.1 dev,
  professionally **translated** into ten languages including Thai. When used as
  "XQuAD-th retrieval" the corpus is 240 paragraphs with one gold each — a
  toy-scale retrieval set, and the Thai is translationese, not native.
- **iApp-WikiQA**
  ([github.com/iapp-technology/iapp-wiki-qa-dataset](https://github.com/iapp-technology/iapp-wiki-qa-dataset),
  [HF card](https://huggingface.co/datasets/iapp/iapp_wiki_qa_squad)):
  **5,761 / 742 / 739** questions over **1,529 / 191 / 192** Thai Wikipedia
  articles, SQuAD format, annotator-written.
- **ThaiSum** ([repo](https://github.com/nakhunchumpolsathien/ThaiSum),
  [HF](https://huggingface.co/datasets/nakhun/thaisum)): **358,868** news
  articles with summaries. This is a **summarization** corpus — it has no
  queries and no relevance judgments, and cannot serve as a retrieval baseline.
- **Wisesight** ([repo](https://github.com/PyThaiNLP/wisesight-sentiment)):
  **26,737** Thai social-media messages labelled positive/neutral/negative/question.
  **Sentiment classification.** Also not retrieval.

### 1.7 Thai RAG / retrieval evaluation papers, 2023–2026

WELL ESTABLISHED — and this is the section that most constrains the novelty claim.

- **NitiBench** — *"NitiBench: Benchmarking LLM Frameworks on Thai Legal Question
  Answering Capabilities"*, **EMNLP 2025 main conference**
  ([ACL Anthology](https://aclanthology.org/2025.emnlp-main.1739/),
  [arXiv 2502.10868](https://arxiv.org/abs/2502.10868)). Two datasets:
  **NitiBench-CCL**, 3,730 test queries over Thai financial law with a *single*
  relevant section per query; **NitiBench-Tax**, 50 real tax cases averaging
  *three* referenced legal sections each. Compares BM25, BGE-M3, JinaAI ColBERT
  v2, Jina Embeddings v3, NV-Embed v1, Cohere, plus two fine-tuned BGE-M3
  variants. Reports section-based (hierarchy-aware) chunking beating naive
  chunking, and several **negative** results: long-context LLMs underperform RAG,
  the cross-referencing component "does not improve E2E performance", and
  fine-tuned retrievers show "no clear effect". Data is open-sourced.
  **This is the single most important comparator paper**: it proves a Thai
  domain-specific RAG benchmark is publishable at a top-tier NLP venue, and it
  gets there first for the legal domain.
- **SEA-BED** — *"SEA-BED: How Do Embedding Models Represent Southeast Asian
  Languages?"* ([arXiv 2508.12243](https://arxiv.org/abs/2508.12243)):
  **169 datasets, 9 tasks, 10 SEA languages** including Thai; **120 of 169** were
  authored by native speakers. It spans 17 domains "including academic, blogs,
  medical, and subtitles". Its **4 new Thai datasets (3,147 samples)** are STS,
  NLI and multi-label classification — **not retrieval**. So SEA-BED widens Thai
  *embedding* evaluation without adding a Thai retrieval test collection.
- **Thai-Sentence-Vector-Benchmark**
  ([github.com/mrpeerat/Thai-Sentence-Vector-Benchmark](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark)):
  its retrieval component is **XQuAD, MIRACL and TyDiQA** only. Checked
  explicitly: none of its datasets covers governmental or institutional documents.
- **NTCIR-19 `RegCom` (pilot task) includes Thai** — and this is the closest
  *structured-document* Thai evaluation found
  ([NTCIR-19 task list](https://research.nii.ac.jp/ntcir/ntcir-19/tasks.html)).
  RegCom is "automatic assessment of regulatory compliance in Environmental,
  Social, and Governance (ESG) reports across multiple languages, countries, and
  industries", over six languages — English, French, Japanese, Korean, Chinese
  and **Thai** — with two subtasks (full-report compliance matching against SASB
  metrics, and single-page metric verification). **It is corporate ESG reporting,
  not public-sector minutes, and it is compliance matching rather than ad-hoc
  retrieval** — so it does not pre-empt this work. But it does show Thai entering
  structured-document IR evaluation *right now*, and it should be cited as
  concurrent work. NTCIR-19 itself is 8–10 December 2026, NII, Tokyo.

### 1.8 Domain neighbours outside Thai — the ones a reviewer will find

WELL ESTABLISHED.

- **Japanese administrative meeting minutes.** *"GraphRAG with Knowledge Graphs
  for Question Answering on Administrative Meeting Records"*, Ushio, Tsuji &
  Kobashi, **ISWC 2025 Companion Volume**
  ([CEUR Vol-4085, paper 54](https://ceur-ws.org/Vol-4085/paper54.pdf)). Uses the
  minutes of Japan's Financial Services Agency "Expert Panel on Sustainable
  Finance"; builds a Person/Utterance/Meeting knowledge graph; evaluates on
  **50 hand-written questions with answer correctness assessed manually**.
  **No qrels, no IR metrics, no significance testing, one panel.** This is the
  closest published work by domain, and it is a short workshop paper — it does
  *not* pre-empt a test-collection contribution, but it does mean
  "administrative meeting minutes + RAG" is no longer an untouched phrase.
- **Greek government decisions.** *"A Greek Government Decisions Dataset for
  Public-Sector Analysis and Insight"*
  ([arXiv 2512.05647](https://arxiv.org/abs/2512.05647)), sourced from the
  Diavgeia open-government platform, released **CC-BY-4.0**. A directly
  analogous corpus in another language — and, unlike ours, publicly released.
- **Low-resource-language IR benchmark, structurally identical to ours.**
  *"Optimized Text Embedding Models and Benchmarks for Amharic Passage
  Retrieval"*, Mekonnen, Alemneh & de Rijke, **Findings of ACL 2025**
  ([aclanthology.org/2025.findings-acl.543](https://aclanthology.org/2025.findings-acl.543.pdf)).
  Builds language-specific dense retrievers, trains a **ColBERT-based late
  interaction** model, benchmarks "against both sparse and dense retrieval
  baselines to systematically assess retrieval effectiveness", and **publicly
  releases dataset, codebase and models**. This is the best structural template
  for our paper *and* the sharpest reminder of what our non-releasable corpus costs.
- **Entity-anchored query shape.** *"DBpedia-Entity v2: A Test Collection for
  Entity Search"*, SIGIR 2017
  ([ACM DL](https://dl.acm.org/doi/10.1145/3077136.3080751),
  [preprint](http://www.cs.cmu.edu/~callan/Papers/sigir17-Faegheh-Hasibi.pdf),
  [repo](https://github.com/iai-group/DBpedia-Entity)): **467 queries**, over
  **49K** judged query–entity pairs on a three-point scale. Entity-oriented
  retrieval collections exist; they are English and over DBpedia, not over
  institutional prose.

### 1.9 What this means for the novelty claim

**Scope it to the intersection, and state the intersection explicitly.**

- ❌ *"First Thai retrieval evaluation"* — **false**, MIRACL and Mr. TyDi both
  include Thai.
- ❌ *"First Thai RAG benchmark"* — **false**, NitiBench (EMNLP 2025) is exactly that.
- ❌ *"First entity-anchored retrieval test collection"* — **false**,
  DBpedia-Entity v2 (2017).
- ❌ *"First RAG over administrative meeting minutes"* — **weak**, the ISWC 2025
  Japanese FSA workshop paper exists.
- ✅ *Defensible*: **the first IR-style test collection over Thai institutional /
  administrative-council records**, with entity-anchored queries and
  rule-derived, re-derivable relevance judgments; every prior Thai retrieval
  resource is Wikipedia, news, social media or statute law.
- ✅ *Defensible*: the **systematic 4×9×3 factorial comparison with paired
  bootstrap + Holm correction and a power analysis reporting an MDE for every
  tie** — none of the Thai resources above runs a comparison at that scale with
  that statistical machinery (NitiBench compares 8 retrievers but reports no
  significance testing that we could find in the sections we read).

### UNCERTAIN / COULD NOT VERIFY (§1)

- **Whether any Thai administrative / meeting-minutes IR test collection
  exists.** Searched for Thai government-document retrieval corpora, Thai
  council-resolution datasets, and Thai administrative NLP resources; found only
  OCR resources ([ThaiOCRBench](https://opentyphoon.ai/blog/en/thaiocrbench),
  Thai national-document Tesseract repos) and pretraining corpora
  ([Mangosteen](https://arxiv.org/pdf/2507.14664)). **No test collection found** —
  but absence of evidence from ~8 searches is not proof of absence, especially
  for work published only in Thai or at Thai-language venues not indexed by the
  search engine used.
- **NitiBench's statistical testing.** We read the arXiv HTML v1 dataset and
  findings sections; we did **not** confirm whether it reports significance tests.
  Do not assert "it has no significance testing" without reading the EMNLP camera-ready.
- **MLDR-th details.** Confirmed MLDR covers 13 languages including Thai via the
  [BGE-M3 paper](https://arxiv.org/abs/2402.03216) and
  [FlagEmbedding repo](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/BGE_M3/README.md);
  did **not** verify its Thai query count or judgments-per-query.

---

## 2. Judgment depth norms — is "9.87 relevant/query" defensible, and against what?

### 2.1 The measurement trap, stated first

**Three different quantities get quoted as "judgments per query" and they are not
comparable:**

1. **Relevant documents per query** (what our 9.87 is; what BEIR's `Avg. D/Q` is).
2. **Total judged documents per query**, relevant *and* non-relevant (what
   MIRACL's 9.23 is; what TREC's qrels-per-topic figures are).
3. **Total judgments in the collection** (a size figure, not a depth figure).

Our gold set has **1,046 judgments over 106 queries = 9.87**, and — because the
judgments are rule-derived positives — **all 1,046 are relevant**. We therefore
have **zero judged non-relevants**. That is a genuine asymmetry against
MIRACL and TREC, and §2.5 says what to do about it.

### 2.2 MS MARCO passage dev — yes, ~1.1, and it is well documented

WELL ESTABLISHED. The MS MARCO passage dev ("small dev") set has **6,980
queries and 7,437 qrels**
([Anserini docs](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md)),
i.e. **1.07 relevant passages per query**. BEIR's Table 1 lists MS MARCO at
**Avg. D/Q = 1.1** ([BEIR](https://arxiv.org/abs/2104.08663)).

Arabzadeh, Vtyurina, Yan & Clarke, *"Shallow pooling for sparse labels"*
(Information Retrieval Journal 2022,
[arXiv 2109.00062](https://arxiv.org/abs/2109.00062),
[Springer](https://link.springer.com/article/10.1007/s10791-022-09411-0)) is the
canonical citation for the critique: MS MARCO "employ[s] substantially more
queries with substantially fewer known relevant items per query", and **94% of
the ~7,000 MS MARCO passage dev queries have only a single known relevant
passage**. Their crowdsourced preference study found top items from a modern
neural ranker often *preferred* to the judged-relevant item — i.e. the sparse
labels are actively misleading, not merely thin.

### 2.3 BEIR — the per-dataset table, which is the right comparator

WELL ESTABLISHED. From [BEIR](https://arxiv.org/abs/2104.08663) Table 1,
`Avg. D/Q` is **average relevant documents per query** — directly comparable to
our 9.87:

| Dataset | Test queries | Corpus | Avg. D/Q |
|---|---|---|---|
| MS MARCO | 6,980 | 8,841,823 | 1.1 |
| TREC-COVID | 50 | 171,332 | **493.5** |
| NFCorpus | 323 | 3,633 | **38.2** |
| BioASQ | 500 | 14,914,602 | 4.7 |
| NQ | 3,452 | 2,681,468 | 1.2 |
| HotpotQA | 7,405 | 5,233,329 | 2.0 |
| FiQA-2018 | 648 | 57,638 | 2.6 |
| Signal-1M (RT) | 97 | 2,866,316 | **19.6** |
| TREC-NEWS | 57 | 594,977 | **19.6** |
| Robust04 | 249 | 528,155 | **69.9** |
| ArguAna | 1,406 | 8,674 | 1.0 |
| Touché-2020 | 49 | 382,545 | **19.0** |
| CQADupStack | 13,145 | 457,199 | 1.4 |
| Quora | 10,000 | 522,931 | 1.6 |
| DBPedia | 400 | 4,635,922 | **38.2** |
| SCIDOCS | 1,000 | 25,657 | 4.9 |
| FEVER | 6,666 | 5,416,568 | 1.2 |
| Climate-FEVER | 1,535 | 5,416,593 | 3.0 |
| SciFact | 300 | 5,183 | 1.1 |

*(Note the denominator: BEIR's prose says "18 datasets", but Table 1 lists the
**19 rows** reproduced above — MS MARCO is included in the table as the in-domain
reference. Count 18 or 19 consistently and say which you meant; a reviewer who
recounts will otherwise think one of the two is wrong.)*

**The precise, defensible sentence:** *our 9.87 relevant documents per query is
deeper than **12 of the 19** datasets in BEIR's Table 1 and shallower than the
other **7*** (NFCorpus, Signal-1M, TREC-NEWS, Robust04, Touché-2020, DBPedia,
TREC-COVID).

Two bonus facts from the same table, both useful:

- **Query-set size.** BEIR datasets run from **49** (Touché) and **50**
  (TREC-COVID) to **57** (TREC-NEWS), **97** (Signal-1M), **249** (Robust04),
  **300** (SciFact), **323** (NFCorpus), **400** (DBPedia). **106 queries sits
  squarely inside the BEIR range**, which is the primary-source backing for the
  claim already in `docs/eval-validity-threats.md`.
- The deep ones are precisely the **traditional TREC-derived** collections
  (Robust04, TREC-NEWS, TREC-COVID) — depth is a property of pooled TREC-style
  assessment, not of modern web-scale QA collections.

### 2.4 TREC deep-pooled collections — deep, but in the OTHER unit

WELL ESTABLISHED. From the [TREC 2021 Deep Learning Track
overview](https://arxiv.org/pdf/2507.08191), **judgments (qrels) per query**:

| Year | Document task | Passage task |
|---|---|---|
| 2019 | 50 queries, 29,545 qrels — **590.9/query** | 50 queries, 13,520 qrels — **270.4/query** |
| 2020 | 54 queries, 74,474 qrels — **1,379.9/query** | 54 queries, 51,896 qrels — **961.0/query** |
| 2021 | 53 queries, 138,629 qrels — **2,615.3/query** | 54 queries, 138,629 qrels — **2,567.2/query** |

These are **total judged**, on a four-point graded scale (Perfectly relevant /
Highly relevant / Related / Irrelevant, per the
[TREC 2019 overview](https://arxiv.org/abs/2003.07820)), from depth pooling
across all submitted runs plus classifier-identified extras. **Do not compare
9.87 against 2,615.3** — one counts relevant documents, the other counts
assessments.

Note also the query-set sizes: TREC DL judges **50–54 topics**. Depth in TREC is
bought by *judging more per topic*, not by having more topics.

### 2.5 MIRACL — the comparison that needs the most care

MIRACL's headline "Avg # Labels/Q = **9.23**" ([Table
1](https://aclanthology.org/2023.tacl-1.63/)) looks like a near-exact match to
our 9.87, and **it is not the same number**. MIRACL's `#J` is defined in Table 2
as labels "relevant **and non-relevant**", produced by judging the **top-10**
candidates of a BM25+mDPR+mColBERT ensemble. Its relevant-only average is
necessarily lower than 9.23 and is **not published** (see the UNCERTAIN block).

**If you quote both numbers in the paper, you must say which unit each is in**,
or a reviewer who knows MIRACL will read the comparison as either naive or
flattering.

### 2.6 Two more depth anchors

- **DBpedia-Entity v2** (SIGIR 2017, [ACM
  DL](https://dl.acm.org/doi/10.1145/3077136.3080751)): **467 queries**, over
  **49K** judged query–entity pairs ≈ **105 judged per query**, graded 0/1/2,
  crowdsourced with expert adjudication of disagreements. The natural
  entity-search comparator.
- **TREC-COVID**: 50 topics, **493.5 relevant/topic** per BEIR — the deepest
  thing in common circulation, and an outlier driven by broad topical queries.

### 2.7 So: is "unusually deep at 9.87" defensible?

**Yes, with the comparator named.** Defensible phrasings, in decreasing strength:

- ✅ "**~9× deeper than MS MARCO passage dev** (9.87 vs 1.07 relevant/query), the
  collection most modern retrievers are tuned on" — strongest, cleanest, primary-sourced.
- ✅ "**~10× deeper than Mr. TyDi**, the only other Thai monolingual retrieval
  collection with human labels (1.02 labels/query)."
- ✅ "**Deeper than 12 of BEIR's 19 datasets** in relevant documents per query."
- ⚠️ "Comparable in depth to MIRACL" — **only true in the wrong unit.** Avoid, or
  state it as *"MIRACL judges ~9–10 passages per Thai query, of which an
  unpublished fraction are relevant."*
- ❌ "Deep by TREC standards" — **false.** TREC deep pooling is 270–2,615
  *assessments* per topic. Our collection is deep in *relevant* documents and
  **shallow in total assessment**.

### 2.8 The honest framing to use instead

The strongest available claim is not depth alone but **depth × derivability**:
1,046 relevance judgments at 9.87 relevant/query, **rule-derived and
re-derivable** rather than human-judged — which sidesteps the single-annotator
threat entirely and makes the qrels reproducible from the corpus. Pair it with
the residual-relevance measurement already in
`docs/eval-validity-threats.md` (~19–22% residual, CIs overlapping across arms →
incomplete-but-not-directionally-biased), because **the missing half of our
depth story is judged negatives**, and that measurement is what stands in for them.

### UNCERTAIN / COULD NOT VERIFY (§2)

- **MIRACL's average number of RELEVANT (positive-only) labels per query.**
  Checked: the TACL paper's Tables 1 and 2 (both report labels, explicitly
  including non-relevant); the
  [project-miracl/miracl GitHub README](https://github.com/project-miracl/miracl)
  statistics table (states "the judgments include both positive and negative
  labels"); and the
  [HuggingFace `miracl/miracl` dataset card](https://huggingface.co/datasets/miracl/miracl)
  (same `#Q`/`#J` table). **The positive-only count is not published in any of
  the three.** It would have to be computed from the qrels files. Until it is,
  do not state or imply a MIRACL relevant-per-query figure.
- **TREC DL relevant-per-topic** (as opposed to judged-per-topic). We extracted
  qrels-per-query from the TREC 2021 overview but did not find a relevant-only
  breakdown; the [TREC 2019 overview](https://arxiv.org/abs/2003.07820) pages we
  read give the scale and pooling method but not that split.
- **TREC-COVID's per-round judgment counts.** [NIST's Round 5
  page](https://ir.nist.gov/trec-covid/qrels5.html) was located but the exact
  per-topic judged counts were not extracted; the 493.5 figure comes from BEIR,
  not from NIST directly.

---

## 3. Negative results and reproducibility in IR — precedent and venues

### 3.1 The two tracks that explicitly accept "it did not work"

WELL ESTABLISHED, and these two quotes are the ones to put in the cover letter.

**ECIR Reproducibility track** — runs as a **separate call** from Full, Short,
Resource and IR4Good papers, and has done so continuously for years.
From the [ECIR 2027 call for reproducibility
papers](https://www.ecir2027.co.uk/call-for-reproducibility-papers) (identical
wording in the [ECIR 2025 call](https://ecir2025.eu/call-for-reproducibility-papers/)):

> "A successful reproduction of the work is not a requirement, but it is crucial
> to provide a precise and rigid evaluation of the process to allow lessons to be
> learned for the future."

It solicits "replicability (different team, same experimental setup) and
reproducibility (different team, different experimental setup)" papers, and
**excludes** same-team repetitions. **12 pages + references** (appendices count
toward the limit).

**SIGIR Reproducibility track** — a distinct track (alongside Full, Short,
Resources, Low-Resource Environments, Perspectives and Demo). It accepts papers
that "repeat, reproduce, generalise, or analyse prior work impacting information
retrieval", and explicitly welcomes reporting both "assumptions of the original
work that they found to hold up, and **the ones that could not be confirmed**".
Length **9 pages excluding references**. Historical corroboration in the
[SIGIR 2022 call](https://sigir.org/sigir2022/call-for-reproducibility-track-papers/):
papers should "analyze to which extent assumptions of the original work held up,
and elaborate error modes and unexpected conclusions".

**ACL Rolling Review** — the clearest blanket statement in NLP. From the
[ARR CFP](https://aclrollingreview.org/cfp):

> "Both positive and negative results for experimental studies are welcome, and
> have the same challenge of justifying to the program committee why this
> particular result is interesting and important."

ARR also names the shapes it accepts: "non-reproducibility or
non-generalizability of previously published results, their misattribution
('right for the wrong reasons'), or … an idea that seemed great … but didn't work."

**Workshop on Insights from Negative Results in NLP** — a dedicated venue,
**six editions 2020–2025**, all indexed in the ACL Anthology
([aclanthology.org/venues/insights](https://aclanthology.org/venues/insights/)):
2020 (19 papers), 2021 (21), 2022 (26), 2023 (15), 2024 (20), 2025 (17). Its
[CFP](https://insights-workshop.github.io/2021/cfp/) solicits, among other
shapes, "ablation studies of components in previously proposed models, showing
that their contributions are different from the initially reported" and
"datasets or probing tasks showing that previous approaches do not generalize to
other domains or language phenomena" — which describes our HyDE, reranker and
ColBERT results almost exactly.

### 3.1b Pre-registration in IR — it exists, and it is CLEF

**This corrects a search that first came back empty.** No IR venue advertises a
"pre-registration track" under that name, and searches for a CLEF
pre-registration lab, an ECIR pre-registration workshop and SIGIR/CHIIR
registered reports all returned nothing. But the **CLEF conference paper track
runs a two-stage results-blind review**, which is a registered report in all but
name ([CLEF 2026 call for
papers](https://clef2026.clef-initiative.eu/calls/papers/), verified verbatim):

- **Stage 1** — authors submit a version with **results and discussion removed**,
  and result mentions stripped from the abstract. Reviewers assess "the
  importance of the problem addressed and the soundness of the methodology".
- **Stage 2** — surviving papers submit the full manuscript, and *"The final
  decision will not be based on whether results are positive or beat a
  baseline."*

Alongside it, CLEF states outright that it "welcomes papers that describe
rigorous hypothesis testing regardless of whether the results are positive or
negative", and that "negative results and failed experiments are explicitly
welcome".

**This is the single best structural match in the landscape for a study built on
frozen pre-run predictions.** It also means the honest claim about our own
practice can be stronger than "we wrote it down first": the design documents
frozen before each run are exactly what a CLEF stage-1 submission consists of.

### 3.2 The canonical weak-baseline / neural-hype literature

WELL ESTABLISHED. These are the citations that make a negative-results paper
look like methodology rather than sour grapes.

- **Yang, Lu, Yang & Lin (2019)**, *"Critically Examining the 'Neural Hype':
  Weak Baselines and the Additivity of Effectiveness Gains from Neural Ranking
  Models"*, **SIGIR 2019 short paper**
  ([ACM DL](https://dl.acm.org/doi/10.1145/3331184.3331340),
  [arXiv 1904.09171](https://arxiv.org/abs/1904.09171)). From the abstract:
  *"We do not find evidence of an upward trend in effectiveness over time. In
  fact, the best reported results are from a decade ago and no recent neural
  approach comes close. … While there appears to be merit to neural IR
  approaches, at least some of the gains reported in the literature appear
  illusory."*
- **Lin (2018)**, *"The Neural Hype and Comparisons Against Weak Baselines"*,
  ACM SIGIR Forum 52(2)
  ([DOI 10.1145/3308774.3308781](https://dl.acm.org/doi/10.1145/3308774.3308781)).
  Bibliographic facts verified; full text was not re-fetched (403).
- **Armstrong, Moffat, Webber & Zobel (2009)**, *"Improvements That Don't Add Up:
  Ad-Hoc Retrieval Results Since 1998"*, **CIKM 2009**
  ([ACM DL](https://dl.acm.org/doi/10.1145/1645953.1646031),
  [author's abstract page](https://people.eng.unimelb.edu.au/ammoffat/abstracts/amwz09cikm.html)):
  finds "little evidence of improvement in ad-hoc retrieval technology over the
  past decade" and that "baselines are generally weak".
- **Sakai (2014)**, *"Statistical Reform in Information Retrieval?"*, ACM SIGIR
  Forum 48(1)
  ([PDF](https://sigir.org/files/forum/2014J/2014J_sigirforum_Article_TetsuyaSakai.pdf)):
  argues p-values alone are insufficient because they conflate sample size with
  effect size, and asks "should IR journal editors and SIGIR PC chairs require
  (rather than encourage) reporting of effect sizes and confidence intervals?"
- **Sakai (2016)**, *"Statistical Significance, Power, and Sample Sizes: A
  Systematic Review of SIGIR and TOIS, 2006–2015"*, **SIGIR 2016**
  ([ACM DL](https://dl.acm.org/doi/10.1145/2911451.2911492)): reviewed **840
  SIGIR full papers and 215 TOIS papers**; found that "many papers either lack
  significance testing or fail to report p-values and/or test statistics, which
  prevents power analysis". **This is the citation that turns our power analysis
  from housekeeping into a contribution.**

### 3.3 Published papers whose central contribution is a negative result

WELL ESTABLISHED (bibliographic facts verified; framing is our reading).

| Paper | Venue | Negative finding | How it is framed to be publishable |
|---|---|---|---|
| Armstrong et al., *Improvements That Don't Add Up* ([CIKM 2009](https://dl.acm.org/doi/10.1145/1645953.1646031)) | CIKM full | A decade of ad-hoc IR gains do not accumulate | **Meta-analysis across a literature**, not an attack on one paper |
| Yang et al., *Critically Examining the Neural Hype* ([SIGIR 2019](https://dl.acm.org/doi/10.1145/3331184.3331340)) | SIGIR short | Neural gains over strong baselines largely illusory | **Empirical test of a stated hypothesis**, balanced by a positive additivity result |
| Kamphuis, de Vries, Boytsov & Lin, *Which BM25 Do You Mean?* ([ECIR 2020](https://link.springer.com/chapter/10.1007/978-3-030-45442-5_4), [preprint](https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf)) | ECIR **Reproducibility** | **No significant effectiveness differences** across 8 BM25 scoring variants | A pure null result, published **because it was submitted to the reproducibility track** |
| Thakur et al., *BEIR* ([NeurIPS 2021 D&B](https://arxiv.org/abs/2104.08663)) | Datasets & Benchmarks | BM25 is a robust baseline; neural retrievers often underperform out-of-domain | The negative result **rides inside a resource contribution** |
| Weller et al., *When do Generative Query and Document Expansions Fail?* ([Findings of EACL 2024](https://aclanthology.org/2024.findings-eacl.134/)) | ACL Findings | Expansion **harms stronger retrievers** | Framed as **the first comprehensive analysis** of when a popular technique works — a *conditional law*, not a null |
| Yoon et al., *Hypothetical Documents or Knowledge Leakage?* ([Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.980/)) | ACL Findings | HyDE-style gains partly a benchmark-leakage artifact | Framed as a **mechanism diagnosis** of a popular technique |
| Jacob, Lindgren, Zaharia, Carbin, Khattab & Drozdov, *Drowning in Documents* ([arXiv 2411.11767](https://arxiv.org/abs/2411.11767), ReNeuIR@SIGIR 2025) | Workshop | Rerankers **degrade quality** past a certain candidate count; often worse than the retriever | Framed as **a scaling law with a turning point** |
| *A Reproducibility Study of Graph-Based Legal Case Retrieval* ([arXiv 2504.08400](https://arxiv.org/abs/2504.08400), SIGIR 2025) | SIGIR **Reproducibility** | Could not reproduce CaseLink's reported results | **Reproducibility-track framing** carries it |

### 3.4 The transferable framing rules

Every publishable negative result above does at least one of these. Ours should
do **all four**:

1. **Submit to a track whose call names the outcome.** A null in the ECIR/SIGIR
   reproducibility track is on-topic *by definition*; the same null in a full
   track is a rejection risk.
2. **Report a bound, not an absence.** "Rules out an improvement larger than X"
   is a finding; "no significant difference" is not. Our power analysis already
   does this for all 42 ties — that is exactly the Sakai (2014/2016) prescription.
3. **State the condition, not the verdict.** Weller et al. is publishable because
   its result is *"expansion helps weak retrievers and harms strong ones"*, not
   *"expansion doesn't work"*. Our RRF rule has the same shape and should be
   written the same way.
4. **Pre-register and say so.** A prediction frozen before the run converts "we
   found nothing" into "we tested a specific claim and it failed", which is the
   difference between a null and a result.

### UNCERTAIN / COULD NOT VERIFY (§3)

- **A track *named* "pre-registration" at any IR venue.** Searched for a CLEF
  pre-registration lab, an ECIR pre-registration track/workshop, and SIGIR/CHIIR
  registered reports; **nothing found under that name.** The CLEF results-blind
  mechanism in §3.1b is the functional equivalent and is verified — but do not
  call it "pre-registration" in a paper without describing it, because CLEF does
  not use that word either. Every individual CLEF lab CFP was not checked.
- Exact prose of Lin (2018) SIGIR Forum and its 2019 "recantation"
  ([p088.pdf](https://www.sigir.org/wp-content/uploads/2019/december/p088.pdf)) —
  bibliographic existence confirmed, full text not fetched (403).
- The **byte-exact** SIGIR 2026/2027 reproducibility-track wording: the official
  SIGIR page is JS-rendered and resisted direct fetching; substance was
  corroborated against the SIGIR 2022 and 2024 calls, which use near-identical
  language. Treat the ECIR quotes as verbatim and the SIGIR quotes as
  high-confidence paraphrase.

---

## 4. Are our specific negative findings already known?

Verdict key: **CONTRIBUTION** = not found in the literature; **CONFIRMATION** =
already reported, ours adds a new setting; **CONTRADICTION** = literature reports
the opposite.

### (a) HyDE failing on entity-anchored / named-entity queries

**Verdict: CONFIRMATION of a general law, CONTRIBUTION on the specific mechanism.**

The general result is **already established and strongly so**. Weller, Lo,
Wadden, Lawrie, Van Durme, Cohan & Soldaini, *"When do Generative Query and
Document Expansions Fail? A Comprehensive Study Across Methods, Retrievers, and
Datasets"*, **Findings of EACL 2024**
([ACL Anthology](https://aclanthology.org/2024.findings-eacl.134/),
[arXiv 2309.08541](https://arxiv.org/abs/2309.08541),
[code](https://github.com/orionw/LM-expansions)):

> "there exists a strong negative correlation between retriever performance and
> gains from expansion: expansion improves scores for weaker models, but
> generally harms stronger models. We show this trend holds across a set of
> eleven expansion techniques, twelve datasets with diverse distribution shifts,
> and twenty-four retrieval models."

Their hypothesised mechanism — expansions "add additional noise that makes it
difficult to discern between the top relevant documents" — is **the same
dilution mechanism** our HyDE notes record (29 of 30 generated documents still
contain the queried name; the signal is diluted, not deleted).

A second, independent line also exists: Yoon, Jung, Yoon & Park, *"Hypothetical
Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion"*,
**Findings of ACL 2025** ([ACL
Anthology](https://aclanthology.org/2025.findings-acl.980/)), which argues part
of HyDE's reported gain is benchmark knowledge leakage. Note its testbed is
**fact verification**, not entity queries.

**What is still ours:** (i) the finding on a **non-English, entity-anchored**
query set, which neither paper covers; (ii) the **magnitude** — dense recall@10
−0.1898 at Holm 0.0000 is a large, significant *loss*, not a shrunken gain;
(iii) the *"less to lose is not something to gain"* refinement — that damage is
2.6× smaller where the lexical signal is weak, which is a sharper statement than
"it harms strong models"; (iv) the **P3 split result** (feeding the hypothetical
document to BM25 as well costs a further −0.2735).

**Framing advice:** do **not** present this as a new discovery. Present it as
*"we test Weller et al.'s law in a setting it does not cover — a non-English,
entity-anchored, OCR'd institutional corpus — and it holds, with the loss
concentrated exactly where the lexical signal is strongest."* Citing Weller et
al. yourself is far better than a reviewer citing it at you.

### (b) Off-the-shelf cross-encoder rerankers HURTING ranking quality

**Verdict: CONFIRMATION in general; CONTRIBUTION for Thai / this domain.**

Two primary sources report rerankers actively hurting:

- Jacob, Lindgren, Zaharia, Carbin, Khattab & Drozdov, *"Drowning in Documents:
  Consequences of Scaling Reranker Inference"*
  ([arXiv 2411.11767](https://arxiv.org/abs/2411.11767); ReNeuIR 2025 workshop @
  SIGIR 2025): "the best existing rerankers provide diminishing returns when
  scoring progressively more documents and actually degrade quality beyond a
  certain limit", and "rerankers often score completely irrelevant documents …
  very highly". Mechanism = **candidate-set size**.
- Apurba, Hasan, Shehab & Azad, *"SciRet: A Compute-Aware Empirical Study of
  Retrieval and Reranking for Scientific RAG"*
  ([arXiv 2608.03860](https://arxiv.org/html/2608.03860v1), Aug 2026): "an MS
  MARCO-trained cross-encoder reduces precision on the scientific corpus,
  suggesting that domain mismatch can outweigh the benefits of stronger
  query–passage interaction" — P@5 falls **0.600 → 0.404** on CORD-19.
  Mechanism = **domain mismatch**. Note this is a recent, non-peer-reviewed
  preprint; verify its status before leaning on it.

**What is still ours:** (i) a **non-English** instance — both papers above are
English; (ii) the **specific metric asymmetry** we measured (significantly hurts
hybrid *MRR* at 0.7730 → 0.6940, Holm 0.0240, while being ns on recall@10 and
nDCG@10 — a reordering harm, not a recall harm); (iii) the **oracle bound**
showing the evidence *was* reachable (routed P=50 pool holds 0.9054 of gold;
perfect selection delivers 0.8331, i.e. +0.1520 against the off-the-shelf
model's −0.0098), which converts "the reranker failed" into "the *wiring and the
model* failed, the axis did not" — a much stronger and more publishable claim;
(iv) the fine-tuned counter-result (+0.0730 over the router, Holm 0.0000) that
closes the loop.

**Framing advice:** the oracle bound plus the fine-tune is what makes this
publishable rather than anecdotal. Lead with *"the null belongs to the wiring
and the model, not the axis"*.

### (c) ColBERT / late interaction behaving lexically

**Verdict: CONFIRMATION of the mechanism, CONTRIBUTION of the consequence.**

The mechanism is well established and by the ColBERT-analysis authors themselves:

- Formal, Piwowarski & Clinchant, *"A White Box Analysis of ColBERT"*,
  **ECIR 2021 (Best Short Paper)**
  ([Springer](https://link.springer.com/chapter/10.1007/978-3-030-72240-1_23),
  [arXiv 2012.09650](https://arxiv.org/abs/2012.09650)): ColBERT "is able to
  capture a notion of term importance and relies on exact matches for important
  terms".
- Formal, Piwowarski & Clinchant, *"Match Your Words! A Study of Lexical Matching
  in Neural Information Retrieval"*, **ECIR 2022**
  ([Springer](https://link.springer.com/chapter/10.1007/978-3-030-99739-7_14),
  [arXiv 2112.05662](https://arxiv.org/abs/2112.05662)): asks whether neural IR's
  out-of-domain shortcomings follow from an inability to perform lexical matching.
- Independent Thai corroboration exists in
  [MIRACL's own Table 5](https://aclanthology.org/2023.tacl-1.63/): on Thai,
  **mColBERT nDCG@10 = 0.481 vs BM25 0.484** — a tie with the lexical baseline,
  which is exactly the "inherits one side" shape.

**What is still ours:** the specific *conjunctive* framing — that ColBERT
**inherits** one side of a person/program lexical-vs-semantic split rather than
covering it (person 0.8360 ≈ BM25 0.8053; program 0.2749 ≈ BM25 0.3278 vs dense
0.6086), and that the aggregate would have *passed* (0.5555 overall, the highest
in the table) had it been written as an aggregate. **No source was found that
states the split-inheritance result in those terms.** The
*"a conjunction refuses an aggregate win"* methodological point looks genuinely
novel and is arguably the most transferable thing in the whole ColBERT section.

**Framing advice:** the finding is not "ColBERT is lexical" (known since 2021) —
it is *"a per-entity-type decomposition turns an apparent aggregate win into a
pre-registered failure"*. Sell the decomposition, cite Formal et al. for the
mechanism.

### (d) Thai normalization and word-segmentation-aware chunking having no effect

**Verdict: CONTRIBUTION — the literature appears SILENT on this exact question.**

What exists is **classical Thai IR indexing work** on whether to segment at all.
The canonical citation is Theeramunkong, Sornlertlamvanich, Tanhermhong &
Chinnan, *"Character cluster based Thai information retrieval"*, **IRAL 2000**
(5th International Workshop on Information Retrieval with Asian Languages), ACM,
pp. 75–80 ([DOI 10.1145/355214.355225](https://dl.acm.org/doi/10.1145/355214.355225)),
which proposes character clustering precisely "to reduce the ambiguity of word
boundary in Thai documents and improve searching efficiency"; later Thai work on
frequent-max-substring indexing continues the line, and n-gram indexing is
generally reported to outperform segmentation-based indexing at a storage and
latency cost. That literature is **pre-neural and about sparse indexing**, and it
points the *opposite* way from a null — it says the choice mattered.

What exists on the tooling side is segmentation-quality benchmarking, not
retrieval impact: PyThaiNLP documents `newmm` as fastest with the lowest
tokenization quality and `deepcut` as most accurate
([AttaCut survey](https://pythainlp.org/attacut/survey.html),
[PyThaiNLP paper](https://arxiv.org/html/2312.04649v1)); Thai word segmenters are
routinely evaluated on F1 against segmentation gold standards, never on
downstream retrieval effectiveness.

**No study was found** that measures the retrieval effect of (i)
`pythainlp.util.normalize()`-style Thai normalization or (ii) word-boundary-aware
chunking, on **dense** or **hybrid** retrieval. Given modern multilingual
embedders use subword tokenizers that never see the word boundary, a null is
mechanistically unsurprising — but *unsurprising is not published*, and this is
the finding here with the cleanest claim to being new.

**Framing advice:** this is a small but genuinely novel null, and it is best
framed **against the classical Thai IR literature** ("segmentation mattered for
sparse indexing; we show it does not measurably matter for subword-tokenized
dense and hybrid retrieval on this collection, bounded at Holm-adj p ≥ 0.264"),
not against nothing. That framing also gives it a natural home in a Thai/regional
venue.

### (e) RRF helping the weaker arm and taxing the stronger one

**Verdict: CONFIRMATION of a closely analogous law; CONTRIBUTION for RRF specifically.**

- **Weller et al. (Findings of EACL 2024)** — quoted in (a) — is the same law one
  level up: gains accrue to weak systems and harm strong ones. Any reviewer who
  knows it will connect the two.
- **RRF's own origin**: Cormack, Clarke & Buettcher, *"Reciprocal Rank Fusion
  Outperforms Condorcet and Individual Rank Learning Methods"*, SIGIR 2009
  ([ACM DL, DOI 10.1145/1571941.1572114](https://dl.acm.org/doi/10.1145/1571941.1572114);
  bibliographic record confirmed via
  [Google Research](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/):
  SIGIR '09, pp. 758–759) — a **two-page** paper claiming RRF beats Condorcet,
  CombMNZ and learning-to-rank methods. It **does not** characterise when fusion hurts.
- **Bruch, Gai & Ingber**, *"An Analysis of Fusion Functions for Hybrid
  Retrieval"*, **ACM TOIS 42(1), 2023**
  ([ACM DL](https://dl.acm.org/doi/10.1145/3596512),
  [arXiv 2210.11934](https://arxiv.org/abs/2210.11934)): finds RRF "is sensitive
  to its parameters" and that a tuned convex combination "outperforms RRF in
  in-domain and out-of-domain settings". This is the closest thing to a
  critical treatment of RRF, and it is about **parameter sensitivity and
  score-vs-rank fusion**, not about the weak-arm/strong-arm asymmetry.
- **MIRACL** itself uses an untuned α = 0.5 BM25+mDPR hybrid "without tuning"
  and reports it as the strongest zero-shot baseline on average — consistent
  with our finding that 0.50 is already sane where the arms are comparable.

**What is still ours:** the asymmetry stated *as a rule about fusion*, with the
consequences derived — (i) "hybrid beats dense-alone for every embedder" is
**entity-anchored-specific**, and on thematic queries hybrid significantly
*hurts* two embedders; (ii) "do not naively RRF a weak embedder with BM25"
(m2v, sct significantly worse than BM25 alone on all three metrics); (iii) the
per-`entity_type` decomposition showing BM25 carries `person` (0.8147, beating
every embedder's dense-alone score) and collapses on `program` (0.3497). **No
source was found stating the weak-arm/strong-arm rule for RRF specifically.**

**Framing advice:** cite Weller et al. and Bruch et al. and position ours as the
**fusion-side instance** of a known expansion-side law, with the per-category
decomposition as the new evidence. Do not present the rule as unprecedented.

### 4.1 Summary table

| Finding | Already known? | Verdict | Must-cite |
|---|---|---|---|
| (a) HyDE fails on entity-anchored queries | The general law, yes | **Confirmation + new setting** | Weller et al. EACL-F 2024; Yoon et al. ACL-F 2025 |
| (b) Off-the-shelf reranker hurts | Yes, twice | **Confirmation + oracle bound is new** | Jacob et al. 2024; SciRet 2026 |
| (c) ColBERT behaves lexically | Mechanism yes; consequence no | **Confirmation + new decomposition** | Formal et al. ECIR 2021, ECIR 2022 |
| (d) Thai normalization / segmentation null | **Not found** | **Contribution** | classical Thai IR indexing literature (as contrast) |
| (e) RRF taxes the stronger arm | Analogue yes; RRF-specific no | **Confirmation + new instance** | Cormack et al. 2009; Bruch et al. TOIS 2023; Weller et al. |

### UNCERTAIN / COULD NOT VERIFY (§4)

- **Whether any paper reports HyDE failing specifically on *named-entity*
  queries.** Weller et al. analyse by retriever strength and dataset shift; the
  ACL Anthology abstract page does not break results down by query type, and we
  did not read the full paper's per-dataset analysis. **Read Weller et al. §
  results in full before claiming the entity-query angle is uncovered.**
- **SciRet's peer-review status** ([arXiv
  2608.03860](https://arxiv.org/html/2608.03860v1)) — no venue is stated. Treat
  as a preprint.
- **Whether NitiBench reports a reranker null for Thai.** It uses
  BGE-reranker-v2-m3 in the WangchanX pipeline lineage; we did not confirm
  whether it reports a reranking ablation. If it does, that is a *direct* Thai
  precedent for finding (b) and must be cited.
- **Any Thai-language-venue publication** (iSAI-NLP, JCSSE, ECTI-CON, TCI
  journals) reporting any of (a)–(e) for Thai. These proceedings are poorly
  indexed by general web search; **a manual sweep of IEEE Xplore and TCI-ThaiJo
  is owed before claiming (d) is unstudied.**

---

## 5. Venue map

### 5.0 Timing reality check — read this first

**Today is 2026-08-24, and that eliminates most of the obvious answers.** SIGIR
2026 (Melbourne, July), ACL 2026 (San Diego, July), ECIR 2026 (Delft,
March–April), LREC 2026 (Palma, May), CLEF 2026 (Jena, September — deadline
passed), CIKM 2026 (Rome, November — deadline passed), JCSSE 2026 (June) and
SIGIR-AP 2026 (deadline 29 July) have all either happened or closed. Verified
status of every venue *right now*:

| Venue | Status as of 2026-08-24 |
|---|---|
| **ECIR 2027 — Full Papers** | **OPEN.** Abstract **21 Sep 2026**, paper **5 Oct 2026** |
| **ECIR 2027 — Reproducibility** | **OPEN.** Abstract 12 Oct 2026, paper **19 Oct 2026** |
| **ECIR 2027 — Resource** | **OPEN.** Paper **2 Nov 2026** |
| **iSAI-NLP 2026** | **OPEN, closing fast — 1 Sep 2026** (extended from 15 Aug) |
| **ECTI-CIT / JCST / ASEP (journals)** | **OPEN — rolling, no deadline** |
| ARR next cycle | 12 Oct 2026 — but commits to **NAACL/COLING 2027**, not ACL/EMNLP |
| SIGIR 2027, SIGIR-AP 2027, CIKM 2027, CLEF 2027 (conference papers), ACL/EMNLP 2027, JCSSE 2027, ECTI-CON 2027 | **No CFP published — could not verify any deadline** |
| LREC | **Biennial, even years. No 2027 exists.** Next is LREC 2028; no site yet |
| Insights@NLP 2026 (7th) | **Closed** — direct 8 Jun 2026, ARR commit 25 Jun 2026 |

### 5.1 International venues

#### ECIR 2027 — *the only major venue with a live deadline, and a good fit*

[ecir2027.co.uk](https://www.ecir2027.co.uk/) · Southampton, UK, 21–25 Mar 2027 ·
Springer LNCS.

| Track | Length | Deadline |
|---|---|---|
| [Full](https://www.ecir2027.co.uk/call-for-full-papers) | **12 pp** + unlimited refs (appendices count) | abstract 21 Sep, paper **5 Oct 2026** |
| [Short](https://www.ecir2027.co.uk/call-for-short-papers) | 6 pp + refs | see calls page |
| [Reproducibility](https://www.ecir2027.co.uk/call-for-reproducibility-papers) | 12 pp + refs | paper **19 Oct 2026** |
| [Resource](https://www.ecir2027.co.uk/call-for-resource-papers) | 12 pp + refs | paper **2 Nov 2026** |

Full-paper topics explicitly include *"Evaluation research, including new
metrics, benchmarks and novel methods for the measurement and evaluation of
retrieval and/or recommendation systems"* — a direct match. **No data-release
requirement is stated for the full-paper track.**

**Non-releasable corpus: fine for Full/Short, a problem for Resource.** The
[resource call](https://www.ecir2027.co.uk/call-for-resource-papers) asks
whether the resource is "released in a permanent repository for easy access by
researchers" and requires "the licensing and terms of use sufficiently open to
allow most academic and industry researchers to access and use the resource".
Encouraged rather than mandated — but it is an explicit *review criterion*.

**Negative results: the reproducibility track says so in as many words** —
*"A successful reproduction of the work is not a requirement."* Note the
mismatch, though: that track is for re-testing **someone else's** published
method and explicitly excludes same-team repetition. Our own pre-registered
nulls fit the **Full Paper** track's "evaluation, analysis" framing better.

#### CLEF — *the best negative-results fit anywhere, and it has a pre-registration mechanism*

[clef2026.clef-initiative.eu/calls/papers](https://clef2026.clef-initiative.eu/calls/papers/)
(2026 cycle, deadline elapsed) · CLEF 2027 Bucharest, 14–17 Sep 2027, conference
CFP **not yet published**.

Two quotes, both verified verbatim:

> "CLEF welcomes papers that describe rigorous hypothesis testing regardless of
> whether the results are positive or negative."

> "negative results and failed experiments are explicitly welcome."

And the mechanism that makes it real — **two-stage results-blind review**:

- **Stage 1** — authors submit a version with **results and discussion removed**
  (and result mentions stripped from the abstract). Reviewers assess "the
  importance of the problem addressed and the soundness of the methodology".
- **Stage 2** — papers passing stage 1 submit the full manuscript. *"The final
  decision will not be based on whether results are positive or beat a baseline."*

**This is a registered-report mechanism at an IR venue**, and it is the natural
home for a study that froze its predictions before each run. Long research
papers **12 pp + refs**, short **6 pp + refs**. Data/code: share with reviewers
via anonymous repositories, "encouraged" — **no public release required**.

**CLEF *Labs* are the wrong shape** and should not be confused with the
conference: labs are shared tasks run on organiser-released common datasets
producing working notes, and none of the CLEF 2026 labs matches this work.

#### EMNLP / ACL (via ACL Rolling Review) — *clearest CFP language, but no open cycle*

[EMNLP 2026 CFP](https://2026.emnlp.org/calls/main_conference_papers/) ·
[ACL 2026 CFP](https://2026.aclweb.org/calls/main_conference_papers/). Both carry
the identical sentence:

> "papers may contribute negative findings, survey an area, announce the creation
> of a new resource, argue a position, report novel linguistic insights … and
> reproduce, or fail to reproduce, previous results."

Long **8 pp**, short **4 pp**, +1 page on acceptance, **unlimited references and
an unlimited Limitations section**. Findings is the overflow venue.

**Non-releasable corpus: this is the most permissive family.** The ARR
Responsible NLP Checklist has **no question demanding public data or code
release** — section B only asks authors to *document* licensing, intended use,
PII handling and coverage, all answerable in prose. ARR's own stated policy is
that "not answering positively to a question is not grounds for rejection" given
a justification. **Caveat flagged as unverified:** checklist answers are
reported to become a public camera-ready appendix, which would make an
institutional/privacy statement publicly visible — confirm before drafting.

**Timing:** ACL 2026 and EMNLP 2026 are both closed. The next open ARR cycle
(12 Oct 2026) commits to **NAACL 2027 / COLING 2027**, not ACL/EMNLP.

#### SIGIR and SIGIR-AP

**SIGIR** — [sigir2026.org](https://sigir2026.org/en-AU) concluded; SIGIR 2027
site is a placeholder with no CFP. 2026 structure as a guide: Full **9 pp** +
refs, Short **4 pp** + refs, **Resources 6 pp**, **Reproducibility 9 pp**, plus a
**Low-Resource Environments** track (a 2-page presentation proposal only, not a
paper venue). Reproducibility accepts analysis of "the extent to which the
assumptions of the original work hold up". **The Resources track states the
resource "must be available to reviewers at the time of submission" with open
licensing and no stated waiver — avoid it.**

**SIGIR-AP** — [sigir-ap.org/sigir-ap-2026](https://www.sigir-ap.org/sigir-ap-2026/call-for-papers/index.html),
4th edition, Singapore, **13–15 Dec 2026**; deadlines (abstract 22 Jul, paper 29
Jul 2026) **already elapsed**. Structurally the friendliest major: a **single
track, 2–9 pages** + unlimited refs, judged on contribution-to-length fit, with
named categories for *original research, resource, reproducibility, industry and
perspective* papers, plus a SIGIR-Revise-and-Resubmit route for SIGIR/ICTIR
rejections. Scope is "the same as that of SIGIR". Data policy is the mildest of
the majors — authors are merely *"encourage[d] … to make as many of the resources
associated with a paper publicly available"*. **Watch for the 2027 CFP; and the
SIGIR-RR route means a SIGIR 2027 rejection is not wasted.**

#### CIKM

[cikm2026.diag.uniroma1.it](https://cikm2026.diag.uniroma1.it/) · Rome, 7–11 Nov
2026, deadlines elapsed (abstract 16 May, paper 23 May 2026). Full Research
**10 pp + 2 pp refs**, Short **4 pp**, Resource **4 pp**. No explicit
negative-results language. Full Research only "strongly encourages" sharing data
with reviewers — **but the Resource track requires public DOI'd release, so avoid
it.** CIKM 2027 (Sydney) is rumoured via the organisers' social account only —
**unverified**; expect a ~May deadline by analogy.

#### LREC — *biennial, and the worst fit for a private corpus*

[lrec2026.info](https://lrec2026.info/third-call-for-papers/) — held 11–16 May
2026. **LREC runs in even years only; there is no LREC 2027.** Next possible is
**LREC 2028, with no site or CFP in existence.** 2026 format: 4–8 pp submission,
up to 9 pp camera-ready, optional 10 pp appendix.

LREC is a **language-resources** venue, so the non-releasable corpus bites
hardest here. The 2026 CFP frames resource sharing as an opportunity —
*"authors are also offered the opportunity to share related language resources
with the community"* via the LRE Map — softer than LREC's historical
mandatory-registration-with-opt-out reputation. **Not a hard blocker as written,
but this is the one venue where a reviewer is most likely to ask "why can we not
have the corpus?" and treat the answer as disqualifying.**

#### NTCIR — worth knowing about

[NTCIR-19](https://research.nii.ac.jp/ntcir/ntcir-19/tasks.html), NII Tokyo,
**8–10 Dec 2026**. Not a paper venue for this work (it is a shared-task
evaluation forum), but its `RegCom` pilot task covers Thai structured documents
(§1.8) — relevant as concurrent work and as a possible future home for a Thai
administrative-document task proposal.

#### Workshop on Insights from Negative Results in NLP

[insights-workshop.github.io](https://insights-workshop.github.io/) · 7th
edition, co-located with **EMNLP 2026, Budapest, 22–29 Oct 2026**. **Deadlines
have passed** (direct 8 Jun 2026, ARR commit 25 Jun 2026). Short papers **4 pp**
excluding references, plus 1–2 page non-archival abstracts for work published
elsewhere. Solicits exactly our shapes — "ablation studies", "generalization
failures", "cross-lingual analyses", "reproducibility concerns". **Small venue,
4-page limit — a poor home for the whole study, but an excellent home for a
single finding (e.g. the Thai normalization/segmentation null), and the
non-archival abstract route lets it double-dip after a main-venue acceptance.**

### 5.2 Regional / Thai venues

#### iSAI-NLP 2026 — *open now, closing 1 September 2026*

[isai-nlp2026.aiat.or.th](https://isai-nlp2026.aiat.or.th/) · 21st edition,
Bangkok, **19–21 Nov 2026**, in-person only, organised by the AI Association of
Thailand. **Paper deadline 1 Sep 2026** (extended from 15 Aug), notification
3 Oct, camera-ready 10 Oct. Ten tracks including an explicit **Natural Language
Processing** track. **~6 pages**; prior editions are indexed in **IEEE Xplore**.
**No data-availability policy exists** — the corpus constraint is a non-issue.

This is the only Thai venue with an open deadline, and it is one week away.

#### JCSSE 2026 / 2027

[jcsse2026.org](https://jcsse2026.org/) · 23rd edition, KMUTT + IEEE Thailand
Section, Bangkok, 24–27 Jun 2026 — **elapsed** (three rounds: 31 Jan / 30 Mar /
24 Apr 2026). Extended abstract 1–2 pp; full paper **≤6 pp**; optional
journal-extension route to **ECTI-CIT** at 8–12 pp. Twenty tracks including an
explicit **Natural Language Processing** track. Proceedings → **IEEE Xplore**;
extended papers → **Scopus + TCI**. No data policy. 2027 dates not yet published.

#### ECTI-CON 2026 — weakest fit

23rd edition, Chonburi, 27–30 May 2026, IEEE Xplore-indexed; ~6 pp. **Eleven
general EE/CS tracks with no dedicated IR or NLP track** — this work would sit
generically under "Computers" or "Information Technology". No data policy.
**Partially verified only**: the official host returned a TLS certificate error
and details come from search-indexed snippets of the same official domain; a
human should re-check `eng.buu.ac.th/ecti-con2026/` directly.

#### Thai journals (rolling — always open)

| Journal | Indexing | Length | Fit |
|---|---|---|---|
| **[ECTI-CIT](https://ph01.tci-thaijo.org/index.php/ecticit)** | **Scopus** (2024 CiteScore 1.6) + **TCI Tier 1** | 8–12 pp | **Best Thai option.** Scope explicitly names *"Natural language processing, pattern recognition, data mining"* and *"Vision Language Models, Large Language Models"*. Rolling submission, ~6.5-week review, ~23.75% acceptance |
| **[JCST](https://ph04.tci-thaijo.org/index.php/JCST/about)** (Rangsit) | **Scopus Q2** + TCI Tier 1 + ACI | — | Has a **dedicated NLP subcategory** with on-genre precedent (Thai sentiment analysis, BERT summarization). Fee-based (~USD 100 review + USD 280 publication) |
| **[ASEP](https://ojs.kmutnb.ac.th/index.php/ijst)** (KMUTNB) | Scopus **Q1 Engineering** + TCI1 | — | "Information Technology" subtopic only; no NLP/IR category. Moderate fit |
| **[Engineering Journal](https://engj.org/)** (Chula) | Scopus Q3 | — | **Scope page returned 403; CS/IR fit unconfirmed** |

**None of the four states any data or artifact availability policy** — checked
on every page fetched. For a non-releasable corpus, the Thai journals are the
lowest-friction path in the entire landscape.

### 5.3 The non-releasable-corpus verdict, across every venue

**No venue researched disqualifies a non-releasable corpus for a regular
research-paper track.** The friction is concentrated in exactly three places:

1. **Resource tracks** (SIGIR, ECIR, CIKM) — all require reviewer-accessible
   data and open licensing, **with no stated waiver for restricted data**.
2. **LREC** — a language-resources venue by constitution; softest current
   wording, hardest cultural expectation.
3. **Reproducibility tracks** — which ask for code *and datasets* available to
   reviewers, and which in ECIR's case are for re-testing *other people's* work
   anyway.

**ARR (ACL/EMNLP/NAACL) is the most explicitly favourable**: no data-release
question exists on the Responsible NLP Checklist, and a truthful "cannot release
due to institutional constraints" is stated not to be grounds for rejection.
**CLEF and SIGIR-AP are next**, both encouraging rather than requiring.

**Rule: submit to the regular / full-paper track everywhere. Do not submit this
to a resource track unless a releasable derivative is built first** — the qrels,
the 106-query set, and either hashed/redacted chunks or a synthetic surrogate
corpus would turn the weakest part of the submission into an asset.

### 5.4 Recommended sequence

1. **iSAI-NLP 2026 (1 Sep 2026)** — a 6-page regional slice (e.g. the Thai
   normalization/segmentation null plus the chunker/embedder comparison) to get
   the work into IEEE Xplore and into the Thai community fast. Low risk, no data
   policy, one week out.
2. **ECIR 2027 Full Papers (abstract 21 Sep, paper 5 Oct 2026)** — the main
   12-page submission. Only live major deadline; explicit evaluation/benchmark
   scope; no data-release requirement on the full track.
3. **CLEF 2027 conference papers (CFP not yet out, expect ~May 2027)** — the
   single best ideological fit given the pre-registered nulls and the
   results-blind review. Watch [clef2027.clef-initiative.eu](https://clef2027.clef-initiative.eu/).
4. **SIGIR-AP 2027 / SIGIR 2027** — watch for CFPs; SIGIR-AP's 2–9-page single
   track and SIGIR-RR route make it a good second home.
5. **ECTI-CIT** — the rolling-deadline fallback and the natural home for an
   extended journal version, including the deployment/serving characterization
   that will not fit in a 12-page conference paper.

### UNCERTAIN / COULD NOT VERIFY (§5)

- **Every 2027 deadline except ECIR 2027.** SIGIR 2027, SIGIR-AP 2027, CIKM 2027,
  CLEF 2027 conference papers, ACL 2027, EMNLP 2027, JCSSE 2027 and ECTI-CON 2027
  have **no published CFP**. Previous-cycle dates are given above **as timing
  guides only** and must not be treated as deadlines.
- **LREC 2028** — no site, no CFP, no confirmed host. Whether the LRE Map
  resource-registration step is currently mandatory-with-opt-out or fully
  optional **could not be confirmed** from the 2026 pages fetched.
- **ECTI-CON 2026** — official site not directly fetchable (TLS certificate
  error); page limit inferred from the 2024 edition, deadline from search-indexed
  snippets. Re-check manually.
- **Engineering Journal (Chula)** scope page — HTTP 403; CS/IR suitability unconfirmed.
- **CIKM 2027** — Sydney venue reported only via the organisers' social media
  account, not an official conference page.
- **Whether ARR checklist answers become a public camera-ready appendix** — this
  matters if the data-availability answer must name institutional constraints.
  Reported via a non-official source; **verify before drafting**.
- **iSAI-NLP 2026 page limit and IEEE Xplore indexing for the 2026 edition** —
  the ~6 pp figure and Xplore indexing come from prior editions and the
  conference's general description, not from a 2026 author-instructions page we
  fetched.
- **SIGIR 2026 reproducibility/resources track wording** was obtained through a
  reader proxy after the JS-rendered official pages resisted direct fetching;
  substance corroborated against the SIGIR 2022 and 2024 calls.

---

## What would make our novelty claim weaker than we think

Every risk found, stated at full strength.

1. **NitiBench got there first for Thai domain RAG, at EMNLP main.** A Thai
   domain-specific RAG benchmark with multiple retrievers, chunking analysis and
   negative results is already published at a top venue
   ([aclanthology.org/2025.emnlp-main.1739](https://aclanthology.org/2025.emnlp-main.1739/)).
   The domain differs (law vs institutional minutes) and our judgment depth is
   ~10× theirs, but *"first Thai domain RAG benchmark"* is gone, and a reviewer
   familiar with Thai NLP will know this paper.

2. **The corpus cannot be released, and that is worse than it looks for a
   resource claim.** Our strongest structural precedent, the Amharic passage
   retrieval paper ([Findings of ACL
   2025](https://aclanthology.org/2025.findings-acl.543.pdf)), releases
   "dataset, codebase and models". NitiBench open-sources both datasets. The
   Greek government-decisions corpus is CC-BY-4.0
   ([arXiv 2512.05647](https://arxiv.org/abs/2512.05647)). **We would be the
   outlier.** ECIR's resource track "strongly encourages" a permanent URL and
   licensing "sufficiently open to allow most academic and industry researchers
   to access and use the resource"
   ([ECIR 2027 resource call](https://www.ecir2027.co.uk/call-for-resource-papers)) —
   encouraged, not mandated, but it is an explicit *review criterion*. **A
   resource-track submission is the wrong shape for this paper unless a
   releasable derivative (qrels + query set + hashed/redacted chunks, or a
   synthetic surrogate corpus) can be produced.**

3. **Three of our five negative results are confirmations, not discoveries.**
   (a), (b) and (e) all have prior art (§4). If the paper is sold as "five novel
   negative results", three of them will be knocked down in review, and the
   knock-down will discredit the two that stand. Sell it as *"we test five
   widely-adopted techniques against a pre-registered protocol in a setting none
   of them was validated in, and four fail"* — a **replication-in-a-new-setting**
   claim, which is exactly what the reproducibility tracks want.

4. **The judgment depth claim is fragile in the wrong unit.** "Unusually deep at
   9.87" is true against MS MARCO (1.07) and Mr. TyDi (1.02) and false against
   TREC deep pooling (270–2,615 judged/topic). It is also *not* a match for
   MIRACL's 9.23, which counts non-relevants. **A reviewer who knows MIRACL and
   sees an unqualified depth comparison will read the whole evaluation section
   as insufficiently careful.**

5. **We have no judged negatives at all.** Every one of our 1,046 judgments is a
   positive derived by rule. MIRACL deliberately judges negatives and argues they
   are "quite valuable"; TREC pools them. Our residual-relevance study (~19–22%
   residual across arms) is the substitute and it is a good one — but it is a
   *sample*, and it must be foregrounded rather than buried, because the first
   question any IR reviewer asks a rule-derived qrels set is "what did the rule
   miss, and is the miss correlated with the systems you are comparing?"

6. **The circularity in the entity arms is a known-fatal shape if under-stated.**
   The qrels and the `entity_lookup`/`entity_boost` retrieval modes read the same
   dictionaries. This is already documented internally, but in a paper it must
   appear *next to the number*, not in a limitations section — and the recall =
   0.9449 figure must never be printed in a column next to recall@10 values.

7. **Rule-derived judgments will be challenged as "not relevance".** Deriving
   qrels by string containment is what BM25 does, which structurally penalises
   dense retrieval for being right — and it points straight at our own headline
   "BM25 carries `person` (0.8147)". The residual-relevance measurement answers
   it, but expect the challenge, and expect it to be aimed at the
   BM25-vs-dense comparison specifically.

8. **The chunker axis is getting crowded fast.** Systematic chunking studies
   appeared through 2025–2026 — e.g.
   [*Rethinking Chunk Size for Long-Document Retrieval*](https://arxiv.org/abs/2505.21700),
   [*Chunk Twice, Embed Once*](https://arxiv.org/html/2506.17277v1) (25 chunking
   configurations × 48 embedding models). Our chunker contribution is a **tie
   with a bound**, not a win; against that literature the interesting part is the
   **opposing signal between entity-anchored and thematic queries**, not the
   comparison itself.

9. **"Administrative meeting minutes + RAG" is no longer an empty phrase.** The
   ISWC 2025 Japanese FSA workshop paper
   ([CEUR Vol-4085 p54](https://ceur-ws.org/Vol-4085/paper54.pdf)) exists. It is
   methodologically far weaker (50 questions, manual accuracy, no qrels), so it
   does not pre-empt a test collection — but it must be cited, and citing it
   ourselves is much better than being shown it.

10. **The deployment/serving characterization has no measured network hop.**
    App, embedder and engine are one process on one box. Any venue with a
    systems or industry orientation will treat a single-box characterization as
    a lab result, and the paper must say so before a reviewer does.

11. **Our "pre-registration" is self-attested, and only one venue can check it.**
    No IR venue runs a track called pre-registration; CLEF's two-stage
    results-blind review (§3.1b) is the only mechanism found that would actually
    *verify* a frozen prediction, and it does so by reviewing the methodology
    before the results exist. Everywhere else, "pre-registered" means "we wrote
    it down in our own repository first" — honest, useful, and **self-attested**.
    A sceptical reviewer is entitled to discount it unless the frozen design
    documents are released with timestamps. Releasing those documents costs
    nothing (they contain no corpus) and is the cheapest credibility available.

12. **The publication window is awkward and that is itself a risk.** As of
    2026-08-24 the only open major deadline is **ECIR 2027 (paper 5 Oct 2026)**,
    and the only open Thai deadline is **iSAI-NLP 2026 (1 Sep 2026)**. Everything
    else — SIGIR, SIGIR-AP, CIKM, CLEF, ACL, EMNLP, LREC — is either past or has
    no published CFP. That creates pressure toward whichever venue happens to be
    open rather than whichever fits, and pressure to fragment the work across a
    fast regional paper and a slower major one. **If the study is split, the
    regional paper must not consume the headline result the major paper needs**,
    and the overlap must be declared to both.

13. **Several 2026 findings will be a year old by the time they appear.** ECIR
    2027 publishes in March 2027; a CLEF 2027 paper appears in September 2027.
    The reranker, ColBERT and HyDE axes were measured against rebuild #4 in
    August 2026, and this field moves fast enough that "off-the-shelf
    cross-encoder" and "ColBERT checkpoint" will both mean different artifacts by
    then. **The bounds and the mechanisms transfer; the model-specific levels do
    not.** Write the claims so that a newer checkpoint does not falsify them —
    which the existing "the null belongs to the wiring and the model, not the
    axis" framing already does, and which a bare "reranking hurts" would not.

14. **Thai-venue prior art is under-searched.** iSAI-NLP, JCSSE, ECTI-CON and
    TCI-indexed Thai journals are poorly covered by the search engine used here.
    There may be a Thai-language paper on Thai administrative-document retrieval
    that no search in this session would surface. **A manual IEEE Xplore +
    TCI-ThaiJo sweep is owed before any "first" claim is committed to print.**
