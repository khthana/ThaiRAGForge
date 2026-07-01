# Context Glossary — RAG Indexing Experimentation Framework

A shared glossary for this project. Terms only — no implementation details.

## Corpus & Documents

- **Corpus** — The full body of source material under test: KMITL academic-council
  resolutions spanning Buddhist-era years 2564–2569.
- **Resolution (มติ)** — The atomic source document: one academic-council agenda
  item / decision. Currently stored as an OCR-produced Markdown file. *(Unit to be
  confirmed — a file may contain one or several มติ.)*
- **Session (ครั้งที่ N)** — A numbered council meeting within a given year that
  groups the resolutions decided in it.

## Pipeline Units

- **Chunk** — A sub-segment of a Resolution's text; the unit that retrieval returns
  and displays. Every Chunk carries a stable `resolution_id` back to its source
  Resolution (chunk identities are not stable across Chunkers; Resolution identity is).
- **Loader / Chunker / Embedder / Retriever** — The four swappable strategy stages.
  Loader also enriches metadata (year, session, เรื่อง/title, NER entities) that the
  Retriever may use for lexical or filtered search.
- **Combination** — One concrete tuple (Loader × Chunker × Embedder × Retriever)
  under test.
- **Index artifact** — The persisted, cached output of the expensive Index-build
  phase for one (Loader × Chunker × Embedder) triple: chunks + embeddings + a lexical
  (BM25) index + metadata. Retrievers operate over an Index artifact; they never
  re-embed.
- **Index-build phase** — Offline, expensive: Loader → Chunker → Embedder produces an
  Index artifact. Cached on (document set, chunker params, embedder) so nothing is
  re-embedded needlessly.
- **Retrieval phase** — Query-time, cheap: a Retriever ranks Chunks from a chosen
  Index artifact for a Query.
- **Dev subset** — A small default slice of the Corpus used for fast iteration; the
  full 1,215-Resolution corpus is an explicit opt-in.
- **Experiment run** — A single execution of the framework that produces persisted,
  comparable artifacts for one or more Combinations.

## Retrieval

- **Query (คำค้น)** — A user-supplied search string entered to test retrieval.
- **Dense retrieval** — Ranking Chunks by embedding (cosine) similarity to the Query.
- **Lexical retrieval (BM25)** — Ranking Chunks by term overlap, using Thai-aware
  tokenization (pythainlp). Requires a lexical index over chunk text.
- **Hybrid retrieval** — Fusing Dense and Lexical rankings (e.g. RRF or weighted sum).
- **Metadata filter** — An orthogonal pre-filter (by year / faculty / session) applied
  before any Retriever ranks, narrowing the candidate Resolutions.
- **Retrieval unit** — The Chunk. Retrieval ranks and returns top-k Chunks.
- **Relevance unit** — The Resolution. Whether a retrieval result is "correct" is
  judged at the Resolution level, not the Chunk level: a Query's ground truth is
  "Resolution R is relevant to Q." Chunk-level labels are deliberately avoided
  because they do not survive a change of Chunker.
- **Hit** — For a Query, a retrieved top-k Chunk whose `resolution_id` belongs to a
  Resolution labelled relevant to that Query. Metrics (recall@k, MRR, nDCG) are
  computed over Hits at the Resolution level.
- **Silver query set** — Auto-generated Query→Resolution pairs: each Resolution's
  เรื่อง/title is a Query whose one relevant Resolution is itself. Cheap, immediate,
  but "easy" (title wording overlaps the document).
- **Gold query set** — A small set of hand-written realistic Queries with manually
  labelled relevant Resolutions. Harder and truer to real use than the Silver set.
