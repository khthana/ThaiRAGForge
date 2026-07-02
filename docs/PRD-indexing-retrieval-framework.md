# PRD: Framework ทดลอง Indexing + Retrieval สำหรับ RAG (มติสภาวิชาการ)

**วันที่:** 2026-07-01
**เจ้าของ:** thana.ho@kmitl.ac.th
**เอกสารอ้างอิง:** `requirements/req0.md` (spec เดิม), `CONTEXT.md` (glossary),
`docs/adr/0001` และ `0002` (เหตุผลการตัดสินใจ), `docs/grill-session-2026-07-01.md`
(บันทึกการสัมภาษณ์)

**สถานะ (2026-07-02):** Implemented — issues #1–#11 ปิดหมดแล้ว ดู `README.md` §Status
สำหรับข้อจำกัดที่ตั้งใจเหลือไว้ (APIEmbedder เป็น stub, HybridEmbedder ไม่ได้สร้าง,
SemanticChunker threshold ยังไม่ tune). เอกสารนี้คงไว้ตามแผนเดิม ไม่แก้ไขตามของจริงที่สร้าง.

---

## Problem Statement

ผู้ใช้ (นักวิจัย/ผู้พัฒนา) มีคลังเอกสารมติสภาวิชาการ KMITL ที่ผ่าน OCR แล้ว 1,215 ไฟล์
Markdown (`academic_resolutions/`) และต้องการสร้างระบบ RAG แต่ **ยังไม่รู้ว่าองค์ประกอบ
ชุดไหนดีที่สุด** สำหรับข้อมูลภาษาไทยชุดนี้ — ควรแบ่ง chunk แบบไหน, ใช้ embedding model
ตัวใด, ค้นแบบ dense/BM25/hybrid ดีกว่ากัน

ทุกวันนี้ยังไม่มีวิธีที่จะ **สลับองค์ประกอบทีละส่วนแล้วเทียบผลได้อย่างเป็นระบบและทำซ้ำได้**
การทดลองด้วยสคริปต์แยกๆ (เช่น `indexing_pipline.py`) ทำให้ผลลัพธ์กระจัดกระจาย เทียบกัน
ไม่ได้ และไม่รู้ว่าปรับอะไรแล้ว retrieval ดีขึ้นจริงหรือไม่

## Solution

สร้าง **framework แบบ modular** ที่แยกไปป์ไลน์เป็นองค์ประกอบสลับได้ (pluggable) 4 แกน
คือ **Loader × Chunker × Embedder × Retriever** ผู้ใช้เลือก combination ผ่าน **Streamlit
UI** (หรือ YAML/CLI) สั่งรัน ระบบจะ **สร้างและ cache ดัชนี (Index artifact)** แล้วให้ผู้ใช้
**ใส่คำค้นเพื่อดู top-k chunk ที่เกี่ยวข้อง เทียบหลาย combination เคียงกัน** พร้อมเก็บผล
ทุกครั้งแบบถาวรเพื่อนำมาเปรียบเทียบและวัด metric (recall@k/MRR/nDCG) ย้อนหลังได้

จุดสำคัญของสถาปัตยกรรม (ดู ADR-0001):
- แยก **Index-build phase** (ออฟไลน์ แพง cache) ออกจาก **Retrieval phase** (query-time ถูก)
- core เป็น Python package ล้วน **ไม่ผูก Streamlit** จึง scriptable และ unit-test ได้
- ตัดสินความถูกของ retrieval ที่ **ระดับ Resolution (มติ)** ไม่ใช่ระดับ chunk (ดู ADR-0002)

## User Stories

1. ในฐานะนักวิจัย ฉันต้องการเลือก Loader/Chunker/Embedder/Retriever จากรายการใน UI เพื่อ
   ประกอบ pipeline โดยไม่ต้องแก้โค้ด
2. ในฐานะนักวิจัย ฉันต้องการปรับพารามิเตอร์ของแต่ละองค์ประกอบ (chunk_size, overlap,
   breakpoint threshold, model name, น้ำหนัก hybrid) ผ่าน UI/YAML เพื่อทดลองหลายค่า
3. ในฐานะนักวิจัย ฉันต้องการสั่งรันทุก combination (cartesian) หรือเฉพาะคู่ที่ระบุ (paired)
   เพื่อควบคุมปริมาณการทดลอง
4. ในฐานะนักวิจัย ฉันต้องการให้ระบบรันบน **dev subset เป็นค่าเริ่มต้น** และเลือก full corpus
   เป็น opt-in เพื่อ iterate ได้เร็วบน RTX 3060
5. ในฐานะนักวิจัย ฉันต้องการเห็น progress bar ระหว่างรัน batch เพื่อรู้ว่าเหลืออีกเท่าไร
6. ในฐานะนักวิจัย ฉันต้องการให้ combination ที่ fail (เช่นโหลด model ไม่ได้) ถูก log แล้วรัน
   ตัวอื่นต่อ ไม่ทำให้ทั้ง batch พัง
7. ในฐานะนักวิจัย ฉันต้องการใส่คำค้นภาษาไทยแล้วเห็น top-k chunk ที่เกี่ยวข้องที่สุด เพื่อ
   ประเมิน retrieval ด้วยตา (eyeball)
8. ในฐานะนักวิจัย ฉันต้องการเห็นผล retrieval ของหลาย combination **เคียงข้างกัน** สำหรับคำค้น
   เดียวกัน เพื่อเทียบว่าแบบไหนดึงมติที่ถูกขึ้นมาได้
9. ในฐานะนักวิจัย ฉันต้องการเห็นว่าแต่ละ chunk มาจาก **มติ (Resolution) ไหน** พร้อมหน้า/คะแนน
   เพื่อยืนยันความเกี่ยวข้อง
10. ในฐานะนักวิจัย ฉันต้องการเลือกวิธีค้น Dense / BM25 / Hybrid ต่อดัชนีเดิมได้โดยไม่ต้อง
    สร้างดัชนีใหม่ เพื่อเทียบวิธีค้นอย่างประหยัด
11. ในฐานะนักวิจัย ฉันต้องการกรองผลด้วย metadata (ปี/คณะ/ครั้งที่) ก่อนจัดอันดับ เพื่อจำกัด
    ขอบเขตการค้น
12. ในฐานะนักวิจัย ฉันต้องการให้ระบบ **เก็บผลทุกครั้งที่รัน** (คำค้น, combination, อันดับ
    chunk พร้อมคะแนน, timestamp) เพื่อนำมาเทียบภายหลังโดยไม่ต้องรันซ้ำ
13. ในฐานะนักวิจัย ฉันต้องการให้ทุก run บันทึก config ที่ใช้จริง + git hash + timestamp เพื่อ
    reproducibility
14. ในฐานะนักวิจัย ฉันต้องการวัด recall@k/MRR/nDCG อัตโนมัติด้วย **silver query set** (เรื่อง
    ของมติเป็นคำค้น) เพื่อได้ตัวเลขเทียบทันทีโดยไม่ต้อง annotate
15. ในฐานะนักวิจัย ฉันต้องการเพิ่ม **gold query set** ที่เขียนเองพร้อม label มติที่เกี่ยวข้อง
    เพื่อวัดผลกับคำค้นจริงที่ยากขึ้น
16. ในฐานะนักวิจัย ฉันต้องการให้ metric คำนวณที่ **ระดับ Resolution** (map chunk → resolution)
    เพื่อให้ label ใช้ซ้ำได้แม้เปลี่ยน chunker
17. ในฐานะนักวิจัย ฉันต้องการให้ Loader สกัด metadata (ปี, ครั้งที่, เรื่อง, source_url จาก
    ไฟล์ `_LINK`, entities จาก NER ไทย) เพื่อป้อนให้ retriever แบบ filter/lexical
18. ในฐานะนักวิจัย ฉันต้องการให้ chunker เคารพขอบเขต `## Page` ของ OCR เป็น hard boundary
    เพื่อไม่ให้ chunk คร่อมหน้าอย่างไม่มีความหมาย
19. ในฐานะนักวิจัย ฉันต้องการ chunker ที่รองรับการตัดประโยคภาษาไทย (pythainlp) เพราะภาษาไทย
    ไม่มีช่องว่างระหว่างคำ
20. ในฐานะนักวิจัย ฉันต้องการใช้ embedding model ภาษาไทยแบบ local (BGE-M3, multilingual-
    e5-large) บน GPU โดยไม่ต้องเสียค่า API
21. ในฐานะนักวิจัย ฉันต้องการให้มี interface สำหรับ API embedder ไว้ล่วงหน้า เพื่อเปิดใช้เมื่อ
    ใส่ key ภายหลังโดยไม่แก้สถาปัตยกรรม
22. ในฐานะนักวิจัย ฉันต้องการให้ embedding ถูก cache เพื่อไม่คำนวณซ้ำเมื่อ config เดิม
23. ในฐานะนักวิจัย ฉันต้องการนิยาม experiment เป็นไฟล์ YAML ที่เป็น source of truth เพื่อแชร์/
    เก็บ/รันซ้ำได้ โดย UI เป็นแค่ตัวแก้และสั่งรัน
24. ในฐานะผู้พัฒนา ฉันต้องการเพิ่ม strategy ใหม่ได้แค่สร้างไฟล์ + register ใน registry โดยไม่
    แตะ runner (Open/Closed)
25. ในฐานะผู้พัฒนา ฉันต้องการรัน pipeline ผ่าน CLI (typer) สำหรับงาน batch/อัตโนมัติ นอกเหนือ
    จาก UI
26. ในฐานะนักวิจัย ฉันต้องการดู metric สรุปต่อ combination (จำนวน chunk, การกระจายขนาด chunk,
    เวลา load/chunk/embed) เพื่อเข้าใจต้นทุนของแต่ละแบบ
27. ในฐานะนักวิจัย ฉันต้องการเพิ่ม SemanticChunker ภายหลังโดยกำหนด embedding model ตายตัว
    เพื่อไม่ให้ปนเปื้อน (confound) แกน Embedder
28. ในฐานะนักวิจัย ฉันต้องการให้ระบบเตือน/เข้าใจว่าคะแนน similarity ข้าม embedder เทียบตรงๆ
    ไม่ได้ จึงจัดอันดับ/normalize ภายใน combination เท่านั้น

## Implementation Decisions

**สถาปัตยกรรม 2 phase (ADR-0001)**
- Index-build: `Loader → Chunker → Embedder` → **Index artifact** (chunks + embeddings +
  BM25 index + metadata) cache ด้วย key `(document_set, chunker_params, embedder)`
- Retrieval: `Retriever` ทำงาน query-time บน Index artifact ที่มีอยู่ ไม่ re-embed
- จำนวน build ที่แพง = Loader×Chunker×Embedder (~24) ไม่ใช่ cartesian เต็ม 4 แกน (~72)

**Core เป็น package ไม่ผูก Streamlit** — UI (Streamlit 2 โหมด) และ CLI (typer) เป็น entry
point บางๆ บน core เดียวกัน

**4 โมดูลสลับได้ (registry แบบ decorator ต่อ stage)**
- **Loader** — `load(path) -> Resolution`; อ่าน Markdown, ทำความสะอาด artifact OCR (เก็บ
  `## Page` เป็น page marker), สกัด metadata (ปี/ครั้งที่จาก path, เรื่องจากชื่อไฟล์+บรรทัดแรก,
  `source_url` จากไฟล์ `_LINK.txt` ข้างเคียง, entities จาก pythainlp NER). กลยุทธ์:
  Plain / Metadata / NER
- **Chunker** — `chunk(Resolution) -> list[Chunk]`; ทุกตัวเคารพ `## Page` เป็น hard boundary
  และ chunk ทุกตัวถือ `resolution_id` เสมอ. กลยุทธ์เริ่ม: Fixed (char, มี param `unit:
  char|token`), Recursive (langchain-text-splitters + Thai separators), Sentence
  (pythainlp sent_tokenize). Semantic เพิ่มภายหลัง **โดย fix embedding model** (ADR-0001)
- **Embedder** — `embed(list[Chunk]) -> list[EmbeddedChunk]`; local-first ผ่าน
  sentence-transformers บน GPU: `BAAI/bge-m3`, `intfloat/multilingual-e5-large` (ใส่ prefix
  `query:`/`passage:` สำหรับ e5), batch + cache. `APIEmbedder` มี interface แต่ inert จนกว่า
  จะมี key
- **Retriever** — `retrieve(query, index, k) -> RetrievalResult`; Dense (cosine),
  BM25 (rank_bm25 + pythainlp tokens), Hybrid (RRF เป็นค่าเริ่ม, weighted-sum เป็น option) +
  `MetadataFilter` (ปี/คณะ/ครั้งที่) เป็น pre-filter orthogonal ใช้ได้กับทุก retriever

**Schema (Pydantic v2)**
- `Resolution`: resolution_id, source_path, source_url, year, session, title, raw_text,
  metadata
- `Chunk`: chunk_id, **resolution_id**, text, chunk_index, metadata (start/end char, page,
  chunker)
- `EmbeddedChunk`: chunk_id, embedding, embedding_model, embedding_dim
- `RetrievalResult`: query, combination_id, ranked [(chunk_id, resolution_id, score, rank)],
  top_k, retriever, timestamp — **persist ทุก run**

**Corpus** — แหล่งข้อมูล = 1,215 ไฟล์ `.md`; ไฟล์ `_LINK.txt` = URL ต้นฉบับ (provenance) ไม่ใช่
เนื้อหา; แต่ละ `.md` = 1 Resolution. Dev subset เป็น default, full corpus opt-in

**Config** — YAML เป็น source of truth (`req0.md` §5 ขยายด้วยบล็อก `retrievers:` และ
`corpus.subset`); ทุก run เขียน manifest (resolved config + git hash + timestamp + doc-set
hash)

**Evaluation** — silver query set (เรื่อง→มติตัวเอง) + gold query set (เขียนเอง); metric
คำนวณที่ระดับ Resolution โดย map chunk→resolution_id (ADR-0002); metric layer อ่านจากไฟล์
`RetrievalResult` ที่ persist ไว้ จึงวัดผล run เก่าได้โดยไม่ต้อง re-index

## Testing Decisions

**หลักการ:** ทดสอบเฉพาะ **external behavior** ของ interface โมดูล ไม่ทดสอบรายละเอียดภายใน
(implementation detail) ใช้ fixture เอกสารเล็กที่มีภาษาไทย/อังกฤษปน และ qrels จำลอง ทุกเทสต้อง
deterministic (ไม่พึ่งการโหลด embedding model จริงหรือเครือข่าย)

**โมดูลที่จะเขียนเทส (ผู้ใช้ยืนยันทั้ง 4):**
- **Chunkers** — input Resolution เล็ก → assert: การตัดตาม chunk_size/overlap, การเคารพ
  `## Page` เป็น hard boundary, การตัดประโยคไทย, และที่สำคัญ **`resolution_id` คงเดิมข้าม
  chunker ทุกตัว** (รากฐานของ ADR-0002)
- **Metrics** — ป้อน RetrievalResult จำลอง + qrels → assert recall@k / MRR / nDCG ที่คำนวณ
  ที่ระดับ Resolution (รวมเคส: chunk หลายตัวจากมติเดียวใน top-k ต้องนับเป็น hit เดียว)
- **Retriever fusion** — ป้อน ranking ย่อย 2 ชุด (dense/bm25) จำลอง → assert ผล RRF/weighted
  และการทำงานของ metadata filter (ตัดผู้สมัครก่อนจัดอันดับ) โดยไม่ต้องมี embedding จริง
- **Loader metadata + Index artifact store** — assert การ parse ปี/ครั้งที่/เรื่อง/source_url
  จาก path และไฟล์ `_LINK`, และ save/load round-trip ของ Index artifact + ความถูกต้องของ
  caching-key (config เดิม → cache hit, เปลี่ยน param → cache miss)

**Prior art:** ยังไม่มีชุดเทสในโปรเจกต์ (มีแต่สคริปต์ OCR) — จึงตั้ง `tests/` ใหม่ด้วย pytest
เป็นแบบแผนอ้างอิงสำหรับ strategy ที่จะเพิ่มในอนาคต

**ไม่เขียน unit test (ทดสอบด้วย smoke/manual):** Embedder ตัวจริง (พึ่ง model, ทดสอบแค่ cache/
batching interface), Streamlit UI (ทดสอบด้วยการรันจริง)

## Out of Scope

- Vector database (Milvus/Qdrant/Chroma) — ยังคง serialize เป็นไฟล์ (`req0.md` §10)
- Re-ranking / query transformation / query expansion
- Generation phase ของ RAG (LLM ตอบคำถาม) — โฟกัสแค่ indexing + retrieval
- การประเมินด้วย RAGAS หรือ metric ที่ต้องมี LLM
- SemanticChunker และ HybridEmbedder — เลื่อนเป็น milestone ท้าย
- API embedder แบบใช้งานจริง — มี interface แต่ยังไม่เปิด (ไม่มี key/งบ)
- OCR pipeline — ทำเสร็จแล้วใน `ocr2.py` เป็น input ให้ framework นี้

## Further Notes

- **Hardware:** RTX 3060 12GB (bge-m3 ~2.3GB / e5-large ~2.2GB ลง GPU ได้สบาย),
  Python 3.13, ยังไม่มี `OPENAI_API_KEY`, มี Ollama อยู่แล้ว (ใช้ตอน OCR)
- **ข้อควรระวังเชิงวิธีวิจัย:** คะแนน cosine ข้าม embedder เทียบตรงๆ ไม่ได้ → metric layer ต้อง
  จัดอันดับ/normalize ภายใน combination; silver set เป็น benchmark "ง่าย" (เรื่องซ้ำเนื้อใน
  เอกสาร) จึงควรอ่านคู่กับ gold set
- **ลำดับสร้างแนะนำ:** schema+registry+io → loaders → chunkers → local embedder + index-build
  + cache → runner + YAML ตัวอย่าง → retrievers (dense→bm25→hybrid+filter) → eval (silver +
  metrics) → Streamlit UI + CLI → semantic chunker / api embedder (ท้าย)
- **การตัดสินใจที่ยังเปิด (reversible):** ไลบรารี/การจูน BM25 ไทย, นิยาม dev-subset ที่แน่นอน
  (แนะนำปี 2568 หรือ stratified ~80 docs), รายละเอียด Parquet vs JSON
