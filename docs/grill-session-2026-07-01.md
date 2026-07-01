# Grilling Session Log — RAG Indexing Experimentation Framework

**วันที่:** 2026-07-01
**ผู้ใช้:** thana.ho@kmitl.ac.th
**Input:** `requirements/req0.md` + คำอธิบายเพิ่มเติมในแชต (ต้องการ indexing pipeline
แบบ modular, ข้อมูลจาก `academic_resolutions/`, มี UI เลือกโมดูล/รัน/เก็บผล, ผลลัพธ์คือ
"ใส่คำค้น → หา chunk ที่ relevant ที่สุด")

เอกสารนี้บันทึกคำถาม–ตัวเลือก–คำตอบที่เลือก ระหว่าง session grill-with-docs ไว้อ่านย้อนหลัง
สรุปการออกแบบที่ตกผลึกอยู่ท้ายไฟล์ ส่วน glossary อยู่ที่ `CONTEXT.md` และเหตุผลของการตัดสินใจ
ใหญ่ๆ อยู่ที่ `docs/adr/`.

---

## สิ่งที่พบจากการสำรวจ codebase (ก่อนถาม)

- `academic_resolutions/` = มติสภาวิชาการ KMITL ที่ผ่าน OCR แล้ว จัดโครงสร้าง
  `ปีพ.ศ.(2564–2569)/ครั้งที่ N/<ชื่อมติ>.md`
- **Corpus จริง = 1,215 ไฟล์ `.md`** (ผลจาก `ocr2.py`, Ollama + typhoon-ocr) มีโครงสร้าง
  OCR `# Document:`, `## Page N`, เลขไทย (๑/๒๕๖๙)
- ไฟล์ `.txt` (1,017 ไฟล์ `_LINK.txt`) = **ไม่ใช่เนื้อหา** แต่เป็น URL Google Drive ของ
  PDF ต้นฉบับ (provenance) — median 70 bytes
- ขนาด `.md`: median ~7.4KB, p90 ~42KB, max ~214KB (UTF-8 ไทย ~3 bytes/char →
  median ~2,500 ตัวอักษร, max ~70k) → **chunking มีผลจริงกับเอกสารใหญ่**
- `indexing_pipline.py` = ตัวอย่าง LangChain (cricket) ทิ้งได้ / framework จริงยังไม่มี
- Hardware: **RTX 3060 12GB**, Python 3.13, ไม่มี `OPENAI_API_KEY`, มี Ollama (ใช้ OCR อยู่)

---

## คำถามและคำตอบที่เลือก

### Q1 — Scope (ขอบเขต)
`req0.md` §10 บอก retrieval/UI เป็น out-of-scope แต่แชตบอกว่าต้อง query→retrieve
**เลือก: เพิ่ม query→retrieve (ตามแชต)** — retrieval อยู่ในขอบเขต, ตัดสิน "combination
ไหนดี" จากคุณภาพ retrieval ไม่ใช่แค่สถิติ indexing

### Q2 — Evaluation (วิธีวัดผล)
เทียบ 2 combination ด้วย query เดียวกัน จะตัดสินอย่างไร
**เลือก: เริ่มด้วย eyeball แล้วค่อยทำ labeled set** (staged)
→ ข้อบังคับที่ตามมา: ทุก run ต้อง persist query + combination + ranked chunk list +
scores แบบถาวร เพื่อให้ score ย้อนหลังด้วย labeled set ได้โดยไม่ต้อง re-index

### Q3 — UI
`req0.md` บอก CLI-only แต่แชตต้องการ UI
**เลือก: Streamlit แอพเดียว 2 โหมด** (โหมด A = Build/Run experiment, โหมด B = Query &
Compare) → core ต้องเป็น library ที่ไม่ผูก Streamlit, scriptable + testable

### Q4 — Relevance unit (หน่วยตัดสินความถูก)
**เลือก: แสดง chunk แต่ตัดสินที่ระดับ Resolution** — retrieval คืน top-k chunks แต่
"hit" นับที่ระดับมติ (top-k มี chunk จากมติที่ถูกไหม); ground truth = "query Q → มติ R";
chunk ทุกตัวต้องมี `resolution_id` → ดู ADR-0002

### Q5 — Loader role
metadata จะไม่มีผลถ้าไม่ถูกใช้ตอน embed/rank
**เลือก: Hybrid/filter** — metadata (year/คณะ/เรื่อง/entities) มีผลต่อ retrieval ผ่าน
lexical + filter → Loader ป้อนสัญญาณให้ Retriever

### Q6 — Retriever (แกนที่ 4 ใหม่)
**เลือก: Dense + BM25 + Hybrid + filter (เต็มชุด)**
- Dense (cosine), BM25 (pythainlp tokenize), Hybrid (RRF/weighted)
- metadata pre-filter (year/คณะ/session) เป็น option แยกใช้ได้กับทุก retriever

### Q7 — Embedder
**เลือก: Local-first (BGE-M3 + multilingual-e5-large), API เป็น plug-in**
บน GPU ด้วย sentence-transformers; APIEmbedder มี interface พร้อมแต่เปิดใช้เมื่อใส่ key

### Q8 — Chunker
**เลือก: Fixed/Recursive/Sentence ก่อน, Semantic แยก + fix model**
- Fixed (char, มี param char/token), Recursive (LangChain + Thai separators),
  Sentence (pythainlp) = 3 ตัวที่ clean (ไม่ผูก embedder)
- Semantic เพิ่มทีหลังโดย fix embedding model ตายตัว (ไม่ให้ confound Embedder axis)
- ทุก chunker respect `## Page` เป็น hard boundary

### Q9 — Cost model
**เลือก: แยก phase + dev-subset เป็น default**
- Index-build cache ตาม (loader, chunker, embedder); Retriever ทำงาน query-time
- default รันบน dev subset (เช่น 1 ปี / ~50–100 docs), full corpus เป็น opt-in flag
- cache key = (document set, chunker params, embedder) → ไม่ embed ซ้ำ → ดู ADR-0001

### Q10 — Config source
**เลือก: YAML เป็น source of truth, UI เป็น editor/launcher**
UI สร้าง/แก้/สั่งรัน YAML; ทุก run เก็บ snapshot ของ resolved config + git hash +
timestamp ลง manifest

### Q11 — Query set
**เลือก: ทั้ง silver (จากเรื่อง) + hand-written**
- Silver: เรื่อง/title ของแต่ละมติ = query, relevant = มตินั้นเอง → recall@k อัตโนมัติทันที
- Gold: เขียน query จริง 20–30 ข้อ (paraphrase/ถามเชิงเนื้อหา) ไว้วัดของจริง

---

## สรุปการออกแบบที่ตกผลึก (Shared Understanding)

**สถาปัตยกรรม 2 phase**
```
Index-build (offline, แพง, cache)          Retrieval (query-time, ถูก)
Loader → Chunker → Embedder  ──►  Index artifact  ──►  Retriever → top-k chunks
(3)       (3–4)     (2)          (chunks+emb+BM25+meta)  (Dense/BM25/Hybrid+filter)
= ~24 index builds                              = ฟรีทุก retriever ต่อ index
```

**4 แกน swappable**
| แกน | ตัวเลือกเริ่มต้น | หมายเหตุ |
|---|---|---|
| Loader | Plain / Metadata / NER | enrich metadata ป้อน retriever (hybrid/filter) |
| Chunker | Fixed / Recursive / Sentence (+ Semantic ทีหลัง) | respect `## Page`; Thai via pythainlp |
| Embedder | BGE-M3 / e5-large (local) (+ API plug-in) | บน RTX 3060 |
| Retriever | Dense / BM25 / Hybrid (+ metadata filter) | query-time, ไม่ re-embed |

**หลักการ**
- Core = Python package ไม่ผูก Streamlit (scriptable + unit-testable); UI + Typer CLI
  เป็น entry point บางๆ
- YAML = source of truth; ทุก run snapshot config + git hash เพื่อ reproducibility
- Relevance ตัดสินที่ระดับ Resolution (chunk มี `resolution_id` เสมอ)
- Persist ranked results ทุก run เพื่อ score ย้อนหลังด้วย labeled set
- ระวัง: score ข้าม embedder เทียบตรงๆ ไม่ได้ → normalize/rank ภายใน combination

**เอกสารที่เกี่ยวข้อง**
- `CONTEXT.md` — glossary (คำศัพท์กลาง)
- `docs/adr/0001-...` — เหตุผลการขยาย scope + แยก phase
- `docs/adr/0002-...` — เหตุผล relevance ระดับ Resolution

**ยังไม่ได้ตัดสิน (ต่อได้ทีหลัง, reversible):** รูปแบบ storage ละเอียด (Parquet/JSON),
กลไก registry/plugin, ไลบรารี BM25 ไทย, นิยาม dev-subset ที่แน่นอน
