# RAG Lab — โครงงานทดลอง Indexing และ Retrieval สำหรับ RAG

![Python](https://img.shields.io/badge/python-3.13-blue)
![uv](https://img.shields.io/badge/managed%20by-uv-6340ac)
![Status](https://img.shields.io/badge/status-research%20tooling-yellow)

เฟรมเวิร์กแบบโมดูลาร์สำหรับทดลองขั้นตอน **indexing** และ **retrieval** ของระบบ RAG
บนเอกสารมติสภาวิชาการภาษาไทย (มติสภาวิชาการ KMITL) สลับองค์ประกอบแต่ละส่วนได้อิสระ —
**Loader × Chunker × Embedder × Retriever** — รันหลาย combination พร้อมกัน แล้วเทียบว่า
ชุดไหนดึง chunk ที่เกี่ยวข้องกับคำค้นได้ดีที่สุด

## สารบัญ

- [สถานะปัจจุบัน](#สถานะปัจจุบัน)
- [สถาปัตยกรรม](#สถาปัตยกรรม)
- [โครงสร้างโปรเจกต์](#โครงสร้างโปรเจกต์)
- [คลังข้อมูล](#คลังข้อมูล)
- [เริ่มต้นใช้งาน](#เริ่มต้นใช้งาน)
- [เอกสารประกอบ](#เอกสารประกอบ)
- [สภาพแวดล้อม](#สภาพแวดล้อม)

## สถานะปัจจุบัน

Issue ทั้งหมด (#1–#11) พัฒนาเสร็จและปิดแล้ว ครบทั้ง 4 แกน (Loader / Chunker / Embedder /
Retriever) หลาย strategy ต่อแกน, Streamlit UI ทั้งสองโหมด, batch runner/CLI, และชั้น
ประเมินผล (evaluation) ดู [`docs/PRD-indexing-retrieval-framework.md`](docs/PRD-indexing-retrieval-framework.md)
สำหรับแผนเดิม และ GitHub issues ที่ปิดแล้วสำหรับรายละเอียดที่สร้างจริงในแต่ละส่วน

นี่คือเครื่องมือวิจัย ไม่ใช่ผลิตภัณฑ์ที่สมบูรณ์ — มีบางจุดที่ตั้งใจปล่อยไว้ไม่สมบูรณ์:

| ส่วน | สถานะ |
| --- | --- |
| `APIEmbedder` | มี interface พร้อมใช้ (raise error ชัดเจนเมื่อไม่มี key, ถูก isolate โดย batch runner ไม่ทำให้ทั้ง batch พัง) แต่ยังไม่มีการเรียก API จริง — ต้องเลือก provider แล้วใส่ key ก่อน (ดู comment ปิด issue #7) |
| `HybridEmbedder` | **ไม่ได้สร้าง** — มีแค่ `HybridRetriever` (ผสมอันดับผลลัพธ์จาก Dense + BM25 ด้วย RRF) อย่าสับสนสองอย่างนี้ PRD เลื่อน HybridEmbedder ไว้ตั้งแต่แรก |
| `SemanticChunker` | `breakpoint_threshold` ค่าเริ่มต้นเป็นจุดเริ่มต้น ไม่ใช่ค่าที่ tune แล้ว — ทดสอบกับ bge-m3 จริงพบว่าอาจไม่เกิด breakpoint เลยในบางข้อความ (ดู comment ปิด issue #11) |
| Evaluation layer | โค้ด + unit test พร้อมใช้ (silver/gold query set, recall@k/MRR/nDCG) แต่ยังไม่เคยรันเป็น batch จริงกับ corpus ทั้งหมดเพื่อได้ตัวเลขเทียบ combination — เป็นขั้นต่อไปด้านงานวิจัย ไม่ใช่งานวิศวกรรมที่ค้างอยู่ |

## สถาปัตยกรรม

แบ่งไปป์ไลน์เป็น 2 เฟส (ดู [ADR-0001](docs/adr/0001-scope-retrieval-and-index-retrieve-split.md)):

1. **Index-build** — ออฟไลน์ ราคาแพง มี cache: `Loader → Chunker → Embedder` ผลิต
   **Index artifact** (chunks + embeddings + BM25 index + metadata)
2. **Retrieval** — query-time ราคาถูก: `Retriever` จัดอันดับ chunk จาก Index artifact ที่
   สร้างไว้แล้ว ไม่ re-embed

4 แกนสลับได้ผ่าน registry แบบ decorator (เพิ่ม strategy ใหม่ = สร้างไฟล์ + register โดยไม่
ต้องแก้ runner — Open/Closed):

- **Loader** — `Plain` (baseline) / `Metadata` (สกัดปี ครั้งที่ เรื่อง source_url จากไฟล์
  `_LINK.txt`) / `NER` (สกัด entities ด้วย pythainlp)
- **Chunker** — `FixedSize` / `Recursive` (แยกตามลำดับชั้น separator, langchain-text-splitters)
  / `Sentence` (ตัดประโยคไทยด้วย pythainlp) / `Semantic` (หา breakpoint จาก embedding
  similarity โดย fix embedding model ตายตัว ไม่ผูกกับแกน Embedder เพื่อเลี่ยง confound)
- **Embedder** — `Hashing` (baseline ไม่ต้องใช้โมเดล) / `Local` (bge-m3 บน GPU) / `E5`
  (multilingual-e5-large พร้อม prefix `query:`/`passage:`) / `API` (interface พร้อมใช้ รอ
  provider + key)
- **Retriever** — `Dense` (cosine) / `BM25` (rank_bm25 + pythainlp tokens) / `Hybrid`
  (RRF) + `MetadataFilter` (กรองตามปี/คณะ/ครั้งที่ก่อนจัดอันดับ ใช้ร่วมกับ retriever ไหนก็ได้)

ความถูกต้องของการค้นตัดสินที่ **ระดับ Resolution (มติ)** ไม่ใช่ระดับ chunk (ดู
[ADR-0002](docs/adr/0002-resolution-level-relevance.md)) เพราะขอบเขต chunk เปลี่ยนไปตาม
chunker ที่เลือก แต่ `resolution_id` คงที่เสมอข้าม chunker ทุกตัว

## โครงสร้างโปรเจกต์

```
├── CONTEXT.md                # อภิธานศัพท์โดเมน (เริ่มอ่านที่นี่)
├── CLAUDE.md                 # คำสั่ง dev + convention ของการทำงานในโปรเจกต์นี้
├── docs/
│   ├── PRD-indexing-retrieval-framework.md   # แผนเดิม (ไทย) — เอกสารประวัติศาสตร์
│   ├── adr/                  # architecture decisions (ทำไมถึงตัดสินใจแบบนี้)
│   ├── agents/                # convention ต่อ skill (issue tracker, triage labels, domain docs)
│   ├── grill-session-2026-07-01.md           # บันทึกสัมภาษณ์ออกแบบ — เอกสารประวัติศาสตร์
│   └── req0-original-spec.md # สเปกเดิม — ถูกแทนที่ด้วย PRD, เอกสารประวัติศาสตร์
├── tools/corpus_prep/        # pipeline scrape → OCR → clean ที่สร้าง corpus
├── src/rag_lab/              # ตัวเฟรมเวิร์ก — ไม่ผูก Streamlit, import/test ได้อิสระ
│   ├── loaders/               # Plain / Metadata / NER
│   ├── chunkers/               # FixedSize / Recursive / Sentence / Semantic
│   ├── embedders/               # Hashing / Local (bge-m3) / E5 / API
│   ├── retrievers/               # Dense / BM25 / Hybrid (RRF) + MetadataFilter
│   ├── pipeline.py, runner.py, query_service.py   # index-build + retrieval + batch
│   ├── metrics.py, query_sets.py                  # evaluation (silver/gold, recall@k/MRR/nDCG)
│   └── config.py, factory.py, registries.py       # YAML config + Open/Closed strategy wiring
├── app/                       # Streamlit UI: Mode A (Build/Run), Mode B (Query & Compare)
├── config/experiments/       # ตัวอย่างไฟล์ config การทดลอง (YAML)
├── tests/                    # unit test (deterministic) + smoke test จริงที่ gate ด้วย RAG_LAB_SMOKE
└── academic_resolutions/     # ตัว corpus (gitignored; ~1,215 ไฟล์ .md)
```

## คลังข้อมูล

`academic_resolutions/` เก็บมติที่ผ่าน OCR แล้ว จัดเรียงเป็น
`ปี(พ.ศ.)/ครั้งที่ N/<เรื่อง>.md` แต่ละไฟล์ `.md` ถูกนับเป็นหนึ่ง **Resolution** ในปัจจุบัน
(หน่วยนี้ยังเป็นคำถามเปิด — ไฟล์เดียวในทางทฤษฎีอาจมีหลายมติ ยังไม่มีส่วนใดของ implementation
ที่ยืนยันเรื่องนี้ ดู `CONTEXT.md`) ไฟล์ `_LINK.txt` ข้างเคียงเก็บ URL ของ PDF ต้นฉบับบน
Google Drive (ใช้เพื่อ provenance ไม่ใช่เนื้อหา) ดูวิธีที่ corpus ถูกสร้างได้ที่
`tools/corpus_prep/README.md`

## เริ่มต้นใช้งาน

```bash
# ติดตั้ง (เฉพาะ corpus-prep)
uv sync

# ติดตั้งเต็ม (framework + pytest)
uv sync --extra lab

# รันเทสต์ทั้งหมด
.venv/Scripts/python.exe -m pytest

# สร้าง index ทีละ batch จาก YAML config
PYTHONPATH=src python -m rag_lab.cli run --config config/experiments/dev_smoke.yaml

# เปิด Streamlit UI (Mode A: Build/Run, Mode B: Query & Compare)
streamlit run app/streamlit_app.py
```

รายละเอียดคำสั่งทั้งหมด (การ smoke test โมเดลจริงด้วย `RAG_LAB_SMOKE=1`, คำสั่ง CLI
ระดับล่าง ฯลฯ) ดูที่ **`CLAUDE.md`** — เก็บไว้ที่นั่นเป็นแหล่งเดียว (single source of truth)
เพื่อไม่ให้ไฟล์นี้กับ `CLAUDE.md` เขียนคำสั่งซ้ำแล้วข้อมูลเพี้ยนไปคนละทาง

## เอกสารประกอบ

- **[`CONTEXT.md`](CONTEXT.md)** — อภิธานศัพท์ร่วม (Resolution, Chunk, Index artifact,
  Retriever, Dense/BM25/Hybrid, Silver/Gold query set, …)
- **[`docs/PRD-indexing-retrieval-framework.md`](docs/PRD-indexing-retrieval-framework.md)**
  — แผนเดิม: ปัญหา, ทางแก้, user stories, การตัดสินใจด้าน implementation/testing เป็นเอกสาร
  ประวัติศาสตร์ — สะท้อนเจตนา ณ ตอนนั้น ไม่จำเป็นต้องตรงกับทุกรายละเอียดที่ implement จริง
- **[ADR-0001](docs/adr/0001-scope-retrieval-and-index-retrieve-split.md)** — ขยายขอบเขตถึง
  retrieval; แยกเฟส Index-build กับ Retrieval
- **[ADR-0002](docs/adr/0002-resolution-level-relevance.md)** — ตัดสินความเกี่ยวข้องที่ระดับ
  Resolution ไม่ใช่ระดับ chunk

## สภาพแวดล้อม

- Python 3.13 จัดการด้วย `uv`
- เฉพาะ corpus-prep: `uv sync`
- เต็มเฟรมเวิร์ก: `uv sync --extra lab`
- การ embed แบบ local รันบน GPU (พัฒนาบน RTX 3060 12GB) ส่วน OCR ใช้โมเดล Ollama แบบ local
  (ดู `tools/corpus_prep/`)
