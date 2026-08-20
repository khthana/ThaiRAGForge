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

*ปรับปรุงล่าสุด 7 ส.ค. 2569*

**เฟรมเวิร์ก**: issue ทั้งหมด (#1–#11) พัฒนาเสร็จและปิดแล้ว ครบทั้ง 4 แกน (Loader / Chunker /
Embedder / Retriever) หลาย strategy ต่อแกน, Streamlit UI ทั้งสามหน้า, batch runner/CLI และชั้น
ประเมินผล ดู [`docs/PRD-indexing-retrieval-framework.md`](docs/PRD-indexing-retrieval-framework.md)
สำหรับแผนเดิม และ GitHub issues ที่ปิดแล้วสำหรับรายละเอียดที่สร้างจริงในแต่ละส่วน

**งานวิจัย**: ตอนนี้เลยเฟส "สร้างเครื่องมือ" ไปไกลแล้ว — คอร์ปัส 2,854 ไฟล์ผ่านการกระทบยอด
และการซ่อม OCR ครบ, Gold query set 106 ข้อ (1,046 relevance judgment), เทียบ chunker × embedder
9 ตัว × BM25 × hybrid ด้วย bootstrap + Holm correction, RQ3 (preprocessing ablation) และ RQ4
(คุณภาพคำตอบปลายทาง) ปิดครบ, และประเมิน validity ของ eval เอง (power, pooling bias, circularity)
แล้ว · index ถูก rebuild รอบที่ 3 เสร็จ 5 ส.ค. 2569 และ refresh ทุกเส้นทางการค้นแล้ว
**ถ้าอยากรู้ว่าได้ผลอะไรบ้าง อ่าน [`docs/project-journey.html`](docs/project-journey.html)
(เชิงเล่าเรื่อง) หรือ [`docs/paper-results-summary.md`](docs/paper-results-summary.md)
(ตัวเลขพร้อมอ้างอิง)**

นี่คือเครื่องมือวิจัย ไม่ใช่ผลิตภัณฑ์ที่สมบูรณ์ — มีบางจุดที่ตั้งใจปล่อยไว้ไม่สมบูรณ์:

| ส่วน | สถานะ |
| --- | --- |
| `APIEmbedder` | มี interface พร้อมใช้ (raise error ชัดเจนเมื่อไม่มี key, ถูก isolate โดย batch runner ไม่ทำให้ทั้ง batch พัง) แต่ยังไม่มีการเรียก API จริง — ต้องเลือก provider แล้วใส่ key ก่อน (ดู comment ปิด issue #7) เป็นการตัดสินใจเชิงขอบเขต: ทั้งโปรเจกต์รันโมเดล local ล้วน ไม่ส่งข้อมูลออกนอกเครื่อง |
| `HybridEmbedder` | **ไม่ได้สร้าง** — มีแค่ `HybridRetriever` (ผสมอันดับผลลัพธ์จาก Dense + BM25 ด้วย RRF) อย่าสับสนสองอย่างนี้ PRD เลื่อน HybridEmbedder ไว้ตั้งแต่แรก |
| `SemanticChunker` | `breakpoint_threshold` ค่าเริ่มต้นเป็นจุดเริ่มต้น ไม่ใช่ค่าที่ tune แล้ว — ทดสอบกับ bge-m3 จริงพบว่าอาจไม่เกิด breakpoint เลยในบางข้อความ (ดู comment ปิด issue #11) ภายหลังพบ fragmentation defect และแก้แล้ว (`e8f4b80`) |
| `HybridRetriever` แบบ `weighted` | **ยังไม่ได้ ship แต่ไม่ใช่คำถามที่ค้างแล้ว** — ตัวเลข hybrid ที่ตีพิมพ์ทุกตัวคือ RRF 50:50 และ default ก็ยังเป็น 50:50 อยู่ ส่วนที่วัดไปแล้ว: `hybrid_alpha_sweep.py` (2026-08-08) กวาด `alpha` บนสาขา `rrf` และ `hybrid_weighted_fetch_depth.py` (2026-08-12) วัดคู่ `weighted` × `fetch_depth` จนยกการ์ดออกได้ (ที่ F=200 `weighted` เสีย −0.0609 macro recall@10 ≈ 18 เท่าของ `rrf`) |
| ประสิทธิภาพตอนค้น | แก้ไปครึ่งหนึ่งแล้ว: `BM25Okapi` ถูก memoise ไว้บน `Index.lexical_scorer` (2026-08-09, ตัดออก ~1.0 วินาที/query) และการดึงผลทั้งคอร์ปัสถูกจำกัดด้วย `fetch_depth` ที่**ชั้น query-time เท่านั้น** (Streamlit default F=200; default ของคลาสยังเป็น k=n เพื่อให้ตัวเลขที่ตีพิมพ์ reproduce ได้) → routed hybrid p50 **475.6 ms** ที่ยังค้าง: ไม่มี batching ตอน query และ embedder ถูกโหลดใหม่ทุกครั้งที่เรียก `route_query` ซึ่งเป็นชั้นที่อิ่มตัวก่อนเสมอเมื่อมีผู้ใช้พร้อมกัน |
| `lexical_containment` | **ship แล้ว แต่ต้องอ่านคู่กับ circularity เสมอ** — arm L′ (2026-08-20): รัน `hybrid` แล้วแบ่งกลุ่ม top-50 แบบ stable ว่า entity ที่ `detect_entities` เจอในคำถาม ปรากฏใน chunk จริงไหม ไม่ใช้ GPU ไม่ใช้โมเดล · ชนะ router ที่ ship อยู่อย่างมีนัยสำคัญทุกเมตริก (recall@10 **+0.0489**, MRR +0.0437, nDCG@10 +0.0714) แต่ qrels ของ `person`/`program`/`faculty` **ถูกสร้างมาด้วยการจับสตริงเหมือนกัน** — คะแนนจึงวัดกฎของ arm นี้เองอยู่ส่วนหนึ่ง **ห้ามอ้างว่า lexical ชนะ learned ranking** · opt-in ด้วยชื่อ **ไม่มีอะไร default มาที่มัน** |
| ชื่อสาขาเปล่าๆ | คนค้นหาพิมพ์ `วิศวกรรมคอมพิวเตอร์` ไม่ใช่ชื่อ canonical 60 ตัวอักษร · `match_programs_by_field` (2026-08-20) คลี่เป็น**ทุกหลักสูตรที่เปิดสาขานั้น** ไม่เดาระดับปริญญา และ `programme_groups` ยุบพจนานุกรม 253 → 250 entry (คู่เปลี่ยนชื่อปริญญาของโคเซ็น 3 คู่) · **Gold set วัดเรื่องนี้ไม่ได้เชิงโครงสร้าง** — คำถาม `program` ทั้ง 30 ข้อเขียน canonical เต็ม จึงเป็น deployment fix และ**ห้ามอ้างตัวเลข retrieval ให้มันทั้งสองทาง** · แหล่งที่มาของตัวเลขคือ `tests/test_program_field_matching.py` ไม่ใช่รายงานใด |
| Qdrant | คำเตือน 20k-point ปิดไปแล้วด้วยการวัด — embedded mode เป็น **brute force ไม่ใช่ ANN** จึงไม่เคยเป็นคำถามที่มันตอบได้ ตอนนี้เป็น server จริง (`qdrant/qdrant` 1.18.0) และ `qdrant_hybrid` ถูกต่อเข้ากับ `route_query` ที่ ship จริงแล้ว (2026-08-13, served 0.6827 vs published 0.6835) ที่ยังค้าง: ingest ยังเป็นสคริปต์ต่อ combo, มีแต่ filter `resolution_id_in`, และยัง**ไม่เคยวัดข้าม network hop** — และ **ไม่มีอะไร default มาที่มัน** |

## สถาปัตยกรรม

แบ่งไปป์ไลน์เป็น 2 เฟส (ดู [ADR-0001](docs/adr/0001-scope-retrieval-and-index-retrieve-split.md)):

1. **Index-build** — ออฟไลน์ ราคาแพง มี cache: `Loader → Chunker → Embedder` ผลิต
   **Index artifact** (chunks + embeddings + BM25 index + metadata)
2. **Retrieval** — query-time ราคาถูก: `Retriever` จัดอันดับ chunk จาก Index artifact ที่
   สร้างไว้แล้ว ไม่ re-embed

4 แกนสลับได้ผ่าน registry แบบ decorator (เพิ่ม strategy ใหม่ = สร้างไฟล์ + register โดยไม่
ต้องแก้ runner — Open/Closed):

- **Loader** — `Plain` (baseline) / `Metadata` (สกัดปี ครั้งที่ เรื่อง source_url จากไฟล์
  `_LINK.txt`) / `NER` (สกัด entities ด้วย pythainlp) / `EntityTags` (รวมผลของ tagger
  4 ชนิด — person / program / course / faculty — ลง metadata) / `Normalized` (normalize
  เลขไทย + `pythainlp.util.normalize()` สำหรับ RQ3 ablation)
- **Chunker** — `FixedSize` / `Recursive` (แยกตามลำดับชั้น separator, langchain-text-splitters)
  / `Sentence` (ตัดประโยคไทยด้วย pythainlp) / `Semantic` (หา breakpoint จาก embedding
  similarity โดย fix embedding model ตายตัว ไม่ผูกกับแกน Embedder เพื่อเลี่ยง confound)
- **Embedder** — `Hashing` (baseline ไม่ต้องใช้โมเดล) และโมเดลจริง 9 ตัวที่เอามาเทียบกัน
  ผ่าน 4 คลาส: `Local` (sentence-transformers ทั่วไป — bge-m3, ConGen-PhayaThaiBERT,
  simcse-thai, model2vec) / `E5` (multilingual-e5-large / -small พร้อม prefix
  `query:`/`passage:`) / `Qwen3` (Qwen3-Embedding-0.6B / -4B) / `JinaV5` ·
  `API` เป็น interface พร้อมใช้ รอ provider + key
- **Retriever** — `Dense` (cosine) / `BM25` (rank_bm25 + pythainlp tokens) / `Hybrid`
  (RRF, และมี `weighted` ที่ยังไม่ ship) / `EntityLookup` (ค้นจาก entity tag ตรงๆ
  ใช้กับโหมด `entity_lookup`/`entity_boost`) / `Qdrant` · `QdrantHybrid`
  (ยิง dense + sparse ไปที่ Qdrant server แล้ว fuse ด้วย `fuse_rrf` ตัวเดียวกับ
  `HybridRetriever` — collection ถูก resolve ตอน query จาก
  `Index.provenance["index_dir"]` จึงใช้ spec เดียวครอบทั้ง 4 route) / `MetadataFilter`
  (กรองตามปี/คณะ/ครั้งที่ก่อนจัดอันดับ ใช้ร่วมกับ retriever ไหนก็ได้ — ยกเว้น
  retriever ที่ไม่อ่านแถวของ `Index` เอง ซึ่งจะ **ปฏิเสธ** ด้วย `ValueError`
  แทนที่จะเงียบ) และมี
  `CrossEncoderReranker` เป็น stage เสริมตอน query (วัดแล้ว: **ทำให้ hybrid MRR แย่ลง**
  อย่างมีนัยสำคัญ ดู `docs/reranker-hybrid-interaction-research.md`)

ทับบนแกนทั้งสี่มี **hard routing** (`router.py`): `classify_query` แยกคำถามเป็น 5 route
(person / program / course / faculty / unmatched) แล้ว `query_service.route_query` ค้นจาก
index ของ route นั้นตัวเดียว โดยเลือก target จาก `route_targets(retriever_type)` — map
คนละชุดสำหรับ `dense` กับ `hybrid` เพราะ index ที่ดีที่สุดของแต่ละ route ขึ้นกับ retriever
**อ่านผลของมันว่าเป็นเรื่อง coverage ไม่ใช่ accuracy**: router 5 route ชนะ router 3 route
เดิมอย่างมีนัยสำคัญ (+0.0958 dense recall@10) แต่ไม่มี arm ไหนที่ deploy ได้ชนะการใช้ combo
เดี่ยวที่ดีที่สุดกับทุกคำถามอย่างมีนัยสำคัญ (ดู `data/results/routing_eval.md`)

ความถูกต้องของการค้นตัดสินที่ **ระดับ Resolution (มติ)** ไม่ใช่ระดับ chunk (ดู
[ADR-0002](docs/adr/0002-resolution-level-relevance.md)) เพราะขอบเขต chunk เปลี่ยนไปตาม
chunker ที่เลือก แต่ `resolution_id` คงที่เสมอข้าม chunker ทุกตัว

## โครงสร้างโปรเจกต์

```
├── CONTEXT.md                # อภิธานศัพท์โดเมน (เริ่มอ่านที่นี่)
├── CLAUDE.md                 # คำสั่ง dev + convention ของการทำงานในโปรเจกต์นี้
├── docs/
│   ├── project-journey.html  # ★ สรุปเชิงเล่าเรื่องทั้งโปรเจกต์ (เริ่มอ่านที่นี่)
│   ├── paper-results-summary.md              # ★ ตัวเลขพร้อมอ้างอิง (อัปเดตเมื่อ headline เปลี่ยน)
│   ├── chunker-embedder-comparison-log.md    # บันทึกกระบวนการเทียบ — append-only
│   ├── eval-validity-threats.md              # power / pooling bias / circularity
│   ├── rq4-design.md                         # การออกแบบ + build log ของ RQ4
│   ├── pipeline-invariant-audit.md           # รายงาน 25 invariant ล่าสุด
│   ├── doc-claims-audit.md                   # รายงานการตรวจ prose เทียบรายงาน (ตระกูล D)
│   ├── llm-ocr-scan-log.md                   # การตรวจ + ซ่อม OCR ทั้งสองกลไก
│   ├── entity-extraction-and-gold-eval-log.md  # tagger 4 ชนิด + การสร้าง Gold set
│   ├── PRD-indexing-retrieval-framework.md   # แผนเดิม (ไทย) — เอกสารประวัติศาสตร์
│   ├── adr/                  # architecture decisions (ทำไมถึงตัดสินใจแบบนี้)
│   ├── agents/                # convention ต่อ skill (issue tracker, triage labels, domain docs)
│   ├── grill-session-2026-07-01.md           # บันทึกสัมภาษณ์ออกแบบ — เอกสารประวัติศาสตร์
│   └── req0-original-spec.md # สเปกเดิม — ถูกแทนที่ด้วย PRD, เอกสารประวัติศาสตร์
├── tools/corpus_prep/        # pipeline scrape → OCR → clean ที่สร้าง corpus + ซ่อม OCR
├── tools/eval/               # สคริปต์ประเมินผลทั้งหมด (retrieval, significance test,
│                             #   power, cost/latency, audit invariant, RQ3/RQ4)
├── src/rag_lab/              # ตัวเฟรมเวิร์ก — ไม่ผูก Streamlit, import/test ได้อิสระ
│   ├── loaders/               # Plain / Metadata / NER / EntityTags / Normalized + tagger 4 ชนิด
│   ├── chunkers/               # FixedSize / Recursive / Sentence / Semantic
│   ├── embedders/               # Hashing / Local / E5 / Qwen3 / JinaV5 / API
│   ├── retrievers/               # Dense / BM25 / Hybrid (RRF) / EntityLookup / Qdrant / QdrantHybrid + MetadataFilter
│   ├── colbert/                 # late interaction (encoder + store + scoring) — วัดแล้วและ**ไม่รับเข้า**
│   ├── router.py                # classify_query + ROUTE_COMBO_BY_RETRIEVER (hard routing ที่ ship จริง)
│   ├── pipeline.py, runner.py, query_service.py   # index-build + retrieval + batch
│   ├── metrics.py, query_sets.py                  # evaluation (silver/gold, recall@k/MRR/nDCG)
│   └── config.py, factory.py, registries.py       # YAML config + Open/Closed strategy wiring
├── app/                       # Streamlit UI: Mode A (Build/Run), Mode B (Query & Compare),
│                              #   Chunk Inspector
├── config/experiments/       # ตัวอย่างไฟล์ config การทดลอง (YAML)
├── config/eval/              # Gold query set (ใช้ `gold_query_set_73det.yaml` 106 ข้อ)
├── data/results/             # ผลลัพธ์การค้นที่ persist ไว้ + รายงาน significance test
├── tests/                    # unit test (deterministic) + smoke test จริงที่ gate ด้วย RAG_LAB_SMOKE
└── academic_resolutions/     # ตัว corpus (gitignored; 2,854 ไฟล์ .md ใต้โฟลเดอร์ปี)
```

## คลังข้อมูล

`academic_resolutions/` เก็บมติที่ผ่าน OCR แล้ว **2,854 ไฟล์** จัดเรียงเป็น
`ปี(พ.ศ.)/ครั้งที่ N/<เรื่อง>.md` (การประชุมวาระพิเศษใช้ `ครั้งที่ Ns`)
แต่ละไฟล์ `.md` ถูกนับเป็นหนึ่ง **Resolution** ในปัจจุบัน
(หน่วยนี้ยังเป็นคำถามเปิด — ไฟล์เดียวในทางทฤษฎีอาจมีหลายมติ ยังไม่มีส่วนใดของ implementation
ที่ยืนยันเรื่องนี้ ดู `CONTEXT.md`) ไฟล์ `_LINK.txt` ข้างเคียงเก็บ URL ของ PDF ต้นฉบับบน
Google Drive (ใช้เพื่อ provenance ไม่ใช่เนื้อหา) ส่วน `meeting_manifest.json` ต่อการประชุม
เป็น **แหล่งความจริงของ metadata** (ชื่อเรื่อง/URL) — ห้ามเข้ารหัส metadata ลงในชื่อไฟล์
(ADR-0003) และบัญชีรวมที่กระทบยอดแล้วอยู่ที่ `academic_resolutions/master_list.csv`
ดูวิธีที่ corpus ถูกสร้างได้ที่ `tools/corpus_prep/README.md`

คอร์ปัสผ่านการซ่อม OCR มาแล้วสองกลไก (รวม ~2,750 หน้าที่เขียนกลับ) และผ่านการตรวจ
invariant 25 ข้อด้วย `tools/eval/audit_pipeline_invariants.py` — รันสคริปต์นี้ก่อนเชื่อ
ตัวเลข eval ใดๆ หลังแตะคอร์ปัสหรือ manifest

ชั้นถัดขึ้นมาคือ **prose**: `tools/eval/audit_doc_claims.py` ตรวจว่าทุกตัวเลข 4 ตำแหน่ง
ใน `CLAUDE.md`/`docs/paper-results-summary.md` หาเจอในรายงานสักฉบับ, รายงานไม่เก่ากว่า
สคริปต์ที่สร้างมัน และรายงานทุกฉบับประกาศว่าใครสร้าง — รันหลังแก้สองเอกสารนั้น
และหลัง eval refresh ทุกครั้ง (รายงาน: `docs/doc-claims-audit.md`)

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

# เปิด Streamlit UI (Mode B: Query & Compare เป็นหน้าหลัก,
# Mode A: Build/Run และ Chunk Inspector อยู่ใน sidebar)
# --server.fileWatcherType none กันไม่ให้ auto-reload watcher ไปเดินสำรวจ
# submodule ของ transformers (เช่น zoedepth) แล้วขึ้น ModuleNotFoundError:
# torchvision -- แค่ warning ไม่เป็นอันตราย แต่ log รกไม่จำเป็น
.venv/Scripts/streamlit.exe run app/streamlit_app.py --server.fileWatcherType none
```

รายละเอียดคำสั่งทั้งหมด (การ smoke test โมเดลจริงด้วย `RAG_LAB_SMOKE=1`, คำสั่ง CLI
ระดับล่าง ฯลฯ) ดูที่ **`CLAUDE.md`** — เก็บไว้ที่นั่นเป็นแหล่งเดียว (single source of truth)
เพื่อไม่ให้ไฟล์นี้กับ `CLAUDE.md` เขียนคำสั่งซ้ำแล้วข้อมูลเพี้ยนไปคนละทาง

## เอกสารประกอบ

- **[`docs/project-journey.html`](docs/project-journey.html)** — **เริ่มอ่านที่นี่ถ้าอยากเข้าใจ
  ภาพรวมทั้งโปรเจกต์**: สรุปเชิงเล่าเรื่อง 13 บทว่าทำอะไรไปบ้างตามลำดับเวลา แต่ละขั้นได้ผล
  อย่างไร ข้อสรุปไหนถูกถอนไปแล้วและเพราะอะไร บทเรียนเชิงกระบวนการ และภาคผนวกอธิบาย
  metric/วิธีการทางสถิติ — render เป็น PDF ได้ (คำสั่งอยู่ใน `docs/project-journey.md`)
- **[`CONTEXT.md`](CONTEXT.md)** — อภิธานศัพท์ร่วม (Resolution, Chunk, Index artifact,
  Retriever, Dense/BM25/Hybrid, Silver/Gold query set, …)
- **[`docs/PRD-indexing-retrieval-framework.md`](docs/PRD-indexing-retrieval-framework.md)**
  — แผนเดิม: ปัญหา, ทางแก้, user stories, การตัดสินใจด้าน implementation/testing เป็นเอกสาร
  ประวัติศาสตร์ — สะท้อนเจตนา ณ ตอนนั้น ไม่จำเป็นต้องตรงกับทุกรายละเอียดที่ implement จริง
- **[`docs/paper-results-summary.md`](docs/paper-results-summary.md)** — ตัวเลขที่พร้อมนำไป
  อ้างอิงในเปเปอร์ (อัปเดตทุกครั้งที่ headline เปลี่ยน) คู่กับ
  [`docs/chunker-embedder-comparison-log.md`](docs/chunker-embedder-comparison-log.md)
  ซึ่งเป็นบันทึกกระบวนการแบบ append-only (ตัวเลขในนั้นอาจเป็นของรอบเก่าโดยตั้งใจ)
- **[`docs/eval-validity-threats.md`](docs/eval-validity-threats.md)** — อ่านก่อนจะปกป้อง
  ตัวเลขใดๆ ในโปรเจกต์นี้: statistical power, pooling bias, circularity, single annotator
- **[ADR-0001](docs/adr/0001-scope-retrieval-and-index-retrieve-split.md)** — ขยายขอบเขตถึง
  retrieval; แยกเฟส Index-build กับ Retrieval
- **[ADR-0002](docs/adr/0002-resolution-level-relevance.md)** — ตัดสินความเกี่ยวข้องที่ระดับ
  Resolution ไม่ใช่ระดับ chunk
- **[ADR-0003](docs/adr/0003-meeting-manifest-metadata.md)** — `meeting_manifest.json` เป็น
  แหล่งความจริงของ metadata ไม่ใช่ชื่อไฟล์
- **[ADR-0004](docs/adr/0004-curriculum-bundle-splitting.md)** — แยกไฟล์รวมหลักสูตรเป็นมติย่อย
  โดยเก็บต้นฉบับไว้กู้คืนได้

## สภาพแวดล้อม

- Python 3.13 จัดการด้วย `uv`
- เฉพาะ corpus-prep: `uv sync`
- เต็มเฟรมเวิร์ก: `uv sync --extra lab`
- การ embed แบบ local รันบน GPU (พัฒนาบน RTX 3060 12GB) ส่วน OCR ใช้โมเดล Ollama แบบ local
  (ดู `tools/corpus_prep/`)
