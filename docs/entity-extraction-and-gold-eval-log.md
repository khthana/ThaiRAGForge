# บันทึกงาน Entity Extraction + Gold Query-Set Evaluation (16 ก.ค. 2569)

บันทึกนี้ครอบคลุมงานที่ยังไม่มี log เฉพาะมาก่อน: entity-tagging pipeline ทั้ง 4 ตัว
(NER GPU, Faculty, Program, Person) และ gold query-set candidate generation ที่ต่อยอด
จากมัน — เขียนไว้เพื่อเป็นวัตถุดิบสรุปเป็นผลงานวิจัยในภายหลัง งาน OCR remediation/
curriculum-bundle splitting/chunker-embedder comparison มี log แยกอยู่แล้วที่
`docs/llm-ocr-scan-log.md`, `docs/corpus-reconciliation-log.md`,
`docs/chunker-embedder-comparison-log.md` — ไม่ซ้ำในนี้

## ภาพรวม: ทำไมต้อง entity extraction

RAG baseline ของโปรเจกต์ (`plain` loader) ไม่ทำอะไรกับ metadata นอกจาก
`resolution_id`/`page`/`chunk_index` — การค้นหาที่ต้องพึ่ง entity เช่น "ใครเคยเป็น
กรรมการหลักสูตร" หรือ "หลักสูตรนี้เปลี่ยนแปลงกี่ครั้ง" ต้องมีชั้น entity resolution
มาก่อน ทั้ง 4 ตัวใช้ pattern เดียวกัน: **entity dictionary (json) + Loader class
(`.entities(text)` หรือ `match_*(text)`) + batch-tag script (`tag_*.py`) ที่เขียนผล
เป็น `_by_file.json` (ทุกไฟล์ → entity ที่เจอ) + `_report.md` (coverage summary)**
ทั้งหมดเป็น read-only ต่อคลังเอกสารจริง เขียนออกไปที่
`academic_resolutions/entity_tags/` (gitignored เหมือน corpus เอง)

## 1. NER GPU Loader (`src/rag_lab/loaders/ner_loader.py`)

`NERLoader` มี tagger 3 engine หลัง interface เดียวกัน `.entities(text)`,
เลือกด้วย `NERLoader(engine=...)`:

- **`_PyThaiNLPTagger`** — engine `thainer` ของ pythainlp ตรงๆ, CPU
- **`_WangchanBERTaTagger`** (`engine="wangchanberta-thainer"`) — เขียน inference
  ของ pythainlp's wangchanberta ใหม่เอง (word_tokenize → is_split_into_words → IOB
  tags) เพื่อรันบน CUDA ได้ เพราะ class เดิมของ pythainlp hardcode CPU ไว้ ตัดประโยค
  ที่เกิน `max_position_embeddings - 2` แบบ bisect ตาม token จริง
- **`_PhayaThaiBERTTagger`** (`engine="phayathaibert-thainer"`) — ใช้ HF
  `TokenClassificationPipeline` กับ `aggregation_strategy="first"` (ไม่ใช่ default
  `"simple"` ของ pythainlp ซึ่งตัดชื่อสถาบันของสถาบันเองเป็น entity แยก 8 คำ)

Checkpoint: `Porameht/wangchanberta-thainer-corpus-v2-2`,
`Pavarissy/phayathaibert-thainer` (cached local, CUDA บน RTX 3060 12GB)

**ผลตรวจสอบกับเอกสารจริง**: ทั้ง 3 engine โหลด checkpoint และรันได้ปกติ รวมถึงไฟล์ที่
ยาวเกิน position limit แต่มี**ข้อจำกัดที่รู้แล้วและยอมรับ**: ORGANIZATION span
แตกกระจาย (เช่น ชื่อสถาบันที่ตัดข้ามบรรทัด) และ PERSON false positive/span-splitting
เป็นครั้งคราว — เหตุผลที่ Faculty/Program/Person tagger ด้านล่างใช้ rule-based
matching แทน ไม่ใช้ NER โดยตรงสำหรับ 3 entity type นี้ `NERLoader` ยังมีประโยชน์
สำหรับ entity type อื่นที่ 3 ตัวข้างล่างไม่ครอบคลุม (LOCATION, DATE, LAW ฯลฯ)

Eval tooling: `tools/corpus_prep/evaluate_ner.py`. Commit: `fc7dde4`.

## 2. Faculty Tagger

`data/entity_dictionaries/faculties.json` (20 คณะ/สำนัก/วิทยาลัย/วิทยาเขต) +
`FacultyLoader` + `tools/corpus_prep/tag_faculties.py`

**Coverage** (2,865 ไฟล์ที่ยังใช้งาน, `academic_resolutions/entity_tags/faculties_report.md`):
matched ≥1 คณะ 2,466 ไฟล์ (86%), matched 0 คณะ 399 ไฟล์, ครบทั้ง 20/20 คณะ

Top 3 ความถี่: คณะวิศวกรรมศาสตร์ (772), คณะครุศาสตร์อุตสาหกรรมและเทคโนโลยี (325),
คณะวิทยาศาสตร์ (315)

Commit: `7b78177`.

## 3. Program (หลักสูตร) Tagger

`data/entity_dictionaries/programs.json` (253 หลักสูตร — **สร้างจาก title ใน
meeting_manifest ไม่ใช่สแกน body text** แต่ละ entry มี `canonical`
(= `หลักสูตร{degree} สาขาวิชา{field}`), `prefix_type`, `degree`, `field`) +
`ProgramLoader` (`match_programs(text)`) + `tools/corpus_prep/tag_programs.py`

**Coverage**: matched ≥1 หลักสูตร 1,649/2,865 ไฟล์ (58%), matched 250/253 หลักสูตร
(`academic_resolutions/entity_tags/programs_report.md`)

Commit: `a7211e0`.

## 4. Person Tagger + Canonicalization

`PersonLoader` (`match_people(text)`, regex จับคำนำหน้า+ชื่อ+นามสกุลที่มีสระ/พยัญชนะ
ไทย 2-18 ตัวติดกัน, รองรับ separator แบบ markdown table linebreak `<br/>` ด้วย) —
**Coverage ระดับ raw spelling**: matched ≥1 คน 2,518/2,866 ไฟล์ (88%), 3,153 distinct
raw spelling (`academic_resolutions/entity_tags/people_report.md`)

`tools/corpus_prep/canonicalize_people.py` clustering ปัญหา OCR ทำให้คนเดียวกันสะกด
ชื่อไม่ตรงกันหลายแบบข้ามเอกสาร — clustering ด้วย edit-distance/prefix-relation แบบ
union-find: **2,808 raw spelling → 2,179 canonical entries, 384 entries มี variant
ที่ merge เข้าด้วยกัน ≥1 ตัว** (`academic_resolutions/entity_tags/people_canonicalization_report.md`)

ตัวอย่าง merge ที่ตรวจสอบด้วยมือแล้วถูกต้อง (spot-check เคส given_ratio ต่ำที่ดูน่าสงสัย):
- "รวิภัทร"/"ทร"/"ดร" ลาภเจริญสุข — OCR เพี้ยนหนักจน 7 ตัวอักษรเหลือ 2 ตัว
- "ปิอิ้ง"/"บิอิ้ง"/"บีอิ๋ง"/"บีอิ้ง" แซ่อัง — ชื่อทับศัพท์จีน สะกดแปรผันสูงตามธรรมชาติ

**ข้อควรระวังเวลาใช้**: `people.json` (canonical identity) กับ `metadata['people']`
ในแต่ละ chunk (raw mention รายจุด) ใช้ต่างวัตถุประสงค์กัน — ระบุตัวตนใช้ตัวแรก ไม่ใช่
ตัวหลัง

Commits: PersonLoader `c196935`, canonicalize_people.py `c77f737`.

## 5. Gold Query-Set Candidate Generation (16 ก.ค. 2569)

### แรงจูงใจ

Framework มี eval infra อยู่แล้ว (`src/rag_lab/query_sets.py` + `metrics.py`,
resolution-level recall@k/MRR/nDCG ตาม ADR-0002) แบ่งเป็น **Silver set**
(auto-gen จาก title ของแต่ละ resolution — เร็วแต่ "ง่ายเกินไป" เพราะคำใน query
ตรงกับเอกสารพอดี) กับ **Gold set** (คำถามจริงเขียนมือ + label ความเกี่ยวข้องเอง —
แต่ก่อนหน้านี้**ยังไม่มีไฟล์ gold set อยู่จริง**ในโปรเจกต์เลย)

โจทย์เดิม: จะสร้าง gold set ได้อย่างไรให้เพียงพอทั้งปริมาณและความสมจริง โดยไม่ต้อง
label มือทั้งหมด ผู้ใช้เสนอแนวทาง bootstrap ด้วย LLM (Claude Opus/Fable) อ่านมติ
ย้อนหลัง 3-6 เดือนเพื่อช่วยสร้าง ground truth เบื้องต้น

### ตัวอย่าง query จริงที่ผู้ใช้ให้มา (ใช้เป็นข้อกำหนด design)

1. "ผศ. ธนา หรือ อ. ธนา มีประวัติเป็นกรรมการหลักสูตรใดบ้าง เป็นช่วงไหน"
2. "หลักสูตรวิศวกรรมคอมพิวเตอร์ มีการเปลี่ยนแปลงกรรมการหลักสูตรกี่ครั้งในรอบ 5 ปีนี้"
3. "มีการเพิ่มหรือเปลี่ยนแปลงในหลักสูตรวิศวกรรมคอมพิวเตอร์อะไรบ้าง ในรอบ 5 ปี"
4. "หลักสูตรวิศวกรรมคอมพิวเตอร์ มีการเปลี่ยนแปลงกี่ครั้งและแต่ละครั้งมีอะไรบ้าง"

ทั้ง 4 ตัวอย่างมีจุดร่วม: ผูกกับ **entity ที่ระบุตัวได้ (บุคคล/หลักสูตร)** และคำตอบ
ที่ถูกต้องมัก**ครอบคลุมหลาย resolution** ไม่ใช่แค่ 1 ฉบับ — ต่างจาก Silver set ที่
เป็น 1-query-1-resolution เสมอ

### พบ insight สำคัญที่เปลี่ยนแผนทั้งหมด: ไม่ต้องใช้ LLM ตัดสิน relevance

เพราะ entity dictionary ของงานข้อ 2-4 (`programs.json`/`people.json` +
`*_by_file.json`) มีอยู่แล้วและตรวจสอบได้จริง (ไม่ใช่ black-box) จึง**หา ground
truth ของ query แบบ entity-anchored ได้แบบ deterministic โดยไม่ต้องให้ LLM อ่าน
แล้วเดา relevance เอง** — ลด hallucination risk ของ LLM-generated qrels ไปเกือบ
ทั้งหมดสำหรับ query 2 ชนิดนี้ (program-history, person-history) บทบาทของ LLM
เหลือแค่การแต่งถ้อยคำ query ให้เป็นธรรมชาติ/หลากหลาย ไม่ใช่ตัดสินว่าอะไรเกี่ยวข้อง
— ส่วนแนวทาง LLM-อ่านช่วงเวลาสด ยังเก็บไว้ใช้กับ query แบบ "เชิงประเด็น" ที่ไม่ผูก
entity ชัดเจน (ยังไม่ได้สร้าง)

### สคริปต์: `tools/corpus_prep/build_gold_candidates.py` (commit `4fb54fd`)

- **Program history**: relevant_resolution_ids = ทุก resolution ที่**ชื่อเรื่อง**
  (ไม่ใช่ body) มี canonical program string (`หลักสูตร{degree} สาขาวิชา{field}`)
  ปรากฏอยู่
- **Person history**: relevant_resolution_ids = ทุก resolution ที่คนคนนั้น
  (ผ่าน canonical alias index จาก `canonicalize_people.py`) ถูกกล่าวถึงใน
  `people_by_file.json` โดยมี context filter คัดข้อความลายเซ็นออก (ดูบั๊กข้อ 2)

### บั๊กจริง 2 จุดที่พบจากการรันกับคลังเอกสารทั้งหมด (2,865 ไฟล์) — เป็น finding เชิงระเบียบวิธีที่มีค่าต่อรายงานวิจัย

**บั๊ก 1 — field substring ชนกับชื่อคณะ**: match program ด้วย substring ของ `field`
เดี่ยวๆ ทำให้ field ที่เป็นคำเดียวกับชื่อคณะ (เช่น "ครุศาสตร์อุตสาหกรรม" ซึ่งเป็นทั้ง
ชื่อ field ของหลักสูตรและส่วนหนึ่งของชื่อคณะ "คณะครุศาสตร์อุตสาหกรรมและเทคโนโลยี")
ไปแมตช์ทุกไฟล์ของคณะนั้นโดยไม่เกี่ยวกับหลักสูตรที่ถามจริง — ทดสอบกับหลักสูตร
"ครุศาสตร์อุตสาหกรรมดุษฎีบัณฑิต สาขาวิชาครุศาสตร์อุตสาหกรรม" ได้ 183 hit ปลอม
**แก้โดยแมตช์ด้วย canonical string เต็มแทน field เดี่ยว**

**บั๊ก 2 — ลายเซ็นเลขานุการปนกับเนื้อหาจริง**: body-tag ของบุคคลนับทุกครั้งที่ชื่อ
ปรากฏ รวมถึงบรรทัดลายเซ็นท้ายมติที่พิมพ์ซ้ำแทบทุกฉบับ
("(รองศาสตราจารย์ ดร.ปุณณมา ศิริพันธ์โนน) ผู้ช่วยเลขานุการ ทำหน้าที่แทนกรรมการและ
เลขานุการ") ทำให้ผู้ช่วยเลขานุการที่ลงชื่อรับรองบ่อยคนหนึ่งขึ้นเป็น 696/2,865 (~24%
ของทั้งคลัง!) ทั้งที่เป็นการรับรองเอกสารทางธุรการ ไม่ใช่ตัวชี้วัดว่าเกี่ยวข้องกับ
เนื้อหา **แก้ด้วยการเช็ค context 80 ตัวอักษรหลังชื่อ ถ้ามีคำว่า "เลขานุการ" ตามหลัง
ไม่นับเป็น hit ของไฟล์นั้น (นับเฉพาะไฟล์ที่มีการกล่าวถึงแบบไม่ใช่ลายเซ็นอย่างน้อย
1 ครั้ง)**

ทั้งสองบั๊กเป็นตัวอย่างที่ชัดเจนว่า**การ join metadata แบบ naive (substring/
body-presence) สร้าง false positive สูงในคลังเอกสารราชการที่มีข้อความ boilerplate
ซ้ำ** — ประเด็นที่ควรพูดถึงถ้าจะเขียนเป็นระเบียบวิธีในรายงานวิจัย

### ผลลัพธ์หลังแก้บั๊ก

| | ก่อนแก้ | หลังแก้ |
|---|---|---|
| Program candidates | 191 (max hit 183 — ผิดปกติ) | 148 (max hit 24 — สมเหตุสมผล) |
| Person candidates | 1,239 (max hit 696 — ผิดปกติ) | 1,226 (max hit 24 — สมเหตุสมผล) |

ตรวจสอบด้วยตัวอย่างจริงจาก query 1 ของผู้ใช้: **ผศ.ธนา หงษ์สุวรรณ** → 5 resolution
(2564, 2567, 2568×3) ทั้งหมดเกี่ยวกับหลักสูตรวิศวกรรมคอมพิวเตอร์คณะวิศวกรรมศาสตร์
จริง — ตรงกับสิ่งที่ query ตัวอย่างต้องการเป๊ะ

### สถานะ / งานที่เหลือ

Output เป็น **candidate pool** ไม่ใช่ gold set สำเร็จรูป:
`academic_resolutions/entity_tags/gold_candidates.json` (148 program + 1,226
person, gitignored) — ยังต้องคัดเลือกมือ ~30-50 รายการ (ผสม program/person,
hit_count หลากหลาย) ลงเป็น `config/eval/gold_query_set.yaml` (ตำแหน่งใหม่ ยังไม่มี
ไฟล์จริง แต่ `load_gold_query_set` อ่านตำแหน่งไหนก็ได้ที่ระบุ path เข้าไป)

ยังไม่ได้ทำ:
- คัดกรอง gold set จริงจาก candidate pool
- รัน Silver-set eval (ฟรี ไม่ต้อง label) เทียบ 4 chunker ใน
  `data/index/chunker_compare_full/` เป็น baseline ก่อน
- ออกแบบ query แบบ "เชิงประเด็น" ที่ไม่ผูก entity เดียว (ต้องพึ่ง LLM อ่านช่วงเวลา
  ตามที่ผู้ใช้เสนอไว้ตอนแรก — ยังไม่ได้ทำ)
- Vector DB (Qdrant, เลือกไว้แล้วแทน LanceDB) สำหรับ metadata filtering แบบ
  list-membership — งานนี้ค้างอยู่ก่อนจะแวะมาทำ gold-set thread ยังไม่มีโค้ด, ยังไม่
  ตัดสินใจ deployment mode (embedded local vs server) และ ingestion path
