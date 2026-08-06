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

**บั๊ก 3 — ไฟล์ report ที่ไม่ใช่คลังจริงปนเข้ามาใน `rglob("*.md")` (พบ 17 ก.ค. 2569,
ระหว่างขั้นตอน human-review ของ gold candidate)**: `tag_people.py`, `tag_programs.py`,
`tag_faculties.py`, `canonicalize_people.py` ทั้ง 4 ตัวเดินไฟล์ด้วย
`corpus_root.rglob("*.md")` แบบไม่กรอง — แต่ `academic_resolutions/` (gitignored)
ไม่ได้มีแค่มติจริง มันมีไฟล์ report/working ของ pipeline อื่นปนอยู่ด้วย (เช่น
`entity_tags/*.md`, `llm_ocr_scan/full_review_*.md` รวม 20 ไฟล์ ณ เวลาที่พบบั๊ก)
ไฟล์เหล่านี้บางไฟล์ควอตข้อความ OCR ที่มีชื่อคนจริงปนอยู่ (จาก LLM OCR-scan) จึงถูกนับ
เป็น "การกล่าวถึง" ปลอมของคนนั้น กระทบ **131/1,226 person candidates (18 รายการเป็น
ของปลอมล้วนๆ ไม่มี hit จริงเลย)**; program candidates ไม่กระทบ (0/148) เพราะ
resolution_id fallback ของไฟล์เหล่านี้ (full path) ไม่บังเอิญมี canonical program
string ปนอยู่

**แก้โดยเพิ่มฟังก์ชันกลาง `iter_corpus_files()` ใน `src/rag_lab/loaders/common.py`**
ที่กรองด้วยโครงสร้างจริงของคลัง (`<ปี 4 หลัก>/<ครั้งที่ N>/*.md`, เงื่อนไขเดียวกับที่
`make_resolution_id` ใช้ตัดสินว่าจะสร้าง id จริงหรือ fallback เป็น path) แทนการกรอง
ด้วยรายชื่อไฟล์ที่รู้จัก (blocklist) — กันไม่ให้ไฟล์ report ใหม่ที่จะเกิดขึ้นในอนาคต
หลุดเข้ามาอีก แล้วแก้ทั้ง 4 สคริปต์ให้เรียกใช้ฟังก์ชันนี้แทน `rglob` ตรงๆ รัน pytest
ผ่านหมด แล้ว rerun ทั้ง 4 สคริปต์ + `build_gold_candidates.py` ใหม่ทั้งหมด — ยืนยันด้วย
สคริปต์ตรวจนับว่าปนเปื้อนเหลือ 0/0 ทั้งสอง pool

ทั้งสามบั๊กเป็นตัวอย่างที่ชัดเจนว่า**การ join metadata แบบ naive (substring/
body-presence/ไม่กรอง directory scope) สร้าง false positive สูงในคลังเอกสารราชการที่มี
ข้อความ boilerplate ซ้ำ และในโปรเจกต์ที่ working directory กับ corpus ปนกัน** —
ประเด็นที่ควรพูดถึงถ้าจะเขียนเป็นระเบียบวิธีในรายงานวิจัย

### ผลลัพธ์หลังแก้บั๊ก

| | ก่อนแก้บั๊ก 1-2 | หลังแก้บั๊ก 1-2 (เดิม) | หลังแก้บั๊ก 3 (rerun 17 ก.ค.) |
|---|---|---|---|
| Total live files | 2,865-2,866 (ปนไฟล์ report) | เท่าเดิม | **2,853 (ตรงกับ master_list.csv)** |
| Program candidates | 191 (max hit 183) | 148 (max hit 24) | 148 (ไม่เปลี่ยน — ไม่เคยกระทบ) |
| Person candidates | 1,239 (max hit 696) | 1,226 (max hit 24) | **1,171 (ตัด 18 รายการปลอมล้วนๆ + รายการที่ hit_count ตกต่ำกว่า min_hits ออก)** |
| Person canonical entries | - | 2,179 | **2,158** |

ตรวจสอบด้วยตัวอย่างจริงจาก query 1 ของผู้ใช้: **ผศ.ธนา หงษ์สุวรรณ** → 5 resolution
(2564, 2567, 2568×3) ทั้งหมดเกี่ยวกับหลักสูตรวิศวกรรมคอมพิวเตอร์คณะวิศวกรรมศาสตร์
จริง — ตรงกับสิ่งที่ query ตัวอย่างต้องการเป๊ะ

### Human review รอบแรก (17 ก.ค. 2569): สุ่ม 42 รายการ ตรวจมือ ไม่พบ false positive

หลังแก้บั๊ก 3 สุ่มตัวอย่างแบบ stratified ตาม hit_count (ต่ำ ≤3 / กลาง 4-8 / สูง 9+)
ฝั่งละ 7 × 3 bucket × 2 ประเภท entity (program/person) = 42 รายการ (seed=42)
ตรวจ context ทีละรายการ (อ่านข้อความรอบชื่อในไฟล์จริง) — **ไม่พบ false positive แม้แต่
รายการเดียว** ยืนยันว่าวิธี matching (title เต็มสำหรับ program, exact given+surname
+ กรอง secretarial window สำหรับ person) แม่นจริง

พบ**ประเด็นเชิงระเบียบวิธี 2 เรื่อง** ที่ไม่ใช่บั๊ก แต่เป็น judgment call:

1. **Bundle-split inflation**: มติเดียวที่ถูก ADR-0004 แตกเป็น N ไฟล์ (curriculum
   bundle) นับเป็น N hit แยกกันสำหรับคนคนเดียวที่โผล่ใน preamble ที่ใช้ร่วมกันทุกไฟล์
   — ถูกต้องในระดับ retrievable-unit แต่ทำให้ hit_count สูงเกินจริงเมื่อใช้แทน "จำนวน
   เหตุการณ์/มติที่แตกต่างกันจริง" (สำคัญกับ query แบบ "กี่ครั้ง")
2. อาจารย์พิเศษสอนเกินร้อยละ 50 เป็น hit ที่ถูกต้องแต่เป็นความเกี่ยวข้องคนละแบบกับ
   กรรมการหลักสูตร — ผู้ใช้ยืนยันแล้วว่าเก็บไว้ได้ (17 ก.ค. 2569)

### แก้ประเด็นที่ 1: เพิ่ม `event_count` แยกจาก `hit_count` (17 ก.ค. 2569)

ผู้ใช้ขอให้นับ "ต่อ resolution" แทนต่อไฟล์จริง ("อยากให้เป็นต่อ resolution มากกว่า
ครับ") ก่อนแก้ ตรวจสอบทั้งคลัง (707 ไฟล์ `__N.md` ทั้งหมด) พบว่า:
- title ใน `meeting_manifest.json` ของทุกไฟล์ split (707/707) มีรูปแบบ
  `<title เดิม> — <title หลักสูตรย่อย>` เสมอ (em dash คั่นแน่นอน ไม่มีข้อยกเว้น)
- เลข `__N` เรียงต่อเนื่อง 1..N เสมอ ไม่มีเคสที่ใช้เลขไม่ต่อเนื่อง (ตัดความเป็นไปได้ว่า
  `__N` ถูกใช้เพื่อแก้ปัญหา title ชนกันแทนการ split จริง)

จึงปลอดภัยที่จะ group `resolution_id` กลับเป็น "มติต้นฉบับ" ด้วยการตัด string ที่ " — "
ตัวแรก (`_event_key()` ใน `build_gold_candidates.py`) — ตรวจยืนยันซ้ำด้วย 2 เคสจริง
(ผศ.ดร.อาทิตย์ เพชรศศิธร: hit_count 24→event_count 18; ผศ.ดร.กนกนุช ทรงสุวรรณกิจ:
hit_count 18→event_count 8) โดยเปิดไฟล์ `__1`/`__2`/... ของ bundle เดียวกันเทียบ
เนื้อหาจริง — snippet รอบชื่อเหมือนกันตัวต่อตัวข้าม 11 ไฟล์ ยืนยันว่าเป็น preamble
เดียวกันที่ถูกก็อปปี้ซ้ำจริง ไม่ใช่ 11 เหตุการณ์แยกกัน

**ผลคือ**: `gold_candidates.json` แต่ละรายการมีทั้ง `hit_count` (จำนวนไฟล์จริง ใช้เกรด
recall การค้นคืน) และ `event_count` (จำนวนมติต้นฉบับหลัง collapse bundle-siblings
ใช้สำหรับ bucket stratification และคำตอบคาดหวังของ query แบบ "กี่ครั้ง") —
`relevant_resolution_ids` ยังคงลิสต์ไฟล์จริงครบทุกไฟล์เหมือนเดิม (ไม่ตัดออก เพราะทุก
ไฟล์ยังเป็น retrieval target จริงที่มีชื่อคนนั้นปรากฏจริง) เกณฑ์ `min_hits` ก็เปลี่ยนไป
กรองด้วย `event_count` แทน `hit_count` ด้วย (มติเดียวที่ถูกแตกเป็นไฟล์เดียวไม่ควรนับผ่าน
threshold จากจำนวนไฟล์). ผลลัพธ์หลัง rerun: program candidates 148→147 (1 รายการมี
event_count < min_hits), person candidates 1,171→1,139

### Query shape ที่ 3: Faculty adjunct-instructor aggregate (17 ก.ค. 2569)

ผู้ใช้เสนอโจทย์ใหม่ (ยอมรับเองว่า "อาจหายากหน่อย"): **"หลักสูตรใดของ
คณะวิศวกรรมศาสตร์ใช้อาจารย์พิเศษมากที่สุด กี่คน ใครบ้าง"** — ถามว่าจะ scope
ไปทางไหน มี 2 ทาง: (ก) สร้าง analytics report คำนวณคำตอบจริงจาก tag ที่มีอยู่
(แม่นยำ 100% แต่ไม่ผ่าน retrieval) หรือ (ข) ทำเป็น gold query จริงเพื่อทดสอบว่า
RAG pipeline ตอบคำถามรวม/นับข้ามหลายเอกสารได้มั้ย — ผู้ใช้เลือก **(ข)**

ตรวจ `src/rag_lab/query_sets.py` ก่อนลงมือ (สำคัญ เกือบทำเกินจำเป็น): eval
เกรดแค่ `relevant_resolution_ids` (recall@k/MRR/nDCG@k, ADR-0002) — `QuerySetEntry`
ไม่มี field เก็บ "คำตอบ" เลย แปลว่า**ไม่ต้องรู้ว่าใครอยู่หลักสูตรไหนในไฟล์เดียวกัน**
(ไม่ต้อง parse ตารางแยกตามหลักสูตรในไฟล์ multi-program) — สิ่งที่ต้องมีจริงๆ
คือ**ไฟล์ filing ทั้งหมดของคณะนั้น** (ต้องดึงครบทุกไฟล์เพื่อเทียบข้ามหลักสูตรได้)
ซึ่งมีอยู่แล้วจาก `faculties_by_file.json`

เพิ่ม `faculty_adjunct_candidates()` ใน `build_gold_candidates.py`: หา resolution
ที่ title ตรงกับ filing "อาจารย์พิเศษสอนเกินร้อยละ 50" (`_is_adjunct_filing_title` —
ต้องมีทั้ง "อาจารย์...พิเศษ" และ "ร้อยละ" กันชนกับเอกสารนโยบายทั่วไปที่พูดถึง
"อาจารย์พิเศษ" เฉยๆ เช่น ข้อบังคับตำแหน่ง "ผู้ช่วยศาสตราจารย์พิเศษ" หรือ
"ไม่ใช่อาจารย์พิเศษ" — ตรวจสอบด้วยมือแล้วว่า pattern นี้ตรงกับ 196 ไฟล์เดิม
เป๊ะ ไม่ over/under-match) group ตาม faculty ที่ tag ไว้แล้ว ได้ **14 faculty
candidates** เช่น คณะวิศวกรรมศาสตร์ (43 filing), คณะบริหารธุรกิจ (30),
คณะเทคโนโลยีสารสนเทศ (22) — ตัวอย่างของผู้ใช้ (คณะวิศวกรรมศาสตร์) เป็น
candidate ที่มี hit_count มากที่สุดพอดี

**ข้อควรระวังที่บันทึกไว้ใน docstring**: query shape นี้ต่างจาก 2 อันแรกตรงที่
relevant set ใหญ่และแบนราบโดยเจตนา (ต้องดึงครบทุก filing ของคณะ ไม่ใช่แค่ของ
หลักสูตรที่ถูกถามถึง) เป็นโจทย์ยากจริงสำหรับ retrieval (recall@k ข้ามเอกสาร
หลายสิบไฟล์) — ของแบบนี้คือสิ่งที่ผู้ใช้อยากทดสอบจริงๆ ไม่ใช่บั๊ก

### Gold set เสร็จสมบูรณ์ + รัน eval แล้ว (17 ก.ค. 2569)

คัดจาก candidate pool (147 program + 1,139 person + 14 faculty-adjunct) พร้อม
rephrase คำถามให้เป็นธรรมชาติ (หมุนเทมเพลตต่อประเภท ไม่แตะ `relevant_resolution_ids`)
เขียนลง `config/eval/gold_query_set.yaml` แล้วรัน `tools/eval/run_gold_chunker_eval.py`
เทียบ 4 chunker บน e5-large

รอบแรก 37 รายการ (12 program + 12 person + 13 faculty-adjunct) ผลไม่ชี้ทางเดียว
เหมือน Silver แต่พอเช็ค paired significance (fixed_size vs semantic) ที่ n=12/category
ได้ `|t|~1.6-1.7` (ต้องการ >2.20 ที่ df=11) — **ทิศทางถูกแต่ยังไม่แน่นพอทางสถิติ**
ผู้ใช้เลยขอให้ขยาย **program 12→30, person 12→30** (faculty-adjunct คงที่ 13 เพราะ
ใกล้เพดาน pool ที่มี hit_count>=3 อยู่แล้ว) รวมเป็น **73 รายการ** — ฟรี ไม่ต้อง
label เพิ่มเพราะ candidate pool เหลือเยอะ รันใหม่ (144.6s สำหรับ 73×4 combo) ได้
`|t|=2.91` (program) และ `|t|=-3.50` (person) ที่ df=29 — **ผ่านนัยสำคัญทางสถิติจริง
ทั้งคู่ (p<0.01)** ยืนยันว่าไม่ใช่ noise จากตัวอย่างน้อย

ดูผลละเอียดที่ `docs/chunker-embedder-comparison-log.md` § "Gold query set"
สรุปสั้นๆ: `fixed_size` ชนะ program-history จริง, `semantic` ชนะ person-history จริง
(กลับทิศ Silver โดยสิ้นเชิงสำหรับ person), faculty-adjunct-aggregate ติดเพดาน
recall@10 ทางคณิตศาสตร์ (relevant set ใหญ่ถึง 30 ไฟล์) แต่ MRR เกือบเต็มทุก chunker

### สถานะ / งานที่เหลือ

ยังไม่ได้ทำ:
- ออกแบบ query แบบ "เชิงประเด็น" ที่ไม่ผูก entity เดียว (ต้องพึ่ง LLM อ่านช่วงเวลา
  ตามที่ผู้ใช้เสนอไว้ตอนแรก — ตั้งใจเลื่อนไว้หลัง gold set นี้เสร็จ ยืนยันกับผู้ใช้แล้ว
  17 ก.ค. 2569 ว่ายังอยากทำต่อ)
- faculty_adjunct_aggregate ยังมี n=13 เท่านั้น (เกือบเต็ม pool ที่มี hit_count>=3,
  14 รายการทั้งหมด) — ยังไม่ได้เช็ค significance แยกสำหรับ query shape นี้ เพราะ
  ผลระหว่าง chunker ใกล้เคียงกันมาก (0.42-0.50) ไม่ชัดว่ามีผู้ชนะจริง

## 6. ตรวจ 3 รูปแบบตารางที่พบบ่อยเพิ่มเติม + แก้บั๊ก match_people จริง (25 ก.ค. 2569)

หลังปิด Thread B ของ preprocessing review (course-comparison table stripping —
ดู `docs/chunker-embedder-comparison-log.md` § course-comparison table
compaction) ผู้ใช้ระบุตารางอีก 3 รูปแบบที่พบบ่อยในคลังเอกสาร ขอให้ทดสอบว่า
`match_people`/`match_programs`/`match_courses` ที่มีอยู่แล้ว extract ได้ถูกต้องไหม
ก่อนตัดสินใจว่าต้องแก้อะไรเพิ่ม — ตรวจกับไฟล์จริงทั้ง 3 แบบ:

**แบบ 1) แต่งตั้งอาจารย์บัณฑิตประจำ/พิเศษ** — `match_people` ถูก 4/4 แต่
`match_programs` ได้ 0/4 เพราะช่องหลักสูตรในตารางเขียนแบบย่อ (เช่น
`วศ.ด. (วิศวกรรมเคมี)`) ไม่ใช่ชื่อเต็มแบบที่ `programs.json` เก็บ **ผู้ใช้ยืนยันว่า
ไม่ใช่บั๊ก** — ช่องนี้คือวุฒิที่อาจารย์จบมาเอง ไม่ใช่หลักสูตรของสถาบัน ไม่ควร match
`programs.json` อยู่แล้ว ปิดประเด็นนี้โดยไม่แก้อะไร

**แบบ 2) อาจารย์พิเศษสอนเกินร้อยละ 50** — พบบั๊กจริงที่ใหญ่กว่าที่คาด: ตัวอย่างแรก
ที่สุ่มมา (คณะบริหารธุรกิจ, 2564 ครั้งที่ 1) มีแต่ชื่อต่างชาติ (title ภาษาอังกฤษ)
ทำให้เข้าใจผิดในตอนแรกว่าปัญหาคือ title ภาษาอังกฤษไม่รองรับ — แต่พอสุ่มไฟล์อื่น
(คณะศิลปศาสตร์ 2564 ครั้งที่ 7) พบว่า**ชื่อไทยล้วนก็ไม่ถูก match เช่นกัน**
(`อ.อภิชา เชื้อประศิลป์` ได้ 0 ผล) — เจาะจึงพบ root cause จริง: `_TITLE` ใน
`person_loader.py` ไม่มี pattern สำหรับคำนำหน้า "อ." (อาจารย์ ตำแหน่งพื้นฐานที่สุด)
เลย มีแต่ ผศ./รศ./ศ./ดร. — อาจารย์พิเศษ/พาร์ทไทม์ส่วนใหญ่ใช้ตำแหน่งนี้พอดี

**แบบ 3) หัวเป็นปรับปรุงหลักสูตรไม่กระทบโครงสร้าง แต่เนื้อจริงคือเปลี่ยนอาจารย์
ผู้รับผิดชอบหลักสูตร** — ชื่อไทย (มี title ผศ./รศ./อ.) match ได้ครบแม้ตารางจะแน่น
4 คอลัมน์ (เดิม/ใหม่/ผลงานวิชาการ) ยืนยันว่าปัญหาไม่ใช่โครงสร้างตารางที่แน่น แต่คือ
title ที่ไม่รองรับ (แบบ 2) เดียวกัน — ชื่อต่างชาติ (`MR.NAVIDREZ AHADI` เป็นต้น)
ก็หลุดเหมือนแบบ 2 แต่ผู้ใช้บอกว่าคนต่างชาติมีน้อย ไม่ต้องแก้

**แก้จริง (commit `a4e250e`)**: เพิ่ม `อ\.` เป็น title alternative ใน `_TITLE`
**เฉพาะรูปย่อ ไม่เพิ่มคำเต็ม "อาจารย์"** — ตรวจก่อนแก้ด้วยการรัน regex ทดลองกับ
คลังเอกสารทั้งหมด (2,853 ไฟล์) พบว่าคำเต็ม "อาจารย์" หลวมเกินไป จับวลีทั่วไปเช่น
"อาจารย์พิเศษ", "อาจารย์ผู้สอนในรายวิชา", "อาจารย์ประจำหลักสูตร" เป็น "ชื่อคน" ผิด
ส่วนรูปย่อ "อ." เดี่ยวๆ สแกนได้ 1,415 จุดทั่วคลัง ตรวจด้วยตาแล้วสะอาดเกือบทั้งหมด
(ไม่ชนกับ "อำเภอ" ตามที่กังวลไว้) เจอ false positive จริงแค่ 2 คำ ("ผู้สอน", "ผู้รับ"
ตามหลัง "อ." เป็นคำนามทั่วไปไม่ใช่ชื่อ) เพิ่มเข้า `_NOT_A_NAME`

**ผลวัดได้**: ไฟล์ตัวอย่างศิลปศาสตร์ได้ครบ 3/3 คน (จากเดิม 1/3) รวมทั้งคลัง
(deduped ต่อไฟล์): 9,712 → 10,697 คน (+10%); เฉพาะ 166 ไฟล์กลุ่ม "อาจารย์พิเศษ
สอนเกินร้อยละ 50": 1,640 → 2,240 (+37%) เพิ่ม unit test 5 เคสใน
`tests/loaders/test_person_loader.py` (`_TITLE`/`_NOT_A_NAME` ครอบคลุมทั้งกรณี match
และกรณี reject)

**ยังไม่แก้ (ตัดสินใจโดยผู้ใช้)**: ชื่อต่างชาติ/title ภาษาอังกฤษ (Mr./Assoc.Prof.Dr.)
ยังไม่ match เลย — เป็นคนละ root cause จาก "อ." (ต้องเพิ่ม pattern ภาษาอังกฤษแยก)
ผู้ใช้เห็นว่าคนต่างชาติมีสัดส่วนน้อย ไม่คุ้มที่จะทำตอนนี้ ยัง deferred เหมือนบั๊ก
cross-table-cell name split เดิม (ดู § 4 ด้านบน)

**Rebuild ที่ตามมา**: `data/index/entity_tags_full` (index เดียวที่ใช้ `entity_tags`
loader, เป็นฐานให้ entity_boost/entity_lookup ใน UI) build ไว้ก่อนแก้บั๊กนี้
(14:33) ต้อง rebuild ใหม่เพื่อให้ metadata['people'] มีชื่อ "อ." ที่กู้กลับมาได้ —
สั่ง rebuild แล้ว (`PYTHONPATH=src python -m rag_lab.cli run --config
config/experiments/entity_tags_full.yaml`, semantic+bge-m3, คาด~60 นาที)

## 6b. แก้บั๊ก cross-cell name split (26 ก.ค. 2569, commit `e1523b3`)

บั๊กที่ log ไว้ตอน § 5 (ชื่อแยกคนละ `<td>` — title+ชื่อต้นอยู่ช่องหนึ่ง นามสกุลอยู่อีก
ช่อง คั่นด้วย markup `</td><td>` ล้วนๆ ไม่มี `<br/>` หรือช่องว่างคั่น) ตอนนั้นผู้ใช้
สั่งให้บันทึกไว้ก่อน ยังไม่แก้จนกว่าจะ scope ร่วมกัน — วันนี้ scope ผ่าน `/grill-me`:

ก่อนเขียนโค้ด รัน pattern คู่ cell ที่ต้องสงสัยกับคลังเอกสารทั้งหมดก่อน (ไม่ใช่แค่ไฟล์
ตัวอย่างเดียวที่เจอตอนแรก) — พบว่าช่องโหว่จริงเล็กกว่าที่ประเมินไว้ตอน § 5 มาก
("≥4 ไฟล์ น่าจะมากกว่านั้น"): เจอ **8 จุดจริง ใน 4 ไฟล์** ทั้งหมดเป็นตาราง
"รับรองรายงานการประชุม" (แก้ยศ/ตำแหน่ง) เท่านั้น

ทดลอง fix แบบ naive ("เชื่อม `</td><td>` ที่ติดกันเสมอ") กับผลสแกนทั้งคลังเดียวกัน
ก่อนนำมาใช้จริง — เจอ false positive จริง 2 จุดในเอกสารคนละประเภท (รายงาน
"อาจารย์พิเศษสอนเกินร้อยละ 50" ที่ OCR เพี้ยน): ช่องข้อความ prose ที่เพี้ยนถูกอ่านเป็น
นามสกุลปลอม 1 จุด และเศษ OCR 5 ตัวอักษรลอยๆ ("มเชี่") อีก 1 จุด แก้โดยกำหนดเงื่อนไข
เพิ่ม: ช่องถัดไปต้องมี**เฉพาะ**ตัวสมัครนามสกุล (ไม่มีอย่างอื่นปน) และยาว **≥6 ตัวอักษร**
(นามสกุลจริงที่สั้นที่สุดที่เจอคือ "มิตะถา" ยาวพอดี 6 — ผู้ใช้ยืนยันให้คงเกณฑ์ที่ 6 หลัง
ลองเกณฑ์ 5 แล้วเจอเศษ "มเชี่" หลุดกลับเข้ามา)

ไม่ต้องแก้ฝั่ง query — `match_people` ใช้ร่วมกับ `router.classify_query` อยู่แล้ว และ
pattern ใหม่ทำงานเฉพาะกับ HTML markup `</td><td>` ตัวจริงเท่านั้น ไม่มีทางเกิดใน query
ข้อความล้วน ตรวจ diff ทั้งคลังก่อน/หลัง (`data/results/person_cross_cell_fix_review.md`,
gitignored): 4 ไฟล์เปลี่ยน, เพิ่ม 7 tag, ไม่มีตัวไหนหาย — ผู้ใช้ตรวจ diff เองแล้วยืนยันถูก
ทั้ง 7 เคส เพิ่ม unit test 5 เคส สูทเต็มเขียว (435 ผ่าน) ส่วนตารางเปลี่ยนอาจารย์
ผู้รับผิดชอบหลักสูตร (สมมติฐานที่สองของผู้ใช้) สแกนทั้งคลังแล้ว**ไม่พบ pattern นี้เลย**
— ไม่มีอะไรต้องแก้จริง

**Rebuild**: ตอนแก้เสร็จผู้ใช้เลือก**เลื่อน** rebuild `entity_tags_full` ออกไปก่อน (7 tag
ใน 4 ไฟล์ ไม่คุ้มเวลา GPU ~80 นาที ตอนนั้น) — แต่ index ตัวนี้ก็ถูก rebuild อยู่ดีในอีก 2
วันถัดมาด้วยเหตุผลคนละเรื่อง (ดู § 7 ด้านล่าง: OCR-remediation rebuild 28 ก.ค.) ซึ่ง
เป็นการ build คลังใหม่ทั้งหมดด้วยโค้ด loader ปัจจุบัน จึงพา fix นี้ (และ `a4e250e`) เข้า
`entity_lookup`/`entity_boost` ไปโดยอัตโนมัติ ไม่ต้องสั่ง rebuild แยกเพื่อ fix นี้อีก

## 7. Refresh entity_boost/entity_lookup หลัง OCR-remediation rebuild (29 ก.ค. 2569)

ระหว่างกวาดเช็ก `data/results/*` ทั้งหมดเทียบกับ OCR-remediation rebuild
(chunker_compare_full + entity_tags_full ที่ rebuild ไป 28 ก.ค.) พบว่า
`gold_entity_boost_73det_report.md`/`gold_entity_lookup_73det_report.md`
(25 ก.ค.) เก่ากว่า `entity_tags_full` index ตัวจริง
(`chunks.parquet`/`embeddings.npy` มี mtime 28 ก.ค. 11:52) — ค้างมาแล้วรอบ
rebuild หนึ่งรอบ เป็นบั๊กเดียวกับที่เจอกับ BM25/hybrid/cost_latency/multi_k
แค่คนละไฟล์ รันทั้งสองสคริปต์ใหม่:

**entity_boost (Gold 106, k=10)**:

| entity_type | เดิม (25 ก.ค.) | สด (29 ก.ค.) | delta |
|---|---|---|---|
| course | 0.7332 | 0.7164 | −0.017 |
| faculty_adjunct_aggregate | 0.5485 | 0.5089 | −0.040 |
| person | 0.8552 | 0.8182 | −0.037 |
| program | 0.6196 | 0.5781 | −0.042 |

**entity_lookup (exhaustive, k=1000)**: overall recall 0.9291 → **0.9422**
(+0.013; course 0.9804, faculty_adjunct_aggregate 1.0000, person 0.9255,
program 0.8918).

**อ่านผล**: entity_boost ตกทุกประเภท (ทิศทางเดียวกับตัวเลข dense/hybrid อื่นๆ
ในเอกสารนี้หลัง 2 rebuild ติดกัน) แต่ entity_lookup กลับขึ้นเล็กน้อย — สอง
mechanism คนละแบบกัน (exhaustive metadata match ไม่ใช่ ranked retrieval)
จึงไม่แปลกที่ทิศทางต่างกัน ไม่ใช่ความขัดแย้งที่ต้องสงสัย ตัวเลขทั้งหมดยังไม่เคย
significance-test เทียบก่อน/หลัง (ไม่มี test ไหนรองรับ metric แบบนี้อยู่แล้ว)
รายงานเต็ม: `data/results/gold_entity_boost_73det_report.md`,
`data/results/gold_entity_lookup_73det_report.md` อัปเดต memory
`[[project_entity_lookup_next_plan]]` แล้ว

## 8. Rebuild entity_tags_full อีกรอบ เพื่อแก้ resolution_id staleness (5 ส.ค. 2569)

`data/index/entity_tags_full` (build 28 ก.ค.) เก่ากว่าการแก้ `resolution_id`-collision
(30 ก.ค., ดู `resolution_id` bullet ใน CLAUDE.md) — 6 ไฟล์ที่โดน relabel ยังถือ id เดิม
(ก่อนแก้) อยู่ในดัชนีนี้ แม้ `chunker_compare_full` (rebuild #3, เสร็จ 5 ส.ค. 07:56) จะได้
id ที่ถูกต้องแล้วโดยอัตโนมัติเพราะ build ใหม่หมดด้วย `make_resolution_id` เวอร์ชันแก้แล้ว
ก็ตาม — `entity_tags_full` เป็นดัชนีแยกต่างหาก ไม่ได้อยู่ใน rebuild #3 chain จึงต้อง
สั่ง rebuild เองอีกรอบ (ไม่เกี่ยวกับ entity-loader fix ใดๆ ทั้งสอง fix นั้น "ขึ้น" ไปตั้งแต่
28 ก.ค. แล้ว — รอบนี้แก้เฉพาะ id ให้ตรงกับคลังปัจจุบัน)

สั่ง build ใหม่ด้วย `config/experiments/entity_tags_full.yaml` (semantic+bge-m3) ใช้เวลา
1 ชม. 22 นาที เสร็จ 5 ส.ค. 18:32 — ยืนยันว่า `chunks.parquet`/`embeddings.npy`/
`lexical.json`/`manifest.json` ทั้ง 4 ไฟล์มี mtime ตรงกันแล้ว (ก่อนหน้านี้
`chunks.parquet` ค้างที่ 30 ก.ค. ขณะที่อีก 3 ไฟล์ยังเป็น 28 ก.ค. — สัญญาณ staleness
แบบเดียวกับที่ `audit_pipeline_invariants.py` เช็ก)

**Re-run แล้ว (6 ส.ค. 2569 16:03-16:04 — แก้จากที่เคยเขียนผิดเป็น "5 ส.ค.": build
entity_tags_full เสร็จ 5 ส.ค. 18:32 แต่การ re-run entity_boost/entity_lookup จริง ๆ
เกิดวันถัดมา ตาม mtime ของทั้งไฟล์ผลลัพธ์และไฟล์สำรองด้านล่าง)**, เทียบกับรายงาน 29 ก.ค.
เดิม (สำรองไว้ที่ `data/results/_pre_2026_08_05_refresh/` — ชื่อโฟลเดอร์อ้างอิงวันที่ของ
รายงานที่เอามาเทียบ ไม่ใช่วันที่สำรอง):

- **entity_lookup**: แทบไม่ขยับเลย (recall@1000 รวม 0.9422 → 0.9422 เท่าเดิม, ต่างกัน
  แค่ระดับ rounding ในบางเซลล์) — สมเหตุสมผล เพราะ entity_lookup เป็น exhaustive
  metadata match ไม่ใช่ ranked retrieval การ relabel `resolution_id` แค่ 6 ไฟล์ (จาก
  2,853 ไฟล์) ไม่ทำให้ recall/precision เปลี่ยน
- **entity_boost**: recall@10 นิ่งเกือบทั้งหมด (course 0.7164→0.7107, program
  0.5781→0.5765 — ต่างกันระดับ noise) **ยกเว้น program mrr ที่ขยับจริง 0.6238 →
  0.6544 (+0.0306)** — ทิศทางเข้าใจได้: การ relabel แก้ id ที่เคย collide กันของ 2
  ไฟล์ (สมมติว่าเป็นกรณี "รายการวาระซ้ำชื่อ" ที่ § resolution_id เอกสาร CLAUDE.md พูดถึง)
  เปลี่ยนลำดับการจัดอันดับผลลัพธ์ที่เคยผูกกันเป็น id เดียว ให้แยกจากกันได้ชัดขึ้นสำหรับ
  query ประเภท program — ยังไม่ได้ significance-test ตัวเลขนี้ (ไม่มี test รองรับ
  before/after metric เดี่ยวแบบนี้อยู่แล้ว เหมือนที่ § 7 บันทึกไว้) รายงานเต็มอัปเดตแล้วที่
  `data/results/gold_entity_boost_73det_report.md`,
  `data/results/gold_entity_lookup_73det_report.md`
