# บันทึกผล Chunker × Embedder Comparison (7 ก.ค. 2569)

Config: `config/experiments/chunker_compare_full.yaml` (+ resume:
`chunker_compare_full_resume.yaml`) — 4 chunkers × 3 embedders = 12 combos,
`run_mode: cartesian`, `loader: plain`, `subset: full`, `seed: 42`.
Output: `data/index/chunker_compare_full/`.

## Setup

- **GPU enablement**: venv เดิมมี `torch==2.12.1+cpu` (PyPI wheel บน Windows ไม่มี CUDA
  ผูกมาให้ ต่างจาก Linux ที่ดึงผ่าน `nvidia-*` pip packages) → เพิ่ม `torch` เป็น direct
  dependency ใน extra `lab` และตั้ง `[tool.uv.sources]`/`[[tool.uv.index]]` ใน
  `pyproject.toml` ให้ดึงจาก `https://download.pytorch.org/whl/cu130` เฉพาะ
  `sys_platform == 'win32'` (cu130 เพราะ driver รองรับ CUDA 13.1 และมี torch 2.12.1
  ตรงกับที่ lock ไว้อยู่แล้ว) → `uv lock --upgrade-package torch` + `uv sync --extra lab`
  → `torch==2.12.1+cu130`, `torch.cuda.is_available()==True`. โค้ด embedder เดิมไม่ต้องแก้
  เพราะ `device=None` หมายถึง auto-detect CUDA อยู่แล้ว
- **คลังเอกสารจริง**: 2,849 ไฟล์ .md รวม 52,581,621 bytes (~52.6 MB). *(แก้ตัวเลขที่เคย
  รายงานผิดระหว่างทาง — ใช้ `find` ผ่าน bash/MSYS นับได้แค่ 1,293 เพราะสะดุดชื่อไฟล์ภาษาไทย
  ตัวจริงต้องนับด้วย Python `pathlib.rglob` ซึ่งตรงกับวิธีที่ `runner.py` ใช้)*
- **Smoke test ก่อนหน้า** (`chunker_compare_smoke.yaml`, CPU, dev subset 10 ไฟล์แรก) ใช้เป็น
  calibration data point — แต่ 10 ไฟล์แรกมีขนาดเฉลี่ยเล็กกว่าคลังจริงเกือบ 2 เท่า
  (9,399 bytes/ไฟล์ vs 18,456 bytes/ไฟล์เฉลี่ยทั้งคลัง) ทำให้ estimate เวลารันจริงที่คำนวณไว้
  ล่วงหน้าคลาดเคลื่อนไปมาก (ดูหัวข้อ "เหตุการณ์ระหว่างรัน")

## เหตุการณ์ระหว่างรัน

รอบแรก (`chunker_compare_full.yaml`, ทุก 12 combo) ถูก kill หลังรันไป 2:13:01 ชม.
(เสร็จ 6/12 combo: `fixed_size` + `recursive` × 3 embedders — ไฟล์ผลลัพธ์ไม่เสียหาย
เพราะ `ArtifactStore.save` เขียนทีหลัง `build_index` คืนค่าสำเร็จเท่านั้น)
สาเหตุไม่ทราบแน่ชัด (ไม่มี error/traceback ในล็อก, สถานะ "killed" เฉยๆ)

**Resume**: เขียน config ใหม่ (`chunker_compare_full_resume.yaml`) มีแค่ 2 chunker ที่เหลือ
(`sentence`, `semantic`) ชี้ output_dir เดิม — ใช้ได้เพราะ combo id เป็น
`sha256(loader+chunker+embedder)` ไม่ผูกกับตำแหน่งใน config จึงไม่ชนกับ 6 combo ที่มีอยู่แล้ว
รันแบบ detached process (PowerShell `Start-Process`) แยกอิสระจาก session tracking กันเหตุ
ซ้ำ ใช้เวลา 15:04:24 ชม. จนเสร็จครบ 6/6 combo ที่เหลือ

**เวลารวมที่ใช้จริงเพื่อสร้างทั้ง 12 combo**: 2:13:01 + 15:04:24 ≈ **17:17:25 ชม.**
(ไม่นับ smoke test 17 นาทีก่อนหน้าซึ่งรันบน CPU กับ dev subset 10 ไฟล์)

## ผลลัพธ์ต่อ combo

| Chunker | Embedder | n_chunks | chunk len min/mean/max (chars) | chunk_seconds | embed_seconds | รวม (โดยประมาณ) |
|---|---|---|---|---|---|---|
| fixed_size | bge-m3 | 62,018 | 1 / 438.9 / 512 | 0.47 | 1,832.8 | ~30.6 min |
| fixed_size | e5-large | 62,018 | 1 / 438.9 / 512 | 0.29 | 1,841.3 | ~30.7 min |
| fixed_size | ConGen-PhayaThaiBERT | 62,018 | 1 / 438.9 / 512 | 0.28 | 216.4 | ~3.6 min |
| recursive | bge-m3 | 76,555 | 1 / 336.7 / 512 | 1.10 | 1,836.6 | ~30.6 min |
| recursive | e5-large | 76,555 | 1 / 336.7 / 512 | 1.08 | 1,861.7 | ~31.0 min |
| recursive | ConGen-PhayaThaiBERT | 76,555 | 1 / 336.7 / 512 | 1.11 | 235.8 | ~4.0 min |
| sentence | bge-m3 | 58,853 | 2 / 422.8 / 17,420 | 150.3 | 5,681.2 | ~97.2 min |
| sentence | e5-large | 58,853 | 2 / 422.8 / 17,420 | 160.4 | 3,313.8 | ~57.9 min |
| sentence | ConGen-PhayaThaiBERT | 58,853 | 2 / 422.8 / 17,420 | 164.7 | 331.4 | ~8.3 min |
| semantic | bge-m3 | 129,131 | 1 / 192.7 / 19,427 | 14,188.2 | 10,148.9 | ~406.1 min (6.77 hr) |
| semantic | e5-large | 129,131 | 1 / 192.7 / 19,427 | 6,621.7 | 3,522.3 | ~169.1 min (2.82 hr) |
| semantic | ConGen-PhayaThaiBERT | 129,131 | 1 / 192.7 / 19,427 | 9,467.7 | 314.2 | ~163.0 min (2.72 hr) |

หมายเหตุตัวเลข:
- `chunk_seconds` ของ `semantic` คือขั้นตอนหา breakpoint ด้วย bge-m3 embedding ต่อประโยค
  (ใช้ bge-m3 เสมอไม่ว่า final embedder จะเป็นตัวไหน — ดู `embedding_model` ใน meta.json)
  แต่ตัวเลขจริงต่างกันมาก (14,188s vs 6,621s vs 9,467s) ทั้งที่ควรใกล้เคียงกัน — **variance สูง
  ผิดปกติ ไม่ทราบสาเหตุ** (สงสัย thermal throttling หรือ system contention ระหว่างรัน ยังไม่ได้
  สืบสาเหตุลึกกว่านี้)
- `fixed_size`/`recursive` chunk_seconds ≈ 0 (ไม่ใช้ embedding, split แบบ deterministic)
- `sentence`/`semantic` chunk_seconds ที่เหลือ (crfcut sentence split) scale เชิงเส้นกับขนาด
  คลังชัดเจนกว่า (~150-165s ทุก embedder เพราะเป็น CPU-only ไม่ขึ้นกับ final embedder)
- ConGen-PhayaThaiBERT (199 weight-loading steps) เร็วกว่า bge-m3/e5-large (391 steps) ใน
  ขั้น embed_seconds ประมาณ 8-11 เท่า (โมเดลเล็กกว่า)

## ข้อสังเกตที่ควรตามต่อ

**✅ ไล่หาสาเหตุ chunk ยาวผิดปกติแล้ว (7 ก.ค. 2569)** — ไล่ resolution เบื้องหลัง chunk ที่ยาวสุด
30 อันดับแรกใน `semantic` แล้วแยกเป็น 2 กลุ่ม:

1. **บั๊กจริง (OCR repetition-loop) — พบ 1 ไฟล์ตอนแรก, แก้สแกนเนอร์ 2 รอบ, ขยายเป็น 9 ไฟล์**:
   chunk ที่ยาวสุด (19,427 ตัวอักษร) มาจาก `2565\ครั้งที่ 10\...คณะวิศวกรรมศาสตร์.md` ซึ่งมีคำว่า
   `compliance; security;` วนซ้ำ **773 ครั้ง** แทนคำอธิบายรายวิชาจริง (หน้า 5 ของ PDF เดิม) —
   defect class เดียวกับที่เคยแก้ไปแล้ว (commit `dda8154`)

   **รอบแก้ที่ 1 — blind spot ของ "Curriculum Mapping" exclusion**: ตัวสแกน
   `tools/corpus_prep/scan_ocr_repetition.py` เดิม blank เนื้อหา **8,000 ตัวอักษรแบบเหมารวม**
   หลัง heading "Curriculum Mapping" ทุกครั้ง (เพื่อเลี่ยง false positive จากตาราง PLO ที่ตั้งใจให้
   0/1 ซ้ำกันได้ปกติ) — ในไฟล์นี้ window ดันไปกลืน loop ที่อยู่*คนละตาราง*กันเข้าไปด้วยเพราะ heading
   อยู่ใกล้กันในระยะ 8,000 ตัวอักษร ทั้งที่ตาราง PLO จริงจบไปตั้งนานแล้ว **แก้แล้ว**: เปลี่ยนจาก
   blanket window เป็น bound ตาม `<table>` จริงที่ตามหลัง heading (chain ตารางที่ห่างกันไม่เกิน 500
   ตัวอักษรเพื่อรองรับกริดที่ถูกตัดข้ามหน้า PDF) ตกกลับไปใช้ window 8,000 แบบเดิมเฉพาะกรณีไม่มีตาราง
   ตามหลังเลย (tag ปิดไม่สนิท) — ดู `curriculum_map_spans()`
   ผลจากรอบนี้: 5 ไฟล์ (1 prose) → 22 ไฟล์ (15 prose/table-name-loop)

   **รอบแก้ที่ 2 — blind spot ของความยาว cycle**: ผู้ใช้เจอเองด้วยตาว่าไฟล์
   `2568\ครั้งที่ 2\...สำนักวิชาศึกษาทั่วไป.md` มีประโยค `"Basic knowledge of food components,
   including watercarbohydrate food in, ... protein, hina"` (ประโยคที่ hallucinate เพี้ยนอยู่แล้ว)
   วนซ้ำทั้งประโยค **78 ครั้ง** ติดกัน — ที่รอบแก้ 1 ยังจับไม่ได้ (ตอนแรกเข้าใจผิดว่าไฟล์นี้ไม่มีบั๊ก
   จริง เป็นแค่เนื้อหายาวชอบธรรม) เพราะ `find_cyclic_floods()` เดิมเช็ค cycle ได้แค่ 2-8 token/หน่วย
   แต่ประโยคนี้ยาวถึง 26 token เกิน `CYCLE_MAX_PERIOD` ไปมาก **แก้แล้ว**: เพิ่มฟังก์ชันใหม่
   `find_line_repeats()` เช็คบรรทัดที่ไม่สั้นเกินไป (≥20 ตัวอักษร) ซ้ำกันแบบคำต่อคำติดกันหลายบรรทัด
   แทนการนับ token cycle — ครอบคลุม defect ที่เป็น "ประโยคยาวทั้งประโยควนซ้ำทีละบรรทัด" ที่
   token-cycle เช็คไม่ถึง
   ผลจากรอบนี้: 22 ไฟล์ (15 prose) → **29 ไฟล์ (19 prose/table-name-loop)**

   ทั้งสองรอบยืนยันว่า blind spot ของสแกนเนอร์เดิมซ่อน defect จริงในไฟล์อื่นอีกหลายไฟล์ ไม่ใช่แค่
   ไฟล์เดียว — รายชื่อทั้งหมด (รวม dedupe curriculum-split siblings) เขียนไว้ที่
   `academic_resolutions/ocr_repetition_review.md` — **9 source document ต้อง re-OCR ใหม่**:
   - `2564\ครั้งที่ 7\30. เรื่อง...คณะวิศวกรรมศาสตร์.md` (มี piece ย่อยหลายชิ้น) — token `"11"`
     วนซ้ำ 943 ครั้ง
   - `2565\ครั้งที่ 10\...คณะวิศวกรรมศาสตร์.md` — `compliance; security;` x773 (ตัวที่พบตอนแรก)
   - `2565\ครั้งที่ 12\...คณะวิศวกรรมศาสตร์.md` (มี piece ย่อย) — พบอยู่แล้วก่อนแก้
   - `2565\ครั้งที่ 6\...คณะวิศวกรรมศาสตร์.md` (มี piece ย่อย) — วลีซ้ำ x25 (พบจากรอบแก้ 2)
   - `2566\ครั้งที่ 2\...วิทยาลัยการจัดการนวัตกรรมและอุตสาหกรรม.md` — วลีซ้ำ x5 (พบจากรอบแก้ 2)
   - `2567\ครั้งที่ 9\...คณะวิศวกรรมศาสตร์.md` — `ฉบับที่ ๒๕๖๕` x237 + อีก 1 loop
   - `2568\ครั้งที่ 10\...คณะวิศวกรรมศาสตร์.md` — token `"e."` วนซ้ำ 1,705 ครั้ง
   - `2568\ครั้งที่ 2\...สำนักวิชาศึกษาทั่วไป.md` — `"Basic knowledge of food components..."`
     x78 (ผู้ใช้เจอเอง, ตัวจุดชนวนรอบแก้ 2)
   - `2568\ครั้งที่ 5\...คณะวิศวกรรมศาสตร์.md` — วลียาว x155
   **ยังไม่ได้ re-OCR จริง** — รอคิวถัดไป (เครื่องมือ scan เป็น read-only ไม่ได้แก้ไฟล์ใดๆ)

2. **ไม่ใช่บั๊ก — เอกสารประเภทหลักสูตรมีเนื้อหายาวชอบธรรม**: chunk ยาว 9,000–17,000 ตัวอักษรที่
   เหลือ (เช่น `2566/4s คณะทันตแพทยศาสตร์`) ตรวจแล้ว**ไม่พบ repetition-loop แม้หลังแก้สแกนเนอร์ 2
   รอบ** — เป็นคำอธิบายรายวิชาภาษาอังกฤษยาวๆ/ตาราง HTML ขนาดใหญ่ที่ไม่มีจุดจบประโยคชัดเจนแบบที่
   crfcut (Thai sentence tokenizer) จะตัดได้ ทำให้กลายเป็น "ประโยคเดียว" ยาวทั้งบล็อก — เป็น
   **ข้อจำกัดของ chunker กับเอกสารประเภทนี้** ไม่ใช่ข้อมูลเสีย ยังไม่ได้แก้ (ถ้าจะแก้ต้องเพิ่ม
   fallback hard-split ตามความยาวสูงสุดใน `sentence`/`semantic` chunker ไม่ใช่แก้ที่ข้อมูล)
   ⚠️ *บทเรียนจากรอบนี้*: "ไม่พบ repetition-loop" แปลว่า "เครื่องมือที่มีอยู่ตอนนั้นไม่พบ" ไม่ใช่
   "ยืนยันว่าไม่มี" — เห็นได้จากกรณี 2568/2 ที่กลับกลายเป็นบั๊กจริงหลังแก้สแกนเนอร์รอบ 2

3. **✅ ทำแล้ว — ตัดตาราง Curriculum/SKILL Mapping ออกจาก pipeline จริง**: ผู้ใช้เสนอว่าตาราง
   "SKILL MAPPING" (คล้าย Curriculum Mapping แต่เป็นชื่อเฉพาะของเอกสารสำนักวิชาศึกษาทั่วไป) มี
   โอกาสถูกค้นหาต่ำมาก ควรตัดออกจากข้อมูลที่ index จริง (คนละเรื่องกับการ exclude ออกจาก
   OCR-repetition scanner ซึ่งเป็นแค่ diagnostic tool)
   เพิ่มฟังก์ชัน `strip_mapping_tables()` ใน `src/rag_lab/loaders/common.py` (bound ตาม
   `<table>` จริงด้วย logic เดียวกับที่แก้ scanner รอบ 1 — chain ตารางที่ห่างกันไม่เกิน 500
   ตัวอักษร, fallback เป็น window 8,000 ตัวอักษรเฉพาะกรณีไม่มีตารางตามหลัง) แล้วเรียกใช้ใน
   loader ทั้ง 3 ตัว (`plain`, `ner`, `metadata`) — **`plain` เปลี่ยน docstring จาก "verbatim, no
   cleaning" เป็น "no cleaning ยกเว้นตัด mapping table"** (การเปลี่ยนสัญญาของ baseline loader ที่
   ควรรู้ไว้)
   ตรวจสอบแล้วว่า bound ไม่ไปกลืน defect ที่อยู่คนละตาราง (เทียบกับไฟล์ compliance/security
   ที่ยังอยู่ครบหลัง strip) และไฟล์สำนักวิชาศึกษาทั่วไปที่ตาราง SKILL MAPPING ถูกตัดจริง (70,287 →
   21,756 ตัวอักษร) — เพิ่ม unit test 5 เคสใน `tests/loaders/test_loaders_common.py`, รัน full suite ผ่าน
   หมด (104 passed, 3 skipped)
   **⚠️ ยังไม่ได้ rebuild index**: 12 combo ที่ build ไว้ใน `data/index/chunker_compare_full/`
   สร้างก่อนการแก้นี้ ยังมีตาราง Mapping ปนอยู่ในทุก combo — ถ้าต้องการให้ผลสะท้อนการตัดตารางนี้
   ต้องรัน build ใหม่ทั้งหมด (การแก้นี้ไม่กระทบ defect ทั้ง 9 ไฟล์ข้างต้น เพราะ OCR-repetition
   garbage อยู่คนละตำแหน่งกับตาราง Mapping — ยังต้อง re-OCR แยกต่างหากเหมือนเดิม)

## Retrieval-quality eval: Silver query set (16 ก.ค. 2569)

หลังจากมีแค่การเปรียบเทียบเชิงคุณภาพ (chunk-comparison artifact) จึงรัน eval เชิงปริมาณจริง
ด้วย Silver query set (title ของแต่ละ resolution เป็น query, ตอบตัวเองเป็น relevant —
ฟรี ไม่ต้อง label มือ, ดู `docs/entity-extraction-and-gold-eval-log.md` สำหรับ Gold set
ที่ยากกว่า) เทียบ 4 chunker บน e5-large embedder เดียวกัน (ตัดตัวแปร embedder ออก)

**พบบั๊ก perf จริงระหว่างเตรียมรัน**: `run_query_set` (framework code เดิม) reload
embedder + Index ทั้งก้อนใหม่ทุก query — จาก smoke test คำนวณได้ว่ารัน Silver set เต็ม
(2,873 query) จะใช้เวลา **~36 ชั่วโมง** แก้เป็นโหลด Index/embedder ครั้งเดียวต่อ combo
แล้ว loop query ในหน่วยความจำแทน (`src/rag_lab/query_sets.py`, commit `5f84654`) เหลือ
**~66 นาที** จริง (2,873 query × 4 combo, script: `tools/eval/run_silver_chunker_eval.py`)

**ผลลัพธ์** (k=10):

| Chunker | recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| fixed_size | **0.8398** | **0.6238** | **0.6754** |
| recursive | 0.8188 | 0.6206 | 0.6679 |
| sentence | 0.8267 | 0.6197 | 0.6693 |
| semantic | 0.7602 | 0.5371 | 0.5903 |

**ข้อสังเกตที่ขัดกับสามัญสำนึก**: `semantic` — chunker ที่ซับซ้อนและแพงที่สุด (ใช้
bge-m3 embedding หา breakpoint) — **แพ้ทั้ง 3 metric ให้ทุกตัวรวมถึง `fixed_size` ที่
เป็นวิธีง่ายที่สุด** สอดคล้องกับที่ chunk-comparison artifact เคยแสดงไว้เชิงคุณภาพ:
`semantic` ให้ chunk ขนาดไม่สม่ำเสมอที่สุด (1 ถึง 19,427 ตัวอักษรในเอกสารเดียวกัน) ซึ่ง
น่าจะเจือจาง/ฝังเนื้อหาที่สะท้อน title ไว้ในก้อนใหญ่เกินไป ขณะที่ `fixed_size` ตัดแบบ
กลไกสม่ำเสมอ ทำให้เนื้อหาต้น ๆ ที่มักย้ำ subject ของมติอยู่ใน chunk ที่โฟกัสกว่า

**ข้อจำกัดของผลนี้**: Silver เป็น self-retrieval แบบ "ง่าย" (title→เอกสารตัวเอง) วัด
ความสามารถ retrieve ด้วยคำที่ทับซ้อนกับ title เป็นหลัก ไม่ใช่ query แบบ paraphrase/
เชิงประเด็นที่ผู้ใช้จริงจะถาม (ดู Gold candidate pool ใน
`academic_resolutions/entity_tags/gold_candidates.json`) — ผลอาจพลิกได้เมื่อทดสอบกับ
Gold set ที่ยากกว่า ยังไม่ได้ทำ

Raw retrieval results: `data/results/silver_chunker_compare/` (gitignored, 11,460 ไฟล์),
summary: `data/results/silver_chunker_compare_report.md`

## Retrieval-quality eval: Gold query set (17 ก.ค. 2569)

หลัง curate `config/eval/gold_query_set.yaml` จาก candidate pool (แรกเริ่ม 37
entries: 12+12+13 ดู `docs/entity-extraction-and-gold-eval-log.md` สำหรับที่มาของ
3 query shape) รัน `tools/eval/run_gold_chunker_eval.py` เทียบ 4 chunker บน
e5-large เดียวกัน พบผลไม่ชี้ทางเดียวชัดเจนเหมือน Silver ต่างกันตาม query shape —
แต่ n=12 ต่อ category บาง จึงเช็ค paired significance (fixed_size vs semantic,
recall@10) พบ `|t| ~1.6-1.7` ที่ df=11 (ต้องการ >2.20 ถึงจะ p<0.05) — ทิศทางถูก
แต่ยังไม่แน่นทางสถิติ **ขยายเป็น program 30 + person 30 (faculty-adjunct คงที่ 13
เพราะใกล้เพดาน pool แล้ว) รวม 73 entries** แล้วรันใหม่ (ฟรี ไม่ต้อง label เพิ่ม
เพราะ candidate pool มีเหลือเยอะ — คัดจาก 147 program + 1,139 person candidates)

**ผลรวม** (k=10, 73 query):

| Chunker | recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| semantic | **0.4590** | **0.6994** | **0.5132** |
| fixed_size | 0.4272 | 0.6281 | 0.4698 |
| sentence | 0.4185 | 0.6526 | 0.4680 |
| recursive | 0.3926 | 0.6569 | 0.4573 |

**แยกตาม query shape**:

| Chunker | program recall@10 | person recall@10 | faculty-adjunct recall@10 |
|---|---|---|---|
| fixed_size | **0.487** | 0.335 | 0.501 |
| recursive | 0.296 | 0.456 | 0.470 |
| semantic | 0.312 | **0.622** | 0.423 |
| sentence | 0.401 | 0.429 | 0.435 |

**Paired significance ที่ n=30/category (fixed_size vs semantic, recall@10, df=29,
critical `|t|`≈2.05 ที่ p<0.05, ≈2.76 ที่ p<0.01)**:

| Query shape | mean diff (fixed−semantic) | t | ชนะ fixed / semantic / เท่า |
|---|---|---|---|
| program | +0.175 | **2.91** (p<0.01) | 15 / 6 / 9 |
| person | −0.286 | **−3.50** (p<0.01) | 6 / 19 / 5 |

ทั้งสองทิศผ่านนัยสำคัญทางสถิติจริง ไม่ใช่ noise จากตัวอย่างน้อย:

- **Program-history query**: `fixed_size` ชนะจริง (0.49 vs 0.31-0.40) — สอดคล้องกับ
  Silver: query ประเภทนี้ผูกกับชื่อหลักสูตรที่ปรากฏชัดใน title ตรงกับจุดแข็งของ
  fixed_size ที่เคยพบมาแล้ว
- **Person-history query**: กลับกัน — `semantic` ชนะจริง (0.62 vs 0.30-0.46) คนละทิศกับ
  Silver โดยสิ้นเชิง คนถูกกล่าวถึงกระจายอยู่ในเนื้อหา (ตาราง กรรมการ ผู้รับผิดชอบ)
  ไม่ใช่ title เหมือน program จึงต้องพึ่ง embedding ที่จับ semantic boundary ของ
  เนื้อหาจริงมากกว่าการตัดแบบกลไก
- **Faculty-adjunct-aggregate**: recall@10 ทุก chunker แน่นอยู่แถว 0.42-0.50 (เพดาน
  ทางคณิตศาสตร์: relevant set มีถึง 30 ไฟล์ในบางคำถาม แต่ k=10 ดึงได้สูงสุดแค่ 10
  resolution ต่อ query ต่อให้แม่นทุกอัน recall ก็ทะลุ 10/30≈0.33 ไม่ได้มาก) แต่
  **MRR สูงเกือบเต็ม (0.90-1.00 ทุก chunker)** — เอกสารที่เกี่ยวข้องมีสัดส่วนในคลัง
  สูงพอที่ผลอันดับ 1 มักจะ hit เกือบเสมอ ยืนยันสมมติฐานที่ตั้งไว้ตอน scope งานนี้
  (แนะนำโดย advisor ก่อนสร้าง): เป็นโจทย์ retrieval ที่ยากจริงในมิติ recall แม้ตัว
  ค้นหาจะ "เจอของแรก" ได้ง่ายก็ตาม

**สรุป**: chunker ที่ดีที่สุดไม่ใช่ตัวเดียวคงที่ — ขึ้นกับ query shape จริงตามสถิติ
ไม่ใช่แค่ทิศทางที่ดูสมเหตุสมผล เป็นเรื่องที่ Silver (self-retrieval ล้วน) มองไม่เห็น
เลย เป็นเหตุผลที่ต้องมี Gold set แยกต่างหาก

Raw retrieval results: `data/results/gold_chunker_compare/` (gitignored),
summary: `data/results/gold_chunker_compare_report.md`

---

## Addendum (18 ก.ค. 2569) — SemanticChunker fragmentation fix: rebuild + re-eval

หลัง dogfood Smart Routing UI ในเซสชันก่อนหน้า (17 ก.ค.) เจอบั๊ก: `SemanticChunker`
ใช้ `pythainlp.tokenize.sent_tokenize(engine="crfcut")` ตัดประโยค ซึ่งตีความจุดใน
คำย่อวิชาการไทย (ผศ. ดร. ภ.สถ.ม. วศ.บ.) เป็นขอบเขตประโยค และไม่รู้จัก HTML table
markup (`</td><td>`, มาจากตารางกรรมการที่ OCR ไว้) เลย ทำให้เกิด "ประโยค" เศษ
2-8 ตัวอักษร ซึ่งกลายเป็น chunk เดี่ยวๆ ที่ไม่มีอะไรมาเจือจาง cosine similarity
เวลามี query ตรงกับ token เศษนั้นพอดี — อธิบายทั้งอาการ "คนผิดขึ้นอันดับ 1" และ
"ผลลัพธ์เป็นแค่ `ผศ.ดร.` ไม่มีชื่อ" ที่ผู้ใช้เจอจริง รายละเอียด root-cause analysis
เต็มอยู่ที่ memory `[[project_semantic_chunker_fragmentation]]`

**สิ่งที่กังวลตอนนั้น**: recall@10 ตัดสินที่ระดับ Resolution (ADR-0002) ดังนั้น
chunk ขยะจากเอกสารที่ถูกต้องก็ยังนับเป็น hit ได้ — แปลว่าข้อสรุป "semantic ชนะ
person-history, p<0.01" (หัวข้อก่อนหน้า) และการที่ `[[project_hybrid_routing]]`
เลือก semantic+bge-m3 เป็น route สำหรับ person **อาจเป็นผลพลอยได้จากบั๊กนี้**
ไม่ใช่ความสามารถจริงของ chunker — เป็นคำถามค้างที่ต้อง verify หลัง fix

### สิ่งที่ทำวันนี้

1. **Fix**: `_merge_short_fragments()` ใน `src/rag_lab/chunkers/semantic.py` —
   รวมประโยคสั้นต่อกันจนถึง `min_sentence_chars` (ค่าเริ่มต้น 15) ก่อนแยก chunk
   เทสต์ใหม่ 2 ตัวใน `tests/chunkers/test_semantic_chunker.py`, suite ผ่านทั้งหมด —
   commit `e8f4b80`
2. **Rebuild index**: `data/index/chunker_compare_full/plain__semantic__*`
   ทั้ง 3 embedder (bge-m3, e5-large, ConGen-PhayaThaiBERT) เขียนทับ directory
   เดิมตาม content-hash เดิม (combo id hash มาจาก YAML spec ไม่ใช่ runtime
   params เลย hash ไม่เปลี่ยน) ได้ 81,489 chunks ทั้ง 3 ตัว ใช้ config ใหม่
   `config/experiments/_history/chunker_compare_full_semantic_rebuild.yaml` (ตาม pattern
   เดียวกับ `_resume.yaml`) รันแบบ background 2 รอบ: รอบแรกโดนระบบ kill หลังทำ
   bge-m3 เสร็จ (~3.5 ชม.), รอบสอง resume 2 combo ที่เหลือจนจบ (~2:40 ชม.)
   รวม **~7 ชม.**
3. **Re-eval**: รัน `tools/eval/run_gold_chunker_eval.py` ซ้ำบน 73-entry
   deterministic Gold set (program 30 + person 30 + faculty-adjunct 13,
   ไม่รวม thematic 179 ตัวที่รู้อยู่แล้วว่า discriminate แทบไม่ได้ — ดู
   `[[project_thematic_query_bootstrap]]`) เทียบกับตัวเลขเดิมในหัวข้อ
   "Retrieval-quality eval: Gold query set (17 ก.ค. 2569)" ด้านบน

### ผลรวม (k=10, 73 query, e5-large embedder คงที่)

| Chunker | recall@10 (เดิม → ใหม่) | MRR (เดิม → ใหม่) | nDCG@10 (เดิม → ใหม่) |
|---|---|---|---|
| **semantic** | 0.4590 → **0.4675** | 0.6994 → **0.7143** | 0.5132 → **0.5255** |
| fixed_size | 0.4272 → 0.4272 | 0.6281 → 0.6281 | 0.4698 → 0.4698 |
| sentence | 0.4185 → 0.4185 | 0.6526 → 0.6526 | 0.4680 → 0.4680 |
| recursive | 0.3926 → 0.3926 | 0.6569 → 0.6569 | 0.4573 → 0.4573 |

`fixed_size`/`sentence`/`recursive` เท่าเดิมเป๊ะทุก digit เพราะ index ของ 3 ตัวนั้น
ไม่ถูกแตะ — เป็น sanity check ว่า rebuild กระทบเฉพาะ `semantic` จริงตามที่ตั้งใจ
`semantic` ดีขึ้นทั้ง 3 metric (เล็กน้อยแต่ทิศทางเดียวกันหมด ไม่ใช่ noise) และยังคง
เป็น chunker ที่ดีที่สุดในชุดนี้เหมือนเดิม สรุปว่าการตัดเศษ chunk ขยะออกไม่ได้ทำร้าย
คะแนน ตรงข้ามคือช่วยเล็กน้อย

Raw results รอบนี้: `data/results/gold_chunker_compare_73det/` (gitignored),
summary: `data/results/gold_chunker_compare_73det_report.md` (ยังรันชุด
252-full ที่รวม thematic ด้วยรอบแรกเป็น sanity check ก่อนสังเกตว่า thematic เจือจาง
สัญญาณ — เก็บไว้ที่ `data/results/gold_chunker_compare_report.md`/
`data/results/gold_chunker_compare/` แต่**อย่าใช้เทียบค่าเดิม** เพราะ query set
ไม่ตรงกับที่ใช้สร้างตัวเลขเดิม)

### ตอบคำถามค้าง: "semantic ชนะ person-history" เป็นผลจากบั๊กหรือเปล่า?

รัน `tools/eval/gold_eval_breakdown.py` (per-entity_type paired t-test,
fixed_size vs semantic, recall@10) บนผลชุด 252-full ซ้ำ:

| entity_type | n | fixed_size | semantic | mean_diff (fixed−semantic) | t | ชนะ fixed/semantic/เท่า |
|---|---|---|---|---|---|---|
| program | 30 | 0.4871 | 0.2952 | +0.1919 | **+3.11** (p<0.01) | 16/7/7 |
| person | 30 | 0.3353 | **0.6542** | **−0.3189** | **−4.51** (p<0.01) | 4/20/6 |
| faculty_adjunct_aggregate | 13 | 0.5014 | 0.4346 | +0.0669 | +1.71 (n.s.) | 7/2/4 |

เทียบกับตัวเลขเดิม (17 ก.ค., ก่อน fix): program `t=+2.91`, person `t=−3.50`
(ทั้งคู่ p<0.01 เหมือนกัน) — **ทั้งสองข้อสรุปรอดหลัง fix จริง ไม่ใช่ artifact ของบั๊ก**
โดยเฉพาะ person-history ยิ่งชัดขึ้นด้วยซ้ำ (recall 0.622→0.654, `t` แรงขึ้น
−3.50→−4.51) แปลว่า `[[project_hybrid_routing]]` ที่เลือก semantic+bge-m3 เป็น
route สำหรับ person query **ยังคงเป็นทางเลือกที่ถูกต้อง ไม่ต้องแก้ routing ใดๆ**

ปิดคำถามค้างจาก memory `[[project_semantic_chunker_fragmentation]]` ครบทุกข้อแล้ว

---

## Addendum (20 ก.ค. 2569) — ขยายเป็น 6 embedder (qwen3, jina_v5, m2v) + Gold eval เบื้องต้น

เพิ่ม 3 embedder ใหม่เข้าไปใน matrix เดิม (4 chunker × 3 embedder เดิม) ทำให้ครบ
**4 chunker × 6 embedder = 24 combo**: `Qwen3-Embedding-4B` (LLM-based, กลุ่ม C
ตามกรอบวิจัยใน `docs/research-framework-gap-analysis.md`), `jina-embeddings-v5-
text-small-retrieval`, และ `Thaweewat/jina-embedding-v3-m2v-1024` (Model2Vec
static distillation) — commit `2aeeabf` (wiring) + `7735164` (bound batch/seq
length + `release()` แก้ VRAM OOM บนการ์ด 12GB)

### Build: กระบวนการซ้ำๆ ของการโดน kill กลางคัน

เหมือนรอบ semantic-chunker rebuild — build เต็มคลังโดนระบบ kill แบบเงียบๆ
(ไม่มี error/traceback) หลายครั้งติดกัน ต้องไล่ทำ resume แบบ content-hash
(combo id ไม่ผูกตำแหน่งใน config) **7 รอบติดกัน**: `..._resume.yaml` →
`resume2` → `resume3` → `resume4` (ครอบ jina_v5/m2v × 4 chunker) แล้ว
`..._resume_qwen3.yaml` → `resume_qwen3_2` → `resume_qwen3_3` (ครอบ qwen3 ×
4 chunker) — ตัวสุดท้าย (`semantic × qwen3`, 81,489 chunks) ใช้เวลา **6:23:10 ชม.**
เดี่ยวๆ (embed step ช้าเพราะ batch_size ถูกจำกัดไว้กัน OOM + Qwen3-Embedding-4B
เป็นโมเดลหนักสุดในชุด) จบแล้วไม่มี error/OOM — build ครบ 12 combo ใหม่จริง
(ยืนยันด้วยการนับ dir ใน `data/index/chunker_compare_full/`)

### ผลลัพธ์ดิบ (k=10, ทุก combo, e5-large embedder ผสมกับตัวใหม่)

รัน `tools/eval/run_gold_chunker_eval.py --embedder-filter plain` (filter
`"plain"` แปลว่าไม่กรองอะไรเลยเพราะทุก combo ขึ้นต้นด้วย loader `plain` —
เจตนาให้ผ่านทั้ง 24 combo) **⚠️ รันบน query set เต็ม 252 ข้อ (73 deterministic +
179 thematic) ไม่ใช่ 73-deterministic ที่ addendum ก่อนหน้าใช้** — thematic
เจือจางสัญญาณตามที่บันทึกไว้แล้ว (`[[project_thematic_query_bootstrap]]`)
ดังนั้น**ตัวเลขชุดนี้เป็นค่าเบื้องต้น ยังไม่ใช่ตัวเลขที่ควรอ้างอิงในเปเปอร์**
ต้องรันซ้ำแบบ `--gold-query-set` ชี้ไปที่ 73-deterministic ก่อนถึงจะเทียบกับ
ตัวเลขเดิม (0.4272-0.4675 ระดับ) ได้ตรงๆ

**⚠️ บั๊กเล็กใน `render_report()`**: header ในไฟล์ output เขียน hardcode ว่า
`"embedder = e5-large (held fixed...)"` ทั้งที่รอบนี้รันครบ 6 embedder จริง
(ข้อความสืบทอดมาจากตอน script นี้เขียนไว้ให้ตรึง embedder เดียว) — **เป็นบั๊ก
cosmetic ใน header string เท่านั้น ไม่กระทบตัวเลขที่คำนวณจริง** ยังไม่ได้แก้
โค้ด แค่บันทึกไว้กันสับสนเวลาอ่าน raw report

Decode hash → ชื่อโมเดลจริง (`local` type มี 3 โมเดลปนกันในชื่อ type เดียว
แยกด้วย hash):

| combo hash suffix | โมเดลจริง | กลุ่มตามกรอบวิจัย |
|---|---|---|
| `e5__*` | `intfloat/multilingual-e5-large` | B. multilingual |
| `jina_v5__*` | `jinaai/jina-embeddings-v5-text-small-retrieval` | B. multilingual |
| `qwen3__*` | `Qwen/Qwen3-Embedding-4B` | C. LLM-based |
| `local__ceea7536`/`e05efbb8`/`8aae9bcd`/`bf8b7ebb` | `BAAI/bge-m3` | B. multilingual |
| `local__7cceab27`/`d04f22ee`/`87fee2dc`/`5f573c4f` | `kornwtp/ConGen-BGE_M3-model-phayathaibert` | A. ไทยเฉพาะทาง |
| `local__f71f693a`/`0fdddac0`/`834c4336`/`5528285a` | `Thaweewat/jina-embedding-v3-m2v-1024` | static distilled |

**ค่าเฉลี่ยต่อ embedder (เฉลี่ยข้าม 4 chunker, query set 252 เต็ม — ค่าเบื้องต้น)**:

| Embedder | กลุ่ม | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| BAAI/bge-m3 | B | **0.3308** | **0.3843** | **0.2998** |
| multilingual-e5-large | B | 0.3136 | 0.3649 | 0.2785 |
| Qwen3-Embedding-4B | C | 0.2994 | 0.3299 | 0.2666 |
| ConGen-PhayaThaiBERT | A | 0.2957 | 0.3533 | 0.2669 |
| jina-v5-small | B | 0.2771 | 0.3038 | 0.2424 |
| Thaweewat/m2v (static) | — | 0.1448 | 0.1631 | 0.1191 |

ผลดิบเต็ม 24 combo: `data/results/gold_full_embedder_matrix_report.md`
(gitignored), raw retrieval: `data/results/gold_full_embedder_matrix/`

### ข้อสังเกตเบื้องต้น (ยังไม่ทดสอบนัยสำคัญทางสถิติ — เป็นสมมติฐานตั้งต้นเท่านั้น)

- **bge-m3 นำในค่าเฉลี่ยทุกเมตริก** สอดคล้องกับ `[[project_embedder_comparison]]`
  เดิมที่สรุปว่า bge-m3 balanced สุดในบรรดา embedder เดิม 3 ตัว
- **Qwen3-Embedding-4B (โมเดลใหญ่สุดในชุด) ไม่ชนะในค่าเฉลี่ย** แพ้ทั้ง bge-m3
  และ e5-large ที่เล็กกว่ามาก — ทิศทางตรงกับที่กรอบวิจัยตั้งคำถามไว้ใน RQ2
  ("โมเดลใหญ่คุ้มจริงไหม") แต่ **combo เดี่ยวที่คะแนนสูงสุดในตารางทั้งหมดคือ
  `semantic × qwen3`** (recall=0.3702, nDCG=0.3344) — Qwen3 ดูจะมี interaction
  กับ semantic chunker โดยเฉพาะ ไม่ใช่ผู้ชนะสม่ำเสมอข้าม chunker ยังไม่ได้ยืนยัน
  ด้วย significance test
- **ConGen-PhayaThaiBERT ไม่ชนะ multilingual ในค่าเฉลี่ยรวม** — **แต่ไม่ขัดกับ
  `[[project_embedder_comparison]]` เดิม** ที่พบว่า ConGen เป็น "specialist"
  (ชนะ program ทุก chunker แต่แพ้ person หนักมาก) ค่าเฉลี่ยรวมที่เจือจางด้วย
  person+thematic ย่อมกลบจุดแข็งด้าน program ไว้ — **ยังไม่ได้รัน entity-type
  breakdown สำหรับ embedder ใหม่ 3 ตัว** เพื่อดูว่า pattern specialist/generalist
  เดิมยังอยู่กับ qwen3/jina_v5/m2v ไหม เป็นงานที่ควรทำต่อ
- **m2v (static distillation) แพ้ทุกตัวขาดลอย** (recall 0.145 vs ~0.28-0.33
  ตัวอื่น) — จ่ายคุณภาพแพงมากแลกความเร็ว จุด Pareto ต้องดูคู่กับตัวเลข
  throughput/latency (ยังไม่ได้รวบรวมเป็นตาราง ดู Tier 1 ใน gap-analysis)

### งานที่เหลือก่อนตัวเลขพร้อมอ้างอิงในเปเปอร์

1. ~~รันซ้ำบน 73-deterministic Gold set~~ **ทำแล้ว — ดูหัวข้อถัดไป**
2. รัน `gold_embedder_breakdown.py`-style per-entity_type สำหรับ qwen3/jina_v5/m2v
   (ตอนนี้มีแค่ e5/bge-m3/ConGen)
3. ทดสอบนัยสำคัญทางสถิติ (อย่างน้อย paired t-test แบบเดิม; กรอบวิจัยแนะนำ
   อัปเกรดเป็น bootstrap + Holm correction เพราะเทียบ 6 embedder = 15 คู่พร้อมกัน)
4. แก้บั๊ก header string ใน `render_report()`

### Re-run บน 73-deterministic (20 ก.ค. 2569, ต่อเนื่องวันเดียวกัน) — ตัวเลขสะอาด

ไฟล์ 73-only เดิมถูก merge เข้า 252 ไปแล้วตอนเพิ่ม thematic (`add_thematic_to_gold_set.py`,
73→252) จึงสร้างใหม่จาก `config/eval/gold_query_set.yaml` โดยกรอง
`entity_type != thematic` → เก็บไว้ที่ `config/eval/gold_query_set_73det.yaml`
(tracked ใน repo รันซ้ำได้ในอนาคตไม่ต้อง derive ใหม่) ยืนยันจำนวนตรง: 30
program + 30 person + 13 faculty_adjunct_aggregate = 73

รัน `run_gold_chunker_eval.py --gold-query-set config/eval/gold_query_set_73det.yaml
--embedder-filter plain` (ครบ 24 combo, 955.8s) ผลดิบ:
`data/results/gold_73det_full_embedder_matrix_report.md` (gitignored)

**ค่าเฉลี่ยต่อ embedder (73-deterministic, สะอาด — เทียบกับค่า 252-diluted ก่อนหน้า)**:

| Embedder | recall@10 (252-diluted → 73det) | MRR (252→73det) | nDCG@10 (252→73det) |
|---|---|---|---|
| **Qwen3-Embedding-4B** | 0.2994 → **0.5155** | 0.3299 → **0.7848** | 0.2666 → **0.5912** |
| **BAAI/bge-m3** | 0.3308 → **0.5108** | 0.3843 → 0.7543 | 0.2998 → 0.5717 |
| jina-v5-small | 0.2771 → 0.4503 | 0.3038 → 0.7057 | 0.2424 → 0.5168 |
| multilingual-e5-large | 0.3136 → 0.4264 | 0.3649 → 0.6630 | 0.2785 → 0.4801 |
| ConGen-PhayaThaiBERT | 0.2957 → 0.4134 | 0.3533 → 0.6535 | 0.2669 → 0.4727 |
| Thaweewat/m2v (static) | 0.1448 → 0.1472 | 0.1631 → 0.3107 | 0.1191 → 0.1845 |

**ค่าเฉลี่ยต่อ chunker (73-deterministic, ข้ามทั้ง 6 embedder)**:

| Chunker | recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| **semantic** | **0.4939** | **0.7184** | **0.5483** |
| recursive | 0.3922 | 0.6135 | 0.4488 |
| fixed_size | 0.3786 | 0.6251 | 0.4417 |
| sentence | 0.3776 | 0.6243 | 0.4393 |

**Combo เดี่ยวที่ดีสุดทั้งตาราง**: `semantic × Qwen3-Embedding-4B`
(recall@10=**0.6581**, MRR=**0.8831**, nDCG@10=**0.7339**) — ทิ้งห่างอันดับ 2
(`semantic × jina_v5` = 0.5845/0.8104/0.6493) ชัดเจน

### ข้อสังเกตหลังตัด thematic (เทียบกับข้อสังเกตเบื้องต้นด้านบน)

- **การเจือจางจาก thematic มีจริงและมีขนาดใหญ่** — ทุก embedder ขยับขึ้น
  ~30-75% เมื่อตัด thematic ออก ยืนยัน `[[project_thematic_query_bootstrap]]`
  อีกครั้งว่า thematic queries ไม่ควรรวมในตัวเลขอ้างอิง
- **อันดับพลิก**: บนชุด 252 ที่เจือจาง `bge-m3` นำ Qwen3 อยู่ (0.331 vs 0.299)
  แต่บนชุด 73-deterministic ที่สะอาด **Qwen3-Embedding-4B แซงขึ้นเป็นอันดับ 1**
  (0.5155 vs bge-m3 0.5108) แม้ระยะห่างจะเล็ก (~0.005, ยังไม่ได้ทดสอบนัยสำคัญ
  — อาจไม่ต่างกันจริงทางสถิติ) แต่ทิศทางกลับด้าน แปลว่า **ข้อสรุปเบื้องต้นที่ว่า
  "โมเดลใหญ่สุดไม่ชนะ" ต้องระงับไว้ก่อนจนกว่าจะมี significance test** — อย่าใช้
  ค่า 252 เป็นข้อสรุปสุดท้ายของ RQ2
- **semantic chunker ชนะชัดเจนขึ้นมากบนชุด 73-det** (0.494 เทียบ 0.38-0.39
  ตัวอื่น) แรงกว่าที่เคยเห็นตอนเทียบแค่ 3 embedder เดิม — ดูเหมือน semantic
  จะ synergize ดีเป็นพิเศษกับ embedder ที่แรงอยู่แล้ว (qwen3, jina_v5, bge-m3
  ทั้ง 3 ตัวได้ผลดีสุดตอนจับคู่กับ semantic) ขณะที่ ConGen/m2v ไม่ได้ผลบวก
  แบบเดียวกัน — สอดคล้องกับ "specialist vs generalist" pattern ที่เคยพบใน
  `[[project_embedder_comparison]]` (ConGen ไม่ได้ประโยชน์จาก semantic
  เท่าตัวอื่น)
- **m2v แทบไม่ขยับ** (0.1448→0.1472) ต่างจากตัวอื่นที่ขยับแรง — สอดคล้องกับว่า
  m2v เป็น static embedding ที่คุณภาพจำกัดมาแต่ต้น ไม่ได้ไวต่อความยากง่ายของ
  query แบบเดียวกับตัวอื่น
- ยังไม่ได้ทำ: per-entity_type breakdown สำหรับ 3 embedder ใหม่ และ
  significance test — คือสิ่งที่ต้องทำก่อนอ้างอิง "Qwen3 ชนะ" ในเปเปอร์ได้จริง

### Significance test (21 ก.ค. 2569) — ยืนยัน: bge-m3 vs Qwen3 ไม่ต่างกันจริงทางสถิติ

สคริปต์ `tools/eval/embedder_significance_test.py` — paired bootstrap
(resample หน่วยเป็น **query**, 73 ตัว, n_boot=10,000, seed=42) เทียบทุกคู่
embedder (15 คู่) บน 3 เมตริก แต่ละคู่: ค่าต่อ query เฉลี่ยข้าม 4 chunker
ก่อน (ตรงกับตารางค่าเฉลี่ยต่อ embedder ที่รายงานไว้ด้านบน) แล้ว bootstrap
ค่าเฉลี่ยผลต่าง, p-value สองด้านแบบ percentile, แก้ multiple-comparison ด้วย
Holm-Bonferroni **แยกต่างหากต่อเมตริก** (15 การทดสอบ/เมตริก ตามที่
gap-analysis แนะนำ) ผลดิบเต็ม: `data/results/embedder_significance_test.md`
(gitignored)

**ข้อสรุปหลัก**:

- **bge-m3 vs Qwen3-Embedding-4B ไม่ต่างกันอย่างมีนัยสำคัญทางสถิติในทั้ง 3
  เมตริก** (recall@10: diff=-0.0048, p=0.837, Holm-adj=1.0; mrr: p=0.187,
  Holm-adj=0.657; ndcg@10: p=0.417, Holm-adj=0.835) — **ยืนยันข้อสงสัยที่ตั้ง
  ไว้ก่อนหน้า: การที่อันดับพลิกระหว่างชุด 252 กับ 73-det เป็น noise ไม่ใช่
  สัญญาณจริง** ข้อสรุป RQ2 ("โมเดลใหญ่สุดคุ้มไหม") **ต้องเขียนว่า "ไม่ต่างกัน
  อย่างมีนัยสำคัญ" ไม่ใช่ "Qwen3 ชนะ" หรือ "bge-m3 ชนะ"**
- **กลุ่มบนที่ต่างจากกลุ่มล่างอย่างมีนัยสำคัญจริง**: {bge-m3, Qwen3} ทั้งคู่
  ชนะ e5-large, ConGen-PhayaThaiBERT, jina-v5-small อย่างมีนัยสำคัญในอย่างน้อย
  1 เมตริก (ส่วนใหญ่ทั้ง 3) — ยกเว้น bge-m3 vs jina_v5 ที่ recall@10 ไม่ผ่าน
  หลัง Holm correction (raw p=0.007 แต่ holm-adj=0.036 < 0.05 จริงๆ ผ่าน — แต่
  mrr/ndcg ไม่ผ่าน) จึงสรุปว่า **{bge-m3, Qwen3} เป็นกลุ่มนำที่ต่างจาก e5/congen
  อย่างชัดเจน แต่เทียบกับ jina_v5 ยังก้ำกึ่ง**
- **congen vs e5-large ไม่ต่างกันเลยสักเมตริก** (p สูงสุด 0.86) — สองตัวนี้
  อยู่ระดับเดียวกันจริง ไม่ใช่แค่บังเอิญตัวเลขใกล้กัน
- **m2v แพ้ทุกตัวอย่างมีนัยสำคัญสุดขั้ว** (p<0.001 ทุกคู่ทุกเมตริก) ตามที่
  คาดไว้ ยืนยันเป็นเชิงสถิติแล้วว่าไม่ใช่แค่ตัวเลขดิบต่างกัน
- **ผลกระทบต่อกรอบวิจัย**: RQ2 เปลี่ยนจาก "โมเดลใหญ่สุดไม่คุ้ม" (สรุปเบื้องต้น
  จากชุด 252) เป็น **"ในกลุ่มโมเดลที่แข็งแรง (bge-m3/Qwen3) ขนาดไม่ใช่ตัวชี้วัด
  ผลต่างที่วัดได้ — แต่ต้นทุน inference ของ Qwen3-4B สูงกว่า bge-m3 มาก โดยไม่
  ได้คุณภาพเพิ่มที่พิสูจน์ได้"** เป็นข้อสรุปที่แรงกว่าเดิม (ตอบคำถาม "คุ้มไหม"
  ได้ตรงประเด็นกว่า "ใครชนะ")

### Entity-type breakdown สำหรับ 6 embedder (21 ก.ค. 2569) — ยืนยัน pattern specialist เดิม + เจอ Qwen3 เป็น generalist ที่แข็งที่สุด

สคริปต์ `tools/eval/gold_embedder_breakdown_73det.py` (สืบทอด logic จาก
`gold_embedder_breakdown.py` เดิมที่ทำแค่ 3 embedder บนชุด 252-diluted) รันบน
73-deterministic ที่สะอาด ครบ 6 embedder × 4 chunker ผลดิบเต็ม:
`data/results/gold_embedder_breakdown_73det.md` (gitignored)

**ค่าเฉลี่ยข้าม chunker ต่อ embedder × entity_type (recall@10)**:

| embedder | faculty_adjunct | person | program | overall |
|---|---|---|---|---|
| bge_m3 | 0.4555 | **0.5694** | 0.4760 | 0.5107 |
| qwen3 | **0.4741** | 0.4807 | 0.5682 | 0.5155 |
| congen | 0.3966 | 0.2608 | **0.5732** | 0.4134 |
| jina_v5 | 0.4130 | 0.4285 | 0.4881 | 0.4503 |
| e5 | 0.4603 | 0.4686 | 0.3697 | 0.4265 |
| m2v | 0.2215 | 0.0572 | 0.2049 | 0.1472 |

**สรุป**:

- **Pattern specialist ของ ConGen ยังยืนยันชัดเจนกับชุดข้อมูลใหม่**: ชนะ
  program ขาดลอย (0.573, สูงสุดในตาราง) แต่แพ้ person หนักที่สุดในบรรดา
  embedder ปกติทั้งหมด (0.261 — เหนือแค่ m2v) ตรงกับที่พบครั้งแรกใน
  `[[project_embedder_comparison]]` (3-embedder เดิม)
- **จุดที่ต่างจากเดิม (สำคัญ)**: bge-m3 กับ Qwen3 เฉลี่ยรวมใกล้กันมาก
  (0.5107 vs 0.5155, ยืนยันแล้วว่าไม่ต่างกันจริงทางสถิติจาก significance
  test ด้านบน) **แต่โปรไฟล์ต่างกันชัดเจน**: bge-m3 เป็น person-specialist
  (ชนะ person ขาด แต่ program กลางๆ) ส่วน **Qwen3 เป็น generalist ที่แข็งสุด
  ในตาราง** — อันดับ 2 ของ person (ตามหลัง bge-m3 ไม่มาก), อันดับ 2 ของ
  program (ตามหลัง ConGen ไม่มาก), และ**ชนะ faculty_adjunct_aggregate สูงสุด**
  — อธิบายได้ว่าทำไมค่าเฉลี่ยรวมของทั้งคู่ถึงมาบรรจบกัน ทั้งที่โปรไฟล์คนละแบบ
  กันเลย
- **นัยเชิงปฏิบัติ**: ถ้าเลือก embedder ตาม use-case เฉพาะทาง (ไม่ใช่ค่าเฉลี่ย
  รวม) — ระบบที่เน้นค้นหาบุคคลควรใช้ bge-m3, เน้นหลักสูตร/สาขาวิชาควรใช้ ConGen
  (เล็กกว่ามาก ถูกกว่ามาก ในเมื่อ program คือ use-case เดียว), ส่วน Qwen3
  เหมาะกับระบบที่ต้องรับมือคำถามหลายประเภทปนกันโดยไม่รู้ล่วงหน้าว่าจะเป็น
  ประเภทไหน (สอดคล้องกับ [[project_hybrid_routing]] ที่ routing ตาม entity
  type อยู่แล้ว — ถ้า routing ทำงานสมบูรณ์ Qwen3's generalist strength ก็ไม่ได้
  ให้ประโยชน์เพิ่มเหนือ specialist-per-route)
- **m2v ล้มเหลวหนักเป็นพิเศษที่ person** (0.057, เกือบเป็นศูนย์) มากกว่าที่
  เห็นตอนดูค่าเฉลี่ยรวม (0.147) — เป็น static embedding ที่ดูเหมือนจับ named-
  entity context แทบไม่ได้เลย ต่างจาก program (0.205) ที่ยังพอจับ topical
  similarity ได้บ้าง
### Significance test แยกต่อ entity_type (21 ก.ค. 2569, ต่อเนื่อง) — Qwen3 คือ generalist ที่ tie กับ specialist ทั้งสองด้านพร้อมกัน

สคริปต์ `tools/eval/embedder_significance_test_by_entity_type.py` — logic
เดียวกับ significance test รวม แต่รันแยกต่อ entity_type (resample เฉพาะ
query ในกลุ่มนั้น) แล้ว Holm-correct แยกใน 15 คู่ต่อ (entity_type, เมตริก)
ผลดิบเต็ม: `data/results/embedder_significance_test_by_entity_type.md`
(gitignored)

**ข้อสรุปหลัก — ยืนยันด้วยสถิติ ไม่ใช่แค่ตัวเลขเชิงพรรณนาแล้ว**:

- **person (n=30)**: bge-m3 ชนะ ConGen, e5, jina_v5, m2v **อย่างมีนัยสำคัญ
  ทั้งหมด** (Holm-adj p<0.001 ทุกคู่) **แต่ bge-m3 vs Qwen3 ไม่ต่างกันอย่างมี
  นัยสำคัญ** (recall@10 raw p=0.033, Holm-adj=0.131 — ไม่ผ่าน) ConGen แพ้ทุก
  ตัวยกเว้น m2v อย่างมีนัยสำคัญ (แพ้หนักสุดในกลุ่ม normal embedder ยืนยัน
  ชัดเจน)
- **program (n=30)**: ConGen ชนะ bge-m3, e5, jina_v5, m2v **อย่างมีนัยสำคัญ
  ทั้งหมด** **แต่ ConGen vs Qwen3 ไม่ต่างกันอย่างมีนัยสำคัญ** (recall@10
  diff=+0.005, raw p=0.881 — เท่ากันจริงๆ ไม่ใช่แค่ใกล้เคียง) bge-m3 แพ้
  Qwen3 อย่างมีนัยสำคัญที่ program (Holm-adj p=0.015)
- **faculty_adjunct_aggregate (n=13, เล็กสุด)**: ส่วนใหญ่ยังแยกไม่ออกเพราะ
  ตัวอย่างเล็ก (ยกเว้น m2v ที่แพ้ทุกตัวชัดเจนแม้ n เล็ก) แต่ Qwen3 ชนะ ConGen
  และ jina_v5 อย่างมีนัยสำคัญที่ recall@10 — สัญญาณไปทางเดียวกับที่เห็นใน
  entity type อื่น
- **ข้อสรุปที่แรงที่สุด**: **Qwen3-Embedding-4B tie กับ specialist ทั้งสองด้าน
  พร้อมกัน** — tie กับ bge-m3 (person-specialist) ที่ person, **และ** tie กับ
  ConGen (program-specialist) ที่ program ในเวลาเดียวกัน ไม่มี embedder ตัวไหน
  อื่นที่ทำได้แบบนี้ (bge-m3 แพ้ ConGen ที่ program อย่างมีนัยสำคัญ, ConGen แพ้
  bge-m3 ที่ person อย่างมีนัยสำคัญ — ทั้งคู่มีจุดอ่อนที่พิสูจน์ได้ทางสถิติ
  คนละด้าน) **นี่คือหลักฐานเชิงสถิติที่หนักแน่นที่สุดที่สนับสนุนกรอบ "Qwen3 =
  generalist แข็งสุด" ที่ตั้งไว้จาก breakdown เชิงพรรณนาก่อนหน้า**
- **นัยต่อ RQ2**: ถ้า deployment มี routing ตาม entity type ที่แม่นยำ
  (`[[project_hybrid_routing]]`) → ใช้ specialist-per-route (bge-m3 สำหรับ
  person, ConGen สำหรับ program) จะได้ผลเท่ากับ/ดีกว่า Qwen3 ในต้นทุนที่ต่ำกว่า
  มาก แต่ถ้า routing ไม่แม่นยำ/ไม่มี หรือคำถามมีลักษณะผสม Qwen3 คือตัวเลือก
  เดียวที่ไม่มีจุดอ่อนที่พิสูจน์ได้ในทั้งสองประเภทหลัก — เป็น trade-off เรื่อง
  ความเชื่อมั่นใน routing มากกว่าคุณภาพ embedder ล้วนๆ

## BM25 lexical baseline (21 ก.ค. 2569) — ผลที่พลิกกรอบ RQ2 ทั้งหมด

โค้ด `BM25Retriever` (`src/rag_lab/retrievers/bm25.py`, ใช้ `rank_bm25.BM25Okapi`
บนโทเคนคำจาก PyThaiNLP `word_tokenize` engine `newmm`) มีอยู่แล้วในโปรเจกต์
แต่ไม่เคยรัน eval จริง — index ที่ build ไว้แล้วทุก combo มี `lexical.json`
ติดมาด้วยอยู่แล้ว (คำนวณตอน build ไม่ขึ้นกับ embedder) จึง**ไม่ต้อง build
index ใหม่** สคริปต์ `tools/eval/run_gold_bm25_eval.py` หยิบ index ตัวแทน
1 embedder ต่อ chunker (เลือก e5 variant, ไม่กระทบผลเพราะ BM25 ไม่สนใจ
embedder) รันบน Gold 73-det ครบ 4 chunker (432.9s รวม)

**ผลลัพธ์ (BM25, ต่อ chunker)**:

| chunker | recall@10 | mrr | ndcg@10 |
|---|---|---|---|
| semantic | **0.5902** | 0.7690 | **0.6174** |
| sentence | 0.5801 | 0.7955 | **0.6379** |
| recursive | 0.5526 | 0.7491 | 0.5889 |
| fixed_size | 0.5476 | 0.8019 | 0.6126 |

**เทียบกับค่าเฉลี่ย dense-embedder ต่อ chunker (ข้าม 6 embedder, จากตารางก่อน
หน้า)**:

| chunker | dense เฉลี่ย recall@10 | BM25 recall@10 | ผลต่าง |
|---|---|---|---|
| fixed_size | 0.3786 | 0.5476 | **BM25 ชนะ +0.169** |
| recursive | 0.3922 | 0.5526 | **BM25 ชนะ +0.160** |
| semantic | 0.4939 | 0.5902 | **BM25 ชนะ +0.096** |
| sentence | 0.3776 | 0.5801 | **BM25 ชนะ +0.203** |

**BM25 ชนะค่าเฉลี่ย dense-embedder ทุก chunker ขาดลอย** และเทียบกับ embedder
เดี่ยวที่ดีที่สุดต่อ chunker (จากตาราง breakdown ก่อนหน้า) — BM25 ยังชนะ
เกือบทุกกรณี ยกเว้น **semantic × Qwen3-Embedding-4B** (0.6581) ที่ยังสูงกว่า
BM25×semantic (0.5902) — เป็น**combo เดียว**ในทั้งตารางที่เอาชนะ BM25 ได้
ชัดเจน (fixed_size ที่ดีสุดของ dense คือ qwen3=0.4438 แพ้ BM25 0.5476 ขาด,
recursive ดีสุด qwen3=0.4709 แพ้ BM25 0.5526, sentence ดีสุด qwen3=0.4893
แพ้ BM25 0.5801)

**ทำไมถึงเป็นแบบนี้ (สมมติฐาน, ยังไม่ยืนยัน)**: Gold query set เป็น
entity-anchored (`[[project_gold_query_set]]`) — แม้ตัวคำถามจะ rephrase
หนีจาก title wording ของเอกสาร แต่**ชื่อเฉพาะของ entity ยึด** (ชื่อคน/ชื่อ
หลักสูตร/ชื่อคณะ) **ยังคงอยู่ในคำถามแบบคำต่อคำ** เพราะเป็นสิ่งเดียวที่ระบุ
ตัวตนได้ ทำให้ BM25 ที่ match คำตรงตัวได้เปรียบธรรมชาติสำหรับงานประเภทนี้
โดยเฉพาะ ต่างจาก paraphrase/thematic query ที่ dense embedding ควรจะได้เปรียบ
กว่า (แต่ thematic queries มี discrimination ต่ำมากอยู่แล้วตาม
`[[project_thematic_query_bootstrap]]` เลยไม่มีในชุด 73-det นี้)

**ผลต่อกรอบ RQ2 ทั้งหมด**: ข้อสรุปก่อนหน้า ("Qwen3 ไม่ชนะจริงเทียบ bge-m3",
"Qwen3 = generalist ที่แข็งสุด") **ยังคงจริงในกรอบ "เทียบ dense embedder
ด้วยกันเอง"** แต่ต้องเพิ่มบริบทใหม่ที่สำคัญกว่า: **dense embedding ธรรมดา
(ไม่ hybrid) แพ้ lexical baseline ง่ายๆ ในงานนี้แทบทุกกรณี** ยกเว้น
combo ที่แพงที่สุดตัวเดียว (semantic×Qwen3-4B) คำถามที่ควรตอบต่อไม่ใช่แค่
"embedder ไหนดีสุด" แต่คือ **"คุ้มไหมที่จะใช้ dense retrieval เลย ถ้า BM25
เปล่าๆ ก็ชนะเกือบทุก combo"** — เป็นคำถามที่แรงกว่า RQ2 เดิมมาก

### Significance test BM25 vs dense embedder (21 ก.ค. 2569, ต่อเนื่อง) — ต้องปรับข้อสรุปให้ระมัดระวังขึ้น

สคริปต์ `tools/eval/bm25_vs_embedder_significance_test.py` — paired bootstrap
เดียวกันกับ significance test อื่น (resample 73 query, ค่าต่อ query เฉลี่ย
ข้าม 4 chunker ก่อน — **BM25 ก็มีค่าเฉลี่ยข้าม 4 chunker ของตัวเองเหมือนกัน**
เพื่อเทียบแบบเดียวกับ embedder อื่น) Holm-correct แยกใน 6 คู่ (bm25 vs
embedder แต่ละตัว) ต่อเมตริก ผลดิบเต็ม:
`data/results/bm25_vs_embedder_significance_test.md` (gitignored)

**ผล**:

| A vs B | recall@10 diff | Holm-adj p | significant |
|---|---|---|---|
| bm25 vs m2v | +0.4205 | 0.0000 | **yes** |
| bm25 vs congen | +0.1543 | 0.0064 | **yes** |
| bm25 vs e5 | +0.1411 | 0.0000 | **yes** |
| bm25 vs jina_v5 | +0.1174 | 0.0132 | **yes** |
| bm25 vs bge_m3 | +0.0569 | 0.1832 | no |
| bm25 vs qwen3 | +0.0521 | 0.1884 | no |

**ต้องปรับข้อสรุปจากหัวข้อก่อนหน้า**: ในกรอบ**เฉลี่ยข้าม chunker แบบเดียวกับ
ที่ใช้เทียบ embedder ทุกตัว** (ไม่ใช่ point comparison ต่อ chunker) **BM25
ชนะ embedder กลุ่มกลาง-ล่างอย่างมีนัยสำคัญจริง** (ConGen, e5, jina_v5, m2v)
**แต่ไม่ต่างจาก top tier (bge-m3, Qwen3) อย่างมีนัยสำคัญ** (p=0.183, 0.188
ตามลำดับ) — ตัวเลขดิบ BM25 สูงกว่า (0.568 vs 0.511/0.516) แต่ interval ยัง
กว้างพอที่จะไม่ผ่าน Holm-corrected threshold ที่ n=73

**สรุปที่แม่นยำกว่าเดิม**: **BM25 tie กับ dense embedder ที่ดีที่สุด (bge-m3,
Qwen3) และชนะ embedder อื่นๆ ทั้งหมดอย่างมีนัยสำคัญ** — ไม่ใช่ "BM25 ชนะแทบ
ทุก combo" อย่างที่ตัวเลขดิบต่อ chunker (ในหัวข้อก่อนหน้า) ทำให้ดูเหมือน
ตอนยังไม่ได้ทดสอบนัยสำคัญ ข้อสรุปที่ยัง**ยืนกรานได้อยู่**คือ **BM25 (ฟรี ไม่
ต้อง GPU ไม่ต้อง train) ให้คุณภาพเทียบเท่า embedder ที่ดีที่สุดในชุดนี้ได้
โดยไม่มีข้อแตกต่างที่พิสูจน์ได้** — เป็นคำถามที่แรงพอกันแต่ตอบตรงกว่าเดิม
(ไม่ใช่ "dense ไม่คุ้มเลย" แต่เป็น "embedder ที่ดีพอจะคุ้ม ต้องดีในระดับ
bge-m3/Qwen3 เท่านั้น ต่ำกว่านั้นสู้ BM25 ฟรีไม่ได้จริง")

**ยังไม่ได้ทำ (สำคัญก่อนอ้างอิงในเปเปอร์)**:
1. **Hybrid (BM25 + dense ผ่าน RRF)** มีโค้ดอยู่แล้ว
   (`src/rag_lab/retrievers/hybrid.py`, ผูกกับ `[[project_hybrid_routing]]`
   ไปแล้วบางส่วน) — ยังไม่ได้วัดผลบน Gold 73-det ชุดใหม่นี้ คำถามที่น่าสนใจ
   คือ hybrid (BM25+bge-m3 หรือ BM25+Qwen3) จะเก่งกว่าทั้งคู่แยกกันไหม หรือ
   ไม่ต่างเพราะ signal ซ้ำกันเยอะอยู่แล้ว (ตอนนี้รู้แล้วว่า BM25 กับ top-tier
   dense ให้คุณภาพใกล้เคียงกันมาก อาจ correlate สูงจน RRF ไม่ได้ช่วยมาก)
2. Point comparison ต่อ chunker (BM25 เทียบ embedder ที่ chunker เดียวกัน
   โดยไม่เฉลี่ยข้าม chunker) ยังไม่ได้ทดสอบนัยสำคัญแยก — อาจมีรายละเอียดที่
   ต่างจากภาพรวม เช่น semantic×Qwen3 (0.6581) vs BM25×semantic (0.5902)
   ที่ดูต่างกันชัดในตัวเลขดิบ
3. ผลดิบเต็ม: `data/results/gold_bm25_73det_report.md`,
   `data/results/gold_bm25_73det/`,
   `data/results/bm25_vs_embedder_significance_test.md` (ทั้งหมด gitignored)

## Hybrid (RRF: BM25 + Dense) (21 ก.ค. 2569) — ผลบวกชัดเจน, พบ combo ที่ดีที่สุดในทั้งโปรเจกต์

`HybridRetriever` (`src/rag_lab/retrievers/hybrid.py`, RRF ค่า default
`rrf_k=60`) รวม dense + BM25 บน index เดียวกัน (ไม่ต้อง build index ใหม่
เพราะทุก index มี `embeddings.npy` + `lexical.json` อยู่แล้ว) รันครบ
24 combo (`tools/eval/run_gold_hybrid_eval.py`, ~1h23m บน 73 query)

**ค่าเฉลี่ยข้าม chunker ต่อ embedder (recall@10) เทียบ hybrid/dense/bm25**:

| embedder | hybrid | dense เดี่ยว | bm25 เดี่ยว |
|---|---|---|---|
| bge_m3 | **0.6472** | 0.5107 | 0.5676 |
| congen | 0.6426 | 0.4134 | 0.5676 |
| jina_v5 | 0.6383 | 0.4503 | 0.5676 |
| e5 | 0.6264 | 0.4265 | 0.5676 |
| qwen3 | 0.6235 | 0.5155 | 0.5676 |
| m2v | 0.3244 | 0.1472 | 0.5676 |

**Significance test** (`tools/eval/hybrid_significance_test.py`, bootstrap +
Holm แยก 2 family: hybrid vs dense-เดียวกัน, hybrid vs bm25 — ผลดิบเต็ม
`data/results/hybrid_significance_test.md`):

- **hybrid ชนะ dense-alone อย่างมีนัยสำคัญทุก embedder ทุกเมตริก** (Holm-adj
  p<0.01 เกือบทั้งหมด, ยกเว้น qwen3 บน mrr ที่ไม่ผ่าน) — เป็นผลที่ robust
  ที่สุดในทั้ง session นี้ **การเพิ่ม BM25 เข้าไปช่วย dense เสมอ ไม่มีข้อยกเว้น
  (นอกจาก MRR ของ qwen3)**
- **hybrid ชนะ BM25-alone อย่างมีนัยสำคัญที่ recall@10 สำหรับ embedder ที่ดี
  ทั้ง 5 ตัว** (bge_m3, congen, jina_v5, e5, qwen3 — Holm-adj p≤0.015 ทุกตัว)
  **แต่ไม่ค่อยผ่านที่ mrr** (ไม่มีตัวไหนผ่านหลัง correction) **และผ่านแค่
  บางส่วนที่ ndcg@10** (bge_m3/congen/jina_v5/qwen3 ผ่าน, e5 ไม่ผ่านเฉียดๆ)
  — แปลว่า**การเพิ่ม dense เข้าไปช่วย BM25 ตรง recall (เจอเอกสารที่ถูกใน
  top-10 มากขึ้น) มากกว่าช่วยเรื่อง ranking precision (การจัดอันดับให้ตัวที่
  ถูกที่สุดขึ้นอันดับ 1)**
- **m2v ทำให้แย่ลงอย่างมีนัยสำคัญเมื่อ hybrid กับ BM25** (recall diff=-0.243,
  p<0.001) — เป็นตัวอย่าง failure mode ของ RRF จริง: เมื่อสัญญาณหนึ่ง
  (m2v dense, คุณภาพเดี่ยวต่ำมาก 0.147) เกือบสุ่ม การรวมแบบ RRF ที่ให้น้ำหนัก
  เท่ากัน (`rrf_k` เดียวกันทั้งสอง list) จะดึงเอกสารที่ผิดขึ้นมาปนกับ BM25 ที่
  ถูกต้องอยู่แล้ว ทำให้แย่กว่า BM25 เดี่ยวๆ **ข้อควรระวังสำหรับ deployment: RRF
  ไม่ปลอดภัยเสมอไปถ้า embedder อ่อนเกินไป**
- **Combo ที่ดีที่สุดในทั้ง 24+24+4 การทดลองทั้งหมด (dense/hybrid/BM25) คือ
  `semantic × bge-m3 × hybrid`** (recall@10=**0.6845**, mrr=0.8495,
  ndcg@10=0.7264) แซง `semantic × qwen3 × dense-alone` เดิม (0.6581) และ
  `semantic × qwen3 × hybrid` (0.6797) เฉียดๆ — **น่าสนใจว่าตอนเป็น hybrid
  bge-m3 กลับแซง qwen3 ทั้งที่ dense-alone ทั้งคู่ tie กัน** (อาจเป็นเพราะ
  bge-m3 กับ BM25 มี error pattern ที่ complement กันดีกว่า qwen3 กับ BM25 —
  ยังไม่ได้สืบสาเหตุลึกกว่านี้)

**สรุปสำหรับเปเปอร์**: ระบบที่ดีที่สุดโดยรวมคือ **semantic chunking + hybrid
retrieval (BM25+bge-m3 ผ่าน RRF)** ไม่ใช่ dense-alone หรือ BM25-alone
เดี่ยวๆ — ยืนยันว่า lexical กับ dense signal **complement กันจริง ไม่ได้
ซ้ำซ้อนกันอย่างที่สงสัยไว้ก่อนรัน** ยกเว้นกรณี embedder อ่อนเกินไป (m2v)
ที่ RRF กลับทำร้ายมากกว่าช่วย

ผลดิบเต็ม: `data/results/gold_hybrid_73det_report.md`,
`data/results/gold_hybrid_73det/`, `data/results/hybrid_significance_test.md`
(ทั้งหมด gitignored)

## บั๊ก: ConGen/SCT ถูก truncate เนื้อหาทิ้งแบบเงียบๆ (21 Jul, พบระหว่างตรวจ
เรื่อง context-length diversity)

ระหว่างตอบคำถาม user ว่า embedder matrix ครอบคลุมความยาว context ที่หลากหลาย
(512-8192 tokens) พอหรือยัง ไล่เช็ค `sentence_bert_config.json` ของแต่ละโมเดลจริง
(ไม่ใช่เดาจาก model card) พบว่า `kornwtp/ConGen-BGE_M3-model-phayathaibert` และ
`kornwtp/SCT-KD-BGE-M3-model-phayathaibert` (ทั้งคู่ backbone PhayaThaiBERT ที่
`max_position_embeddings=512`) ตั้ง `max_seq_length: 128` ไว้ใน repo ของตัวเอง —
เป็นค่าที่ผู้เขียนโมเดล (kornwtp) กำหนด ไม่ใช่ค่าที่โปรเจกต์นี้ตั้ง

ตรวจสอบเชิงประจักษ์ด้วย tokenizer ของ ConGen เองกับ chunk จริงที่ build ไว้แล้ว:
- **fixed_size** (512 ตัวอักษร): เฉลี่ย 172 tokens (max 515) — **72.5% ของ
  chunk ยาวเกิน 128 tokens** โดนตัดทิ้งแบบเงียบๆ (sentence-transformers ไม่
  error ไม่เตือน)
- **semantic**: เฉลี่ย 164 tokens แต่ max ถึง **3116 tokens** — 33% เกิน 128,
  กรณีเลวร้ายสุดโดนตัดทิ้งเนื้อหาไป 96%

**นี่กระทบทุกผลลัพธ์ ConGen ที่ significance-test ไปแล้วทั้งหมด** รวมถึงข้อสรุป
"ConGen = program-query specialist" ([[project_embedder_comparison]]) — ตัวเลข
เหล่านั้นวัดจากโมเดลที่อ่านได้แค่ 1-4 ประโยคแรกของแต่ละ chunk ไม่ใช่ทั้ง chunk
เหมือนโมเดลอื่น `sct` ยังไม่มีผลลัพธ์ให้แก้เพราะกำลัง build ตอนที่เจอบั๊กพอดี

**แก้แล้ว**: `LocalSTEmbedder` มี parameter `max_seq_length` รองรับ
การ override อยู่แล้ว (`src/rag_lab/embedders/local_st_embedder.py:24`) แค่ไม่
เคยตั้งเพราะไม่รู้ว่า repo นี้ cap ต่ำผิดปกติ สร้าง config ใหม่
`config/experiments/_history/chunker_compare_full_fix_congen_sct_maxseqlen.yaml`
เข้าคิวรันหลัง build ปัจจุบัน (sct + qwen3_0.6b) และ e5-small เสร็จ (GPU เดียว
รันทีละงาน) — เนื่องจาก combo id เป็น hash ของ params ทั้งหมดใน YAML
(`combos.py:BuildCombo.id`) การเพิ่ม `max_seq_length` เป็น key ใหม่ทำให้ได้
combo directory ใหม่ ไม่ทับของเดิม (ของเดิมเก็บไว้เทียบ before/after ได้ แต่
ห้ามอ้างอิงเป็นตัวเลขจริงต่อไป)

**พลาดรอบแรก**: ตั้ง `max_seq_length: 512` ตรงกับ `max_position_embeddings`
ใน config.json ของ backbone — crash ทั้ง 8 combo ด้วย `CUDA error:
device-side assert triggered` (illegal position-embedding index) สาเหตุคือ
RoBERTa สงวน position slot ไว้ 2 ตำแหน่งสำหรับ padding offset ทำให้ตำแหน่งที่
ใช้ได้จริงคือ 512-2=**510** ไม่ใช่ 512 — ยืนยันจาก `tokenizer_config.json`
ของทั้ง congen และ sct ที่ระบุ `model_max_length: 510` ตรงกันทั้งคู่ (เห็นค่านี้
ตอนสำรวจ context-length ตั้งแต่แรกแต่ตอนเขียน config ใช้ค่าจาก config.json แทน
โดยไม่ทันสังเกต) แก้เป็น 510 แล้วรันใหม่สำเร็จ

**ผล before/after (`tools/eval/congen_sct_truncation_fix_eval.py`, paired
bootstrap บน Gold 73-det)**: ไม่ได้เป็นบั๊กเดียวกันทั้งคู่อย่างที่คิด

| model | 128-cap | 510-cap | diff | สรุป |
|---|---|---|---|---|
| sct | 0.1374 | **0.1519** | +0.0144, p<0.0001 | 510 ดีขึ้นจริง — **เปลี่ยนไปใช้ 510** |
| congen | **0.4134** | 0.3836 | -0.0298, p=0.0016 | 510 **แย่ลง** อย่างมีนัยสำคัญ — **คง 128 ไว้เหมือนเดิม** |

**sct**: cap 128 ตัดเนื้อหาทิ้งจริง แก้เป็น 510 ช่วยได้จริง (ตัวเลข sct ชุดแรก
ที่มีใช้ได้)

**congen**: ผลตรงข้าม — ให้ context ยาวขึ้นกลับทำให้แย่ลง เดาว่าเป็นเพราะ
ConGen เป็น pure distillation จาก `paraphrase-multilingual-mpnet-base-v2`
(teacher ที่ปกติใช้กับ input สั้นระดับประโยคคู่) การยืด input ไปถึง 510 tokens
เลยหลุดจาก length distribution ที่โมเดลถูก distill มา เป็น train/test
mismatch ไม่ใช่ปัญหาเนื้อหาถูกตัดทิ้ง **สรุป: cap 128 ของ ConGen ถูกต้องอยู่แล้ว
ไม่ใช่บั๊ก** — ข้อสรุปเกี่ยวกับ ConGen ที่ significance-test ไปก่อนหน้านี้ทั้งหมด
(รวม "program-query specialist") **ไม่ต้องแก้ไขอะไร** วัดถูกต้องมาตั้งแต่แรก

ป้ายเตือนใน `docs/paper-results-summary.md` แก้เป็นสรุปผลจริงแล้ว (section
"Resolved 2026-07-21")

## รัน full 9-embedder matrix (21 Jul, ต่อเนื่องจากบั๊ก ConGen/SCT)

ก่อนรัน significance test เต็มรูปแบบ เจอความเสี่ยงจาก script เดิม
(`embedder_significance_test.py`) — logic ติด label ตาม `type` เฉยๆ ไม่แยกตาม
`model_name` แปลว่าถ้าเอามารันตรงๆ e5-large กับ e5-small (ทั้งคู่ type="e5")
หรือ Qwen3-4B กับ Qwen3-0.6B (ทั้งคู่ type="qwen3") จะถูกนับรวมเป็นตัวเดียวกัน
เงียบๆ เขียน script ใหม่ `tools/eval/embedder_matrix_9way.py` แก้ label ให้
แยกตาม (type, model_name) และ exclude combo ที่ superseded แล้วอย่างชัดเจน
(sct แบบ 128-cap เก่า, congen แบบ 510-cap ที่ปฏิเสธไปแล้ว)

**ผลลัพธ์ significance test เต็ม 9 embedder (36 คู่, Holm-corrected)**:
- **qwen3_0.6b vs qwen3(4B): ไม่ต่างกันนัยสำคัญเลยสักตัวชี้วัด** (Holm p=1.0
  ทุกตัว) — โมเดลเล็กกว่า ~7 เท่า ให้ผลเท่ากัน top tier กลายเป็น 3 ทาง:
  {bge_m3, qwen3, qwen3_0.6b} เท่ากันหมด
- **e5 vs e5_small: ไม่ต่างกันนัยสำคัญเลยเช่นกัน** — ยืนยัน pattern
  "เล็กเท่าใหญ่" เป็นครั้งที่สองจากคนละตระกูลโมเดล ไม่ใช่เรื่องบังเอิญของ
  ตระกูลเดียว **นี่คือ finding ด้าน cost-efficiency ที่แข็งแรงที่สุดในการ
  เปรียบเทียบทั้งหมด** — เลือกตัวเล็กกว่าในทั้งสองตระกูล ประหยัดโดยไม่เสีย
  คุณภาพที่พิสูจน์ได้
- **sct vs m2v: ไม่ต่างกันนัยสำคัญเลย** — แม้แก้ max_seq_length แล้ว sct
  ก็ยังอยู่ bottom tier เท่ากับ m2v (static, ไม่ใช่ transformer ด้วยซ้ำ)
  person recall ของ sct (0.0571) แทบเท่า m2v (0.0572) — ทั้งคู่แทบทำ
  named-entity retrieval ไม่ได้เลย
- qwen3_0.6b นำทุกตัวใน program (0.6396) แต่ยังไม่ได้ significance-test
  แยกตาม entity_type (ยังเหลืองาน)

ผลเต็ม: `data/results/embedder_significance_test_9way.md`,
`data/results/gold_embedder_breakdown_9way.md` บันทึกลง
`docs/paper-results-summary.md` แล้ว (section "Embedders compared (9 total)")

## Per-entity_type significance test สำหรับ 9-embedder matrix (21 Jul,
ต่อจากด้านบน)

เขียน `tools/eval/embedder_significance_test_by_entity_type_9way.py` (import
label/exclusion logic จาก `embedder_matrix_9way.py` ไม่เขียนซ้ำ) รันเลยเพราะ
ผลลัพธ์ที่ persist ไว้จากรอบ aggregate มีครบแล้ว ไม่ต้อง retrieval ใหม่

**ผลลัพธ์สำคัญที่แก้ไข headline "qwen3_0.6b เท่า qwen3" ให้ละเอียดขึ้น**:
- **program**: กลายเป็น 3-way tie {congen, qwen3, qwen3_0.6b} ไม่ต่างกันนัย
  สำคัญเลยสักคู่ แม้ qwen3_0.6b จะมีค่าเฉลี่ยดิบสูงสุด (0.6396)
- **person**: bge-m3 ชนะ qwen3_0.6b อย่างมีนัยสำคัญ (Holm p<0.0001) แต่
  bge-m3 vs qwen3(4B) ยังไม่ต่างกัน (p=0.374 เหมือนเดิม)
- **สรุป**: **มีแค่ qwen3(4B) เท่านั้นที่ ties ทั้ง 2 specialist พร้อมกัน**
  qwen3_0.6b มีจุดอ่อนจริง (person) ที่ qwen3-4B ไม่มี — คะแนนเฉลี่ยรวมที่
  เท่ากันระหว่าง 0.6B กับ 4B เป็นเรื่องบังเอิญจากการเฉลี่ย (ได้ program
  เสีย person พอดีชดเชยกัน) ไม่ใช่หลักฐานว่า 0.6B ฟรีจริง ถ้าไม่มี routing
  หรือเจอ query ผสม ยังควรเลือก qwen3(4B) มากกว่า
- **sct vs m2v tie ทุก entity_type** (person/program/faculty ทั้งหมด Holm
  p=1.0) — ยืนยันว่า sct อ่อนสม่ำเสมอ ไม่ใช่แค่ค่าเฉลี่ยรวมถูกลากลงจาก
  entity_type เดียว

ผลเต็ม: `data/results/embedder_significance_test_by_entity_type_9way.md`
บันทึกลง `docs/paper-results-summary.md` แล้ว (section "Embedder ×
entity_type profile") ยังเหลืองาน: ขยาย BM25/hybrid ให้ครอบคลุม 3 embedder
ใหม่

## ขยาย BM25/hybrid ให้ครอบคลุม 9 embedder (2026-07-21)

BM25 เองไม่ต้องรันซ้ำ (chunker-only ไม่ผูกกับ embedder) แต่ hybrid (RRF) ต้อง
รัน retrieval ใหม่สำหรับ 12 combo ที่ยังไม่มี: `e5_small`, `qwen3_0.6b`, และ
`sct` ที่ config ที่แก้แล้ว (max_seq_length=510 ไม่ใช่ 128 cap เดิม) × 4
chunker. เขียน `tools/eval/run_gold_hybrid_eval_9way_new.py` ให้ระบุ combo
dir ตรงๆ ทั้ง 12 ตัว (ไม่ใช้ substring filter เพราะคำว่า "sct" จะแมตช์ทั้ง
combo เก่า 128-cap ที่ยังอยู่บนดิสก์ด้วย) เขียนผลลง
`data/results/gold_hybrid_73det` ไดเรกทอรีเดียวกับรอบ 6-embedder เดิม
เพื่อให้สคริปต์ significance test ที่ glob ทั้งโฟลเดอร์เจอทั้งเก่าทั้งใหม่

รันเป็น background task ใช้เวลา 2169 วินาที (~36 นาที) เสร็จสมบูรณ์ครบ
12/12 combo × 73/73 query ตรวจสอบแล้ว จากนั้นรัน 2 สคริปต์ significance
test ที่เตรียมไว้ (import label/exclusion logic จาก `embedder_matrix_9way.py`
เหมือนสคริปต์ 9-way ก่อนหน้า กัน type-label ชนกันระหว่าง e5/e5_small และ
qwen3/qwen3_0.6b):

**`bm25_vs_embedder_significance_test_9way.py`** — BM25 vs 9 embedder:
BM25 ยัง tie 3 ตัวบนสุด (bge_m3, qwen3, **qwen3_0.6b** — ตัวใหม่ที่เข้ามา
tie ด้วย) และชนะอย่างมีนัยสำคัญกับที่เหลือทั้งหมดรวม **sct ตัวใหม่** ด้วย
(diff=+0.4158, Holm p<0.0001) เพราะ sct ที่ 510 token มี recall@10 แค่
0.1519 — พอๆ กับ m2v (0.1472) คือใกล้สุ่มทั้งคู่ ไม่ใช่แค่อ่อนกว่าเฉยๆ

**`hybrid_significance_test_9way.py`** — hybrid vs dense-alone / vs BM25-alone,
9 embedder:
- **hybrid ชนะ dense-alone อย่างมีนัยสำคัญทุก embedder ทุก metric** (ยกเว้น
  qwen3 บน MRR) — ผลนี้ทนต่อการขยายเป็น 9 embedder แม้จะรวม sct/m2v ที่
  dense-alone อ่อนมากๆ เข้าไปด้วย ยังไม่เคยเจอกรณีที่ hybrid แพ้ dense-alone
  เลยสักตัว — เป็นข้อค้นพบที่ robust ที่สุดของงานทั้งหมด
- **hybrid ชนะ BM25-alone บน recall@10 อย่างมีนัยสำคัญ 7/9 embedder**
  (qwen3_0.6b, bge_m3, congen, jina_v5, e5, qwen3, e5_small) — **ยกเว้น
  sct กับ m2v ที่ hybrid แพ้ BM25-alone อย่างมีนัยสำคัญทั้งคู่**
  (sct: diff=−0.0497, Holm p=0.0312; m2v: diff=−0.2433, Holm p<0.0001)
  **นี่คือข้อค้นพบใหม่ที่สำคัญ**: เดิมคิดว่า m2v เป็นกรณีพิเศษตัวเดียวที่
  RRF ล้มเหลว (dense signal อ่อนเกินจนลาก hybrid ต่ำกว่า BM25 เดี่ยว) ตอนนี้
  มีหลักฐานยืนยันซ้ำอีกตัว: sct ก็เป็นกรณีเดียวกัน เพราะ dense-alone ของมัน
  ก็อ่อนพอๆ กับ m2v (recall 0.15 ทั้งคู่) — สรุปแพทเทิร์นได้ชัดขึ้น:
  **RRF ล้มเหลวเมื่อ dense signal อ่อนจนเกือบสุ่ม ไม่ว่าจะมาจากสาเหตุอะไร
  (static embedding อย่าง m2v หรือ transformer ที่ training regime ไม่เข้ากับ
  context ยาวอย่าง sct)**
- ตาราง aggregate hybrid recall@10: `qwen3_0.6b` ขึ้นเป็นอันดับ 1 ตัวเลขดิบ
  (0.6543 > bge_m3 0.6472) แต่ยังไม่ได้เช็คระดับ per-chunker ว่า
  `semantic × qwen3_0.6b × hybrid` จะแซง `semantic × bge-m3 × hybrid`
  (0.6845, best combo เดิม) จริงหรือไม่ — ทิ้งไว้เป็น open item #8 ไม่ควรอ้าง
  ว่า qwen3_0.6b ชนะ best-combo เดิมจนกว่าจะเช็ค

บันทึกผลลง `docs/paper-results-summary.md` แล้ว (sections "BM25 lexical
baseline", "Hybrid retrieval", "Methodology", Open item #7 ปิดแล้ว) ผลเต็ม:
`data/results/bm25_vs_embedder_significance_test_9way.md`,
`data/results/hybrid_significance_test_9way.md`. งานส่วน "ขยาย BM25/hybrid
ให้ครอบคลุม 9 embedder" เสร็จสมบูรณ์

## เพิ่ม MAP + Precision@k + multi-k ใน metrics.py (gap-analysis Tier 1 ข้อ #1, 2026-07-21)

เพิ่ม `precision_at_k` (hit resolutions ภายใน window k หารด้วย k ตรงข้ามกับ
`recall_at_k` ที่หารด้วยจำนวน relevant ทั้งหมด) และ `average_precision_at_k`
(AP มาตรฐานวงการ IR: บวก precision ณ ตำแหน่งที่เจอ relevant resolution
แต่ละตัวครั้งแรก แล้วหารด้วยจำนวน relevant ทั้งหมด — ใช้ denominator แบบ
เดียวกับ `recall_at_k` เพื่อความสอดคล้อง) ลงใน `src/rag_lab/metrics.py`

แก้ `evaluate()` ให้รับ `k` เป็น int เดี่ยว (เหมือนเดิม, backward-compatible
100% กับผู้เรียกทั้ง 4 จุดที่มีอยู่) **หรือ** list ของ k (เช่น `[1,3,5,10]`)
เพื่อรายงานหลาย cutoff ในรอบเดียว — output dict จะมี `recall@{k}`,
`precision@{k}`, `ndcg@{k}` ต่อทุก k ที่ขอ บวก `mrr` และ `map` (map คำนวณ
ครั้งเดียวที่ `max(k)` ไม่ใช่ต่อ cutoff เพราะ AP รวมทั้ง ranking อยู่แล้ว)

เพิ่ม unit test 8 ตัวใหม่ใน `tests/test_metrics.py` (precision_at_k 3 ตัว,
average_precision_at_k 4 ตัว, evaluate() หลาย-k/backward-compat 2 ตัว) รัน
`pytest` ทั้ง suite ผ่านครบ 356 ผ่าน (เดิม 348 + ใหม่ 8), skip 3 ตัวเดิม (bge-m3
smoke test ที่ gate ด้วย env var) ไม่มีอะไรพัง

อัปเดต `render_report()` ใน `run_gold_bm25_eval.py`, `run_gold_chunker_eval.py`,
`run_gold_hybrid_eval.py`, `run_silver_chunker_eval.py` (ทั้ง 4 มี body
เหมือนกันทุกตัวอักษร) ให้แสดงคอลัมน์ precision@k และ map เพิ่ม

**ยังไม่ทำ**: ยังไม่มีการรัน eval script จริงด้วย multi-k list (เช่น
`k=[1,3,5,10]`) — เพิ่มแค่ความสามารถ (capability) เข้าไปเฉยๆ ตัวเลขทุกตัวใน
`docs/paper-results-summary.md` ยังคงเป็น k=10 อย่างเดียวเหมือนเดิม การรันจริง
ด้วย multi-k + อ้างตัวเลข MAP/P@k เป็นงานต่อยอดที่ยังไม่ได้ทำ

บันทึกลง `docs/paper-results-summary.md` (Methodology + Open item #4 ปิด),
`project_research_framework_gap_analysis.md` แล้ว

## Cost/latency Pareto table (gap-analysis Tier 1 ข้อ #4, 2026-07-21)

สร้าง `tools/eval/cost_latency_pareto.py` วัด vector dim, ขนาด index บนดิสก์,
embed throughput (จาก `meta.json` ตอน build), และ query latency p50/p95 —
แยก encode time (ขึ้นกับ embedder) กับ search time (ขึ้นกับขนาด index) — ให้
ครบทั้ง dense, BM25, hybrid บน `semantic` chunker (chunker ที่ paper แนะนำ)

**ร่างแรกมีบั๊กเชิงตัวเลข**: ใช้ recall@10 แบบ cross-chunker aggregate (ค่า
เฉลี่ยข้าม 4 chunker) มาจับคู่กับ latency ที่วัดเฉพาะ `semantic` chunker เท่านั้น
— เทียบกันแบบ apples-to-oranges ไม่ใช่ apples-to-apples ตรวจพบเองก่อนส่งอะไรออก
ไปแล้วขอ advisor ช่วยรีวิวก่อนเขียนรายงานจริง — advisor ยืนยันบั๊กนี้และชี้อีก
ประเด็นที่สำคัญกว่า (ดูหัวข้อถัดไป)

**แก้บั๊ก**: เขียนฟังก์ชัน `compute_semantic_quality()` อ่านผลลัพธ์ retrieval
ที่ persist ไว้แล้ว (`data/results/gold_73det_full_embedder_matrix/`,
`gold_hybrid_73det/`, `gold_bm25_73det/`) กรองเฉพาะ combo ที่ chunker=='semantic'
ผ่าน `build_combo_to_chunker_embedder()` จาก `embedder_matrix_9way.py` แล้วคำนวณ
`recall_at_k()` ตรงๆ ไม่ต้องโหลด GPU/embedder ใหม่ เลิกใช้ dict hardcode เดิม
(`_QUALITY_RECALL10`) ไปเลย เพื่อไม่ให้บั๊กแบบนี้เกิดซ้ำถ้ามีคนรัน script ใหม่
ในอนาคต

**ข้อค้นพบใหม่จากการคำนวณตัวเลขที่ถูกต้อง (ปิด Open item #8)**: `qwen3_0.6b ×
semantic × hybrid` recall@10 = **0.6935** — สูงสุดในทั้ง study เลย สูงกว่า
combo ที่เคยอ้างว่าดีที่สุดคือ `bge_m3 × semantic × hybrid` (0.6845) แบบ
ตัวเลขดิบ (+0.009) แต่**ยังไม่ได้ทดสอบนัยสำคัญระดับ per-chunker** (test ที่มีอยู่
เทียบแค่ cross-chunker aggregate กับเทียบแต่ละ embedder กับ
dense-alone/BM25-alone ของตัวเอง ไม่เคยเทียบ embedder ต่อ embedder ภายใน
chunker เดียว) — advisor ชี้ว่า `bge_m3` เองก็ไม่เคยผ่าน significance test มา
ก่อนเหมือนกัน แค่เป็นตัวเลขสูงสุดที่เจอตอนนั้น ดังนั้นจุดยืนที่ถูกต้องคือ
**ไม่ยกให้ใครเป็นแชมป์** — รายงาน top 5 hybrid combo (qwen3_0.6b, bge_m3,
e5_small, qwen3, jina_v5, ช่วง 0.6796-0.6935) เป็น cluster ที่ยังไม่ผ่านการ
ทดสอบ ระบุ qwen3_0.6b ว่าตัวเลขสูงสุด แต่ headline ของ paper ให้เป็นระดับ
system (semantic chunking + hybrid retrieval) ไม่ใช่ embedder ตัวใดตัวหนึ่ง —
ทิ้งเป็น open item ใหม่ (ต้องทำ per-chunker significance test)

**Advisor ชี้ประเด็นสำคัญกว่าบั๊กตัวเลข**: latency ที่วัดได้เป็น
"implementation-bound" ไม่ใช่ต้นทุนที่หลีกเลี่ยงไม่ได้ของวิธีการ ถ้าไม่บอกให้
ชัดเจน กราฟ Pareto จะเล่าเรื่องผิด (ทำให้ดูเหมือน hybrid ต้องแลกด้วย latency
สูงมากเพื่อคุณภาพที่ดีกว่า ทั้งที่จริงๆ ต้นทุนส่วนใหญ่หลีกเลี่ยงได้) จึงเพิ่ม
ฟังก์ชัน `measure_intrinsic_costs()` วัดตรงๆ (ไม่ต้องโหลด GPU) สองจุดที่เคย
เจอมาก่อนหน้านี้ในเซสชันนี้:

1. `DenseRetriever` คำนวณ `row_norms` ของ embedding ใหม่ทุกครั้งที่ query
   (ทั้งที่ corpus ไม่เปลี่ยน) — วัดได้ 39ms (dim=384) / 119ms (dim=1024) /
   287ms (dim=2560) จาก search time รวม ~270-670ms
2. `HybridRetriever` ขอ k=n (ทั้ง corpus 81,489 chunks) จากทั้ง dense และ BM25
   ก่อน fuse+ตัดเหลือ k=10 จริง — และ `BM25Retriever` เองก็สร้าง `BM25Okapi`
   ใหม่ทุก query (rebuild ~1.0s เทียบกับ `get_scores` อย่างเดียว ~43ms หรือ 23
   เท่า) วัดผลรวม: `DenseRetriever.retrieve(k=n)` ~765ms เทียบกับ
   `retrieve(k=10)` ~277ms ส่วนต่าง ~490ms คือค่าใช้จ่ายสร้าง `RankedChunk`
   (พร้อม text เต็ม) ของ chunk หลายหมื่นตัวที่ไม่มีใครดู — สองจุดนี้รวมกัน
   (ไม่ใช่ RRF fusion เอง) อธิบาย latency วัดได้ของ hybrid ~2.3-2.9 วินาที
   เกือบทั้งหมด (intrinsic cost จริงๆ อยู่ที่ ~130-730ms เท่านั้น)

   **ตัวเลขที่ควรใช้พูดถึง overhead นี้คือส่วนต่างแบบบวก ไม่ใช่อัตราส่วน**:
   measured hybrid − intrinsic hybrid อยู่ที่ ~2.15-2.26 วินาที สำหรับทุก
   embedder ทั้ง 9 ตัว (ตัวเลขที่นิ่งที่สุดในทั้งตาราง) ยืนยันว่า overhead
   นี้มาจากการสแกน corpus ทั้งก้อน (ขึ้นกับขนาด corpus ไม่ใช่ embedder) ถ้า
   มองเป็นอัตราส่วนแทน ตัวเลขจะแกว่งมาก — ~4 เท่าสำหรับ `qwen3` (embedder
   แพงสุด: 2914ms/727ms) ไปจนถึง ~18 เท่าสำหรับ `e5_small` (ถูกสุด:
   2326ms/130ms) — ทั้งที่ overhead ที่บวกเข้าไปจริงๆ เท่ากันแทบทุกตัว ฉบับ
   ร่างแรกใช้คำว่า "7-18 เท่า" ซึ่งชี้นำผิด (ตัดตัวเลขต่ำกว่า 7 ทิ้งไปครึ่งนึง
   ของ embedder ทั้งหมด แถมสื่อว่าเป็นความสัมพันธ์เชิงคูณทั้งที่จริงเป็นค่าคงที่
   เชิงบวก) — advisor ชี้จุดนี้ แก้เป็นเน้นตัวเลขบวก ~2.2 วินาทีแทน

รายงานเป็นลักษณะการทำงานจริงของ implementation ปัจจุบัน ไม่ได้แก้โค้ดในรอบนี้
(ตามแนวทางที่ user ให้ไว้ก่อนหน้านี้ในเซสชัน — ให้รายงานสิ่งที่พบ ไม่ใช่แอบแก้เงียบๆ)

**Deliverable**: รายงานเต็ม `data/results/cost_latency_pareto.md`
(gitignored, รันซ้ำได้ด้วย `--reuse-latency-cache` เพื่อไม่ต้องวัด GPU latency
ใหม่ ~20 นาที), ตัวเลข citation-ready ใน `docs/paper-results-summary.md`
(section "Cost / latency characterization" ใหม่, ปิด Open item #3 และ #8),
กราฟ Pareto แบบ interactive (recall@10 vs cost, dense/hybrid series, สลับ
intrinsic/measured ได้) ตีพิมพ์เป็น Claude Artifact (private)

บันทึกลง `docs/paper-results-summary.md`, `project_research_framework_gap_analysis.md`,
`project_hybrid_rrf_eval.md`, `project_embedder_comparison.md`, `MEMORY.md` แล้ว

## Per-chunker (semantic-only) significance test สำหรับ top-5 hybrid combos (2026-07-22)

จากรอบก่อน (`Cost/latency Pareto table`) พบว่า `qwen3_0.6b × semantic × hybrid`
(recall@10=0.6935) มีตัวเลขสูงกว่า `bge-m3 × semantic × hybrid` (0.6845) ที่เคยอ้างว่า
"ดีที่สุด" แต่ยังไม่เคยทดสอบนัยสำคัญแบบ embedder-vs-embedder ภายใน chunker เดียวกันเลย
(การทดสอบก่อนหน้านี้ทั้งหมดเทียบ hybrid_E vs BM25-alone หรือ hybrid_E vs dense-alone_E
ของ embedder เดียวกันเท่านั้น ไม่เคยเทียบ embedder ต่าง ๆ กันเองภายใต้ hybrid+semantic)

เขียนสคริปต์ใหม่ `tools/eval/hybrid_significance_test_semantic_top5.py` — reuse
`build_combo_to_chunker_embedder` / `bootstrap_pvalue` / `holm_correct` จาก
`embedder_matrix_9way.py` เหมือนสคริปต์ significance test อื่น ๆ ในชุดนี้ กรองเฉพาะ
combo ที่ chunker="semantic" และ embedder อยู่ใน top 5 (`qwen3_0.6b`, `bge_m3`,
`e5_small`, `qwen3`, `jina_v5`) แล้วทำ pairwise bootstrap ทั้ง 10 คู่ x 3 metrics
(recall@10, MRR, nDCG@10) ไม่เฉลี่ยข้าม chunker เหมือนสคริปต์เดิม

ใช้ผลที่ query ไว้แล้วจาก `data/results/gold_hybrid_73det/*.json` (ไม่ต้อง retrieve
ใหม่) รันเสร็จในไม่กี่วินาที

**ผลลัพธ์**: ไม่มีคู่ไหนใน 10 คู่มีนัยสำคัญเลยแม้แต่คู่เดียว หลัง Holm correction บน
metric ไหนก็ตาม (raw p ต่ำสุด = 0.065, คู่ qwen3_0.6b vs jina_v5/e5_small บน MRR/nDCG
— ก็ยังไม่ผ่าน alpha=0.05 หลัง correction) ยืนยันว่า top-5 hybrid combo เป็นกลุ่มที่
เสมอกันทางสถิติจริง ไม่ใช่แค่ยังไม่ได้ทดสอบ — ปิดคำถาม "crown neither" ที่ค้างจากรอบ
cost/latency Pareto ได้อย่างสมบูรณ์

**Deliverable**: `tools/eval/hybrid_significance_test_semantic_top5.py`,
`data/results/hybrid_significance_test_semantic_top5.md`

บันทึกลง `docs/paper-results-summary.md` (Open item #8 ปิดสมบูรณ์, section "Top
single-combo tier" อัปเดต), `CLAUDE.md` (bottom line ของ hybrid), `MEMORY.md` แล้ว

## Multi-k report: ปิดเศษที่ค้างของ Tier 1 ข้อ #1 (2026-07-22)

Tier 1 ข้อ #1 (MAP + Precision@k + multi-k) เสร็จไปแค่ครึ่งเดียวตอน 21 ก.ค. — เพิ่ม
`precision_at_k` / `average_precision_at_k` ใน `metrics.py` และให้ `evaluate()` รับ
`k` เป็น list ได้แล้ว แต่ไม่มี eval script ไหนถูกรันซ้ำด้วย multi-k จริง ตัวเลขทุกตัว
ในเอกสารยังเป็น k=10 ล้วน

ก่อนจะรัน เช็คก่อนว่าจะ "แพง" แค่ไหน (user ถามเพราะ quota Pro เหลือน้อย) — พบว่า
`RetrievalResult.persisted` ทุกตัวถูก retrieve มาที่ `top_k=10` อยู่แล้ว (เช็คจาก JSON
จริงใน `data/results/gold_73det_full_embedder_matrix/`) และ `recall_at_k`/
`precision_at_k` ใน metrics.py แค่ filter `rc.rank <= k` จาก list ที่มีอยู่แล้ว — แปลว่า
k∈{1,3,5,10} ทุกค่า **ไม่ต้อง retrieve ใหม่เลย ไม่ต้องใช้ GPU ไม่ต้องเรียก embedding**
เป็นแค่การคำนวณซ้ำจาก JSON ที่มีอยู่แล้วบนดิสก์

เขียน `tools/eval/multi_k_report.py` ใหม่ — โหลดผลที่ persist ไว้แล้วทั้ง 3 ระบบ (dense
9-embedder matrix, hybrid, BM25), เรียก `evaluate(persisted, qrels, k=[1,3,5,10])`
ครั้งเดียวต่อระบบ แล้วเฉลี่ยข้าม 4 chunker ต่อ embedder ตามธรรมเนียมเดิมที่ตารางอื่นๆ
ในเอกสารใช้ ยืนยันความถูกต้องด้วยการเทียบ recall@10/MRR/nDCG@10 ที่คำนวณใหม่กับตัวเลข
เดิมในเอกสาร — ตรงกันทุกตัว รันจริงใช้เวลาไม่ถึงวินาที

**สิ่งที่เจอเพิ่ม (ไม่ใช่แค่ปิด item เดิม)**: ที่ MAP และ precision@1 (ให้น้ำหนักกับ "อันดับ
ที่เจอ relevant ตัวแรก" มากกว่า recall@10) `bge_m3` มี MAP สูงสุดในกลุ่ม hybrid top-3
(0.5224) แต่ `qwen3_0.6b` มี precision@1/nDCG@1 สูงสุด (0.7671) — สวนทางกับ recall@10
ที่เพิ่งพิสูจน์ว่าเสมอกันทั้งคู่ (ดู section ก่อนหน้า) ยังไม่ได้ทดสอบนัยสำคัญบน MAP/precision@1
เลย บันทึกเป็น open item ใหม่ ไม่ใช่ finding ที่ยืนยันแล้ว — ขนาด gap (+0.013 / −0.017)
ใกล้เคียงกับ gap recall@10 (+0.009) ที่เพิ่งพิสูจน์ว่าเป็น noise ล้วนๆ ต้องระวังแบบเดียวกัน

**Deliverable**: `tools/eval/multi_k_report.py`, `data/results/multi_k_report.md`

บันทึกลง `docs/paper-results-summary.md` (Open item #4 ปิดสมบูรณ์ + section "Multi-k
metrics" ใหม่) แล้ว

## เพดาน recall@10 ตามจำนวนคำตอบที่ถูกต้องต่อคำถาม (2026-07-22)

User ถามว่า recall@10 ~0.6-0.7 ต่ำเกินไปไหม และควรเพิ่ม graph/knowledge-graph
เข้าไปในระบบหรือเปล่า — ก่อนตอบ เช็คโครงสร้างของ Gold set ก่อนว่า "0.6" หมายถึงอะไรจริงๆ

พบว่า **Gold set ไม่ใช่ 1 คำถาม = 1 คำตอบที่ถูกต้อง** — เฉลี่ยมีคำตอบที่ถูกต้อง 8.8
ฉบับต่อคำถาม (ต่ำสุด 2, สูงสุดถึง 43!) โดยเฉพาะหมวด `faculty_adjunct_aggregate`
("รายชื่อการแต่งตั้งอาจารย์พิเศษทั้งหมดของคณะ X") ที่เป็นคำถามแบบ list-all จริงๆ
(เฉลี่ย 16.8 ฉบับ/คำถาม) เพราะ `recall_at_k` หารด้วยจำนวนคำตอบที่ถูกต้องทั้งหมด
ไม่ใช่หารด้วยจำนวนที่เจอ — คำถามที่มีคำตอบถูก 43 ฉบับ ต่อให้ retriever สมบูรณ์แบบ
100% ก็ทำ recall@10 ได้สูงสุดแค่ 10/43 ≈ 0.23 (มีที่แค่ 10 slot ใส่คำตอบที่ถูกได้)

คำนวณ "เพดานสูงสุดที่เป็นไปได้" แยกตาม entity_type:

| entity_type | เฉลี่ยคำตอบถูก | เพดาน recall@10 |
|---|---|---|
| person | 6.0 | 0.976 (เกือบเต็ม) |
| program | 8.2 | 0.900 |
| faculty_adjunct_aggregate | 16.8 | **0.681** |
| รวมทั้ง 73 ข้อ | 8.8 | 0.892 |

**สรุปสำคัญ**: ตัวเลข recall@10 ที่ดูต่ำของหมวด faculty_adjunct_aggregate (เช่น
bge_m3 dense-alone 0.4555) ส่วนหนึ่งเป็นข้อจำกัดของตัว metric เอง ไม่ใช่ retriever
แย่ทั้งหมด — ในขณะที่หมวด person มีเพดานสูงเกือบ 1.0 (0.976) แต่ตัวเลขจริงยังอยู่แค่
~0.5-0.6 แปลว่าหมวด person ยังมีช่องว่างที่ปรับปรุงได้จริงมากกว่าที่ตัวเลขดิบข้างๆ
กันดูเหมือน — สองหมวดนี้เทียบกันตรงๆ ไม่ได้โดยไม่คิดเพดานนี้ก่อน

**คำตอบเรื่อง graph**: ยังไม่เคยทดสอบในโปรเจกต์นี้ แต่มีเหตุผลรองรับดี — คำถามหมวด
"list ทั้งหมด" มีเพดานต่ำโดยธรรมชาติเพราะ top-k similarity search ไม่ใช่ paradigm ที่
เหมาะกับคำถามแบบนี้ตั้งแต่ต้น การใช้ entity tagger ที่มีอยู่แล้ว (people.json,
faculties.json, programs.json) ทำ structured lookup ตรงๆ แทน similarity search
น่าจะดันเพดานหมวดนี้ขึ้นไปใกล้ 1.0 ได้ — แต่เป็นแค่ข้อเสนอที่ยังไม่ได้ทดสอบ ไม่ใช่
finding ที่ยืนยันแล้ว บันทึกเป็น candidate direction ใหม่ ไม่ใช่ส่วนหนึ่งของ Tier 1-3
เดิม

**ข้อจำกัดของการวิเคราะห์นี้**: เพดานคำนวณจาก breakdown ของ dense-alone เท่านั้น
(อันเดียวที่มี per-entity_type breakdown) — BM25 และ hybrid ไม่เคยถูกแบ่งตาม
entity_type เลย เลยยังบอกไม่ได้ว่า hybrid เข้าใกล้เพดาน person (0.976) แค่ไหนจริงๆ

**Deliverable**: ไม่มีสคริปต์ใหม่ (คำนวณ inline จาก `config/eval/gold_query_set_73det.yaml`)

บันทึกลง `docs/paper-results-summary.md` (section ใหม่ "Structural recall@10 ceiling
by entity_type" + Open item #9, #10 ใหม่) แล้ว

## BM25 vs embedder แบบ per-chunker — ปิด Open item #1 (2026-07-22)

Open item #1 ที่ค้างมานาน: ตาราง aggregate (เฉลี่ยข้าม 4 chunker ก่อนเทียบ) บอกว่า
BM25 ผูกสถิติเสมอกับ top tier (bge_m3, qwen3, qwen3_0.6b) แต่ตัวเลขดิบรายchunker
ดูเหมือนจะเอียงไปทาง BM25 มากกว่า — ยังไม่เคยเช็คว่าจริงไหม หรือเป็นแค่ artifact ของ
การเฉลี่ย

เขียน `tools/eval/bm25_vs_embedder_significance_test_per_chunker.py` — เหมือน
`bm25_vs_embedder_significance_test_9way.py` เดิม แต่ไม่เฉลี่ยข้าม chunker เลย
แยกเป็น 4 family อิสระ (ต่อ chunker) x 9 คู่เทียบ, Holm-correct แยกกันต่อ family
ต่อ metric ใช้ผลที่ persist ไว้แล้วทั้งหมด รันเสร็จในไม่กี่วินาที

**ผลลัพธ์ — ยืนยันว่าข้อสงสัยเป็นเรื่องจริง ไม่ใช่แค่จินตนาการ**:

1. **`bge_m3`** (ตัวที่ควรจะเสมอ BM25 ทุกที่ตาม aggregate) **แพ้ BM25 อย่างมีนัยสำคัญ
   เฉพาะตอนใช้ chunker แบบ `sentence`** (Holm-adj p=0.0108, diff +0.1152) ส่วน
   `fixed_size`/`recursive`/`semantic` ยังเสมอกันอยู่ — แปลว่า "เสมอ" ใน aggregate
   table จริงๆ แล้วมาจาก semantic chunker (chunker ที่ dense embedder แข็งแรงสุด)
   ที่ไปช่วยดึงคะแนนขึ้น พอเฉลี่ยรวมกับอีก 3 chunker ที่ BM25 ได้เปรียบกว่า เลยดูเหมือน
   เสมอกันตลอด ทั้งที่จริงแล้วเสมอกันแค่ 3 ใน 4 chunker
2. **`qwen3` และ `qwen3_0.6b` เป็น embedder เดียวที่ margin ของ BM25 ติดลบได้** — เฉพาะ
   ตอนใช้ `semantic` chunker (BM25 แพ้แบบไม่มีนัยสำคัญ −0.068/−0.046) ส่วน chunker
   อื่นๆ BM25 ยังนำอยู่ดี (แค่ไม่ถึงระดับมีนัยสำคัญ)

**สรุปสำหรับเปเปอร์**: "BM25 ผูกกับ embedder ระดับบนสุด" ไม่ใช่ความจริงที่คงที่ทุก
chunker — เป็นจริงที่สุดเฉพาะตอนใช้ semantic chunking (chunker ที่โปรเจกต์แนะนำอยู่
แล้ว) และเป็นจริงน้อยที่สุดตอนใช้ sentence chunking ที่แม้แต่ bge_m3 ก็ยังพิสูจน์ไม่ได้
ว่าคุ้มกว่า BM25 ฟรี — ผลนี้ **สนับสนุนข้อแนะนำ semantic chunking เดิมให้แข็งแรงขึ้น**
ไม่ใช่ทำให้ซับซ้อนขึ้น เพราะเป็น chunker เดียวกันที่ทำให้ embedder แพงๆ คุ้มค่าที่สุดด้วย

**Deliverable**: `tools/eval/bm25_vs_embedder_significance_test_per_chunker.py`,
`data/results/bm25_vs_embedder_significance_test_per_chunker.md`

บันทึกลง `docs/paper-results-summary.md` (Open item #1 ปิดสมบูรณ์ + section "Per-chunker
BM25 vs. embedder" ใหม่) แล้ว

## ทำไม bge-m3 แซง qwen3 เฉพาะตอน hybrid — ทดสอบสมมติฐานเดิม ผลออกมาตรงข้าม (2026-07-22)

Open item #2 ค้างมานาน: "bge-m3 แซง qwen3 ตอนเป็น hybrid ทั้งที่เสมอกันตอน dense-เดี่ยว
— เดาว่าเป็นเพราะ error pattern เข้ากับ BM25 ได้ดีกว่า (complementary) แต่ไม่เคยพิสูจน์"
User ถามตรงๆ ว่าทำไม เลยทดสอบสมมติฐานนี้จริงจังแทนที่จะเดาต่อ

เขียน `tools/eval/bge_qwen_bm25_complementarity.py` — ใช้ผล top-10 ที่ persist ไว้แล้ว
(ไม่ต้อง retrieve ใหม่) วัด 3 อย่าง: (1) rescue rate — ในบรรดาคำตอบที่ dense-เดี่ยวพลาด
BM25 (chunker เดียวกัน) เจอกี่ %, (2) union coverage — รวม dense+BM25 แล้วครอบคลุมกี่ %
ของคำตอบทั้งหมด, (3) correlation ของ recall รายคำถามระหว่าง BM25 กับแต่ละ embedder (ยิ่ง
ต่ำ = ยิ่ง complementary)

**ผลลัพธ์ — สวนทางกับสมมติฐานเดิมทั้ง 3 ตัวชี้วัด**: qwen3 มี rescue rate สูงกว่า (0.3189
vs bge_m3's 0.2995), union coverage สูงกว่า (0.6036 vs 0.5823), และ correlation กับ
BM25 ต่ำกว่าใน 3 ใน 4 chunker — ทุกตัวชี้วัดบอกว่า qwen3 ควรจะ "เข้ากับ BM25" ได้ดีกว่า
bge_m3 แต่ผลจริงกลับตรงข้าม (bge_m3 hybrid 0.6472 ชนะ qwen3 hybrid 0.6235)

**สรุป: สมมติฐาน "error-pattern complementarity" ตกไป ไม่ใช่คำอธิบายที่ถูกต้อง**

**ทำไมยังหาสาเหตุจริงไม่เจอ**: วิธีวัดนี้เช็คแค่ "อยู่ใน top-10 ของแต่ละระบบไหม" (เพราะ
ข้อมูลที่ persist ไว้มีแค่ top-10) แต่ RRF จริงๆ คำนวณจาก **rank ที่แน่นอนในทั้ง corpus**
(HybridRetriever ปัจจุบันดึงข้อมูลทั้ง corpus จากทั้งสองฝั่งมาก่อน fuse ไม่ได้ตัดที่
top-10) — สาเหตุจริงน่าจะอยู่ที่ rank-order effect ละเอียดที่มองไม่เห็นจากข้อมูล top-10
เท่านั้น จะขุดต่อได้ต้อง retrieve ใหม่แบบเก็บ rank เต็ม corpus (ไม่ใช่แค่ recompute ฟรี
แบบที่ผ่านมา ต้องโหลดโมเดล bge_m3/qwen3 ขึ้น GPU จริง)

**Deliverable**: `tools/eval/bge_qwen_bm25_complementarity.py`,
`data/results/bge_qwen_bm25_complementarity.md`

บันทึกลง `docs/paper-results-summary.md` (Open item #2: hypothesis REFUTED, root cause
ยังเปิดอยู่) แล้ว

## ปิด Open item #2 จริง — ไม่ต้องขุดสาเหตุ เพราะ effect ไม่มีอยู่จริง (22 ก.ค. 2569)

ผู้ใช้กดยืนยัน "ทำเลย" ให้เดินหน้าขุดสาเหตุจริงด้วย GPU retrieval (full-corpus rank)
ต่อจากที่สมมติฐาน complementarity ตกไปเมื่อกี้ ก่อนเริ่มงานที่มีต้นทุนจริง ทำ 2 เช็คฟรี
ก่อนเพื่อลด scope:

1. **ดู gap เป็นราย chunker** (recall@10, bge_m3 − qwen3, hybrid): fixed_size +0.0209,
   recursive **+0.0371** (มากสุด), sentence +0.0320, semantic +0.0048 (แทบเป็นศูนย์ —
   ตรงกับที่ top-5 test ยืนยันไปแล้วว่าเสมอกัน) → เลือก `recursive` เป็น case study
2. **ดู swing queries** ใน `recursive` (คำถามที่ bge_m3-hybrid กับ qwen3-hybrid จับ
   resolution ได้ไม่ตรงกัน): bge_m3-only ชนะ 37 คำถาม, qwen3-only ชนะ 34 คำถาม — สัดส่วน
   ใกล้เคียงกันมาก และกระจายอยู่เกือบทั้ง 73 คำถาม ไม่ใช่กลุ่มเล็กๆ ที่ตัดได้ง่าย

สัญญาณ 37-34 ที่ใกล้เคียงกันมากคือลายเซ็นของสองระบบที่ "เสมอกันจริง แลกกันแพ้ชนะ" ไม่ใช่
ระบบหนึ่งชนะอีกระบบอย่างเป็นระบบ — ปรึกษา advisor ก่อนเริ่มงานที่มีต้นทุน GPU แล้วได้
ข้อเสนอแนะให้ทดสอบนัยสำคัญของ gap นี้ก่อน (paired bootstrap, bge_m3 vs qwen3 ภายใต้
hybrid) เพราะ premise "bge_m3 แซง qwen3" เองยังไม่เคยผ่านการทดสอบนัยสำคัญเลย — งานนี้
**ฟรี** (recompute จากผลที่ persist ไว้แล้ว top_k=10 พอ ไม่ต้องรัน GPU)

**ผลการทดสอบ** (paired bootstrap n_boot=10,000, seed=42, Holm-correct ข้าม 5 การทดสอบ —
4 chunkers + aggregate):

| chunker | mean(bge−qwen) | raw p | Holm-adj p |
|---|---|---|---|
| sentence | +0.0320 | 0.0474 | 0.2370 |
| recursive | +0.0371 | 0.0572 | 0.2370 |
| aggregate | +0.0237 | 0.1156 | 0.3468 |
| fixed_size | +0.0209 | 0.3178 | 0.6356 |
| semantic | +0.0048 | 0.8238 | 0.8238 |

**ไม่มีคู่ไหนนัยสำคัญเลย แม้แต่ก่อน Holm correction** (ค่าที่ใกล้สุดคือ sentence
raw p=0.047 แต่หลัง Holm-correct กลายเป็น 0.237) สรุปว่า **premise "bge_m3 แซง qwen3
ตอน hybrid" เองไม่เคยเป็นจริงในเชิงสถิติ** — เป็นตัวเลขดิบที่ไม่เคยผ่านการทดสอบ เหมือน
top-5 hybrid tie ก่อนที่จะถูกทดสอบนั่นแหละ ไม่มี effect ให้อธิบาย จึงไม่ต้องเสีย GPU
time ไปขุดสาเหตุที่ไม่มีอยู่จริง — ปิด Open item #2 โดยไม่แตะ GPU เลย

บันทึกลง `docs/paper-results-summary.md` (Open item #2 เปลี่ยนจาก "INVESTIGATED, root
cause unresolved" เป็น "CLOSED, premise was false") แล้ว

---

## RQ3 ablation — รัน build จริงเต็มคอร์ปัสแล้ว (23 ก.ค. 2569)

โค้ดเขียน+smoke-test ไว้แล้วตั้งแต่ 22 ก.ค. (commit `5a06c5b`, ดู
`docs/research-framework-gap-analysis.md` §8 ข้อ 7) — วันนี้รัน build จริงทั้ง 3
ablation แบบ background ทีละตัว (segmentation → chunksize sweep → normalize,
เรียงจากเบาไปหนัก) ครบทั้ง build + significance test, exit code 0 ทุกขั้นตอน
เร็วกว่าที่ประมาณไว้มาก (คาด normalize ~6-7 ชม. จากตัวเลข `semantic × bge-m3`
เก่าก่อนแก้ fragmentation bug — รันจริงเสร็จเร็วกว่านั้นมาก น่าจะเป็นผลจาก
[[project_semantic_chunker_fragmentation]] ที่ปิดไปแล้ว 18 ก.ค. ลด breakpoint
count ลงมาก ยังไม่ได้วัด wall-clock แยกส่วนยืนยัน)

**ผลลัพธ์ทั้ง 3 ablation (paired bootstrap n_boot=10000, seed=42, Holm-correct,
Gold 73-det):**

1. **Normalization** (`normalize_thai_text` ผ่าน loader `normalized`, จับคู่
   `semantic × bge-m3`) — **ไม่มีผลนัยสำคัญเลยสักตัว** (Holm-adj p ≥ 0.414 ทุก
   metric ทั้ง dense/hybrid) ตัวเลขดิบสลับทิศทางกันเองระหว่าง metric (บาง
   metric normalized ดีขึ้นเล็กน้อย บางอันแย่ลงเล็กน้อย) — สรุป: การ normalize
   เลขไทย/วรรณยุกต์ **ไม่ช่วยและไม่ทำร้าย** retrieval quality อย่างมีนัยสำคัญ
   บน combo ที่ทดสอบ
2. **Segmentation** (`fixed_size_wordaware` ตัดขอบคำด้วย newmm เทียบ
   `fixed_size` ตัดดิบ, จับคู่กับ bge-m3) — **ไม่มีผลนัยสำคัญเลยสักตัวเช่นกัน**
   (Holm-adj p = 1.0 ทุก metric) จำนวน/ขนาด chunk แทบไม่ต่างกัน (61,766 vs
   62,018 chunks, mean length 447 vs 439 ตัวอักษร) — สรุป: การ snap ขอบ chunk
   ให้ตรงขอบคำภาษาไทยไม่กระทบผลลัพธ์อย่างมีนัยสำคัญ (การตัดกลางคำแบบดิบไม่ได้
   สร้างความเสียหายมากอย่างที่อาจคาดไว้)
3. **Chunk-size sweep** (256/512/1024, `fixed_size` + bge-m3) — **มีผลจริงและ
   มีนัยสำคัญ** โดยเฉพาะ recall@10: 256 > 1024 อย่างมีนัยสำคัญทั้ง dense
   (Holm-adj p=0.0012) และ hybrid (p=0.0006); 512 > 1024 นัยสำคัญบน dense
   recall@10 (p=0.0132); 256 > 512 นัยสำคัญบน hybrid recall@10 (p=0.0056)
   เท่านั้น (dense 256 vs 512 ไม่นัยสำคัญ) ตัวเลข recall@10 เฉลี่ยไล่ตามขนาด
   ชัดเจน: dense 256=0.510, 512=0.480, 1024=0.395; hybrid 256=0.661,
   512=0.607, 1024=0.570 — **chunk เล็กกว่าดีกว่าอย่างสม่ำเสมอสำหรับ recall**
   แต่ MRR/nDCG@10 ส่วนใหญ่ไม่นัยสำคัญหลัง Holm correction (สัญญาณอ่อนกว่า
   recall มาก)

**สรุปรวม RQ3**: ตัวแปรเดียวที่ทดสอบแล้วมีผลจริงคือ **chunk_size** (เล็กกว่า
ดีกว่าสำหรับ recall) — normalization กับ segmentation ไม่มีผลนัยสำคัญทั้งคู่
ผลลัพธ์เต็มอยู่ที่ `data/results/rq3_{normalize,segmentation}_significance_test.md`
และ `data/results/rq3_chunksize_sweep_report.md`

---

## Cross-encoder reranker (Tier 3 ข้อ 8) — สร้าง+รันจริง, ผลลบมีนัยสำคัญต่อ hybrid (23 ก.ค. 2569)

สร้าง stage ใหม่ `CrossEncoderReranker` (`BAAI/bge-reranker-v2-m3`) เป็น
query-time stage ทางเลือก ต่อจาก `pipeline.retrieve()` — ขยาย candidate pool
จาก retriever (`rerank_pool_size=50`) แล้ว rerank ตัดเหลือ `k=10` จริง
สถาปัตยกรรม: registry pattern เดิม (`reranker_registry`, `BaseReranker`
มิเรอร์ `BaseRetriever`), ไม่แตะ `runner.py`/`combos.py` (retrieval เป็น
query-time only เหมือน retriever เดิม) เพิ่ม unit test 16 ตัว (รวม 372 ตัว
ผ่านหมด) รายละเอียด implementation เต็มอยู่ใน plan ที่อนุมัติ (session นี้)

**ทดสอบจริง** (paired bootstrap, Holm-corrected, Gold 73-det, semantic×bge-m3
combo `plain__fixed_size__local__ceea7536`, script
`tools/eval/reranker_significance_test.py`):

| retriever | metric | ไม่ rerank → rerank | Holm-adj p | ทิศทาง |
|---|---|---|---|---|
| hybrid | MRR | 0.848 → 0.760 | 0.006 | **แย่ลงมีนัยสำคัญ** |
| hybrid | nDCG@10 | 0.675 → 0.617 | 0.030 | **แย่ลงมีนัยสำคัญ** |
| hybrid | recall@10 | 0.607 → 0.584 | 1.000 | แย่ลงแต่ไม่มีนัยสำคัญ |
| dense | ทั้ง 3 metric | — | ไม่มีนัยสำคัญ | ไม่มีผล |

ยืนยันแล้วว่าไม่ใช่บั๊ก implementation (smoke test ด้วยมือ: reranker ให้คะแนน
สมเหตุสมผลจริง — chunk เรื่องค่าเทอมได้คะแนนสูงสุดสำหรับ query ลดค่าเทอม)
latency ของ reranker เอง (ไม่รวม model load) ที่ pool=50: p50 1191ms, p95
1522ms — ไม่ถูกด้วยซ้ำ ยิ่งตอกย้ำว่าไม่คุ้ม

**ส่งงานวิจัย literature review ไปหาคำอธิบาย** (background agent, primary
source เท่านั้น) ผลอยู่ที่ `docs/reranker-hybrid-interaction-research.md`
ประเด็นสำคัญ: paper "Drowning in Documents" (ReNeuIR 2025, arXiv:2411.11767)
ทดสอบ **`bge-reranker-v2-m3` ตัวเดียวกับที่เราใช้** พบว่า underperform
baseline ที่แข็งอยู่แล้วบ่อยครั้ง ตั้งชื่อ failure mode "phantom hits" (ให้
คะแนนสูงมั่นใจกับเอกสารที่ไม่เกี่ยวเลย) ตรงกับ fingerprint ผลของเรา (MRR/nDCG
พัง แต่ recall@10 รอด) นอกจากนี้ paper RRF ต้นฉบับ (Cormack et al. 2009) และ
paper HYRR (Google, arXiv:2212.10528) ให้กลไกเพิ่มเติม: RRF ปกป้องสัญญาณ
lexical โดยโครงสร้าง (ไม่สนใจคะแนนดิบ ใช้แค่ rank) ซึ่ง cross-encoder มองไม่
เห็นเลย และ off-the-shelf reranker อาจไม่ transfer ดีกับ candidate
distribution ของ hybrid retriever ที่ไม่เคยถูก train ด้วย **ไม่มี paper ไหน
ทดสอบ pipeline แบบเราเป๊ะๆ (RRF hybrid แล้วค่อย rerank)** — เป็น data point
ใหม่จริง ระบุไว้ตรงๆ ไม่ได้อ้างว่าเป็นการ replicate ของเดิม

**สรุปสำหรับเปเปอร์**: ไม่ควร rerank เส้นทาง hybrid ด้วยการต่อสายแบบปัจจุบัน
— แนวทางทดลองต่อที่วรรณกรรมชี้ไว้ (ยังไม่ได้ทำ): train/validate reranker บน
hybrid-fused candidates โดยเฉพาะ หรือเอาคะแนน reranker ไปเป็น "ระบบที่ 4" ใน
RRF แทนการ truncate-แทนที่ตรงๆ ปิด Tier 3 ข้อ 8 — เหลือแค่ RQ4 (end-to-end
RAG) ที่ยังไม่เริ่มใน Tier 3

---

## บั๊ก corpus discovery ปนเปื้อนทุก full-corpus index ที่เคยสร้างมา (23 ก.ค. 2569)

พบระหว่างการ build index `entity_tags_full` ใหม่ (ไม่เกี่ยวกับ eval หลัก
โดยตรง แต่ build ค้าง 8+ ชั่วโมงที่ไฟล์สุดท้าย ทำให้ต้องสืบสาเหตุ) —
`runner.py::_discover_paths` (ฟังก์ชันที่ full-corpus experiment ทุกตัวใช้หา
ไฟล์ corpus) ทำ `Path(input_dir).rglob("*.md")` แบบไม่กรองอะไรเลย ต่างจาก
`loaders/common.py::iter_corpus_files` (ที่ `tools/corpus_prep/tag_*.py` ใช้)
ซึ่งมี comment เตือนอันตรายนี้ไว้อยู่แล้วแต่ไม่เคยถูกเอาไปใช้ใน `runner.py`
จริง

`academic_resolutions/` มีไฟล์รายงานเครื่องมือ (gitignored) ปนอยู่ ~19 ไฟล์
(`llm_ocr_scan/`, `llm_thematic_scan/`, `entity_tags/`,
`ocr_repetition_review.md`) ที่ path ไม่ตรง pattern `<ปี>/<ครั้งที่>/ไฟล์.md`
จริง — `make_resolution_id` มี fallback เงียบ (คืน path ดิบแทนการ throw error)
สำหรับ path ที่ไม่ตรง pattern เลยทำให้ไฟล์พวกนี้หลุดเข้าไปเป็น "มติปลอม" ใน
ทุก full-corpus build ที่เคยรันมา โดยไม่มี error ใดๆ ให้เห็น

**ตรวจสอบจริงใน index ที่ build ไว้แล้ว** (`data/index/chunker_compare_full/`
— index ที่อยู่เบื้องหลังตัวเลขหลักทั้งหมดใน `docs/paper-results-summary.md`):
chunk ที่มาจากไฟล์ปลอมพวกนี้คิดเป็น **6.87% (fixed_size), 7.03% (recursive),
8.25% (semantic)** ของ chunk ทั้งหมด ไฟล์เดียว (`consensus_priority.md`
รายงาน OCR-scan ขนาด 637KB) สร้าง chunk ถึง 1,517 chunk ใน index semantic —
ราว 50 เท่าของสัดส่วนที่เอกสารจริงทั่วไปควรมี อัตราปนเปื้อนใกล้เคียงกันในทุก
chunker เลยไม่น่าจะเป็นสาเหตุหลักที่ทำให้ semantic ชนะการเปรียบเทียบ
chunker แต่ก็เป็น noise จริงที่แฝงอยู่ในทุกตัวเลขที่รายงานไปแล้ว (9-embedder
matrix, hybrid/BM25 test, reranker eval, RQ3 ablation — ทั้งหมด build ก่อน
บั๊กนี้จะถูกแก้)

**แก้แล้ว** (commit `8c86b63` + ตามด้วยแก้ `cli.py::build` ที่มี pattern
เดียวกันเป๊ะในเซสชันเดียวกัน): เปลี่ยนให้กรองด้วย `parse_path()` ต้องได้ปี+
ครั้งที่จริงก่อนจะรวมไฟล์เข้าไป (ไม่ใช้ `iter_corpus_files`'s relative-to-root
gate ตรงๆ เพราะ `dev_smoke.yaml` ชี้ `input_dir` เข้าไปในโฟลเดอร์ปีอยู่แล้ว จะ
กรองเกิน) ยืนยันแล้ว: `_discover_paths` เจอไฟล์มติจริงครบ 2,853 ไฟล์ ไม่มีการ
ปนเปื้อนเลย test suite ผ่านหมด

**ยังไม่ได้ rebuild index ประวัติศาสตร์** — ลอง build ใหม่ 1 combo (semantic
+ bge-m3) บน corpus ที่กรองแล้วจริง ใช้เวลา **1 ชั่วโมง 17 นาที** (ส่วนใหญ่คือ
`embedder.embed()` แบบ bulk ครั้งเดียวกับ ~70,789 chunk ที่ `batch_size=8` —
ค่า default ที่ตั้งใจให้เล็กเพื่อความปลอดภัยด้าน VRAM) การ rebuild ทั้งชุด
(4 chunker × 9 embedder ≈ 30+ combo) ตามจริงน่าจะเป็นงานหลายวัน สอดคล้องกับที่
การ sweep 9-embedder ครั้งแรกเองก็ต้องใช้ config `_resume*` แยกกันถึง 9 ไฟล์
(เป็นงาน background ที่ทำต่อเนื่องหลายรอบอยู่แล้ว ไม่ใช่ครั้งเดียวจบ) —
ผู้ใช้ตัดสินใจ **เลื่อนการ rebuild ไปก่อน** (23 ก.ค. 2569) เนื่องจากอัตรา
ปนเปื้อนต่ำและใกล้เคียงกันในทุกเงื่อนไขที่เปรียบเทียบ ไม่น่าจะพลิกผลสรุปเชิง
คุณภาพของงานวิจัย

---

## Rebuild index ประวัติศาสตร์ทั้งชุดเสร็จสมบูรณ์ (24-25 ก.ค. 2569)

ผู้ใช้เปลี่ยนใจจากการตัดสินใจ "เลื่อนไปก่อน" ด้านบน — ขอให้ rebuild
`data/index/chunker_compare_full/` ทั้งชุด (4 chunker × 9 embedder = 36
combo) แบ่งเป็น 4 batch ตาม chunker (`fixed_size`, `recursive`, `sentence`,
`semantic` — เรียงจากเร็วไปช้า) เพื่อให้งานหลายวันที่ประเมินไว้ก่อนหน้า
จัดการได้จริง configs อยู่ที่ `config/experiments/_history/rebuild_clean_*.yaml`
(commit `2d36663`)

**ผลลัพธ์**: ทั้ง 4 batch เสร็จสมบูรณ์และตรวจสอบแล้วว่าไม่มีการปนเปื้อน
(0 chunk จากไฟล์ปลอมในทุก 36 combo) — batch 1-2 (`fixed_size`, `recursive`)
รันจบในรอบเดียวไม่มีปัญหา batch 3-4 (`sentence`, `semantic`) โดนขัดจังหวะ
กลางคันจากเครื่องเข้าโหมด sleep ภายนอก (ไม่ใช่ crash หรือบั๊กของโค้ด) —
กู้คืนได้โดยไม่เสียงานที่ทำไปแล้ว เพราะ `pipeline.build_index` เขียนไฟล์
ผลลัพธ์ของแต่ละ combo แบบ atomic ทีเดียวตอนเสร็จเท่านั้น (ไม่มีไฟล์ค้างครึ่งๆ
กลางๆ) จึงสร้าง config `_resume.yaml` ที่ครอบคลุมเฉพาะ combo ที่ยังไม่เสร็จ
แล้ว rerun ต่อโดยไม่ต้อง chunk/embed ซ้ำ

**สิ่งที่ค้นพบระหว่างทาง (ไม่ใช่บั๊ก)**: combo ที่ใช้ Qwen3-4B บน chunker ที่มี
long-tail ยาว (`sentence`, `semantic` — chunk ยาวสุดถึง ~17,000-18,000
ตัวอักษร) จะช้าผิดปกติช่วงเริ่มต้น (~15-125 วินาที/batch แทนที่จะเป็น ~1
it/s) เพราะ `sentence_transformers.encode()` เรียง input จากยาวไปสั้นก่อน
batch (`length_sorted_idx`) รวมกับ Qwen3 มี attention cost O(seq_len²) แบบ
ไม่มี flash attention — เลยประมวลผล chunk ที่ยาวที่สุดของทั้ง corpus ก่อน
เป็นกลุ่มแรก แล้วค่อยเร่งขึ้นเรื่อยๆ เป็น 3-4 it/s ตอนท้าย พฤติกรรมนี้ถูก
เข้าใจผิดว่าเป็น GPU/VRAM contention หลายรอบก่อนจะยืนยันจาก source code ของ
`sentence_transformers` โดยตรง — ไม่ใช่ปัญหาจริง เป็น cost ที่คาดหมายได้และ
front-load ไว้ตอนต้นเท่านั้น

**สถานะ**: `docs/paper-results-summary.md`'s Open item 11 และ caveat ด้านบน
ของไฟล์นั้นอัปเดตแล้วว่า index สะอาดแล้ว แต่ **ตัวเลขในเอกสารนั้นยังไม่ได้
regenerate จาก index ใหม่** — เป็นงานต่อยอดแยกต่างหาก (rerun
`tools/eval/embedder_matrix_9way.py` และสคริปต์พี่น้อง) ที่ยังไม่ได้ทำ ณ
25 ก.ค. 2569 (ผู้ใช้ยังไม่ได้ขอให้ทำ ต้องถามก่อน)

## Eval refresh จาก index สะอาดเสร็จสมบูรณ์ (25 ก.ค. 2569)

ผู้ใช้ขอให้ update เอกสารทั้งหมด (รวม `CLAUDE.md`) แล้วรัน eval script ต่อ
เลย — รัน 7 สคริปต์เรียงกันเป็น chain เดียวใน background
(`embedder_matrix_9way.py` → `run_gold_bm25_eval.py` →
`run_gold_hybrid_eval.py` → `run_gold_hybrid_eval_9way_new.py` →
`embedder_significance_test_by_entity_type_9way.py` →
`bm25_vs_embedder_significance_test_9way.py` →
`hybrid_significance_test_9way.py`) จบทั้งหมด exit code 0 ไม่มี error

**พบระหว่างทาง (ไม่ใช่บั๊ก แต่กระทบเวลาที่ใช้)**: `run_gold_hybrid_eval.py`
ไม่ได้จำกัดแค่ 6 embedder เดิมตามที่คาดไว้ก่อนรัน — `--embedder-filter`
default เป็น string ว่างซึ่ง match ทุก combo และมันสแกน
`data/index/chunker_compare_full` ทั้งโฟลเดอร์ ซึ่งตอนนี้มีครบ 9 embedder
แล้วหลังขยายจาก 6 — เลยกวาดครบทั้ง 36 combo เองในตัว (ไม่ใช่แค่ 24)
ทำให้ `run_gold_hybrid_eval_9way_new.py` (script ที่ตามมา ซึ่ง hardcode
รายชื่อ 12 combo ของ 3 embedder ใหม่ไว้) มาประมวลผลซ้ำ 12 combo ที่ script
ก่อนหน้าทำไปแล้ว — ไม่มีอันตราย (idempotent, เขียนทับไฟล์เดิมด้วยค่าเดิม)
แค่เสียเวลาเพิ่ม ~35-40 นาทีโดยไม่จำเป็น ไม่ได้แก้ไขอะไรเพราะไม่ใช่บั๊ก

**ผลลัพธ์: ทุก conclusion หลักที่เคยสรุปไว้ (pre-rebuild) ยังคงอยู่ ไม่มี
อันไหนพลิกกลับ** ตรวจสอบทีละข้อจริงจากรายงานใหม่:

1. **semantic + hybrid ยังคงเป็นระบบที่ดีที่สุดโดยรวม** — และแข็งแกร่งขึ้น
   ด้วยซ้ำ: combo ที่ดีที่สุดตอนนี้คือ `semantic × qwen3_0.6b` ที่
   recall@10=0.7048 (เดิม 0.6935) นำห่างจาก chunker อื่นชัดเจน
   (`recursive` ดีสุด 0.6800, `sentence` 0.6529, `fixed_size` 0.6322)
2. **hybrid ชนะ dense-alone แทบทุก embedder ทุก metric** — 26/27 test
   มีนัยสำคัญ (ข้อยกเว้นเดียวคือ qwen3 บน MRR, Holm-adj p=0.09) เหมือนเดิม
3. **BM25 tie กับ embedder tier บนสุด (bge_m3/qwen3/qwen3_0.6b) และชนะตัว
   ที่อ่อนกว่า** — เหมือนเดิม แต่ "hybrid ชนะ BM25 บน recall@10" ลดจาก
   7/9 เหลือ 6/9 embedder (jina_v5 หลุดจากนัยสำคัญ Holm-adj p=0.056)
4. **m2v/sct ยังคงทำให้ hybrid แย่ลงเมื่อ RRF กับ BM25** — m2v มีนัยสำคัญ
   ทั้ง 3 metric เหมือนเดิม แต่ **sct อ่อนลงเล็กน้อย**: ยังมีนัยสำคัญบน
   MRR/nDCG@10 แต่ recall@10 ไม่ถึงนัยสำคัญแล้ว (Holm-adj p=0.082 เดิม
   p=0.031 ที่มีนัยสำคัญ) — ทิศทางเดิม แค่ความรุนแรงลดลงเล็กน้อยหลังล้าง
   ข้อมูลปนเปื้อนออก
5. **3-way tie ของ dense-alone (bge_m3/qwen3/qwen3_0.6b)** และ **program
   3-way tie (congen/qwen3/qwen3_0.6b)** — ตรวจสอบทีละคู่จริงจากตาราง
   ใหม่แล้ว ยังคง tie เหมือนเดิมทุกคู่ ไม่มีอันไหนพลิก

**สิ่งที่ยังไม่ได้ verify** (ไม่ได้อยู่ใน 7 script ที่รันรอบนี้): การ
เทียบ top-5 hybrid combo เฉพาะ semantic chunker
(`hybrid_significance_test_semantic_top5.py`) และ per-chunker BM25 breakdown
(`bm25_vs_embedder_significance_test_per_chunker.py`) — สองอันนี้ยังอ้างอิง
ตัวเลข pre-rebuild อยู่ ไม่คาดว่าจะพลิก (เพราะทุกอย่างอื่นที่ตรวจแล้วไม่พลิก)
แต่ยังไม่ได้ยืนยันจริง

**อัปเดตเอกสารครบแล้ว**: `docs/paper-results-summary.md` (ตัวเลข headline
ทุกส่วน + caveat แก้เป็น "resolved"), `CLAUDE.md` (bottom-line ย่อหน้า +
ลบ caveat ออก), memory `project_corpus_discovery_contamination_bug`
("Still outstanding" ปิดแล้ว) — บั๊ก corpus-discovery contamination ที่พบ
23 ก.ค. ถือว่าปิดสมบูรณ์ 100% ตั้งแต่ต้นทาง (พบ+แก้โค้ด) ถึงปลายทาง
(rebuild index สะอาด + regenerate ตัวเลขที่อ้างอิงในเอกสาร) ณ วันนี้

## Sub-analysis 2 ตัวสุดท้ายรันเสร็จเช่นกัน (25 ก.ค. 2569, บ่ายวันเดียวกัน)

ผู้ใช้ขอให้ทำ sub-analysis 2 ตัวที่เหลือค้างจากรอบเช้า
(`hybrid_significance_test_semantic_top5.py`,
`bm25_vs_embedder_significance_test_per_chunker.py`) ทั้งสองสคริปต์อ่านจาก
persisted results (`gold_hybrid_73det`, `gold_bm25_73det`, dense results) ที่
ถูก refresh ไปแล้วในรอบเช้า — ไม่ต้อง rebuild อะไรเพิ่ม แค่รันสคริปต์ใหม่
(pure recompute, วินาทีเดียวเสร็จ)

**ผลลัพธ์ — ทั้งคู่ยืนยันข้อสรุปเดิม เกือบทั้งหมด**:

1. **semantic-only top-5 tie test**: ยัง tie กันจริง ไม่มีคู่ไหนมีนัยสำคัญ
   บน metric ไหนเลย หลังแก้ Holm (raw p ต่ำสุด=0.034, qwen3_0.6b vs
   jina_v5) ค่าเฉลี่ยใหม่: qwen3_0.6b 0.7048, bge_m3 0.6893, e5_small
   0.6871, qwen3 0.6832, jina_v5 0.6703 (recall@10) — เข้มแข็งขึ้นกว่าเดิม
   ทุกตัว แต่โครงสร้าง "tie cluster" เหมือนเดิม
2. **per-chunker BM25 breakdown**: ข้อสรุปทั้ง 2 ข้อเดิมยังจริง — bge_m3
   แพ้ BM25 อย่างมีนัยสำคัญเฉพาะภายใต้ sentence chunking (Holm-adj
   p=0.0354, เดิม p=0.0108); qwen3/qwen3_0.6b เป็น embedder เดียวที่ BM25
   ตกเป็นรองในเชิงตัวเลข และเกิดเฉพาะใน semantic chunking เท่านั้น
   **มีจุดพลิกจุดเดียว**: `congen` ภายใต้ recursive chunking เดิมมี
   นัยสำคัญ (BM25 ชนะ) แต่หลัง refresh กลายเป็นไม่มีนัยสำคัญแล้ว (Holm-adj
   p=0.0620) — ทิศทางเหมือนเดิม (BM25 ยังนำอยู่ในเชิงตัวเลข +0.1490)
   แค่ไม่ผ่านเกณฑ์ 0.05 หลังล้างข้อมูลปนเปื้อนออก

**สรุป**: จากทั้งหมด 7+2 = 9 สคริปต์ eval ที่มีอยู่ในโปรเจกต์นี้ ตอนนี้รัน
ครบทุกตัวกับ index สะอาดแล้ว ไม่มีตัวเลขไหนในเอกสารที่ยังอ้างอิง
pre-rebuild index อีกต่อไป (ยกเว้น `cost_latency_pareto.py` ซึ่งเป็นสคริปต์
วัด cost/latency ไม่ใช่ recall — ยังไม่ได้รันใหม่ แต่ flag ไว้ชัดเจนใน
เอกสารแล้วว่าเป็นคนละกลุ่มกับ eval chain นี้)

**อัปเดตเอกสารเพิ่มเติม**: `docs/paper-results-summary.md` (แก้ caveat
ทั้ง 2 จุดที่เคยบอกว่า "not part of refresh" เป็นผลลัพธ์จริง, อัปเดตตาราง
per-chunker BM25, อัปเดต Open item #1/#8), `CLAUDE.md` (แก้ semantic-top5
caveat, แก้ตัวเลข "beats BM25 for 7/9" ที่ผิดเป็น "6/9" ให้ตรงกับข้อมูลจริง
พร้อมเพิ่มย่อหน้า per-chunker nuance), memory
`project_corpus_discovery_contamination_bug` (ปิดสมบูรณ์รวม sub-analysis
ทั้ง 2) — รอบเช้า commit ไปแล้วที่ `8ac948e`, การแก้ไขรอบนี้ (sub-analysis
2 ตัว) commit แยกเป็นก้อนใหม่.

## Course-comparison table compaction: root-cause ของ chunk ก้อนใหญ่ที่สุด แก้บางส่วนแล้ว ยังไม่ wire (25 ก.ค. 2569)

ต่อจากที่พบไว้ก่อนหน้า (chunk ใหญ่สุดในคลัง 17,077 ตัวอักษร มาจากตาราง
เทียบรายวิชาเดิม/ใหม่ในเอกสารปรับปรุงหลักสูตร — ดู
`docs/preprocessing-review` ใน memory) ผู้ใช้ขอให้ทดลองตัดตารางประเภทนี้จริง
ก่อนตัดสินใจว่าจะ integrate เข้า pipeline หรือไม่

**ฟังก์ชันใหม่**: `strip_course_comparison_tables` + `_compact_course_table`
ใน `src/rag_lab/loaders/common.py` (commit `71764a8`) — ไม่ใช่การลบตารางทิ้ง
แบบ `strip_mapping_tables` เดิม แต่เป็นการ**บีบ**เหลือ `CODE Title` บรรทัดละ
1 วิชา ยึด `<td>` cell boundary จริง (cell ของรหัส → cell ถัดไปคือชื่อวิชา
สั้นๆ, คำอธิบายยาวอยู่คนละ `<tr>` เสมอ — ตรวจสอบกับ HTML จริงแล้ว) ไม่ใช่
flat char-window (วิธีแรกที่ลองแล้วได้ fragment คำอธิบายที่ตัดกลางคำ ไม่มี
ประโยชน์)

**บั๊กจริง 2 จุดที่พบระหว่างทดสอบ**:
1. แถวที่ซ้ำกันหลายสิบครั้ง (เช่น รหัส `20626214` ซ้ำ 15 ครั้ง) **ไม่ใช่ OCR
   อ่านผิด** — ตรวจ raw HTML แล้วพบว่าเนื้อหาที่ซ้ำแต่ละครั้งต่างกันเล็กน้อย
   (ไม่ใช่ copy เป๊ะ) เป็น bug ของตัวแปลง PDF→markdown ที่ทำ `rowspan` โดย
   emit cell เดิมซ้ำทุกแถวที่ span แทนที่จะ merge ครั้งเดียว — แก้ด้วยการ
   dedupe ตาม code (เก็บครั้งแรกที่เจอ)
2. header marker ที่ใช้ตรวจจับตาราง ("รหัส/หน่วยกิต", "Title and Course
   description", "เปลี่ยนเป็น") หลวมเกินไป — "เปลี่ยนเป็น" เป็นคำไทยทั่วไป
   ("เปลี่ยนเป็น" = "changed to") ไปจับตาราง MoA/ค่าธรรมเนียมที่ไม่เกี่ยวกับ
   วิชาเลย แก้ด้วยการเพิ่มเงื่อนไขว่าตารางต้องมีรหัสวิชา/credit-tuple จริง
   ด้วย ไม่ใช่แค่ header ตรงกัน — ยืนยันด้วย full-corpus batch check: 15
   ไฟล์ที่แตะทั้งหมดเป็นเอกสารปรับปรุงหลักสูตร/แก้ไขรหัสวิชาจริง ไม่มี false
   positive เหลือ

**ผลข้างเคียงที่ไม่คาดคิด**: ผู้ใช้ขอให้ตัด credit-tuple (`2 (1-2-3)`) ออก
จากบรรทัดที่บีบด้วย เพราะไม่จำเป็น — พอตัดแล้วพบว่า `CODE Title` (ไม่มีตัวเลข
คั่นกลาง) ผ่านเงื่อนไข "ตามด้วยตัวอักษร" ของ `match_courses` ได้เองพอดี ซึ่ง
raw HTML เดิม (`CODE<br/>credit-tuple`) ไม่เคยผ่านเงื่อนไขนี้เลย — แปลว่า
รหัสวิชาในตารางเทียบวิชาไม่เคยถูก tag ใน `metadata['courses']` เลยตั้งแต่
ต้น (คนละปัญหาจาก course-table stripping) วัดผล: รหัสวิชาที่ tag ได้เพิ่ม
จาก 49 → 131 รหัส ใน 15 ไฟล์ที่แตะ โดยไม่ต้องแก้ `match_courses` เลย ถ้า
strip รันก่อน tag (ลำดับสำคัญ, บันทึกไว้ใน docstring)

**ผลรวม**: 15 เอกสารในคลัง, 200,476 → 139,345 ตัวอักษร (ลด 30.5%) 10 unit
test ใน `tests/loaders/test_loaders_common.py`

**ยังไม่ wire เข้า loader/config ไหนเลย** — ตาม norm ของโปรเจกต์นี้ (วัดผล
ก่อน ตัดสินใจทีหลัง) ก่อน integrate จริงต้อง (1) eval รวม thematic query
set ด้วย ไม่ใช่แค่ course-name gold query (course-name query ไม่แตะ
description เลยจึงตรวจจับ regression จากการตัด description ไม่ได้ — เป็น
ความเสี่ยงเดียวกับที่ reranker เคยพังแบบมองไม่เห็นมาก่อน) (2) build index
คู่ขนานแยกจาก `chunker_compare_full` เดิม ไม่ทับของเดิม

**ยังไม่ทำ**: ตารางประเภทอื่นที่ผู้ใช้ระบุว่าพบบ่อยเหมือนกัน (เช่น ตารางเทียบ
กรรมการหลักสูตร) ต้องมี detector แยก ยังไม่ได้ตรวจโครงสร้างจริง — ระหว่างที่
ตรวจตารางประเภทอื่นๆ (แต่งตั้งอาจารย์บัณฑิต, อาจารย์พิเศษเกิน 50%,
เปลี่ยนอาจารย์ผู้รับผิดชอบหลักสูตร) กลับไปเจอบั๊กจริงใน `match_people` แทน
(title "อ." ไม่รองรับเลย) ดูรายละเอียดที่
`docs/entity-extraction-and-gold-eval-log.md` § 6 (commit `a4e250e`)

## Re-eval หลัง OCR-remediation rebuild 28-29 ก.ค.: เจอ BM25/hybrid ค้าง 3 วัน แก้แล้ว ตัวเลข hybrid/BM25 พลิกจริง (29 ก.ค. 2569)

ต่อจาก kernel-A OCR remediation rebuild (36 combo เต็ม, ดู
`docs/llm-ocr-scan-log.md`) ที่เพิ่งเสร็จ 28 ก.ค. รัน eval suite 5 สคริปต์
ตามแผนเดิม (`embedder_matrix_9way.py` + 4 สคริปต์ significance test) —
ผลลัพธ์แรกดูเหมือนมี conclusion พลิกหลายจุด (qwen3_0.6b แซง bge_m3/qwen3
แบบ cross-chunker, hybrid แพ้ dense-alone เองสำหรับ qwen3_0.6b) ซึ่งแปลกพอที่
จะหยุดเช็คก่อนเขียนเอกสารต่อ — ปรึกษา advisor แล้วชี้จุดที่ต้องเช็คตรง:
**เทียบ mtime ของ `data/results/gold_hybrid_73det` / `gold_bm25_73det` กับ
`gold_73det_full_embedder_matrix`**

**พบว่าเป็นข้อมูลค้างจริง**: `gold_hybrid_73det` และ `gold_bm25_73det` มี
mtime ล่าสุด **25 ก.ค.** (ก่อน rebuild 3 วัน) ในขณะที่ `embedder_matrix_9way.py`
รี trieval ใหม่ทุกครั้ง (verified จากโค้ด, ไม่มี `--skip-retrieval` flag) จึง
ได้ dense-alone ที่สดจริง — เท่ากับว่า eval รอบแรกเอา dense สด (หลัง
OCR-fix) ไปเทียบกับ hybrid/BM25 เก่า (ก่อน OCR-fix) เป็นการเทียบคนละชุดข้อมูล
โดยไม่รู้ตัว conclusion ที่ดู "พลิก" ทั้งหมดจึงเป็น artifact ไม่ใช่ของจริง

**แก้โดยรัน retrieval ใหม่ทั้งคู่** ต่อ index ที่ rebuild แล้ว:
- `run_gold_bm25_eval.py` (default filter `e5`, 8 combo — e5+e5_small ต่อ
  chunker เพราะ BM25 ไม่ขึ้นกับ embedder อยู่แล้ว): 1128s
- `run_gold_hybrid_eval.py` (ไม่ filter, ครบทุก combo ที่มีบน disk รวม 8
  combo เก่าที่ superseded ด้วย): 11573s (~3.2 ชม.)

แล้วรัน 3 สคริปต์ significance test ที่พึ่งข้อมูลนี้ใหม่ทั้งหมด
(`bm25_vs_embedder_significance_test_9way.py`,
`hybrid_significance_test_9way.py`,
`hybrid_significance_test_semantic_top5.py`)

**สิ่งที่พบหลัง refresh จริง (verified fresh ทั้งคู่แล้ว, ปรึกษา advisor
รอบสองยืนยัน methodology):**

1. **BM25 aggregate ขยับขึ้นชัดเจน**: 0.3908 (ค้าง) → **0.4930** (สด) —
   สมเหตุสมผลตามกลไก: lexical matching ไวต่อ OCR corruption ที่ระดับ token
   มากกว่า dense embedding มาก ดังนั้น BM25 ได้ประโยชน์จากการแก้ OCR
   มากกว่า embedder ตัวไหนๆ เป็นทิศทางที่คาดได้ ไม่ใช่เรื่องแปลก — **ตอนนี้
   BM25 ชนะ bge_m3 อย่างมีนัยสำคัญ** (recall@10 diff +0.0840, Holm-adj
   p=0.0216) ซึ่งก่อนหน้านี้เป็นแค่ tie เท่านั้น BM25 ยังคง tie กับ
   qwen3/qwen3_0.6b/jina_v5
2. **Cross-chunker dense-alone**: qwen3_0.6b (0.5263) ชนะ bge_m3 (0.4090,
   diff +0.1173) และ qwen3/4B (0.4777, diff +0.0486) อย่างมีนัยสำคัญ — เป็น
   ผลจาก dense-alone ที่สดจริงตั้งแต่รอบแรก (ไม่ใช่ปัญหาข้อมูลค้าง) 3-way
   tie เดิม (bge_m3/qwen3/qwen3_0.6b) จึง **แตกจริง** ที่ระดับ
   cross-chunker-average
3. **Hybrid vs dense-alone ยังคง 26/27 significant เหมือนเดิม** แต่ตัว
   exception ย้ายที่: เดิมเป็น qwen3 บน MRR (p=0.09) ตอนนี้เป็น qwen3_0.6b
   บน MRR แทน (Holm-adj p=0.3040) — headline เดิมยังถูกต้อง แค่ embedder
   ที่เป็นข้อยกเว้นเปลี่ยนตัว
4. **Hybrid vs BM25-alone มีจุดพลิกจริง 3 จุด**: jina_v5 ตอนนี้ชนะ BM25
   อย่างมีนัยสำคัญบน recall@10 (+0.0901, Holm-adj p=0.0000 — เดิมแค่ tie);
   congen กลายเป็น tie แทน (เดิมชนะ); `sct` recall@10 กลับมามีนัยสำคัญอีก
   ครั้ง (Holm-adj p=0.0000 — เอกสารเดิมบอกว่า "ไม่มีนัยสำคัญแล้ว post-refresh
   ที่ p=0.08" ซึ่งตอนนี้ผิดแล้ว ต้องแก้)
5. **Semantic-only top-5 tie test แตกบางส่วน**: `bge_m3` หลุดจาก tied
   cluster บน **recall@10 และ nDCG@10** (แพ้ qwen3_0.6b/qwen3/jina_v5 อย่าง
   มีนัยสำคัญ) แต่**ยังคง tie บน MRR** (bge_m3 vs qwen3 Holm-adj p=0.058,
   bge_m3 vs qwen3_0.6b p=0.126 — ไม่ถึงเกณฑ์) ที่เหลืออีก 4 ตัว
   (qwen3_0.6b/qwen3/jina_v5/e5_small) ยัง tie กันเองครบทุก pair ทุก metric
   — คำแนะนำเดิม "อย่าฟันธง embedder เดียว" ยังใช้ได้ แค่ cluster เหลือ 4
   ตัวแทน 5

**จุดที่ต้องระวังเป็นพิเศษ — ตัวเลข "top single combo" 0.7048 พังจริง**:
เลข 0.7048 เดิมที่อ้างว่าเป็น `semantic × qwen3_0.6b` hybrid recall@10 ที่ดี
ที่สุดในทั้งโครงการ มาจากตาราง "Per-embedder mean (semantic + hybrid only)"
ใน `hybrid_significance_test_semantic_top5.py` โดยตรง (ยืนยันแล้วว่าคนละ
column กับ dense-alone) — ค่าสดตอนนี้คือ **0.6152** (ลดลง ~0.09) และเมื่อดู
ทุก chunker ของ qwen3_0.6b hybrid พบว่า `sentence` (0.6265) และ `fixed_size`
(0.6154) สูงกว่า `semantic` (0.6152) เล็กน้อยด้วยซ้ำ — **claim เดิมที่ว่า
"semantic ชนะทุก chunker อื่นชัดเจน" ไม่ปรากฏในตัวเลขสดอีกต่อไป** ส่วนต่าง
ระหว่าง sentence/fixed_size/semantic เล็กมาก (~0.01) และไม่มี significance
test ระดับนี้อยู่ (test ที่มีคือ cross-chunker-aggregate กับ
semantic-only-top5 เท่านั้น ไม่มี test เทียบ chunker ต่อ chunker ที่ combo
เดียวกัน) — **ห้ามฟันธงว่า sentence ชนะแทน** แค่บันทึกว่า claim เดิมไม่มี
หลักฐานรองรับแล้ว เป็น open item ใหม่ถ้าจะ pin chunker headline นี้ในเปเปอร์
ต้องทำ significance test เฉพาะจุดนี้เพิ่ม

**สรุป**: ตัวเลขทั้งหมดในเอกสาร (`docs/paper-results-summary.md`,
`CLAUDE.md`) ที่มาจาก section "Hybrid retrieval" และ "BM25 lexical baseline"
(รวมทั้ง 2 sub-analysis ก่อนหน้าที่อ้างอิงข้อมูลเดียวกัน) เป็นค่าที่ผ่านการ
refresh 25 ก.ค. (หลัง contamination-fix) แต่ **ค้างต่อ OCR-remediation
rebuild 28 ก.ค.** ต้องแทนที่ด้วยตัวเลขสดชุดนี้ทั้งหมด — อัปเดตแล้วทั้ง 2
ไฟล์ (ดู commit ถัดไป) ส่วน `cost_latency_pareto.py` ยังไม่แตะ (เป็นคนละ
กลุ่ม เป็น cost/latency ไม่ใช่ recall เหมือนที่ flag ไว้ก่อนหน้า)

**บทเรียนกระบวนการ**: ทุกครั้งที่ rebuild index ใหม่ ต้อง refresh
**ทุก retrieval path ที่มี persisted results** (dense, BM25, hybrid) ไม่ใช่
แค่ตัวที่ eval script เรียก retrieval สดให้อัตโนมัติ — เช็ค mtime ของ
`data/results/*` เทียบกับ index rebuild timestamp ก่อนเชื่อ significance
test ผลลัพธ์ใดๆ เสมอ

## Chunker-vs-chunker significance test — "semantic ชนะ" ไม่ผ่านการทดสอบเลย (29 ก.ค. 2569, บ่ายวันเดียวกัน)

ต่อจากที่พบว่า "top single combo" (semantic × qwen3_0.6b, 0.7048) พังไปแล้ว
ข้างบน และตัวเลขสดของ `qwen3_0.6b` เองก็ไม่ใช่ `semantic` ที่สูงสุดอีกต่อไป
(sentence 0.6265 > fixed_size 0.6154 > semantic 0.6152 > recursive 0.6097)
— แต่ยังไม่มี significance test ไหนเทียบ chunker-ต่อ-chunker ที่ embedder
เดียวกันมาก่อนเลยตลอดทั้งโปรเจกต์ (ที่มีคือ cross-chunker-average กับ
per-chunker-BM25-vs-embedder เท่านั้น) ผู้ใช้ขอให้สร้าง test นี้จริง

**สร้าง `tools/eval/hybrid_chunker_significance_test.py`** — pure recompute
จาก `gold_hybrid_73det` ที่ refresh สดแล้วข้างบน ไม่ต้อง retrieval ใหม่:
family ละ 6 คู่ (fixed_size/recursive/semantic/sentence) ต่อ embedder
(Holm-correct แยกต่อ embedder ต่อ metric) บวกอีก 1 family รวม (เฉลี่ยแต่ละ
chunker ข้าม embedder ทั้ง 9 ตัวก่อน ตามธรรมเนียมเดียวกับที่
`embedder_matrix_9way.py` เฉลี่ยข้าม chunker)

**ผลลัพธ์ (verified ผ่าน advisor's methodology already, pure recompute
เท่านั้น ไม่มีการ retrieval ใหม่ที่ต้องกังวลเรื่อง stale cache)**:

1. **`qwen3_0.6b` (ที่เป็นประเด็นเดิม)**: ทั้ง 4 chunker ผูกกันหมดจริง ไม่มี
   คู่ไหนมีนัยสำคัญเลยสักคู่ในทุก metric (Holm-adj p≥0.44 ทุกคู่) — ยืนยันว่า
   claim เดิม ("sentence" หรือ "semantic" เป็นตัวที่ดีที่สุดสำหรับ combo นี้)
   ไม่มีมูลทั้งคู่
2. **ระดับรวม (เฉลี่ยข้าม 9 embedder)**: มีคู่เดียวที่มีนัยสำคัญคือ
   `fixed_size` แพ้ `recursive` บน nDCG@10 (Holm-adj p=0.0228) — **`semantic`
   ไม่ชนะใครอย่างมีนัยสำคัญเลยสักคู่** ตัวเลขดิบตอนนี้ `recursive` (0.5291
   recall@10) นำ `semantic` (0.5206) ด้วยซ้ำ แค่ไม่ถึงเกณฑ์นัยสำคัญ
3. **ทั่วทั้ง 9 embedder × 3 metric × 6 คู่ + 1 family รวม (163 การทดสอบ)**:
   `semantic` ไม่เคยปรากฏในคู่ที่มีนัยสำคัญเลยแม้แต่คู่เดียว ไม่ว่าจะชนะหรือ
   แพ้ คู่ที่มีนัยสำคัญทั้งหมดเป็น `fixed_size` แพ้ `recursive` (e5-MRR,
   congen/qwen3/m2v-nDCG@10, m2v-recall@10 รายตัว + aggregate-nDCG@10)

**สรุป**: claim หลักของโครงการที่พูดซ้ำมาตั้งแต่รอบเปรียบเทียบแรก
("semantic chunking ชนะทุก metric") เป็นแค่ค่าเฉลี่ยดิบ 6-embedder ที่ไม่เคย
ผ่าน significance test มาก่อนเลย (ดูตาราง "Chunkers compared" ใน
`docs/paper-results-summary.md`) พอทดสอบจริงแล้วไม่ผ่าน — กรอบที่ถูกต้อง
กว่าคือ **`recursive`/`semantic`/`sentence` เป็นกลุ่มบนที่ผูกกันสถิติ ไม่มี
ตัวชนะที่พิสูจน์ได้ ส่วน `fixed_size` เป็นตัวเดียวที่พิสูจน์แล้วว่าด้อยกว่า**
`semantic` ยังเป็นตัวเลือกที่สมเหตุสมผล (ไม่เคยพิสูจน์ว่าแพ้ใคร และยังเป็น
chunker เดียวที่ dense embedder แรงๆ ชนะ BM25 ได้จริงตาม
`bm25_vs_embedder_significance_test_per_chunker.py`) แค่ไม่ใช่ "ตัวที่ดี
ที่สุด" อีกต่อไป

อัปเดต `docs/paper-results-summary.md` ("Chunkers compared" section + Open
item #13), `CLAUDE.md` (bottom-line paragraph) ให้ตรงกันแล้ว

## Rerun `cost_latency_pareto.py` — ปิดช่องโหว่สุดท้ายที่ยัง stale (29 ก.ค. 2569, เย็นวันเดียวกัน)

`cost_latency_pareto.py` ไม่ได้ถูกรวมอยู่ใน 5-script eval suite
(`embedder_matrix_9way.py` + siblings) ที่ refresh ไปตอนเช้า เพราะเป็นสคริปต์
วัด cost/latency คนละ chain กัน — ผลคือ recall/nDCG column ในรายงานของมัน
ค้าง stale มาแล้ว **สองรอบซ้อน** (ทั้งรอบ corpus-discovery-contamination fix
25 ก.ค. และรอบ OCR-remediation rebuild + BM25/hybrid-cache-fix เช้านี้)
โดยที่ latency/cost column ไม่ได้รับผลกระทบ (วัด mechanics ของ
model/index/corpus-size ไม่ใช่เนื้อหา corpus)

**รันใหม่เต็มรูปแบบ** (ไม่ใช้ `--reuse-latency-cache`, background task
`bd9g6naw7`, จบสำเร็จ):

- **Latency/cost แทบไม่เปลี่ยน** ยืนยันว่าตัวเลขกลุ่มนี้ปลอดภัยที่จะเชื่อใน
  ช่วงที่ยังไม่ได้ refresh จริง เช่น `qwen3` encode p50 264.66ms (เดิม 264ms),
  `e5_small` 24.68ms (เดิม 25ms) — hybrid fixed overhead ขยับจาก ~2.1-2.3s
  เหลือ ~1.9-2.0s (การเปลี่ยนแปลงเล็กน้อยจาก corpus ขนาด 74,819 chunks แทนที่
  จะเป็นตัวเลขเก่าก่อน rebuild ไม่ใช่ effect จริง)
- **Quality column ตกลงทุกตัว** สอดคล้องกับทุกตารางอื่นในเอกสารนี้หลัง 2
  rebuild ติดกัน: `qwen3 × semantic` dense recall@10 0.6581→**0.5382**,
  `qwen3_0.6b × semantic` 0.6364→**0.5688** (ตอนนี้สูงกว่า `qwen3` ใน cell
  dense-alone นี้โดยเฉพาะ ตรงกับที่ cross-chunker aggregate เจอด้านบน),
  `jina_v5 × semantic` 0.5845→**0.4705**, `bge_m3 × semantic` 0.5822→**0.4144**
  (ตกมากที่สุดในกลุ่มบน)
- Hybrid recall@10 (semantic) เรียงใหม่: `qwen3_0.6b` 0.6152 > `qwen3` 0.6051
  > `jina_v5` 0.5995 > `e5_small` 0.5877 > `e5` 0.5455 ≈ `bge_m3` 0.5451 >
  `congen` 0.4666 > `sct` 0.3971 > `m2v` 0.3231; BM25-alone 0.4620

**สำคัญ**: เลข `qwen3_0.6b` ที่สูงสุดในตารางนี้**ไม่ได้ถูกอ้างเป็น "combo ที่ดี
ที่สุดในทั้งโปรเจกต์"** อีกต่อไป — ตามผล chunker-vs-chunker significance test
ข้างบน `semantic` ไม่เคยชนะ chunker อื่นอย่างมีนัยสำคัญ ดังนั้นตัวเลขในตาราง
นี้ให้อ่านเป็นแค่ profile ของ combo หนึ่งที่ representative เท่านั้น ไม่ใช่
claim ว่า optimal

อัปเดต `docs/paper-results-summary.md` (caveat paragraph + ตารางเต็ม + "Reading
this table" + overhead ratio ~4.0x-17.9x) และ `CLAUDE.md` (bottom-line cost/
latency paragraph) ให้ตรงกันแล้ว — ปิดช่องโหว่สุดท้ายที่ค้างจากรอบ eval
refresh เช้านี้

## `multi_k_report.md` ก็ค้าง — บั๊กเดิมเป็นครั้งที่ 3, เจอตอนตอบคำถามผู้ใช้ (29 ก.ค. 2569, ค่ำ)

ผู้ใช้ถามว่า open item "MAP/precision@1 ที่ top cluster ยังไม่ผ่าน
significance test" คืออะไร — ระหว่างเช็กที่มา พบว่า `multi_k_report.md`
(`tools/eval/multi_k_report.py`) มี mtime ค้างที่ **22 ก.ค.** ในขณะที่
`gold_hybrid_73det` ถูกเขียนใหม่ 29 ก.ค. — เป็นบั๊ก "ไม่อยู่ใน 5-script
refresh chain" ตัวที่ 3 ต่อจาก BM25/hybrid กับ `cost_latency_pareto.py`

**รันใหม่** (pure recompute จาก retrieval results ที่ persist ไว้แล้ว
ไม่ต้อง retrieve ใหม่ ไม่ต้องใช้ GPU): เรื่องราวที่เอกสารเคยเล่าไว้
("MAP/precision@1 ขัดแย้งกันที่ top 3" — `bge_m3` นำ MAP, `qwen3_0.6b` นำ
precision@1) **หายไปแล้ว** พอเป็นข้อมูลสด `qwen3_0.6b` นำทุก metric พร้อมกัน
(MAP, precision@1/ndcg@1, recall@10) ทั้ง dense-alone และ hybrid — สอดคล้อง
กับที่ `qwen3_0.6b` ทำลาย dense-alone 3-way tie เดิมไปแล้วข้างบน

**ยังไม่ปิด item นี้จริง**: ยังไม่มี significance test สำหรับ MAP/precision@1
เลย (มีแค่ recall@10/MRR/nDCG@10 ที่ทดสอบใน semantic-top5 script) และ scope
ของตารางนี้ (เฉลี่ยข้าม 4 chunker) กับ scope ของ tied-cluster ที่ทดสอบแล้ว
(`semantic` chunker เดี่ยว) ก็คนละ population กัน — แค่ไม่มีข้อขัดแย้งที่ต้อง
อธิบายอีกต่อไปเท่านั้น อัปเดต `docs/paper-results-summary.md` (Multi-k
metrics section ทั้งหมด + Open item #9) แล้ว

## `reranker_significance_test.py` ก็ค้าง — nDCG@10 headline เดิมไม่ผ่านซ้ำ (29 ก.ค. 2569, ค่ำวันเดียวกัน)

ต่อจากการกวาดเช็ก `data/results/*` ทั้งหมด (ผู้ใช้ขอ "เช็คให้หมดครับ แล้วทำ
ให้หมด") พบว่า `reranker_significance_test.py` **retrieve สดทุกครั้ง** จาก
`data/index/chunker_compare_full/plain__fixed_size__local__ceea7536`
โดยตรง ไม่ได้อ่านจาก persisted results — combo นี้ถูก rebuild ไปแล้ว 29 ก.ค.
11:13 แต่รายงาน (`reranker_significance_test.md`) ยังค้างที่ 23 ก.ค. รันใหม่
(ไม่ต้อง build index ใหม่ ใช้ index ที่ rebuild ไว้แล้วโดยตรง):

| metric (hybrid) | เดิม (23 ก.ค.) | สด (29 ก.ค.) |
|---|---|---|
| MRR | 0.848→0.760, Holm-adj p=0.006 **(นัยสำคัญ)** | 0.7775→0.6775, Holm-adj p=0.0048 **(นัยสำคัญ)** |
| nDCG@10 | 0.675→0.617, Holm-adj p=0.030 **(นัยสำคัญ)** | 0.6193→0.5908, Holm-adj p=**0.5676 (ไม่นัยสำคัญ)** |
| recall@10 | 0.607→0.584, ไม่นัยสำคัญ | 0.5570→0.5683 (สลับเครื่องหมาย!), ไม่นัยสำคัญ |
| dense-alone (ทุก metric) | ไม่มีผล | ไม่มีผลเหมือนเดิม |

**นี่คือการเปลี่ยนแปลงระดับข้อสรุปจริง ไม่ใช่แค่ตัวเลขขยับ**: MRR ยังพังอย่างมี
นัยสำคัญเหมือนเดิม แต่ nDCG@10 ที่เคยอ้างว่า "reranker ทำร้ายทั้ง MRR และ
nDCG@10" **ไม่ผ่านซ้ำ** — สรุปใหม่: reranker ทำร้าย **MRR เท่านั้น** อาจจะยิ่ง
เข้ากับกลไก "phantom hits" ในงานวิจัยที่อ้างไว้มากขึ้นด้วยซ้ำ (MRR วัดแค่อันดับ
คำตอบแรก ส่วน nDCG@10 ถ่วงน้ำหนักทั้ง top-10 — ถ้า reranker แค่กวนอันดับต้นๆ
โดยไม่ได้ไล่คำตอบที่ถูกออกจาก top-10 จริง ก็สมเหตุสมผลที่จะเห็นผลใน MRR
ชัดกว่า nDCG@10) latency แทบไม่เปลี่ยน (p50 1191→1170ms) ยืนยันว่าวัด
mechanics ของโมเดล ไม่ใช่เนื้อหา corpus อัปเดต `docs/paper-results-summary.md`
("Cross-encoder reranker results" + headline + Open item #5b),
`docs/reranker-hybrid-interaction-research.md`, และ
`docs/research-framework-gap-analysis.md` แล้ว

## RQ3 ablation rebuild — ล้าง confound clean-vs-dirty, ข้อสรุป chunk_size แคบลงจริง (29 ก.ค. 2569, ค่ำวันเดียวกัน)

รายการสุดท้ายของการกวาดเช็ก `data/results/*` ทั้งหมด และเป็นรายการเดียวที่
**แก้ด้วยการ rerun eval ไม่ได้** — ต้อง build index ใหม่บน GPU จริง

**ปัญหาไม่ใช่แค่ stale แต่เป็น confound**: index ฝั่ง treatment ของ RQ3 ทั้ง 3 ตัว
(`rq3_segmentation_ablation`, `rq3_chunksize_sweep`, `rq3_normalize_ablation`)
build ไว้ 23 ก.ค. ก่อน kernel-A OCR remediation (เสร็จ 27 ก.ค.) และก่อน rebuild
`chunker_compare_full` (28 ก.ค.) แต่ฝั่ง baseline ของทั้ง 3 การทดลอง **reuse combo
จาก `chunker_compare_full` โดยตรง** (`plain__fixed_size__local__ceea7536`,
`plain__semantic__local__8aae9bcd`) ซึ่งถูก rebuild บนข้อความสะอาดไปแล้ว → ทุก
comparison กำลังเทียบ **baseline สะอาดกับ treatment สกปรก** ซึ่งเป็นความผิดพลาด
เชิงระเบียบวิธี ไม่ใช่แค่ตัวเลขเก่า rebuild ครบทั้ง 3 + rerun sig-test ทั้ง 3
(chained job เดียว 6 ขั้น, exit code 0)

**normalization — ข้อสรุปเดิมยืน**: ไม่มีนัยสำคัญบน metric ใดเลย (Holm-adj p≥0.42,
ใกล้สุดคือ dense nDCG@10 raw p=0.0700) ฝั่ง dense เอนลบเล็กน้อยทุกช่อง (−0.012 ถึง
−0.026) ฝั่ง hybrid แทบเป็นศูนย์ — ภาพเดิมเป๊ะ

**segmentation — ข้อสรุปเดิมยืน แต่ไม่ใช่ p=1.0 ทั้งกระดานอีกแล้ว**: ยังไม่มี
นัยสำคัญ (Holm-adj p≥0.4524) แต่ 2 ช่องขยับออกจากพื้น — dense MRR +0.0398
(raw p=0.0754) และ hybrid recall@10 +0.0183 (raw p=0.1054) ทั้งคู่เข้าข้าง
word-aware อ้างเป็น effect ไม่ได้ แต่ถ้อยคำที่ซื่อสัตย์ตอนนี้คือ "ตรวจไม่พบผล"
ไม่ใช่ "เหมือนกันถึงทศนิยมตำแหน่งที่สาม" อย่างที่ตาราง p=1.0 ชุดเก่าสื่อ
chunk stats ยังใกล้เคียงกันมาก (58,655 raw-char vs 58,198 word-aware;
mean len 437.0 vs 444.0) ยืนยันว่าการเปลี่ยนขอบเขตไม่ได้แอบเป็นการเปลี่ยนขนาด

**chunk_size — ข้อสรุปเปลี่ยนจริง (รายการเดียวของ RQ3 ที่เปลี่ยน)**:
สิ่งที่ replicate แข็งแรงคือ **โทษของ 1024** — แพ้ทั้ง 256 และ 512 อย่างมีนัยสำคัญบน
dense recall@10 (Holm-adj p=0.0020/0.0000), hybrid recall@10 (0.0000/0.0028),
hybrid nDCG@10 (0.0072 ทั้งคู่) สิ่งที่ **ไม่ replicate** คือลำดับ 256-vs-512:
ฝั่ง **dense** ตอนนี้ 512 **นำ** 256 เชิงตัวเลข (recall@10 0.4146 vs 0.4103,
diff −0.0043, p=0.8802 = เสมอราบ) กลับทิศจากเดิมที่เป็น 0.510 > 0.480
256 ชนะ 512 อย่างมีนัยสำคัญเฉพาะ **hybrid recall@10** (+0.0509, Holm-adj p=0.0112)
เท่านั้น ไม่ชนะบน hybrid nDCG@10 (p=0.4094) และไม่มีช่อง MRR ไหนมีนัยสำคัญเลย
(เหมือนเดิม)

**headline ใหม่**: chunk_size ยังเป็นตัวแปร RQ3 ตัวเดียวที่มีผลจริง แต่ข้อความที่
อ้างอิงได้แคบลงกว่าเวอร์ชัน 23 ก.ค. — **อ้างได้: "chunk 1024 ตัวอักษรแย่กว่าทั้ง 512
และ 256 อย่างมีนัยสำคัญบน recall@10 และ nDCG@10" อ้างไม่ได้: "recall ลดลงตามขนาด
แบบ monotonic" หรือ "256 คือค่าที่ดีที่สุด"** — 256 กับ 512 เสมอกันทางสถิติบน dense
(512 นำเชิงตัวเลขด้วยซ้ำ) และ 256 ได้เปรียบเฉพาะใต้ hybrid recall@10
**ค่า default 512 ของโปรเจกต์จึงไม่ได้ถูกพิสูจน์ว่าด้อยกว่า มีแค่ 1024 ที่พิสูจน์ว่าผิด**

อัปเดต `docs/paper-results-summary.md` (§ RQ3 + หัวข้อย่อย refresh ใหม่),
`docs/research-framework-gap-analysis.md` (ข้อ 7), memory
`[[project_research_framework_gap_analysis]]` แล้ว ปิดการกวาด `data/results/*`
ทั้งหมดครบทุกรายการ

## ปิด 2 open item สุดท้ายที่เป็น pure recompute: MAP/precision@1 sig-test + BM25/hybrid แยก entity_type (29 ก.ค. 2569, กลางคืน)

เลือกทำ 2 อันนี้เพราะเช็คแล้วว่าเป็น **pure recompute จริง** — ผลที่ persist ไว้เก็บ
ranked results ครบทุก query และทุก combo retrieve ที่ `top_k=10` อยู่แล้ว ส่วน
`entity_type` ก็ join จาก gold YAML ด้วยข้อความ query ได้ (pattern ที่
`embedder_matrix_9way.py:243` ทำอยู่แล้วสำหรับ dense) → ไม่ต้องแตะ GPU เลย

### 1. `map_precision_significance_test.py` — สอง metric สุดท้ายที่ไม่เคยถูกทดสอบ

MAP กับ precision@1 ถูกรายงานใน `multi_k_report.md` และถูกอ้างใน paper summary
มาตลอด แต่**ไม่เคยผ่าน significance test** — เป็นความเสี่ยงแบบเดียวกับที่ทำให้
headline "semantic ชนะ" ต้องถอน และยังมีปัญหา **scope ไม่ตรงกัน** ค้างอยู่ด้วย
(tie test เดิม scope แค่ `semantic` chunker แต่ตาราง multi-k เฉลี่ยข้าม 4 chunker
→ ยืนยันหรือหักล้างกันเองไม่ได้) จึงรัน **ทั้ง 2 scope × ทั้ง 2 retriever**
Holm แยกครอบครัวต่อ (retriever, scope, metric)

ผลสำคัญ 3 ข้อ:

1. **lead ของ `qwen3_0.6b` ฝั่ง dense แข็งกว่าบน MAP/precision@1 มากกว่าบน recall@10** —
   scope aggregate มันชนะ **ทั้ง 8 ตัวที่เหลืออย่างมีนัยสำคัญ** ทั้งสอง metric
   (เทียบกับ recall@10 ที่ชนะแค่ `bge_m3` กับ `qwen3`) → คำกล่าว "qwen3_0.6b นำทุก
   metric" **ยืนยันเป็นข้อสรุปที่ทดสอบแล้วสำหรับ dense-alone** ไม่ใช่แค่ค่าเฉลี่ยดิบ
2. **กลุ่มที่เสมอกันยังเสมอบน 2 metric ใหม่** — ที่ scope ที่ tie ถูกอ้างจริง
   (hybrid, `semantic`) `bge_m3` แพ้บน MAP อีกครั้ง (drop-out แบบเดียวกับ
   recall@10/nDCG@10) ส่วนอีก 4 ตัวเสมอกันหมด และบน **precision@1 ไม่มีคู่ไหน
   มีนัยสำคัญเลยแม้แต่คู่เดียว** ทั้ง 5 ตัว → คำแนะนำ "อย่าตั้งใครเป็น embedder
   ที่ดีที่สุดสำหรับ hybrid" ตอนนี้ครอบคลุมครบ 5 metric แล้ว ไม่ใช่ 3
3. **scope mismatch เป็นเรื่องจริงและมีผล** — `qwen3_0.6b` สูงสุดที่ scope aggregate
   แต่ `qwen3` สูงสุดเชิงตัวเลขที่ scope semantic ทั้งสอง metric (แม้ไม่มีนัยสำคัญ)
   → "qwen3_0.6b นำทุก metric" เป็น **ข้อความระดับ aggregate scope เท่านั้น**
   ต้องระบุ scope กำกับทุกครั้งที่อ้าง

ข้อสังเกตเพิ่ม: **hybrid บีบความต่างระหว่าง embedder ให้แคบลง** — ครอบครัว 9 embedder
เดียวกันไปจาก 8/8 มีนัยสำคัญ (dense) เหลือ 4/8 (hybrid) สอดคล้องกับการที่ BM25
เป็นพื้นร่วมที่ยกทุกตัวขึ้นเท่าๆ กัน

### 2. `bm25_hybrid_entity_type_breakdown.py` — และมันกลับข้อสรุปเดิมเรื่อง headroom

เดิมมีแต่ dense-alone ที่เคยแยก entity_type (ที่มาของ finding specialist/generalist)
ส่วน BM25 กับ hybrid มีแต่ค่า aggregate → ตอบไม่ได้ว่า hybrid เข้าใกล้เพดาน person
0.976 แค่ไหน และปิดช่องว่าง faculty (เพดาน 0.681) ได้หรือไม่ ตอนนี้ตอบได้ทั้งคู่

**ceiling attainment (recall ÷ ceiling — ตัวเลขที่เทียบข้ามหมวดได้จริง)**:

| entity_type | ceiling | hybrid ดีสุด | % ceiling | dense ดีสุด | % | BM25 | % |
|---|---|---|---|---|---|---|---|
| person | 0.9760 | bge_m3 0.8211 | **84.1%** | 0.5735 | 58.8% | 0.8147 | **83.5%** |
| faculty_adjunct_aggregate | 0.6810 | qwen3_0.6b 0.4922 | **72.3%** | 0.4698 | 69.0% | 0.4224 | 62.0% |
| program | 0.9000 | qwen3_0.6b 0.6187 | **68.7%** | 0.6023 | 66.9% | 0.3484 | 38.7% |
| course | 0.8729 | qwen3_0.6b 0.5683 | **65.1%** | 0.5500 | 63.0% | 0.3600 | 41.2% |

**ข้อสรุปเดิมเรื่อง headroom กลับทิศ** — ย่อหน้าเดิมใน paper summary
("person มีช่องว่างให้ปรับปรุงจริงมากกว่าที่ตัวเลขดิบทำให้เข้าใจ") วัดบน
**dense-alone** เท่านั้น พอดูระบบที่แนะนำจริง (hybrid) กลับกลายเป็นว่า **person
คือหมวดที่แก้ไปได้มากที่สุดแล้ว (84.1%)** ไม่ใช่หมวดที่เหลือช่องว่างมากที่สุด —
จุดอ่อน person ของ dense ถูก BM25 ซ่อมจนเกือบหมด หมวดที่เหลือช่องว่างจริงมากสุด
ตอนนี้คือ **`course` (65.1%)** ซึ่งเกิดขึ้นหลังการวิเคราะห์เพดานรอบแรก
ส่วน faculty hybrid ปิดช่องว่างได้จริง (62.0% → 72.3%)

**finding ใหม่ 2 อันที่เห็นได้เฉพาะจากการแยกหมวดนี้**:

1. **หลักฐานตรงของกลไก complementarity ระหว่าง lexical กับ dense** — BM25 เปล่าๆ
   ได้ **0.8147** บน person ซึ่ง**ชนะ dense-alone ของทุก embedder** (ดีสุด bge_m3
   0.5735) แบบขาดลอย แต่ร่วงเหลือ **0.3484** บน program ที่ dense เกือบสองเท่า
   (qwen3_0.6b 0.6023) → **BM25 แบก person (จับชื่อตรงตัว), dense แบก program**
   นี่คือคำอธิบายเชิงกลไกว่าทำไม hybrid ถึงชนะทั้งคู่ และเป็น**หลักฐานตรง**
   ต่างจาก proxy ทางอ้อม (rescue rate, union coverage, correlation) ที่ใช้ตอน
   สอบสวน Open item #2 แล้วได้ผลไม่ชัด
2. **"hybrid ไม่เคยทำให้แย่ลง" เป็นข้อความระดับ aggregate ไม่ใช่ระดับหมวด** —
   เฉพาะ query กลุ่ม person นั้น hybrid **ต่ำกว่า BM25 เปล่าๆ** สำหรับ embedder
   ส่วนใหญ่ (`qwen3_0.6b` 0.7220, `qwen3` 0.7340, `congen` 0.7228, `jina_v5` 0.7382
   เทียบ BM25 0.8147) มีแค่ `bge_m3` (0.8211) ที่แซงได้ และ `e5`/`e5_small`
   ที่ใกล้เคียง → การ fuse สัญญาณ dense ที่อ่อนในหมวดหนึ่ง **ลากหมวดนั้นลงต่ำกว่า
   BM25 ได้** แม้ค่ารวมข้ามหมวดจะดีขึ้น เป็น failure shape เดียวกับเคส `sct`/`m2v`
   แต่เกิด**ภายใน** embedder ที่แข็งโดยรวม แยกตามหมวด — ควรระบุเป็นข้อจำกัดของ
   คำแนะนำ hybrid ที่เป็น headline

อัปเดต `docs/paper-results-summary.md` (§ ceiling + § multi-k + Open item #9/#10 +
รายการสคริปต์), `CLAUDE.md`, memory แล้ว

## รีเฟรชหลัง rebuild #3 (2026-08-06)

`chunker_compare_full` rebuild #3 เสร็จ 2026-08-05T07:56 (แก้ `resolution_id`
ของ 6/2,853 ไฟล์ที่เคยชนกัน) แต่ตอน sync-check เอกสารทั้งชุดวันเดียวกัน พบว่า
chain การเปรียบเทียบหลักนี้ (BM25/hybrid) ยังไม่ได้รันซ้ำ — เห็นจาก mtime ของ
`data/results/gold_bm25_73det/`/`gold_hybrid_73det/` ยังเป็น 2026-07-29 ทั้งที่
RQ3/reranker/power-analysis รีเฟรชไปแล้ว (commit `9c9547a`) นี่คือ pattern
เดิมที่ [[feedback_refresh_all_retrieval_paths_after_rebuild]] เตือนไว้ — เกิดซ้ำเป็นครั้งที่ 3

พอเริ่มรันจริง (สั่งโดยผู้ใช้ "ทำ 1.") พบเรื่องไม่คาดคิด: **retrieval ถูกรันซ้ำไปแล้ว**
ตั้งแต่ 2026-08-05 (`gold_bm25_73det/` 848 ไฟล์ ลง 08:31-08:50, `gold_hybrid_73det/`
ได้ไฟล์ใหม่ 3,816 ไฟล์ 08:53-11:23) — ไม่ใช่ session นี้เป็นคนรัน (เช็คแล้วจาก
transcript) ที่มาไม่ยืนยันแน่ชัดแต่ผลถูกต้อง ครอบคลุมครบ 36 combo ปัจจุบัน สิ่งที่
เหลือให้ session นี้ทำจริงๆ คือรัน significance-test ปลายทาง 7 สคริปต์ใหม่ (เร็ว
ระดับวินาที เพราะอ่านจาก JSON ที่ persist ไว้แล้ว ไม่ retrieve ใหม่) แล้ว diff
verdict ทีละ cell เทียบ baseline 2026-07-29/07-30 (เขียนสคริปต์ parse ตาราง
markdown เป็น key เฉพาะ เพราะ diff ตามตำแหน่งบรรทัดใช้ไม่ได้ — บางสคริปต์เรียง
แถวตาม effect size/p-value ซึ่งเปลี่ยนลำดับได้ทุกรัน)

**ผลสรุป: headline claim ทุกอันในไฟล์นี้/`CLAUDE.md` รอดหมด** — ตาราง 2 อันที่
ใหญ่ที่สุด (`bm25_vs_embedder_significance_test_9way` 9 คู่,
`hybrid_significance_test_9way` 54 คู่, `..._per_chunker` 108 cell) **verdict
ไม่พลิกแม้แต่อันเดียว** ที่ขยับมีแค่ 4 จุดเล็กๆ ระดับ non-headline และขยับไปทาง
"แยกชัดขึ้น" ทั้งหมด: (1) `fixed_size` แพ้ `recursive` (nDCG@10 aggregate, ผลคู่
chunker เดียวที่ significant ในการทดสอบทั้งชุด) ยังอยู่ (Holm-adj p=0.0396 จาก
0.0264); (2) `bge_m3` ใน semantic-top-5 test แพ้ `qwen3`/`qwen3_0.6b` เพิ่มบน MRR
ด้วย (เดิม MRR คือ metric เดียวที่ยังเสมออยู่) — หลุดจาก tied cluster ชัดเจนขึ้น
ทุก metric; (3) `map_precision_significance_test` precision@1 ระดับ hybrid-aggregate
คมขึ้นจาก 4/8 เป็น 5/8 (MAP ยังคง 4/8 เท่าเดิม เลข "8/8 dense → 4/8 hybrid" ที่อ้าง
ใน CLAUDE.md ไม่กระทบ); (4) `bm25_hybrid_entity_type_breakdown` ตัวเลขนิ่งในช่วง
noise (person 84.1%→84.2%, faculty 72.3%→72.5%, program 68.7%→68.7%, course
65.1%→65.6%; BM25 เปล่า person 0.8147 เท่าเดิมเป๊ะ, program 0.3484→0.3497)

**ยังไม่ได้ทำ**: thematic-query arm (`thematic_hybrid_significance_test.md`,
`thematic_vs_deterministic.md`) อ่านจาก `data/results/thematic_{dense,bm25,hybrid}/`
ที่ populate โดย `run_thematic_eval.py` คนละสคริปต์ ไม่ถูกแตะโดยการรัน 08-05 เลย
ยังลง 2026-07-30 อยู่ — ต้องรันแยก คาดว่าหลายชั่วโมง (179 query เทียบชุด 73-det
ที่มี 106) ยังไม่ได้ schedule; RQ4 ก็ยังไม่ได้รีเฟรชเหมือนกัน (รอ request แยก)

รายละเอียดเต็ม: [[project_eval_refresh_2026_08_06]] อัปเดต `CLAUDE.md` +
memory (`project_index_rebuild_pending.md`, `MEMORY.md`) แล้ว

## รีเฟรช thematic arm หลัง rebuild #3 (2026-08-07)

ปิดงานที่ entry ก่อนหน้าค้างไว้ ("ยังไม่ได้ทำ") — รัน `run_thematic_eval.py` ใหม่
ครบทั้ง 3 retrieval path เทียบ `chunker_compare_full` rebuild #3

**Pre-flight ก่อนจ่ายเวลา GPU หลายชั่วโมง**: เช็คก่อนว่า gold id ยังใช้ได้ไหม
หลัง `resolution_id` fix — 550 distinct `relevant_resolution_ids` ของ 179 query
resolve ได้ครบใน `chunks.parquet` ของ index ใหม่ (missing 0, query ที่ gold ตายทั้ง
ชุด 0) ถ้าข้อนี้ไม่ผ่านต้องแก้ gold ก่อน ไม่ใช่รัน eval

**เวลาที่ใช้จริง** (08:01:37 → 13:13:49, exit=0 ทั้งสองคำสั่ง): dense 36 combo
41 นาที, bm25 4 combo ~10 นาที, hybrid 36 combo ~4.5 ชม. — hybrid ช้ากว่า dense
~6.6 เท่าทั้งที่ combo เท่ากัน ตรงกับ overhead ~2 วิ/query ที่ `cost_latency_pareto`
วัดไว้ (BM25Okapi สร้างใหม่ทุก query) ไม่ใช่ค่า RRF เอง
ไฟล์ผลลัพธ์ใหม่หมดจริง ไม่มีของเก่าตกค้าง: 6,444/6,444 + 716/716 + 6,444/6,444

**ผล: verdict ไม่พลิกแม้แต่ cell เดียว** — `thematic_hybrid_significance_test.md`
54 cell, `thematic_eval.md` 27 cell, flip 0 ทั้งคู่ (diff แบบ structural ด้วย
`tools/eval/diff_significance_reports.py` ที่เขียนเป็นสคริปต์ถาวรรอบนี้ แทนที่จะ
เขียนทิ้งใน /tmp เหมือนรอบ 08-06 — key เป็น `(หัวข้อ section, cell ป้ายนำหน้า)`
เพราะทั้งแถวสลับตำแหน่งได้และป้ายซ้ำข้าม section ได้; validate 2 ทาง คือ
self-diff ได้ 0 flip และ diff รอบ 08-06 ซ้ำแล้วได้ 2 flip ตรงกับที่บันทึกไว้)
**effect size ทุกตัวขยับน้อยกว่า 0.02** ที่ขยับจริงมีแต่ p-value และเฉพาะแถวที่
เสมออยู่แล้ว — ขนาดการขยับสอดคล้องกับที่ควรเป็น เพราะ rebuild #3 แตะ 13 ไฟล์ที่มี
course-comparison table ซึ่งไม่มี query thematic ไหนอ้างถึง

headline ที่ยืนยันแล้วว่ายังถูก (ตัวเลขอัปเดตใน `paper-results-summary.md`):
สัญญาณ chunker สวนทางกันสองชุด (fixed_size − semantic = **+0.0258** thematic vs
**−0.0363** entity-anchored, ลำดับ chunker เหมือนเดิมทั้งสองฝั่ง — `semantic`
แย่สุดบน thematic/ดีสุดบน entity-anchored), BM25 อ่อนบน thematic (**0.2990**),
hybrid vs dense แยกสามทางเท่าเดิม (3 significant-for / 4 เสมอ / 2 significant-against
คือ `e5` −0.0449, `qwen3` −0.0516), fixed_size-vs-semantic ยัง **2 of 27** และเป็น
`bge_m3` recall+ndcg คู่เดิม

**พบข้อความผิดใน `paper-results-summary.md` ระหว่างตรวจ (แก้แล้ว)**: ประโยค "the
sign flips almost exactly at BM25's own 0.2988" (และที่ `CLAUDE.md` ย่อว่า
"flipping sign right at BM25's own score") **ตารางใต้ประโยคเองไม่รองรับ และไม่เคย
รองรับ** — `bge_m3` อยู่ที่ dense 0.3959 ซึ่ง *สูงกว่า* BM25 แต่ delta ยังเป็นบวก
(+0.0280) จุดตัดศูนย์จาก OLS อยู่ที่ 0.401 (ตัวเลข 07-30 เดิมก็ 0.399 ไม่ใช่ 0.299)
สิ่งที่ตรงกับคะแนน BM25 จริงๆ คือ **เส้นแบ่ง significance**: embedder ที่คะแนน
ต่ำกว่า BM25 ทุกตัวได้ประโยชน์อย่างมีนัยสำคัญ และที่สูงกว่าไม่มีตัวไหนได้เลย —
ระหว่างนั้นมีย่านที่ fusion ไม่ช่วยและไม่เสียแบบวัดได้ ส่วน r ยังยืน (−0.925 →
**−0.921**) ข้อสรุปเชิงกฎ ("RRF ช่วยแขนที่อ่อน เก็บภาษีแขนที่แข็ง") ไม่กระทบ
เพราะมันวางอยู่บนความ monotone กับ verdict ไม่ใช่บนตำแหน่งจุดตัด

**ยังไม่ได้ทำ**: RQ4 (`docs/rq4-design.md`) ยังไม่ได้รีเฟรชเทียบ rebuild #3

## `BM25Okapi` cache + วัด latency ใหม่บนเครื่องว่าง — ตัวเลขเด่นที่ประกาศตอนเช้าถูกถอนตอนบ่าย (2026-08-09)

สองงานในวันเดียว งานที่สองล้มตัวเลขของงานแรก

**งานที่ 1 — memoise `BM25Okapi` บน `Index`** (`Index.lexical_scorer`, commit
`5cc71a1`): `BM25Retriever` เคยสร้าง scorer ใหม่ทุก query ตอนนี้ cache ไว้บน object
ที่มัน derive มา อายุจึงถูกต้องโดยโครงสร้าง (`Index.select` สร้าง `Index` ใหม่เสมอ
cache จึงรั่วข้าม subset ไม่ได้) · validate ด้วย **persisted results ไม่ใช่ unit test**:
848/848 top-10 ของ BM25 และ 848/848 ของ hybrid reproduce ครบ

ตัวเลขที่ประกาศตอนนั้น: **26.2 เท่า** (1.073 วิ เทียบ 0.041 วิ) และ BM25
`retrieve()` p50 1.094 → 0.050 วิ (~22 เท่า)

**งานที่ 2 — rerun `cost_latency_pareto.py` บนเครื่องว่าง** (ปลดหนี้ที่งานที่ 1
สร้างขึ้นเอง: การแก้ code ทำให้คอลัมน์ latency ทั้งตารางล้าสมัยทันที) — และการวัดรอบนี้
**ถอนตัวเลขข้างบนเกือบทุกตัว**

1. **วัดจาก query ผิดชนิด** — สคริปต์เดิมป้อน slice 8 คำจาก chunk เป็น "query"
   ได้ **3 token** ส่วน Gold query จริง median **20 token** · `rank_bm25` วนลูป
   ตาม *term* ของ query ใน Python (แตะ doc-frequency dict ทั้ง 74,816 ตัวต่อ term)
   การให้คะแนนจึงเป็นเชิงเส้นกับความยาว query ที่ ~12 ms/token ส่วนการ *build*
   ไม่ใช่ · วัดใหม่บน query จริง: build **1035.89 ms** เทียบ `get_scores`
   **253.50 ms** = **4.1 เท่า** และ BM25-alone `retrieve()` p50 **234.45 ms**
   (p95 332.78) · **สิ่งที่โอนข้ามรูปแบบ query ได้คือค่าที่ประหยัดเป็นวินาที
   (~1.0 วิ/query) ไม่ใช่ตัวคูณ** — ตัวคูณคือฟังก์ชันของขนาด input ที่เราเลือกเอง
2. **hybrid 2.269 → 1.361 วิ (1.7 เท่า, −0.907 วิ)** วัดแบบ paired ทั้งสองแขน
   ในโปรเซสเดียวกันบน index ที่โหลดครั้งเดียว · ส่วนต่างระหว่าง BM25 กับ hybrid
   **คือผลจริง**: มันแยก overhead ที่ entry 29 ก.ค. เรียกรวมว่า "~1.9–2.0 วิ"
   ออกเป็นการสร้าง scorer (หายไปแล้ว) กับการดึง `k=n` + fuse (**~1.36 วิ ยังอยู่**)
   — และครึ่งหลังนี้ **ตัดไม่ได้ฟรี** เพราะ `HybridRetriever` ตั้งใจดึง `k=n`
   เพื่อให้ RRF เห็นอันดับครบ การตัดจึงเปลี่ยนผลลัพธ์ เป็นวิธีใหม่ไม่ใช่วิธีเดิมที่เร็วขึ้น
   (ต่างจากการ cache ซึ่งเปลี่ยนผลไม่ได้เลย) · **นี่คือเหตุผลที่ประโยค "overhead
   ~2 วิ/query" ใน entry 2026-08-07 ข้างบนไม่ควรถูกอ้างอีก** — ฐาน hybrid p50
   ตอนนี้คือ **1.21–1.86 วิ**
3. **"k=n over-fetch tax = 66% ทั้งสองรอบ" ถูกถอน** — รอบนี้ได้ **54%**
   (dense k=10 262.46 ms เทียบ k=n 575.58 ms = over-fetch 313 ms) มันไม่ใช่ค่าคงที่
   ของ implementation ต้องอ้างจากรอบที่รันจริง

**วิธีวัดที่เปลี่ยนไป และเหตุผล** — รอบ 7 ส.ค. ถูก reject เพราะ position effect 74.2%
(หน่วยความจำของ `qwen3` 4B ไม่ถูกปล่อยก่อนที่ตัวถัดไปจะถูกจับเวลา) รอบนี้จึงจับเวลา
**แต่ละ embedder ในโปรเซสของตัวเอง** ซึ่งตัดสาเหตุที่ราก · และมี control **3 ชั้น
อยู่ในรายงานเอง เพราะชั้นเดียวไม่พอ — วัดได้ว่าไม่พอ**:
(1) **reference probe** งาน numpy เดียวกันรันในทุก child จับกรณีพื้น CPU ขยับ
(median 156.6 ms, spread 13.5%); (2) **repeat control** วัด embedder ตัวแรกซ้ำเป็น
ตัวสุดท้าย — จับสิ่งที่ probe จับไม่ได้ คือ `search p50` ของ `bge_m3` เอง
ขยับ **245.5 → 257.9 ms (+5.1%)** ใน 45 นาที ขณะที่ probe ขยับ **−0.4 ms (0.3%)**;
(3) **same-dim consistency** 7 embedder ที่ dim=1024 ทำ numpy op เดียวกันบน array
รูปเดียวกัน spread ของมัน (**10.3%**) *คือ* noise floor ไม่ใช่ความต่างระหว่างโมเดล
· **ถือ ~5–10% เป็น resolution ของ rig นี้ อย่าอ่านช่องว่างที่เล็กกว่านั้นว่าจริง**

phase intrinsic ถูก **cache** ลง `cost_latency_raw.json` แล้ว เพราะมันแกว่ง ~15%
ระหว่าง render และ **ตัวเลขที่ตีพิมพ์ต้อง reproduce ได้จาก artifact ที่ตีพิมพ์มัน**
(ซึ่งคือสิ่งที่ `audit_doc_claims.py` D2 ตรวจพอดี) — render 2 ครั้งติดกันได้ผล
byte-identical · gotcha ตอน cache: **key ของ JSON object เป็น string เสมอ** dict
ที่ key เป็น int (`dense_per_dim`) จึงพังตอน round-trip

**บทเรียนที่บันทึกไว้** — ปริมาณที่ reproduce ได้ข้ามสองรอบของ *สคริปต์เดียวกัน*
เป็นหลักฐานที่อ่อนว่ามันเป็นสมบัติของระบบ: ทั้ง "22 เท่า/24 เท่า" และ "66% ทั้งสองรอบ"
ถูกฝากไว้จากรอบที่ถูก reject แล้ว **ล้มทั้งคู่** เพราะทั้งสองรอบป้อน input ผิดแบบเดียวกัน

**เมื่อรันสคริปต์นี้อีก: เครื่องต้องว่าง และเช็ค embedder ที่ dim เท่ากันแต่อยู่คนละ
ตำแหน่งในลูปก่อนเชื่อ timing ใดๆ**

## Rebuild #4 (ปิด I6) — วันเดียวตาย 2 ครั้ง ทั้งคู่ที่ combo เดียวกัน และเสีย GPU ฟรี 5.6 ชม. เพราะ watchdog ผิดชนิด (2026-08-14)

`I6` เป็น warn ตัวเดียวที่เหลือใน `audit_pipeline_invariants.py` มาตั้งแต่ 9 ส.ค.:
**40 index สร้างเมื่อ 2026-08-08 19:33 แต่คอร์ปัสถูกแก้ล่าสุด 2026-08-09 09:53**
(การ re-download + re-OCR ของ `2566/ครั้งที่ 3`) · มันเป็นการเปลี่ยน **ข้อความ**
ไม่ใช่ metadata การ relabel จึงปลดไม่ได้ ต้อง rebuild จริงเท่านั้น · เดิมปล่อยไว้
เพราะ **ไม่มี gold entry ในทั้งสองชุดอ้างถึง resolution จากการประชุมนั้นเลย** ตัวเลขที่
ตีพิมพ์จึงขยับไม่ได้ — ผู้ใช้ตัดสินใจเคลียร์มันเองในวันนี้ ไม่ใช่เพราะความถูกต้องบังคับ

**สถานะเมื่อจบวัน: 7 จาก 40 combo (`suspect dirs: 0`)** ทั้ง 7 เป็น `semantic`
ของ `chunker_compare_full` ใช้เวลา 08:10 → 14:36:44 = **6 ชม. 27 นาที**
(56/41/65/81/42/63/39 นาทีต่อ combo) · ที่เหลือ 33 combo ประเมินจากเวลาต่อกลุ่มที่
วัดไว้ใน rebuild #3 บนเครื่องเดียวกัน ≈ **19 ชม.** — คิดเป็นงานจริงที่เสร็จ **~25%
ของเวลา GPU** ไม่ใช่ 17.5% ตามจำนวน combo เพราะ 7 ตัวแรกเป็นตัวที่ถูกที่สุด

### สิ่งที่เจอ 1 — การตายไม่ได้สุ่ม: 5 ครั้งจาก 5 อยู่ที่ `semantic × qwen3`

รวมของ rebuild #3 สามครั้งกับวันนี้อีกสองครั้ง ทุกครั้งล้มที่ combo เดียวกัน:
#3-a ทำ 7 semantic เสร็จแล้วตายตอนถึง qwen3 · #3-b ตายที่ batch 14/9352 ของ
`Qwen3-Embedding-4B` · #3-c ตายตอน chunking combo เดิม (`nvlddmkm` Event 153) ·
#4-a (วันนี้ 15:15) ทำ 7 ตัวเดิมเสร็จแล้วตายที่ตัวที่ 8 ซึ่งคือ combo เดิม
(`nvlddmkm` 153 เวลา 15:15:43) · #4-b (15:38:51) ตายที่ chunking 46% ของ combo เดิม

**กลไก**: `SemanticChunker` ต้อง embed *ทุกประโยค* เพื่อหาจุดตัด `semantic × Qwen3-4B`
จึงเป็นการรันโมเดล 4B fp16 ต่อเนื่องยาวที่สุดในโปรเจกต์ (~3.4 วิ/it × 2,854 ไฟล์ —
log บอก chunking อย่างเดียว ~1 ชม. 45 นาที) บนการ์ด 12 GB · มันสำเร็จ **ครั้งเดียว**
คือ rebuild #3 รอบที่ 4 ซึ่งเป็นรอบที่ไม่มีการใช้ GPU ทำอย่างอื่นเลย · จาก 33 combo
ที่เหลือมีเพียง **4 ตัวที่ใช้ 4B** (01, 10, 19, 28) ที่เหลือ 29 ตัวเบาทั้งหมด

### สิ่งที่เจอ 2 — watchdog ที่ตั้งไว้ "หยุดตอน 21:00" มองไม่เห็นการตายตอน 15:38

งานตายเวลา 15:38:51 · watchdog ตื่น 20:55 เห็นว่าไม่มีโปรเซสไหนเหลือ เขียน
`watchdog exiting: chain already finished` แล้ว exit 0 — **ซึ่งถูกต้องตามที่มันถูกเขียน
มาทุกประการ** · ผลคือเครื่องนั่งว่าง **5.6 ชม.** และสถานะตอน 21:00 เท่ากับสถานะตอน
15:15 ทุก byte

**"เสร็จแล้ว" กับ "ตายแล้ว" หน้าตาเหมือนกันหมดเมื่อดูจากการมี/ไม่มีโปรเซส** และ
watchdog ที่ตั้งเวลาไว้จะถามคำถามนั้น *ครั้งเดียว* ณ เวลาที่เลือกมาด้วยเหตุผลอื่น
ช่วงก่อนหน้าทั้งหมดจึงไม่มีใครเฝ้า · การตายเงียบยังไม่ทิ้งบรรทัด `FAILED` ไว้ด้วย
เพราะสิ่งที่จะเขียนบรรทัดนั้นคือสิ่งที่ตายไป · **สิ่งที่ต้องเฝ้าคือ artifact ที่พิสูจน์ว่างาน
เดินอยู่ (mtime ของ log, จำนวน manifest) ไม่ใช่การมีอยู่ของโปรเซส** — โปรเซสที่ค้าง
ยังมีชีวิตและไม่ผลิตอะไรเลย

### สิ่งที่เปลี่ยน — เลิกรันข้ามคืน กลับไปรันครั้งละ 1-2 combo

ผู้ใช้เลือกโหมดใหม่: **รันครั้งละ 1-2 combo แบบเฝ้าดูเอง** แทนการปล่อยยาวข้ามคืน

`tools/make_residual_rebuild_config.py --write` แตกงานที่เหลือเป็น YAML **ละ 1 combo**
อยู่แล้ว (33 ไฟล์ใน `config/experiments/_resume/rebuild_2026_08_14/`) และมันคำนวณ
"เหลืออะไร" จาก **timestamp ของ manifest** ไม่ใช่จากไฟล์ `.DONE` — combo ที่รันมือจน
เสร็จจึงหลุดออกจากรายการเอง ไม่ต้องจดสถานะที่ไหน · กฎจัดก้อน: **combo ที่ใช้ Qwen3-4B
(01, 10, 19, 28) รันเดี่ยว ที่เหลือจับคู่ได้** รวมเป็น 19 ก้อน ก้อนละ ~40 นาที ถึง ~3 ชม.
(ก้อนแรกคือ combo 01 ตัวที่ฆ่ามาแล้ว 5 ครั้ง) · **"2 ตัว" หมายถึงรันต่อกัน ไม่ใช่พร้อมกัน**
— GPU 12 GB ไม่พอ

ข้อดีที่แท้จริงไม่ใช่ความสะดวก แต่คือ **มันลบปัญหาทั้งคลาสที่ทำให้เสีย GPU ฟรีวันนี้**:
ไม่ต้องใช้ Task Scheduler ไม่ต้องมี watchdog ไม่ต้องเขียน stall detector เพราะคนเป็น
liveness check เอง และงานแต่ละก้อนสั้นพอจะเห็นได้ว่ามันเดินอยู่จริง · ต้นทุนคือต้อง
กลับมาสั่ง 19 รอบ — คุ้ม เพราะรอบข้ามคืนวันนี้ได้งานจริง 6.4 ชม. จาก 13 ชม. ที่ตั้งใจไว้

**หมายเหตุระหว่างที่ยังไม่จบ**: ต้นไม้ index อยู่ในสภาพ *ผสม* บาง combo เป็น OCR ใหม่
บางตัวเป็นของเก่า — ไม่เป็นอันตรายด้วยเหตุผลข้างบน แต่ **การ refresh eval ต้องรอให้ครบ
ทั้ง 40 ไม่ใช่รอจุดที่หยุดพอดี**
