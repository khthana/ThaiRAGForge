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
   21,756 ตัวอักษร) — เพิ่ม unit test 5 เคสใน `tests/test_loaders_common.py`, รัน full suite ผ่าน
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
