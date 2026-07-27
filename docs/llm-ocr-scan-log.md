# บันทึกการทดลอง LLM OCR-corruption scan (11 ก.ค. 2569) — handoff

เอกสารนี้คือ handoff: บอก Claude ว่า "อ่านไฟล์นี้แล้วทำต่อ" เพื่อกลับมาทำงานนี้ต่อได้เลย

## เป้าหมาย

`tools/corpus_prep/scan_ocr_repetition.py` (ตัวเดิม, regex-based) จับได้เฉพาะ
"อะไรบางอย่างซ้ำกันหลายครั้ง" (repetition-loop hallucination) — เจอ ~112-121 เอกสาร,
แก้หมดแล้วด้วยการ re-OCR (ดู `[[project_curriculum_splitting]]` ใน memory) แต่มัน
มองไม่เห็น**ประโยค/ย่อหน้าที่ OCR อ่านผิดแบบเกิดครั้งเดียว ไม่ซ้ำ** (misread
characters, มโนคำ, ประโยคขาดตอน) โดยโครงสร้างของ regex เอง

ผู้ใช้มีการ์ดจอ + local LLM (phi4-mini, phi4, gemma4:e4b — ดึงไว้ใน Ollama แล้ว)
เลยอยากลองว่า LLM จับ defect class นี้ได้ไหม โดย**ต้นทุนจริงมีแค่ค่าไฟ**
ไม่ใช่มาเติมงาน entity/typo-correction ที่ทำอยู่แล้ว (`ner/` — คนละงานกัน)

## งานที่ทำเสร็จแล้ว

**Script**: `tools/corpus_prep/llm_ocr_scan.py` (ยังไม่ commit — pattern เดียวกับ
`scan_ocr_repetition.py` ตอนเพิ่งเขียนใหม่ๆ) — page-chunk ไฟล์ตาม header
`## Page N`, ส่งแต่ละ chunk ให้ 3 โมเดล local ผ่าน Ollama พร้อม JSON schema
เล็กๆ (`flag`/`span`/`reason` เท่านั้น ไม่มี confidence — โมเดลเล็กประเมิน
confidence เองไม่น่าเชื่อถือ) ถาม "ข้อความนี้ incoherent/garbled ไหม (ไม่ใช่ซ้ำ
กัน — เรื่องซ้ำมีตัวอื่นเช็คอยู่แล้ว)"

**บั๊กที่เจอและแก้แล้ว:**
1. `str.format()` ชนกับ `{}` ตัวอย่าง JSON ที่อยู่ใน prompt เอง (`KeyError: '"flag"'`)
   → เปลี่ยนเป็น placeholder แทน (`%%TEXT%%` + `.replace()`)
2. `gemma4:e4b` เป็น thinking model — ถ้าไม่ปิด จะกิน `num_predict` หมดไปกับการ
   "คิด" ก่อนตอบจริง (content ว่างเปล่า) → ใส่ `think=False`
3. `phi4:latest` บางหน้าตอบยาวเกิน budget เดิม (300 token) → JSON ขาดตอน
   (`Unterminated string`) → เพิ่มเป็น `num_predict=800` + เพิ่ม temperature
   เล็กน้อยตอน retry (retry ที่ temperature=0 เดิมจะได้ผลลัพธ์เดิมซ้ำ ไม่มีประโยชน์)

**พบว่าตัวเลข "26 ไฟล์ที่พังรู้อยู่แล้ว" ที่คุยกันไว้ตอนแรกผิด** — มาจาก bash
`find -iname` ใน Git Bash บน Windows ซึ่งนับไฟล์ชื่อภาษาไทยพวกนี้ผิด (ปัญหา
Unicode/codepage ของสภาพแวดล้อมนั้น ไม่ใช่ของ Python) ตัวเลขจริงจาก
`Path.rglob()` คือ **~121 เอกสารต้นทาง** (ใกล้เคียง ~112 ที่ memory
`[[project_curriculum_splitting]]` บันทึกไว้ตอนเจอครั้งแรก ซึ่งสมเหตุสมผลกว่า)
**ข้อควรระวังทั่วไป: อย่าเชื่อ `find -iname` นับไฟล์ชื่อไทย/ยูนิโค้ดบน Windows
Git Bash — ใช้ Python `pathlib.rglob()` หรือ `Glob` tool แทนเสมอ**

แก้ `bak_files()` ให้:
- ตัด `_LINK.txt.*.bak` sidecar ออก (มีแต่ URL ไม่มีเนื้อความ)
- รวมเอกสารที่ถูก curriculum-split (`__1.md`, `__2.md`, ...) กลับเป็น 1
  ตัวแทนต่อเอกสารต้นทาง (เลือกไฟล์เต็ม/ก่อน-split ถ้ามี เพราะเนื้อหาครบกว่า)
- เพิ่ม `--floor-sample N` (default 20) สุ่มจาก 121 ไฟล์แทนที่จะรันทั้งหมด —
  floor check ควรเป็นแค่ sanity check เล็กๆ

## กรอบความคิดสำคัญ (จาก advisor, อย่าลืมตอนอ่าน report)

- **`.bak` floor set วัด "พื้นล่าง" ไม่ใช่เป้าหมายจริง** — ไฟล์พวกนี้พังแบบ
  repetition (defect class ที่แก้แล้ว) การที่โมเดล flag ไฟล์เหล่านี้ได้แค่
  ยืนยันว่า "โมเดลไม่ได้ตาบอดสนิท" ไม่ได้พิสูจน์ว่าจับ defect แบบไม่ซ้ำได้จริง
- **Deliverable จริงคือ flag ที่เกิดบนไฟล์ที่ `scan_ocr_repetition.py` ผ่านแล้ว
  (สะอาดแล้วตามที่รู้)** — ทุก flag ตรงนั้นคือ false positive หรือ novel-class
  find จริง — รายงานต้องเน้นส่วนนี้เป็นหลัก ไม่ใช่ recall บน `.bak`
- **Verify การทำงานบน `.bak` test file ตัวอย่าง** (เอกสาร 21-piece ที่ split,
  คณะวิศวกรรมศาสตร์ 2564/7): phi4-mini จับ mojibake (`�`) ได้ชัดเจนหลาย
  หน้า แต่ก็ over-flag ข้อความอังกฤษปกติ (course code "PREREQUISITE: NONE",
  citation list) ว่า corrupted ด้วย — เป็นสัญญาณ false-positive rate ที่ต้อง
  ระวังตอนดู sample จริง ไม่ใช่เชื่อทุก flag ทันที
- Re-OCR-diff verification loop (เอาไฟล์ที่โดน flag ใหม่ไปเทียบกับการ re-OCR
  จาก PDF ต้นฉบับ) **ตั้งใจเลื่อนไว้ก่อน** — ใช้ตอนมี novel candidate จาก
  clean sample แล้วเท่านั้น ไม่ใช่ตอนนี้ PDF ต้นฉบับอยู่ที่
  `D:\academic_resolutions (ข้อมูลดิบ + OCR)` ตาม `[[project_curriculum_splitting]]`

## ยังไม่ได้ทำ (ขั้นตอนให้ Claude ทำต่อ) — เดิม, ทำครบหมดแล้ว ดู addendum ด้านล่าง

1. ~~รัน floor sanity check~~ ✅ เสร็จ — phi4-mini flag 156/156 known-bad (แต่
   ดูข้อ 2 ของ addendum: ตัวเลขนี้ไม่มีความหมาย), phi4 28/40 (22 call errors),
   gemma4:e4b 15/20 (2 call errors)
2. ~~รัน experiment จริง `--sample 30`~~ ✅ เสร็จ
3. ~~`--report`~~ ✅ เสร็จ, `academic_resolutions/llm_ocr_review.md`
4. ~~อ่าน flag ด้วยตา ระวัง false positive~~ ✅ เสร็จ — ดู addendum
5. ~~คุยเรื่อง re-OCR-diff verification loop~~ ✅ เสร็จ, validate แล้วบน 2 ไฟล์

ผลลัพธ์ raw (jsonl) เก็บที่ `academic_resolutions/llm_ocr_scan/*.jsonl`
(gitignored, resume ได้ — สคริปต์ข้าม key ที่ทำไปแล้วโดยอัตโนมัติ)

---

## Addendum (12 ก.ค. 2569) — จบ experiment แล้ว เข้าสู่ production scan

รายละเอียดเชิงลึก/citation ทั้งหมดอยู่ที่
`docs/ocr-corruption-detection-strategies.md` (เอกสารแยกที่ background agent
เขียนไว้ตามคำขอผู้ใช้ให้หาคำแนะนำเรื่อง multi-pass/ensemble strategy) —
ที่นี่สรุปเฉพาะสิ่งที่เกิดขึ้นและสถานะปัจจุบัน

### 1. ผล sample experiment (30 ไฟล์, 143 หน้า) ที่ page-level

- **phi4-mini: flag 143/143 หน้า (100%)** — ยืนยันจาก jsonl โดยตรงว่ามัน flag
  ทุกหน้าไม่เว้นแม้แต่ประโยคไทยที่ถูกต้องสมบูรณ์แบบ เป็น constant-true
  classifier ไม่มี discriminative signal เลย **ตัวเลข floor "156/156 recall"
  ของมันจึงไม่มีความหมายย้อนกลับไปด้วย** (constant-true จับได้ทุก positive
  โดยไม่ต้องมองอะไรเลย) → **retire ออกจาก `MODELS` แล้วในโค้ด**
- **phi4: flag 75/143 หน้า (52.4%)** — ยัง noisy, มี false-positive pattern ที่
  ระบุได้ชัด: citation list / course code ภาษาอังกฤษ ("PREREQUISITE: NONE")
- **gemma4:e4b: flag 18/143 หน้า (12.6%)** — เลือกมากสุด อ่านด้วยตาแล้ว ~15/18
  เป็น genuine corruption จริง

### 2. Diff test (page-level เทียบ phi4 vs gemma) — จุดพลิกสำคัญ

ผู้ใช้ขอให้ทำ diff test (ใช้ jsonl เดิม ไม่รันโมเดลใหม่) ก่อนตัดสินใจ
architecture: gemma's 18 flags เกือบทั้งหมด (17/18) ก็ถูก phi4 flag ด้วย —
phi4 เป็น superset เกือบสมบูรณ์ที่ page-level

### 3. File-level rollup — สิ่งที่ล้ม cascade architecture ทิ้งจริงๆ

Advisor (Opus 4.8) ชี้ว่า page-level diff ไม่ใช่คำถามที่ถูกสำหรับ stage-1
filter — recall ระดับ**ไฟล์**ต่างหากที่สำคัญ (อะไรที่ stage 1 พลาด จะไม่มีวัน
ถึง stage 2 เลย) คำนวณใหม่ที่ file-level (30 ไฟล์เดิม):

- gemma4:e4b flag 9/30 ไฟล์, phi4 flag 22/30 ไฟล์
- **gemma พลาดทั้งไฟล์ (ไม่ flag แม้แต่หน้าเดียว) ใน 13/30 ไฟล์ (43%) ที่ phi4
  เจอ defect** — อ่านด้วยตาแล้ว ~9-10 ใน 13 ไฟล์นี้มี defect จริง (บางอันเล็ก
  เช่น "DATATRUCTURE"/"MUSCIANSHIP"/"Digital Quotence" ตัวอักษรหายไป 1-2 ตัว,
  บางอันหนัก เช่น เลขซ้ำยาวๆ ที่ regex scanner ก็พลาดเหมือนกัน — ดูข้อ 4)
- **สรุป: แผน "gemma นำ, phi4 ยืนยันเฉพาะที่ gemma flag" (cascade) ใช้ไม่ได้
  จริง** เพราะจะพลาดไฟล์ที่มีปัญหาจริงไปเงียบๆ ~30-43% → เปลี่ยนเป็น **union**
  (รันทั้งสองโมเดลทั่วคอร์ปัส แยกกัน ไม่ gate กัน)

### 4. Side-finding: `scan_ocr_repetition.py` มี gap เรื่อง intra-token repetition

ไฟล์ `2568\ครั้งที่ 10\...วิทยาลัยวิศวกรรมสังคีต.md` หน้า 2 มีเลข ๒ ซ้ำกัน
20+ ตัวติดกันไม่มีช่องว่างคั่น (`๒๕๕๘๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒`) — phi4
เจอ แต่ regex scanner พลาด เพราะมันเช็ค "token เดียวกันซ้ำ ≥8 ครั้ง" โดย split
ด้วย whitespace (`\S+`) ถ้าตัวอักษรซ้ำติดกันในคำเดียวไม่มีช่องว่างคั่น จะถูก
นับเป็น token เดียว ไม่ใช่ token ซ้ำ N ครั้ง **ยืนยันจากการอ่านโค้ด ไม่ได้รัน
scanner ซ้ำ — เป็น defect คนละ shape (intra-token) จาก class หลักที่ตามหาอยู่
(non-repetitive) ไม่ได้แก้ตอนนี้ เก็บไว้เป็นงานแยกในอนาคต**

### 5. แก้ `tools/corpus_prep/llm_ocr_scan.py` (ไม่ gate กับการตัดสินใจ architecture)

- Retire `phi4-mini:latest` ออกจาก default `MODELS` (เหลือ `[phi4:latest,
  gemma4:e4b]`)
- เพิ่ม `num_predict=1500` เฉพาะ phi4 (เดิม 800 ทำให้ error `Unterminated
  string` ~6% ของ floor set)
- เพิ่ม `elapsed_s` ต่อ call ใน jsonl record (สำหรับวัด throughput จริงในรันถัดไป)
- เพิ่ม `--year YYYY [YYYY ...]` mode: full-corpus production scan แบบไม่สุ่ม
  ทีละปี (resumable ต่อปี, เขียน `academic_resolutions/llm_ocr_scan/
  full_review_<year>.md` หลังแต่ละปี)

### 6. Re-OCR-diff verification — validate แล้วบน 2 ไฟล์ multi-model-consensus

Copy PDF ต้นฉบับจาก `D:\academic_resolutions (ข้อมูลดิบ + OCR)\...` ไปที่
scratch directory (ไม่แตะ D:\ หรือคอร์ปัสจริง) แล้วรัน
`tools/corpus_prep/ocr_pdf_to_md.py` (โมเดล `scb10x/typhoon-ocr1.5-3b`) ใหม่
ทั้งสองไฟล์ — **ยืนยัน genuine corruption 100% ทั้งคู่**:

- `2565\ครั้งที่ 12\...คณะวิศวกรรมศาสตร์.md` หน้า 3: ตารางมี 4 แถวที่ควรมี
  "อาจารย์ผู้รับผิดชอบรายวิชา" + เหตุผล เหมือนกันทุกแถว (คนละคอร์สแต่คน
  รับผิดชอบเดียวกัน) แถว 1-3 ถูกต้อง แต่แถว 4 ในคอร์ปัสปัจจุบันตัวอักษรชื่อ
  กับเหตุผลปนกันมั่วสิ้นเชิง ("รศ.ดร.นัตน์ จรัสโรจ...นัธนศึกษาได้มีโอประ...")
  re-OCR ใหม่ให้ข้อความตรงกับ pattern 3 แถวบน
- `2564\ครั้งที่ 8\...คณะแพทยศาสตร์.md` หน้า 1: "นักศึกษาแลุคลากร" (ควรเป็น
  "นักศึกษาและบุคลากร") ยืนยัน misread ตรงตาม re-OCR; หน้า 6: reference list
  ทั้งตารางถูกตัดขาด/ยุบรวมกันในคอร์ปัสปัจจุบัน re-OCR เผยให้เห็น citation ที่
  ถูกต้องครบ 30 รายการ (ไม่ใช่แค่ misread คำเดียว แต่เป็น truncation ทั้งก้อน)

### 7. สถานะปัจจุบัน: กำลังรัน production union scan แบบ batch รายปี

ผู้ใช้เลือก **union เต็มคอร์ปัส แบ่ง batch ทีละปี** (ไม่ใช่ cascade ที่เร็ว
กว่าแต่รู้แล้วว่าพลาด 30-43%) คอร์ปัสมี ~2,856 ไฟล์ ~10,000 หน้า แบ่งเป็น 6 ปี
(2564: 567, 2565: 509, 2566: 574, 2567: 557, 2568: 534, 2569: 112 ไฟล์)

Throughput จริง (วัดจาก NTFS timestamp ของ jsonl, ไม่ใช่รันใหม่):
gemma4:e4b ~53.6 หน้า/นาที, phi4 ~14.1 หน้า/นาที (รันนี้; เคยวัดได้ ~4.5/นาที
ตอน floor check ภายใต้ load ต่างกัน — ช่วงที่แน่นอนยังไม่นิ่ง) ประมาณเวลารวม
ทั้งคอร์ปัส (union, sequential บน GPU เดียว): **~15-40 ชั่วโมง**

**คำสั่งรัน** (ทำทีละปี, resume ได้ถ้า crash กลางคัน):
```
python tools/corpus_prep/llm_ocr_scan.py --year 2564
python tools/corpus_prep/llm_ocr_scan.py --year 2565
python tools/corpus_prep/llm_ocr_scan.py --year 2566
python tools/corpus_prep/llm_ocr_scan.py --year 2567
python tools/corpus_prep/llm_ocr_scan.py --year 2568
python tools/corpus_prep/llm_ocr_scan.py --year 2569
```

ปี **2564 กำลังรันอยู่** (เริ่ม 12 ก.ค. 2569 ~21:35) ผลลัพธ์จะอยู่ที่
`academic_resolutions/llm_ocr_scan/full_2564__{phi4_latest,gemma4_e4b}.jsonl`
และรายงานที่ `academic_resolutions/llm_ocr_scan/full_review_2564.md`
(gitignored ทั้งคู่) ทำทีละปีต่อไปเรื่อยๆ เมื่อปีก่อนหน้าเสร็จ

## ยังไม่ได้ทำ (หลัง production scan เสร็จครบ 6 ปี) — เดิม, scan เสร็จแล้ว ดู addendum ด้านล่าง

1. ~~รวบรวม `full_review_<year>.md` ทั้ง 6 ปี อ่านด้วยตา~~ scan เสร็จครบแล้ว
   (13 ก.ค. 2569), การอ่านด้วยตายังไม่ได้ทำ — ดู addendum ข้อ 2
2. สำหรับ novel candidate ที่มั่นใจ ใช้ re-OCR-diff loop (ยังไม่ได้ทำ, รอ
   ผลจากข้อ 1)
3. พิจารณาแก้ `scan_ocr_repetition.py` ให้ครอบคลุม intra-token repetition
   (side-finding ข้อ 4 ด้านบน) แยกจากงานหลัก — ยังไม่ได้ทำ

---

## Addendum 2 (13 ก.ค. 2569) — production scan เสร็จครบทั้ง 6 ปีแล้ว

รันแบบ batch ทีละปี ทั้งหมดเป็น detached background process, เริ่ม 12 ก.ค.
2569 ~21:35 จบ 13 ก.ค. 2569 ~12:57 (รวม ~15.5 ชั่วโมง จริง อยู่ในช่วงประมาณ
15-40 ชม. ที่คาดไว้แต่อยู่ที่ขอบล่าง)

### 1. สรุปตัวเลขทั้ง 6 ปี

| ปี | ไฟล์ | ไฟล์ที่มี flag (โมเดลใดก็ได้) | ไฟล์ consensus (ทั้ง 2 โมเดล) |
|---|---|---|---|
| 2564 | 567 | 375 | 117 |
| 2565 | 509 | — | 97 |
| 2566 | 574 | — | 116 |
| 2567 | 557 | — | 118 |
| 2568 | 534 | — | 104 |
| 2569 | 112 | — | 17 |
| **รวม** | **2,853** | — | **569** |

(คอลัมน์ "ไฟล์ที่มี flag" เก็บครบเฉพาะ 2564 — ปีอื่นบันทึกไว้แค่ตัวเลข
consensus ใน memory ระหว่างสแกน ไม่ใช่ปัญหา เพราะ consensus คือ signal ที่จะ
ใช้เรียงลำดับความสำคัญตอน eyeball review อยู่แล้ว)

### 2. Consensus cluster ที่น่าสนใจที่สุด (ควรเช็คก่อนตอน eyeball review)

- `2564\ครั้งที่ 7\33. เรื่อง ...คณะครุศาสตร์อุตสาหกรรมแล__1` ถึง `__6.md` —
  เอกสารเดียวกันที่ถูกตัดเป็น 6 ชิ้น **ทุกชิ้นมี consensus page เท่ากันหมด (6
  หน้า/ชิ้น)** — รูปแบบนี้ (เอกสารต้นฉบับเดียวกัน corrupt ซ้ำในทุก piece หลัง
  ตัด) ก็เจอซ้ำอีกใน 2566 (`ครั้งที่ 2s\...คณะวิศกรรมศาสตร์__1` ถึง `__11`,
  3 หน้า consensus/ชิ้น) — **เอกสารที่ถูกตัดเป็นหลายชิ้นน่าจะเป็นกลุ่มเสี่ยง
  สูงกว่าเฉลี่ย** ควรตรวจเป็นหมวดแยก
- `2567\ครั้งที่ 9\เรื่อง ...วิทยาลัยการจัดการนวัตกรรมและอุตสาหกรรม.md` — 8
  หน้า consensus, สูงสุดในบรรดาไฟล์เดี่ยว (ไม่ใช่ split-cluster) ที่เจอทั้ง
  6 ปี (2568 มีไฟล์หนึ่งเท่ากันที่ 8 เช่นกัน: `ครั้งที่ 11\เรื่อง ...สำนัก
  วิชาศึกษาทั่วไป.md`)
- 2569 (ปีปัจจุบัน, แค่ 112 ไฟล์ที่มีอยู่ตอนสแกน) มี consensus แค่ 17 ไฟล์ —
  สัดส่วนต่ำกว่าปีอื่นๆ (~15% เทียบ ~18-23% ปีอื่น) น่าจะเป็นเพราะเอกสารใหม่
  กว่ายังไม่ผ่าน OCR-repetition fix รอบเก่า หรือแค่ sample size เล็ก ไม่ควร
  อ่านเป็นสัญญาณคุณภาพจริง

### 3. งานที่เหลือ (ตามลำดับ)

1. **อ่าน `full_review_<year>.md` ทั้ง 6 ไฟล์ด้วยตา** — เริ่มจาก consensus
   cluster ที่ระบุไว้ข้างบนก่อน (สัญญาณแรงสุด) แล้วไล่ไฟล์ consensus เดี่ยว
   อื่นๆ ตามลำดับจำนวนหน้า consensus จากมากไปน้อย ระวัง false-positive pattern
   เดิม (course code เดี่ยว, "PREREQUISITE: NONE", citation list ที่ถูกจริง,
   ชื่อวิชาประหลาดแต่มีจริง เช่น "Charm School")
2. **สำหรับ novel candidate ที่มั่นใจ** รัน re-OCR-diff verification
   (copy PDF จาก `D:\academic_resolutions (ข้อมูลดิบ + OCR)\...` ไปที่
   scratch dir, รัน `ocr_pdf_to_md.py`, diff กับคอร์ปัสปัจจุบัน) ก่อนแก้ไข
   คอร์ปัสจริงเสมอ — วิธีนี้ validate แล้วว่าใช้ได้จริงบน 2 ไฟล์ แต่ยัง manual
   ทีละไฟล์ ไม่ได้ automate
3. พิจารณาแก้ `scan_ocr_repetition.py` (intra-token repetition gap) แยกเป็น
   งานภายหลัง ไม่เร่งด่วน

## Addendum 3 (13 ก.ค. 2569) — สร้าง `consensus_priority.md` รวมทั้ง 6 ปี

ผู้ใช้ถามว่า consensus อยู่ที่ไหน และบอกว่าอ่าน `full_review_<year>.md`
ทีละไฟล์แล้วหา flag ที่ผิดปกติจริงไม่เจอ (ตัวอย่างที่ยกมาเป็น phi4-only
flag ซึ่งข้อความอ่านดูปกติมาก — ตรงกับ false-positive rate ที่รู้อยู่แล้วว่า
สูงสำหรับ phi4-only) เพื่อแก้ปัญหานี้ เขียนสคริปต์ parse ทั้ง 6
`full_review_<year>.md`, ดึงเฉพาะหน้าที่ **ทั้ง 2 โมเดล flag ตรงกัน**
(consensus), group ตามไฟล์ แล้วเรียงจากไฟล์ที่มี consensus page เยอะสุดลงมา

ผลลัพธ์: `academic_resolutions/llm_ocr_scan/consensus_priority.md`
(gitignored เหมือนไฟล์อื่นในโฟลเดอร์นี้) — **569 ไฟล์, 872 หน้า consensus
รวมทั้งคอร์ปัส** นี่คือจุดเริ่มต้นที่แนะนำสำหรับ eyeball review แทนการไล่
`full_review_<year>.md` ทีละปี เพราะเรียงความสำคัญไว้ให้แล้วและตัดโอกาส
เจอ false-positive จากไฟล์ที่มีแต่ single-model flag ไปเกือบทั้งหมด

สคริปต์ build ไม่ได้ commit ไว้ (เขียนแบบ one-off ในสคริปต์ scratch) — ถ้า
ต้องรันซ้ำหลังสแกนเพิ่ม ให้เขียนใหม่ตาม logic เดียวกัน: regex
`- \*\*\[(.*?)\]\*\* page \`(.*?)\` -- (.*)` จับ (model, page, reason) จาก
แต่ละบรรทัดใน `full_review_<year>.md`, บรรทัดถัดไปที่ขึ้นต้นด้วย `>` คือ
span, group ตาม page แล้วเก็บเฉพาะ page ที่มี ≥2 โมเดล เป็น consensus

---

## Addendum 4 (14 ก.ค. 2569) — เข้าสู่ remediation pipeline เต็มรูปแบบ (Phase 1+2 เสร็จ)

ผู้ใช้ตัดสินใจ **ข้าม eyeball review ทีละไฟล์ไปเลย**: หลังจากตรวจไฟล์ที่ flag
ด้วยตาเองไปหลายไฟล์แล้วพบว่าผิดจริงทุกไฟล์ ("ok ผมดูแล้วทำไปหลายไฟล์ ผิดจริง
ครับ") สรุปว่าเชื่อได้ว่าทั้ง 569 ไฟล์ consensus น่าจะผิดจริงทั้งหมด ("ผมหมายถึง
ทุกไฟล์เลยครับ ไม่ใช่แค่ 15 ไฟล์... ทำให้ผมเชื่อว่าน่าจะผิดหมดจริง") แล้วขอ
pipeline อัตโนมัติแทน: **re-OCR ใหม่ + เก็บของเดิมไว้ + ให้ local LLM (2 ตัว)
cross-check ว่าตัวไหนถูก** โดยเคส "ทั้งคู่อาจผิด" ให้ defer ไว้ก่อนไม่ต้องแก้
ตอนนี้ ทุกอย่างยัง **staging-only** ไม่มีอะไรเขียนกลับเข้าคอร์ปัสจริงเลย

### 1. Phase 1 — re-OCR สดทุกหน้าที่ถูก flag

`tools/corpus_prep/reocr_consensus_pages.py` — group หน้าที่ถูก flag ตาม
(PDF ต้นฉบับ, เลขหน้าจริง) ไม่ซ้ำ (เอกสารที่ถูกตัดเป็นหลายชิ้นมักแชร์หน้า
เดียวกัน) แล้ว OCR ใหม่ทีละหน้า resumable (JSONL append)

**ผล: 760/760 work item ที่ resolve ได้ staged ครบ**, เหลือ 9 ไฟล์ (จริงๆ
คือ 6 เอกสารต้นทาง หลังยุบไฟล์ split-piece) resolve หา PDF ต้นฉบับไม่เจอ —
ดูข้อ 4 ด้านล่าง

### 2. Phase 2 — dual-model adjudication เก่า vs ใหม่

`tools/corpus_prep/reocr_adjudicate.py` — ให้ `phi4:latest` และ `gemma4:e4b`
ตัดสินอิสระจากกันว่า old หรือ new ดีกว่า (verdict ∈ old/new/both_bad/both_ok)

**บั๊กที่เจอและแก้แล้วระหว่างทาง**: ฟังก์ชันดึงข้อความเดิมจากคอร์ปัส
(`resolve_old_text`) เดิมหาแบบ label ตรงตัว `"Page N"` ซึ่งพังเงียบๆ
(คืน `None`, ข้ามหน้านั้นทิ้งทุกรอบ) สำหรับหน้าที่ `split_pages` ตัดย่อยเป็น
`"Page N.1"`, `"Page N.2"`, ... (เกิดเมื่อเนื้อหาหน้าเดียวเกิน
`PAGE_CHAR_BUDGET` 6000 ตัวอักษร) — กระทบ **21 หน้า** ทุกหน้าถูกข้ามทุกรอบรัน
โดยไม่มีใครสังเกต แก้ด้วย `load_full_page_text` (ประกอบ sub-chunk กลับเป็น
หน้าเดียว) ยืนยันแล้วว่าไม่มีหน้าไหนเหลือ `None` หลังแก้ (หน้าที่เนื้อหายาว
เกิน budget น่าจะเป็นกลุ่มเสี่ยงสูงกว่าเฉลี่ยด้วยซ้ำ เพราะเนื้อหายิ่งยาวยิ่งมี
โอกาส OCR พลาด — เป็น failure mode ที่แย่ที่สุดเท่าที่จะเป็นไปได้สำหรับ
pipeline นี้)

**ผล: 760/760 หน้า adjudicate ครบ, 0 model error, 0 sibling text ไม่ตรงกัน**

| คู่ verdict (phi4, gemma) | จำนวน | ความหมาย |
|---|---|---|
| (new, new) | 729 | ทั้งคู่เห็นตรงกันว่า re-OCR ดีกว่า — apply อัตโนมัติได้ |
| (old, new) / (new, old) | 16 | เห็นไม่ตรงกันจริง |
| (both_ok, new) / (both_bad, new) | 10 | ตัวนึงว่าไม่ต่าง/ยังพัง อีกตัวชอบ new |
| (old, old) | 2 | re-OCR ทำให้แย่ลง |
| (both_ok, both_ok) | 2 | ไม่ต่างกันจริง |
| (both_bad, both_bad) | 1 | เคส "ทั้งคู่ยังพัง" ที่ผู้ใช้บอกให้ defer ไว้ |

### 3. หน้า review UI ใหม่ สำหรับ 31 หน้าที่ไม่ได้ (new,new) ตรงกัน

`tools/corpus_prep/consensus_review/pages/1_reocr_diff_review.py` — โผล่เป็น
หน้าที่ 2 ใน sidebar nav อัตโนมัติ (รันด้วยคำสั่งเดิม `streamlit run
tools/corpus_prep/consensus_review/review_app.py`) แสดงข้อความเดิม vs
re-OCR เทียบข้างกัน มีปุ่ม 3 แบบ (ใช้ข้อความใหม่ / เก็บของเดิม / รอไว้ก่อน)
บันทึกลง `academic_resolutions/llm_ocr_scan/reocr_review_decisions.jsonl`
(append-only เหมือน pattern เดิมของทั้ง pipeline) **ยังไม่มีใครกดรีวิวจริง**

### 4. 9 ไฟล์ที่ resolve ไม่เจอ — เป็นกับดักจริง ไม่ใช่บั๊กชื่อไฟล์ธรรมดา

ตรวจสอบละเอียดแล้ว (14 ก.ค. 2569): ชื่อเรื่องแบบ "ขอความเห็นชอบการปรับปรุง
หลักสูตร (กรณี[ไม่]กระทบกระเทือนโครงสร้าง) คณะ X" **เกิดซ้ำในหลายการประชุม
ต่อปี** (แต่ละคณะยื่นขอปรับปรุงหลักสูตรซ้ำหลายครั้ง) เพราะฉะนั้น **ชื่อเรื่อง
เพียงอย่างเดียวไม่ใช่ตัวระบุที่ไม่ซ้ำกัน** — ยืนยันจากการทดลองจริง: ตัว
fallback ที่ตัด suffix `" (2)"` ของคอร์ปัสออกไปจับ PDF ที่ชื่อตรงเป๊ะและอยู่
ถูกโฟลเดอร์ประชุมจริง แต่กลับเป็น**คนละเอกสาร**ที่สั้นกว่า (10 หน้า ทั้งที่
หน้าที่ถูก flag คือหน้า 23!) แก้โดยเพิ่มการเช็คจำนวนหน้า PDF ก่อนเชื่อผล
resolve (`build_work_items(..., page_count_fn=...)`) จับบั๊กนี้ได้ทันทีและ
revert อัตโนมัติ (ไม่มีข้อมูลเสียหาย)

**ผลสุดท้าย: ทั้ง 9 ไฟล์ (6 เอกสาร) ยังคง unresolved เหมือนเดิม** ต้องหา
ด้วยมือ — header `# Document: <ครั้งที่>-<ปี> <ชื่อไฟล์ต้นฉบับ>` ที่ทุกไฟล์
คอร์ปัสมี ยืนยันได้แค่ว่า**โฟลเดอร์ประชุม**ถูกต้อง แต่ชื่อไฟล์ต้นฉบับที่ header
บันทึกไว้บางอันไม่มีอยู่บนไดรฟ์ต้นทางแล้ว (น่าจะถูกจัดระเบียบ/เปลี่ยนชื่อใหม่
หลัง ingest ครั้งแรก) รายชื่ออยู่ที่
`academic_resolutions/llm_ocr_scan/reocr_unresolved_files.txt`
**(เอกสารนี้เป็น snapshot ค้างจากรอบที่ยังพังอยู่ — ดู Addendum 5 ข้อ 3 ทำไม
ไฟล์ยัง list ครบ 9 ทั้งที่ปัญหาแก้แล้วจริง)**

### 5. งานที่ยังไม่ได้ทำ — เดิม, ทำครบหมดแล้วภายในวันเดียวกัน (16 ก.ค. 2569) ดู Addendum 5

1. ~~Phase 3 (เขียนกลับเข้าคอร์ปัสจริง) ยังไม่ได้เขียนโค้ดเลย~~ ✅ เสร็จ,
   commit `b692480`, backup convention `.pre_reocr.bak`
2. ~~31 หน้าใน review UI ใหม่ — ยังไม่มีใครกดตัดสินจริง~~ ✅ เสร็จ,
   review queue ว่างแล้ว (35/35 ตัดสินแล้ว หลังรวม 4 หน้าเพิ่มจากข้อ 3)
3. ~~9 ไฟล์ (6 เอกสาร) unresolved — ต้องเปิด PDF ต้นทางด้วยมือหาให้เจอ~~ ✅
   ปิดแล้ว, ดู Addendum 5 ข้อ 3
4. ~~`tools/corpus_prep/llm_ocr_scan.py` ยัง untracked ใน git~~ แก้แล้ว
   14 ก.ค. 2569 — commit `c3dd919`

---

## Addendum 5 (16 ก.ค. 2569) — Phase 3 เขียนกลับเข้าคอร์ปัสจริงแล้ว, ปิด blocker ที่เหลือทั้งหมด

ทำต่อจาก Addendum 4 ในวันเดียวกัน (session ต่อเนื่อง) ปิดครบทั้ง 3 ข้อที่ค้าง
ไว้ ("งานที่ยังไม่ได้ทำ" ด้านบน) ในรอบเดียว — commit `b692480` ("Build Phase
3 write-back and close both re-OCR pipeline blockers")

### 1. Phase 3 — `tools/corpus_prep/reocr_apply.py`

Dry-run โดย default (`--apply` เพื่อเขียนจริง), idempotent เมื่อรันซ้ำ รวม
adjudication verdict (Phase 2) เข้ากับ human review decision (ถ้ามี) ต่อ
(pdf, page): auto-apply ถ้าทั้ง 2 โมเดลเห็นตรงกันว่า new ดีกว่า, apply/skip
ตาม human decision ถ้ามี, ไม่งั้น skip เป็น "รอ review" แต่ละไฟล์ที่ถูกแก้
backup ครั้งเดียวเป็น `<name>.md.pre_reocr.bak` (ธรรมเนียมเดียวกับ
`excise_ocr_loops.py` และรอบ re-OCR 2026-07-07)

**รันจริง**: 768 หน้า adjudicated ทั้งหมด → 750 หน้า apply (854 file-write
ทั้งคอร์ปัส เพราะเอกสารที่ถูก split แชร์หน้าเดียวกันหลายไฟล์), 18 หน้า skip
(human decision = keep-old) รันซ้ำทันทีเพื่อ verify idempotent: 0 write ใหม่
(`reocr_apply_report.md`)

### 2. 31 หน้า review queue — ผู้ใช้ตัดสินครบใน UI วันเดียวกัน

13 apply-new, 18 keep-old (31 หน้าแรก) — ดูข้อ 3 สำหรับหน้าเพิ่มอีก 4 หน้า
จากการ resolve 9 ไฟล์ รวมทั้งหมด **35/35 ตัดสินแล้ว** ใน
`reocr_review_decisions.jsonl` (18 keep-old / 17 apply-new)

### 3. 9 ไฟล์ unresolved — ปิดแล้ว, root cause คือ filename truncation ไม่ใช่ metadata หาย

ผู้ใช้เปิด `meeting_manifest.json` url ผ่าน `logic.meeting_info()`
(ฟังก์ชันใหม่ที่เพิ่มใน commit นี้ ไม่ใช่ fuzzy title search แบบเดิมที่เคย
ล้มเหลว) โหลด PDF ต้นฉบับทั้ง 6 เอกสารจาก Google Drive ด้วยมือ วางไว้ที่
โฟลเดอร์ `D:\academic_resolutions (ข้อมูลดิบ + OCR)\<ปี>\<ครั้งที่>\` ที่ถูกต้อง
— **6/9 resolve อัตโนมัติทันที** ผ่าน candidate เดิม (stem ตรง/header-derived
filename) **3/9 ต้อง manual override ใหม่**
(`reocr_manual_pdf_overrides.json`, corpus-relpath → ชื่อไฟล์ PDF จริง) เพราะ
Windows ตัดชื่อไฟล์ยาวไม่เท่ากันทุกครั้งที่ save ใหม่ ทำให้ stem ไม่ตรงไบต์ต่อ
ไบต์ — ทั้ง 3 entry ตรวจสอบด้วย page-count safety check เดิมก่อนเชื่อ (ธรรมเนียม
เดียวกับ automatic path)

Phase 1 re-OCR 8 หน้าใหม่ที่ resolve ได้ (768 unique pairs รวม, จาก 760) →
Phase 2 adjudicate 4 auto-apply + 4 ต้อง human review → Phase 3 apply 4
auto-apply ทันที → ผู้ใช้ review อีก 4 หน้า: 0 keep-old, 3 apply-new → apply
รอบสุดท้ายเก็บ 3 หน้านั้น **ผลสุดท้าย: 753/768 หน้า live, 18 keep-old, queue
ว่าง (35/35)**

`academic_resolutions/llm_ocr_scan/reocr_unresolved_files.txt` (9 รายการ)
เป็น snapshot ที่**ไม่ถูก rewrite ในรอบสุดท้าย** เพราะ `main()` ใน
`reocr_consensus_pages.py` เขียนไฟล์นี้เฉพาะตอน `unresolved` list ไม่ว่าง —
รอบที่แก้เสร็จ (unresolved กลายเป็น 0) เลยไม่ trigger การเขียนทับ **อย่าอ่าน
ไฟล์นี้เป็นสถานะปัจจุบัน** ให้เชื่อ `reocr_manual_pdf_overrides.json` (3
entries) + memory `[[project_reocr_remediation_pipeline]]` แทน

### 4. 10 ไฟล์ header ซ้ำ (`## Page N` ปรากฏ >1 ครั้งในไฟล์เดียว) — ปิดแล้วเช่นกัน

Root cause: bug จริงตอน ingest (page-counter off-by-one) ไม่ใช่ corruption
สุ่ม — ทุก 10 ไฟล์ occurrence แรกคือ boilerplate ทั่วไปที่ซ้ำทุกวาระในชุดเดียวกัน
occurrence ที่สองคือเนื้อหาจริงของหน้านั้น ยืนยันด้วย hard evidence (จับคู่
`span` ที่โมเดล quote มากับเนื้อความจริงของแต่ละ occurrence) — ตรง occurrence
ที่ 2 ทั้ง 10/10 ไฟล์ เพิ่ม `replace_page_text(..., occurrence=N)` +
`reocr_page_occurrence_overrides.json` (10 entries, ทุกตัว occurrence=2) แล้ว
apply — **0 ปัญหาเหลือ, idempotent ยืนยันแล้ว**

### 5. สถานะสุดท้ายของ pipeline นี้

ไม่มี blocker เชิงโครงสร้างเหลือแล้ว — ทุก 872 หน้าที่ถูก consensus-flag
ตอนแรกมีผลลัพธ์ที่ชัดเจนแล้ว (re-OCR live หรือ human keep-old ที่ตั้งใจ)
สิ่งที่เหลือมีแค่งานดูแลตามปกติถ้ามีไฟล์ consensus-flag ใหม่เข้ามาในอนาคต
รายละเอียดเชิง narrative เต็มอยู่ที่ memory
`[[project_reocr_remediation_pipeline]]`

### 6. พบไฟล์ที่น่าจะหลุดจากการสแกนรอบแรก (25 ก.ค. 2569, ยังไม่สอบสวนต่อ)

ระหว่างทดสอบ entity extraction กับตารางประเภท "อาจารย์พิเศษสอนเกินร้อยละ 50"
(ดู `docs/entity-extraction-and-gold-eval-log.md` § 6) เจอไฟล์
`academic_resolutions/2564/ครั้งที่ 1/เรื่อง
ขอความเห็นชอบการปฏิบัติกรณีของอาจารย์พิเศษสอนเกินกว่าร้อยละ 50
คณะบริหารธุรกิจ.md` ที่มี cell ในตารางเป็นตัวอักษรซ้ำเป็นพันตัว
(`ก.ก.ก.ก...`) ซึ่งเป็นรูปแบบ corruption เดียวกับที่ pipeline นี้ออกแบบมา
เพื่อจับ เช็ค `reocr_adjudication.jsonl` แล้ว**ไม่พบไฟล์นี้เลยแม้แต่รายการ
เดียว** ทั้งที่ไฟล์อื่นที่ชื่อเรื่องเหมือนกันทุกตัวอักษร (ต่างแค่ปี/ครั้งที่:
2564/12, 2566/10, 2567/10, 2567/6, 2568/10) **ถูกจับและแก้ไปหมดแล้ว**
(verdict "new" ทุกตัว, บาง reason ระบุตรงๆ ว่าแก้ "garbled text" ใน column
เดียวกันนี้) — อ่านได้ว่าเป็นช่องโหว่ coverage ของการสแกนรอบแรก (ไฟล์นี้ไม่
เคยถูกสแกนเลย) ไม่ใช่กรณีที่ตรวจแล้วตัดสินใจเก็บของเดิม

**ยังไม่ได้ทำ**: ตรวจแค่ไฟล์เดียว ไม่รู้ว่า coverage gap นี้ใหญ่แค่ไหน —
ถ้าจะสืบต่อ ต้อง diff รายชื่อไฟล์เต็มจาก `iter_corpus_files()` กับทุกไฟล์ที่
เคยถูกอ้างถึงใน scan/adjudication jsonl ทั้งหมด ก่อนตัดสินใจว่าต้องสแกนรอบ
ใหม่หรือไม่

### 7. ขนาดจริงของ gap (26 ก.ค. 2569) — ไม่ใช่ 1 ปัญหา แต่เป็น 2 กลไกคนละแบบ

สืบต่อจาก §6 ด้วย full diff ตามที่วางแผนไว้ พบว่า "ไฟล์หลุดจากสแกน" ไม่ใช่
คำอธิบายที่ถูกสำหรับทั้งสองกรณี — ไฟล์ต้นตอของ §6 (`.../อาจารย์พิเศษสอนเกิน
กว่าร้อยละ 50 คณะบริหารธุรกิจ.md`, 2564/1) แท้จริงแล้ว**ถูกสแกนโดยทั้งสอง
โมเดลแล้ว** (มี record ใน `full_2564__phi4_latest.jsonl` และ
`full_2564__gemma4_e4b.jsonl` ครบ) แต่**ทั้งคู่ไม่ flag เลย** ทั้งที่ cell
ในตารางมีตัวอักษร "ก" ซ้ำติดกัน 2,995 ตัว — นี่คือกลไกที่ต่างจาก §6 เดิมโดย
สิ้นเชิง

**กลไก A — consensus-threshold exclusion** (สาเหตุที่แท้จริงของไฟล์ 4 ไฟล์/
2 เอกสารที่เจอระหว่างสืบ item A4: "ปิดหลักสูตรแบบมีเงื่อนไข" 2567/2 และ
"วิทยาเขตชุมพรเขตรอุดมศักดิ์" 2567/1 `__1/__2/__3`). ยืนยันจากตัว docstring
ของ `reocr_consensus_pages.py`: *"re-run OCR fresh on every page that both
phi4:latest and gemma4:e4b flagged"* — เป็นเงื่อนไข AND ที่ตั้งใจออกแบบมา
เพื่อคุมต้นทุน/false-positive ของ phi4 (ซึ่ง flag บ่อยกว่า gemma มาก) ไม่ใช่
บั๊ก แต่ผลข้างเคียงคือหน้าที่โมเดลเดียว flag แล้วจริง ไม่เคยถูกส่งต่อไป
remediate เลย

ขนาด (นับจาก raw per-page jsonl ทั้ง 6 ปี, `flag=true`):

| | phi4:latest | gemma4:e4b |
|---|---|---|
| หน้าที่ flag ทั้งหมด | 5,023 | 928 |
| หน้าที่อีกโมเดล**ไม่**เห็นด้วย (only) | 4,151 หน้า / 1,698 ไฟล์ | 56 หน้า / 54 ไฟล์ |

หน้าที่ทั้งสองโมเดล agree (ไปต่อ remediate ได้): 872 หน้า — ตรงกับ 768
record ใน `reocr_adjudication.jsonl` (872 ไม่เท่า 768 เพราะบางหน้าถูก merge/
skip ระหว่าง phase 1-2)

ที่สำคัญกว่าคือไฟล์ที่**ไม่เคยถูกแตะเลยแม้แต่หน้าเดียว** (ไม่ใช่แค่บาง
หน้า) โดย adjudication pipeline: **1,329 ไฟล์** (1,292 จากฝั่ง phi4-only +
37 จากฝั่ง gemma-only) จาก 2,853 ไฟล์ในคอร์ปัส (~47%)

**ลองหา cheap text-only proxy เพื่อแยก defect จริงจาก noise แล้วไม่เวิร์ก**
(บันทึกไว้กันคนต่อไปลองซ้ำ): filter คำใน `reason` ("repetit"/"duplicat")
ได้ 34 หน้า/30 ไฟล์ แต่ recall ไม่ครบ (พลาดเอกสารที่ยืนยันแล้วว่าเป็น defect
จริง 1 ใน 2); filter เชิงโครงสร้าง (บรรทัดเดียวกันซ้ำติดกัน ≥5 ครั้ง) ก็
recall ไม่ครบเหมือนกัน (defect จริงในเอกสารนั้นซ้ำอยู่ *ในเซลล์ตาราง* ที่
คั่นด้วย `<br/>` ไม่ใช่ทั้งบรรทัด) — และแม้แต่ span ที่ phi4 ชี้มาเป็นหลักฐาน
("Ogcharoen, T., 2020, ...") พอเทียบกับข้อความจริงก็มีโอกาสเป็นการอ้างอิง
เดียวกันที่ถูกอ้างซ้ำใต้ผู้เขียนร่วม 2 คนจริงๆ (เนื้อหาถูกต้อง ไม่ใช่
corruption) — สรุปคือ**การแยกแก้จริง/false-positive ต้องดูเป็นหน้าๆ ไป
(LLM ซ้ำ หรือคนตรวจ) ไม่มี proxy ราคาถูกที่เชื่อถือได้**

ประมาณต้นทุนถ้าจะ**สแกนซ้ำ** 1,329 ไฟล์ที่ยังไม่เคยถูกแตะ (ใช้อีกโมเดลมา
cross-check หน้าที่ถูก flag ฝ่ายเดียว, ไม่รวม cost ของ re-OCR+adjudication
ที่จะตามมาถ้าเจอ defect จริง): เฉลี่ย 2.15 วินาที/หน้า/โมเดล (จาก
`elapsed_s` ใน jsonl เดิม, N=23,682 calls) → **~48 นาที** model time
(รันต่อเนื่อง ไม่ parallelize)

**กลไก B — detection blind spot** (สาเหตุจริงของไฟล์ต้นตอ §6): ทั้งสอง
โมเดลสแกนหน้านั้นแล้ว แต่ไม่มีใคร flag ตัวอักษรเดี่ยวที่ซ้ำติดกันเป็นพันตัว
— ไม่อ่านว่าเป็น "garbled prose" เพราะไม่ใช่คำ/ประโยคที่ผิดหลักไวยากรณ์
เป็นแค่สัญลักษณ์ซ้ำเดิม สแกนหา pattern นี้ทั้งคอร์ปัสด้วย regex ล้วน (ตัว
อักษรเดียวซ้ำติดกัน ≥30 ครั้ง, ไม่ผ่าน LLM เลย) เจอ **47 ไฟล์** แต่ส่วนใหญ่
เกาะอยู่ที่ขอบล่างของ threshold (44 ไฟล์อยู่ที่ 30-49 ตัวอักษรซ้ำ ซึ่งมีโอกาส
สูงที่เป็น dot-leader/placeholder ที่ถูกต้องตามฟอร์แมตเอกสารราชการ ไม่ใช่
corruption) มีแค่ **2 ไฟล์ที่รุนแรงจริง (run ≥1,000 ตัวอักษร)**:
ไฟล์ต้นตอ §6 เอง (2,995 ตัว "ก") และไฟล์ใหม่ที่เพิ่งเจอ
`2568/ครั้งที่ 4/เรื่อง ขอความเห็นชอบโครงการหลักสูตรควบสองปริญญา
วิทยาลัยอุตสาหกรรมการบินนานาชาติ.md` (1,984 ตัว ".")

**สรุปสถานะ (26 ก.ค. 2569)**: ขนาด gap ทั้งสองกลไกวัดได้แล้ว เป็นตัวเลข
จริงพร้อมให้ตัดสินใจ — แต่**ยังไม่ได้ตัดสินใจ/ลงมือ remediate อะไรเพิ่ม**
รอผู้ใช้เลือก scope (เช่น: rescan แค่ 1,329 ไฟล์ที่ไม่เคยถูกแตะ, หรือ
รีวิวมือแค่ 2+1 ไฟล์ severe จากกลไก B, หรือ defer ทั้งหมด)

### 8. สุ่มตัวอย่าง 35 หน้าจาก kernel A ผ่าน pipeline จริง (26 ก.ค. 2569) — ตัวเลขต้นทุนเดิมผิด และวิธีวัดก็มีช่องโหว่

ผู้ใช้เลือกให้สุ่ม ~100 หน้าจาก kernel A (1,329 ไฟล์ที่ไม่เคยถูกแตะ) แล้ว
รันผ่าน Phase 1 (re-OCR สด) + Phase 2 (dual-model adjudication) จริงของ
pipeline เอง ก่อนตัดสินใจเรื่อง remediate เต็มรูปแบบ — สคริปต์แยกต่างหาก
`tools/corpus_prep/sample_kernel_a_check.py` (ไม่แตะ staging/adjudication
jsonl จริง, เขียนไปที่ `sample_kernelA_*` แทน)

**แก้ตัวเลขต้นทุนที่ประเมินผิดใน §7**: 2.15 วินาที/หน้า (จาก `elapsed_s`
ของ scan เดิม) เป็นราคาของ **การเรียก classify ข้อความล้วน** ไม่ใช่ราคา
ของ re-OCR ภาพจริง วัดจาก timestamp delta จริงใน
`reocr_pages_staging.jsonl` (N=766 หลังตัด outlier ที่มี gap คั่นระหว่าง
session >300 วิ) ได้ median ~20.3 วิ/หน้า, mean ~22.0 วิ/หน้า — ต้นทุนจริง
ของการ re-OCR kernel A ทั้งหมด (2,101 work item หลัง resolve PDF path +
dedupe sibling) คือ **~16.4 ชั่วโมง** (re-OCR อย่างเดียว ก่อนรวม Phase 2)
ไม่ใช่ ~48 นาทีตามที่ประเมินไว้ใน §7

**ผลจาก 35/100 ตัวอย่างแรก** (ก่อนจะหยุดกลางคัน — ดูเหตุผลด้านล่าง):

| verdict (phi4, gemma) | จำนวน |
|---|---|
| (new, new) | 29 |
| (new, old) | 2 |
| (both_ok, new) | 2 |
| (old, new) | 1 |
| (both_ok, both_ok) | 1 |

ตัวเลขดิบ "new/new" = 29/35 (~83%) ดูเหมือนบอกว่า phi4-only flag ส่วนใหญ่
เป็น defect จริง — **แต่ตัวเลขนี้วัดผิดสิ่ง**. `ocr_page`/
`load_full_page_text` เทียบ**ทั้งหน้า** เก่ากับใหม่ ในขณะที่ phi4 flag
เฉพาะ **span** (ประโยค/วลีที่ error) หนึ่งจุดในหน้านั้น — โมเดล adjudicate
อาจตัดสิน "new" เพราะ typhoon-ocr สะอาดกว่าโดยรวมทั้งหน้า โดยที่จุดที่ phi4
ชี้ไว้จริงๆ ไม่เคยถูกแก้เลยก็ได้

เช็คด้วยการเทียบ **span ที่ phi4/gemma ชี้ไว้ (มีอยู่แล้วใน raw scan jsonl,
ไม่ต้องเรียก LLM เพิ่ม)** ว่ายังอยู่ใน text ที่ re-OCR ใหม่หรือไม่ (substring
match แบบ normalize whitespace) กับผลทั้ง 35 ตัวอย่าง:

- ทุกตัวอย่างมี span ให้เทียบ (35/35)
- span **หายไป** จาก text ใหม่ (บ่งชี้ defect ถูกแก้จริง): **18/35 (51%)**
- span **ยังอยู่เหมือนเดิม** ใน text ใหม่ (re-OCR ไม่ได้แตะจุดที่ flag เลย):
  **17/35 (49%)**
- ในกลุ่ม verdict (new,new) 29 ตัวอย่าง: มีแค่ **17/29 (59%)** ที่ span
  เปลี่ยนจริง อีก **12/29 (41%)** verdict บอก "new" ทั้งที่ span ที่ถูก flag
  ไม่เปลี่ยนเลย — สัญญาณเท็จเชิงวิธีวัด ไม่ใช่เชิง defect

**สรุป**: อัตรา true-positive ที่แท้จริงของ phi4-only flag (นับจาก span
ที่ถูกยกมาเป็นหลักฐานจริงๆ หายไปหลัง re-OCR) อยู่ที่ **~51%** ไม่ใช่ ~83%
ตามตัวเลขดิบ — คนละครึ่งตัวของภาพเดิม เปลี่ยนสมการต้นทุน/ผลตอบแทนของการรัน
remediate เต็ม 16 ชม. อย่างมีนัยสำคัญ (ครึ่งหนึ่งของงานที่ทำอาจไม่แก้อะไร
เลยที่จุดที่ flag ไว้จริง)

**หยุดที่ 35/100 แทนที่จะรันต่อจนครบ 100** เพราะตัวปัญหาคือ **design ของ
การวัด** ไม่ใช่ขนาด sample — แก้ `sample_kernel_a_check.py` เป็น 2 phase
แยกกัน (`--phase=ocr` รัน OCR ทุกหน้าก่อน โหลดโมเดลเดียว, `--phase=adjudicate`
ค่อย adjudicate ทีหลัง โหลด 2 โมเดล) แก้ปัญหาการสลับโมเดล 3 ตัวทุกๆ 1 หน้า
บน GPU 12GB ได้จริง — phase OCR ที่เหลือ 64 หน้าเสร็จในรอบเดียว (ก่อนหน้านี้
รอบเดียวได้แค่ ~3 หน้าตอนสลับโมเดลทุกครั้ง) แล้วเพิ่ม `--phase=report` เทียบ
span อัตโนมัติ (ไม่เรียก LLM เพิ่ม) รายงานคู่กับ verdict ดิบ

**ผลครบ 100/100 ตัวอย่าง** (คงรูปแบบเดิมจาก n=35 ไว้ ตัวเลขนิ่งขึ้น):

| verdict (phi4, gemma) | จำนวน |
|---|---|
| (new, new) | 85 |
| (both_ok, new) | 6 |
| (new, old) | 3 |
| (old, new) | 2 |
| (both_ok, both_ok) | 4 |

- ตัวเลขดิบ new/new = 85/100 (85%)
- **span-confirmed rate (ทั้ง 100 ตัวอย่าง) = 56/100 (56%)** — คือสัดส่วนที่
  span ที่ถูก flag จริงๆ หายไปหลัง re-OCR
- ในกลุ่ม verdict (new,new) 85 ตัวอย่าง: span เปลี่ยนจริงแค่ **53/85 (62%)**
  อีก **32/85 (38%)** verdict บอก "new" ทั้งที่ span ที่ถูก flag ไม่เปลี่ยน
  เลย (สัญญาณเท็จเชิงวิธีวัด)

ตัวเลขจาก n=100 ยืนยันรูปแบบเดียวกับ n=35 (51%→56% span-confirmed, 59%→62%
ของ new/new ที่ span เปลี่ยนจริง) — ไม่ใช่ noise จาก sample เล็ก
**สรุปสุดท้าย: อัตรา true-positive ของ phi4-only flag ที่วัดแบบตรงจุดคือ
~56%** (ประมาณครึ่งหนึ่งของประชากร kernel A จริงๆ มี defect ที่ re-OCR
แก้ได้ตรงจุดที่ flag ไว้ ส่วนที่เหลือ ~44% คือ false positive ของ phi4
หรือ defect ที่ re-OCR รอบนี้ไม่ได้แก้)

### 9. Kernel-A remediation เต็มรูปแบบ — อนุมัติ, รัน, apply เสร็จสมบูรณ์ (27 ก.ค. 2569)

ผู้ใช้ถามตรงๆ ก่อนตัดสินใจ: "มีอะไรรับประกันว่า re-ocr แล้วความผิดพลาดจะ
ลดลง หรือว่าเอา ocr ทั้ง 2 ครั้งมาเทียบกันเอ" — คำตอบที่เปิดเผยตรงไปตรงมา:
**ไม่มี ground-truth verification กับภาพ PDF ต้นฉบับที่ไหนใน pipeline เลย**
ทั้ง batch เดิม 872 หน้าและ kernel A ใหม่ ตัดสินด้วย 2 โมเดล LLM เทียบ
ความ coherent ของข้อความเก่า-vs-ใหม่เท่านั้น "re-OCR ที่ผิดแบบลื่นไหล" (เช่น
รหัสวิชาอ่านผิดเป็นรหัสอื่นที่มีจริง หรือชื่อคนสะกดผิดแบบที่ไม่ดู garbled)
จะไม่ถูกจับโดยทั้งสองโมเดล เพราะโมเดลตัดสิน coherence ไม่ใช่ accuracy-to-
source ผู้ใช้รับทราบข้อจำกัดนี้แล้วตอบ "ok ทำทั้งหมดไปก่อนก็แล้วกัน" —
อนุมัติให้รัน kernel-A remediation เต็มรูปแบบ (2,101 work items / 1,329
ไฟล์)

**สคริปต์ใหม่**: `tools/corpus_prep/reocr_kernel_a_stage.py` (Phase-1-
equivalent สำหรับ kernel A) เขียนลง `reocr_pages_staging.jsonl` เดียวกับ
pipeline เดิม ทำให้ Phase 2 (`reocr_adjudicate.py`) และ Phase 3
(`reocr_apply.py`) ใช้ได้โดยไม่ต้องแก้โค้ดเลย เพราะทั้งคู่ประมวลผล "ทุกหน้าที่
staged/adjudicated แล้วแต่ยังไม่เสร็จ" อยู่แล้ว

**บั๊กที่เจอกลางทาง (self-shrinking file-level exclusion)**: การ merge ผล
100 sample เข้า `reocr_adjudication.jsonl` จริงก่อนเริ่มรัน ทำให้การคำนวณ
"ไฟล์ที่ยังไม่เคยถูกแตะ" (ซึ่งเทียบกับไฟล์เดียวกันที่ตัวเองกำลังเขียนอยู่)
มองว่าไฟล์ที่มีแค่ 1 หน้าอยู่ใน sample กลายเป็น "touched" ทั้งไฟล์ — ทำให้
หน้าอื่นๆ ในไฟล์เดียวกันที่ไม่เกี่ยวกับ sample เลยหายไปจาก pool เงียบๆ
334 หน้า จาก 102 ไฟล์ (pool เหลือ 1,806 จาก 2,101 ที่ถูกต้อง) แก้โดย exclude
sample-derived keys ออกจาก snapshot ที่ใช้เช็ค exclusion (ดู docstring ใน
`reocr_kernel_a_stage.py::load_kernel_a_items`) แล้ว regenerate cache ยืนยัน
กลับมาที่ 2,101 ตรงกับตัวเลข sizing เดิม งานที่รันไปแล้วภายใต้ scope ผิด
(1,804/1,806) ไม่ถูกทิ้ง เพราะเป็น subset ที่ถูกต้องของ 2,101 อยู่แล้ว —
รันรอบสองเก็บ 194 ที่ขาดหายให้ครบ

**ผลลัพธ์สุดท้าย**:
- Phase 1 (OCR): **2,866 หน้า staged** (มากกว่า 2,101 work items เพราะรวม
  หน้าที่ split-document sibling หลายไฟล์ชี้มาที่ (pdf,page) เดียวกัน)
- Phase 2 (dual-model adjudicate): **2,866/2,866 adjudicated ครบ**
- Phase 3 (apply, `reocr_apply.py --apply`):
  - **1,982 หน้าเขียนจริง ใน 393 ไฟล์** (มี `.pre_reocr.bak` backup ทุกไฟล์
    ก่อนเขียนครั้งแรก)
  - 854 หน้า unchanged (ข้อความใหม่เหมือนเดิมอยู่แล้ว)
  - 80 หน้า missing_header (หา `## Page N` ไม่เจอ — ส่วนใหญ่เป็นไฟล์
    split-piece `__1.md`/`__2.md` ที่ไม่มี header หน้า 1 จริงๆ ในไฟล์นั้น
    ไม่ใช่ error ของ pipeline สคริปต์แค่ปฏิเสธเดาแล้ว report ไว้ ยังไม่ได้
    ตรวจสอบเป็นรายไฟล์ว่าควรมี override เหมือน
    `reocr_page_occurrence_overrides.json` เดิมหรือไม่)
  - **308 หน้ารอ human review** (verdict ไม่ตรงกันระหว่าง 2 โมเดล) ผ่าน
    `consensus_review/pages/1_reocr_diff_review.py` เหมือน batch เดิม —
    ยังไม่ auto-apply
  - 18 หน้า keep-old (มี human decision แล้วจาก batch/sample ก่อนหน้า)

Kernel-A remediation (ทั้ง Mechanism A gap ที่ระบุใน §7) ปิดสถานะ **เสร็จ
สมบูรณ์** ณ จุดนี้ เหลือแค่ 308 หน้ารอ human review ตามปกติของ pipeline
(ไม่ใช่งานค้างของ kernel-A specifically) Mechanism B (2 ไฟล์ severe จาก
§7) ยังไม่ถูกแตะ — ผู้ใช้ยังไม่ได้ขอให้ทำ

**ลด review queue เพิ่มเติมด้วย bulk decision (27 ก.ค. 2569)**: ในกลุ่ม
308 หน้าที่รอ review, breakdown ตาม verdict pair พบว่า **105 หน้าเป็น
(both_ok, both_ok)** — ทั้งสองโมเดลเห็นตรงกันว่า**ไม่มี defect ในทั้งสอง
เวอร์ชัน** เลย ผู้ใช้สังเกตว่า OCR ที่เพี้ยนจริงมักทำให้ความยาว text
เปลี่ยนไปคนละเรื่อง (ไม่ใช่ต่างกันเล็กน้อย) — ใช้เป็นเกณฑ์คัดกรองฟรี (ไม่
เรียก LLM เพิ่ม) สุ่มดู 10/105 ตัวอย่างมือก่อน: 8/10 เหมือนกันแทบทุกตัวอักษร
(ต่างแค่ตัวคั่นหน้า `---` ท้าย text เก่าที่ text ใหม่ไม่มี), 2/10 มีความต่าง
เล็กน้อยจริงแบบ "fluently wrong" ที่เตือนไว้ตอนขออนุมัติ (ชื่อคน "ประวีณ์"
vs "ประวีณ" สะกดคนละแบบ อ่านลื่นทั้งคู่; `---` แทรกผิดที่ในอีกไฟล์หนึ่ง) —
ไม่มีทางบอกได้ว่าอันไหนตรงต้นฉบับโดยไม่เปิด PDF ผู้ใช้ยอมรับความเสี่ยงนี้
(เล็กน้อย, หวังพึ่ง cross-reference จาก person-entity tagging ถ้าชื่อ
เดียวกันปรากฏ >1 ที่ในคอร์ปัส) เช็คสมมติฐาน "ความยาวใกล้เคียง = ปลอดภัย"
กับทั้ง 105 หน้าจริง (ไม่ใช่แค่ 10 ตัวอย่าง): **relative length delta สูงสุด
2.3% (42 ตัวอักษรจาก 1,858), median 0%** — ไม่มีหน้าไหนเข้าใกล้ pattern
การเพี้ยนแบบ OCR-repetition ที่ pipeline นี้ตั้งใจจับเลย

ตัดสินใจ: เขียน bulk **keep-old** decision (ไม่ apply-new) ให้ทั้ง 105 หน้า
ผ่าน `logic.append_reocr_review_decision` ลง
`reocr_review_decisions.jsonl` — เลือก keep-old แทน apply-new เพราะ
`both_ok/both_ok` แปลว่าไม่มี defect ให้แก้อยู่แล้ว apply-new จึงไม่มี
ประโยชน์ แต่มีความเสี่ยงเล็กๆ ไม่คุ้มเปล่าๆ (แบบเคสชื่อคน) แต่ละ record มี
note อธิบายเหตุผล/ตัวเลขที่ใช้ตัดสินใจ ยืนยันผลด้วย dry-run
`reocr_apply.py`: keep-old นับเพิ่มจาก 18 → 123 (+105 ตรงเป๊ะ), queue
รอ review ลดจาก 308 → **203 หน้า** ไม่ต้องรัน `--apply` จริงเพราะ keep-old
ไม่เขียนอะไรทับ corpus อยู่แล้ว

### 10. Round 2/3 tie-break สำหรับ pool ที่เหลือ (203 → 137 → 99 → 88 หน้า, 26–27 ก.ค. 2569)

**203 → 137** (นอกรอบ tie-break ที่บรรยายด้านล่าง, ใช้กฎที่ validate แล้วจาก
§9 ซ้ำ): เอากฎ old-vs-new length-delta (`rel ≤ 1%` = ปลอดภัย, ใช้เกณฑ์
เดียวกับ 105-หน้า both_ok/both_ok ใน §9) กับ span-confirmation
(`build_span_lookup()` ใน `sample_kernel_a_check.py` — เช็คว่า span ที่ถูก
flag ไว้ตอน scan ยังอยู่ใน old text แต่หายไปจาก new text หรือไม่ หลักฐานฟรี
ไม่ต้องเรียก LLM เพิ่ม) กับ pool 137 หน้าที่เหลือหลัง §9 (203 ลบหน้าที่มี
verdict-pair อื่นที่ไม่ใช่ both_ok/both_ok) ได้ 34 หน้า `rel≤1%` → keep-old,
อีก 4 หน้า span หายไปจริง + per-model verdict เอนไปทาง "new" (`round_lean`
helper: majority vote จาก 2 model verdict ต่อหน้า ตัด both_ok/both_bad ออก
นับแค่ new/old) → apply-new รวม **38 หน้าออกจาก queue, เหลือ 137**

**Round 2 (temperature=0.0 ซ้ำเดิม) — พิสูจน์แล้วว่าไม่มีประโยชน์**: แผนเดิม
(อนุมัติไว้ก่อนหน้านี้) คือรัน OCR ซ้ำหน้าเดิมด้วย setting เดิมทุกอย่าง แล้ว
ใช้เป็น "ตัวตัดสินรอบ 3" บนสมมติฐานว่า re-OCR ไม่ deterministic ในทางปฏิบัติ
— **สมมติฐานนี้ผิด**: เขียนสคริปต์วิเคราะห์ใหม่
(`tools/corpus_prep/reocr_tiebreak_analyze.py`) เทียบ round-1 กับ round-2
ทั้ง text และ per-model verdict พบว่า**เหมือนกันทุกตัวอักษร ทั้ง 137/137
หน้า** ทั้งข้อความ OCR และ verdict ของทั้ง 2 โมเดล — `ocr_image()` ที่
`temperature=0.0` deterministic เต็มรูปแบบสำหรับ pipeline นี้ สาเหตุที่
ข้อความคอร์ปัสเดิมต่างจาก re-OCR รอบแรกจึงไม่ใช่ noise ระหว่างรัน (น่าจะเป็น
script/prompt/model-pull drift ตั้งแต่ตอน ingest คอร์ปัสครั้งแรก แต่ไม่ได้
สืบต่อว่าคืออะไรแน่ๆ) — รายงานผู้ใช้ตรงๆ แทนที่จะรัน round 3 แบบเดิมต่อไป
ทั้งที่รู้แล้วว่าไม่มีประโยชน์

**Round 3 ออกแบบใหม่ — perturbed re-OCR**: ผู้ใช้เลือก "Re-OCR แบบ
perturbed" (ไม่ human review ทั้งหมด, ไม่ bulk keep-old) แล้วถามเพิ่มว่าปรับ
DPI ได้ไหม — เพิ่ม `temperature`(default 0.3, อิงจาก retry-temperature
convention เดิมใน `reocr_adjudicate.call_compare_model`) และ `dpi` (ใช้ 400,
ปกติ 300) เป็น parameter ผ่านทั้ง `ocr_image()` → `ocr_page()` →
`reocr_tiebreak_round.py` ยืนยันว่า `preprocess_image()` ไม่ resize รูปเลย
(แค่ brightness/contrast/sharpness) ดังนั้น DPI สูงขึ้นส่งรายละเอียด pixel
จริงเข้าโมเดลมากขึ้น ไม่ใช่แค่แปลงเปล่าๆ ยืนยัน perturbation ทำงานจริง:
round-3 text ต่างจาก round-1 ใน 96/99 หน้า (ตรงข้ามกับ round-2 ที่ 0/137)

**99 → 88** (27 ก.ค. 2569): คำนวณ joint distribution ของ (round1_lean,
round3_lean) จาก 99 หน้า:

| (round1, round3) | จำนวน |
|---|---|
| (new, new) | 17 |
| (old, old) | 2 |
| (new, old) หรือ (old, new) — ขัดแย้งจริง | 2 |
| มี neutral ร่วมด้วย | 78 |

ก่อนสรุปว่า (new,new) 17 หน้า = auto-apply ได้เลย เจอช่องโหว่เชิงวิธีวัด
(คล้าย round-2): **round-3's verdict ตัดสินจาก text ของ round-3 เอง ไม่ใช่
text ของ round-1 ที่ staged ไว้ (ซึ่งเป็นข้อความเดียวที่ `reocr_apply.py`
เขียนได้จริง)** — ถ้า 2 รอบเห็นพ้องกันแค่ "ทิศทาง" (new ดีกว่า old) แต่ตัว
text ไม่ตรงกัน ก็ยังไม่รู้ว่าข้อความไหนถูก เช็ค `rel(t1_len, t3_len)`:
**6/17 เท่านั้น** ที่ 2 รอบ OCR อิสระให้ข้อความเกือบเหมือนกัน (`rel≤1%`) —
6 หน้านี้ถือเป็นหลักฐานแข็งแรง (2 ตัวอย่างอิสระลงเอยที่ข้อความเดียวกัน และ
ทั้งคู่ verdict ว่าดีกว่า old) อีก 11/17 ข้อความต่างกันจริง (สูงสุด 66%) —
เห็นพ้องแค่ทิศทาง ไม่ใช่เนื้อหา ไม่ apply

กลุ่ม "มี neutral ร่วมด้วย" 78 หน้า สุ่มดึง text ตัวอย่าง old/t1/t3 มาเทียบ
ให้ผู้ใช้ดูก่อนตัดสินใจ (ตามที่ขอ) พบ 2 รูปแบบ: (a) ความต่างเป็นแค่ markdown
bold/การตัดบรรทัด ไม่ใช่เนื้อหาจริง, (b) **round-1 OCR หลุดเข้าโหมด
table-repetition** (ความยาวบวมหลายเท่าตัว เช่น old=3,990 ตัวอักษร →
t1=14,452 → t3=3,985 ใกล้เคียง old) — สแกนทั้ง 99 หน้าหารูปแบบ (b) ทั่วไป
(`t1 > 1.5×max(old,t3)` และ `t3` ใกล้ `old` ≤5%) เจอ **3 หน้าเท่านั้น** (คนละ
เอกสารกัน ทั้งหมดเป็นตารางเปรียบเทียบรายวิชาหลักสูตรเก่า/ใหม่ ประเภทเดียวกับ
ที่ `strip_course_comparison_tables` เคยเจอปัญหามาก่อน) — table-repetition
เป็น failure mode เฉพาะของเอกสารประเภทนี้ตอน DPI 300

**สรุปการ apply รอบนี้ (11 หน้าออกจาก queue, 99 → 88)**:
- **6 apply-new** ผ่าน pipeline ปกติ (`reocr_apply.py --apply`, ใช้ text
  ของ round-1 ที่ staged ไว้แล้ว เพราะ round-1≈round-3 อยู่แล้ว)
- **2 keep-old** ผ่าน pipeline ปกติ (round1 & round3 เห็นพ้องกันว่า old ดี
  กว่า)
- **3 หน้า apply ด้วย round-3's text โดยตรง** (ไม่ใช่ผ่าน `reocr_apply.py`
  ซึ่งเขียนได้แค่ text ของ round-1 เท่านั้น) — เขียนตรงเข้าไฟล์คอร์ปัสด้วย
  `apply_to_file()` ตัวเดียวกับที่ `reocr_apply.py` ใช้ข้างใน (import ตรง)
  แล้ว log decision เป็น **keep-old** (ไม่ใช่ apply-new) เพื่อกัน
  `reocr_apply.py` เขียนทับด้วย text ของ round-1 ที่ garbled ซ้ำในอนาคต —
  note อธิบายไว้ชัดว่า corpus ถูกแก้ไปแล้วด้วย round-3's text นอก pipeline
  ปกติ
- เหลือ 88 หน้า ที่หลักฐานที่มี (format-only diff, ไม่มี length-blowup
  signature) ไม่ชัดพอจะ auto-resolve → เข้า human review ตามเดิม

Queue สุดท้าย ณ จุดนี้: **88 หน้ารอ human review**, keep-old สะสม 215,
apply-new เพิ่มอีก 6+3(manual)=9 หน้าในรอบนี้ ไฟล์วิเคราะห์ทั้งหมด (pool
JSON, joint-distribution JSON, ตัวอย่าง text ที่ดึงมาเทียบ) อยู่ใน
scratchpad ของ session นี้เท่านั้น ไม่ได้ commit เข้า repo (เป็นไฟล์
วิเคราะห์ระหว่างทาง ไม่ใช่ pipeline artifact ถาวร)

### 11. Mechanism B -- แก้ 2 ไฟล์ severe เสร็จแล้ว (27 ก.ค. 2569)

ผู้ใช้ขอให้ทำ 2 ไฟล์ severe จาก §7 ตอนนี้เลย (ยืนยันแล้วว่าไม่เกี่ยวกับคิว
88 หน้าใน §10 -- คนละ pipeline entry point กัน) เพราะระหว่างรอรีวิว 88
หน้าเองก็ยังใช้เวลาอีกพอสมควรอยู่แล้ว

หาตำแหน่งหน้าที่แน่ชัดก่อน (ของเดิมในบันทึก §7 บอกแค่ชื่อไฟล์ ไม่ได้ระบุ
เลขหน้า) ด้วยสคริปต์หา "run ของอักขระตัวเดียวกันที่ซ้ำติดกันยาวที่สุด" ต่อ
ไฟล์ พบว่ารูปแบบจริงไม่ใช่ตัวอักษรเดี่ยวล้วนๆ ตามที่ §7 สรุปไว้ (ตัวเลข
นับความถี่รวมทั้งไฟล์ ไม่ใช่ run ต่อเนื่องแท้ๆ) แต่เป็น:
- ไฟล์ 1 (`2564/ครั้งที่ 1/...อาจารย์พิเศษ...คณะบริหารธุรกิจ.md`) หน้า 1:
  สลับ `ก.ก.ก.ก....` ในเซลล์ตาราง (คอลัมน์เหตุผลความจำเป็น) ~1,600 รอบ
- ไฟล์ 2 (`2568/ครั้งที่ 4/...ควบสองปริญญา...การบินนานาชาติ.md`) หน้า 2:
  จุด "." ซ้ำติดกัน 1,984 ตัว ในบรรทัด checkbox เติมคำ (ที่ปกติควรมีแค่
  จุดไม่กี่ตัวเป็น fill-in blank)

ทั้งสองแบบยืนยันว่าเป็น OCR-repetition-loop จริง ไม่ใช่ dot-leader ที่ถูก
ต้องตามฟอร์แมต (ต่างจาก 44 ไฟล์ที่ threshold ต่ำกว่าซึ่งน่าจะเป็นของจริง
ตามฟอร์แมต)

Re-OCR ทั้ง 2 หน้าสด (setting ปกติ, temp=0.0/dpi=300 -- ไม่ใช่ perturbed
เพราะไม่ใช่ tie-break case) แล้วเขียนเข้า `reocr_pages_staging.jsonl`/
`reocr_adjudication.jsonl` จริงตรงๆ (ไม่ใช้สคริปต์ one-off แยก) เพื่อให้
เข้า pipeline เดียวกับทุกหน้าที่เหลือ -- ทั้งสองโมเดล verdict "new" ตรงกัน
ทั้งคู่ (auto-apply ตามเกณฑ์ปกติ ไม่ต้องมีคนตัดสิน) ยืนยันด้วย `decide_action`
โดยตรงก่อน apply ว่า resolve เป็น `apply`/"both models verdict new" จริง
รัน `reocr_apply.py --apply` เขียนเข้าคอร์ปัสจริง แล้วเปิดไฟล์ตรวจซ้ำ:
ไฟล์ 1 เหลือ "ก"=102 ตัวทั้งไฟล์ (จาก 1,598), ไฟล์ 2 เหลือ "."=72 ตัวทั้งไฟล์
(จาก 2,056) -- ข้อความใหม่อ่านได้ปกติทั้งคู่ ไม่มี repetition-loop เหลือ

Mechanism B **ปิดสถานะสมบูรณ์** -- ทั้ง 2 ไฟล์ severe ที่ระบุไว้ใน §7 ได้รับ
การแก้ไขแล้ว ผ่าน quality gate เดียวกับทุกหน้าในทั้ง pipeline (dual-model
adjudication ก่อน apply เสมอ ไม่ข้ามขั้นตอนแม้จะมั่นใจว่าเป็น defect ชัดเจน
อยู่แล้วจากหลักฐาน regex) ส่วน 44 ไฟล์ที่ threshold ต่ำกว่า (30-49 ตัวซ้ำ)
ยังไม่แตะ -- ยังเชื่อว่าเป็น dot-leader/placeholder ที่ถูกต้อง ไม่ใช่
corruption ตามที่ประเมินไว้ใน §7
