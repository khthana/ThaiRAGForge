# dataset สำหรับ fine-tune ภาษาไทย หาจากไหน — และ silver set ของเราใช้ได้มั้ย

Part of: [map.md](../map.md)
Type: research
Status: resolved
Blocked by: —

## Question

ตอบข้อสุดท้ายจากคำถามตั้งต้นของผู้ใช้: **"จะหา dataset จากไหน"**

### สามทางที่ต้องเทียบ

1. **ชุดข้อมูลสาธารณะภาษาไทย** — มีอะไรบ้างที่ใช้เทรน retrieval ได้จริง
   (MIRACL-th, MTEB Thai retrieval tasks, TyDiQA-th, XQuAD-th, Thai Wikipedia QA,
   scb-mt-en-th-2020, Thai-Sentence-Vector-Benchmark) — แต่ละชุด **ขนาดเท่าไหร่, โดเมนอะไร,
   ใบอนุญาตอะไร** และที่สำคัญที่สุด: **ใกล้เคียงกับภาษาราชการ/มติที่ประชุมของสถาบันแค่ไหน**
   (คำตอบที่คาดไว้คือ "ไม่ใกล้เลย" — ถ้าเป็นแบบนั้น ต้องบอกให้ชัดว่ามันยังมีประโยชน์ในฐานะอะไร
   เช่น เทรนขั้นแรกก่อนค่อยปรับด้วยข้อมูลโดเมน)

2. **Silver query set ที่มีอยู่แล้ว** — โปรเจกต์มีตัวสร้างคู่ (เรื่อง/title ของแต่ละมติ →
   ตัวมันเอง) ได้ ~2,853 คู่ฟรีทันที **แต่มีความเสี่ยงที่ต้องประเมินให้ตรง ๆ**:
   คู่เหล่านี้ "ง่าย" เพราะถ้อยคำของ query ทับกับเอกสารเกือบทั้งหมด ในขณะที่ Gold set
   ถูกเขียนใหม่ให้ **หลีกเลี่ยงถ้อยคำในชื่อเรื่องโดยเจตนา** —
   **การเทรนบน silver อาจสอนโมเดลผิดทาง** (สอนให้จับ lexical overlap ซึ่ง BM25 ทำได้ดีอยู่แล้ว
   และเป็นสิ่งที่ dense retrieval ควรจะ *เสริม* ไม่ใช่ *เลียนแบบ*) — ประเมินความเสี่ยงนี้ให้ชัด

3. **สร้างข้อมูลสังเคราะห์จาก corpus ตัวเอง** — ให้ LLM อ่าน chunk แล้วแต่งคำถามภาษาไทย
   ที่คนจะถามจริง จากนั้นขุด hard negative จาก index ที่มีอยู่แล้ว
   ต้องหาคำตอบว่า: วิธีขุด hard negative ที่เป็นมาตรฐาน (top-k จาก retriever ปัจจุบัน
   หักตัวที่เป็น positive ออก), จำนวน negative ต่อ positive ที่ควรใช้,
   ปัญหา **false negative** (มติที่ตอบคำถามได้จริงแต่ไม่ได้ถูก label) —
   ซึ่งใน corpus นี้มีสูงเป็นพิเศษ เพราะมติหลายฉบับมีเนื้อหาซ้ำกันเชิงโครงสร้าง
   (เช่น "อาจารย์พิเศษเกิน 50%" ที่มีทุกคณะทุกปี)

### สิ่งที่ต้องได้จากใบนี้

ข้อเสนอที่เจาะจงพอจะลงมือได้: จะใช้ชุดไหน อย่างละกี่คู่ สร้างยังไง ใช้เวลาเท่าไหร่
และความเสี่ยงข้อไหนที่ยังไม่มีทางลด

## ข้อจำกัดที่ห้ามลืม

- ห้ามส่งเนื้อหามติออกไปยัง API ภายนอก (เหตุผลเดียวกับที่ตัด Group D commercial embedder ออก) —
  ถ้าเสนอให้ใช้ LLM สร้างคำถาม ต้องเป็น LLM ที่รันในเครื่องได้ (โปรเจกต์มี Ollama อยู่แล้ว)
- ข้อมูลเทรนต้อง**ไม่ปนกับ Gold 73** ไม่งั้นผลวัดหลัง fine-tune เชื่อไม่ได้เลย

## Answer

> ตั๋ววิจัย — รวบรวมข้อเท็จจริงพร้อมแหล่งอ้างอิง **ไม่ตัดสินใจแทนผู้ใช้**
> ตัวเลขขนาด/ใบอนุญาตทุกตัวด้านล่างตรวจจาก HF dataset card / `datasets-server` API จริง
> (`/api/datasets/<id>`, `/size`, `/splits`, `/filter`) เมื่อ 2026-07-31 ไม่ได้เดา

---

## บทสรุปสั้น (อ่าน 60 วินาที)

| ทาง | ปริมาณคู่ที่ใช้ได้จริง | ระยะห่างจากโดเมน | คำตัดสิน |
|---|---|---|---|
| 1. ชุดสาธารณะไทย | ~9k–21k คู่ (ที่เป็น query→passage จริง) | **ไกลมาก** (Wikipedia/บทวิจารณ์/บทแปล) ยกเว้น 1 ชุด | ใช้เป็น **stage-1 warm-up เท่านั้น** และมี**หลักฐานในบ้านที่ค้านแรง** |
| 2. Silver 2,853 คู่ | 2,853 (ลบ Gold แล้วเหลือ ~2,372) | ใกล้ที่สุด แต่**ผิดชนิดสัญญาณ** | **ห้ามใช้เป็นสัญญาณหลัก** ใช้ได้เฉพาะเป็น warm-up ที่ผ่านการดัดแปลง |
| 3. สังเคราะห์จาก corpus ด้วย LLM ในเครื่อง | 6k–9k คู่ + hard negatives | ตรงโดเมน 100% | **เป็นทางหลักที่แนะนำให้พิจารณา** ต้นทุน ~1–2 วันเครื่อง |

**ข้อเท็จจริงที่แพงที่สุดในใบนี้**: โปรเจกต์นี้**มีหลักฐานเชิงประจักษ์ของตัวเองอยู่แล้ว**ว่าทาง 1 ไม่พอ —
`sct` (`kornwtp/SCT-KD-BGE-M3-model-phayathaibert`) เทรนบน scb-mt-en-th-2020 ได้ recall@10 = **0.1519**
เสมอทางสถิติกับ m2v (static lookup table, 0.1472) และแพ้ BM25 เปล่า ๆ **+0.4158** (Holm-adj p<0.0001);
`congen` ที่ distill จาก BGE-M3 ได้ 0.4134 ก็ยัง**แพ้ BM25 อย่างมีนัยสำคัญ** (+0.1543, p=0.0080)
[`RAG/docs/paper-results-summary.md`]. นี่คือหลักฐานตรงตัวว่า *corpus ประโยคคู่ภาษาไทยทั่วไป
ไม่ผลิต retriever ที่เก่งบน corpus นี้* — แข็งกว่าการอ้าง literature อย่างเดียว

---

## ทาง 1 — ชุดข้อมูลสาธารณะภาษาไทย (ตรวจจากการ์ดจริง)

### ตารางข้อเท็จจริง

| ชุด | ขนาดที่ตรวจได้ | คู่ query→passage ที่ใช้เทรนได้จริง | โดเมน | ใบอนุญาต |
|---|---|---|---|---|
| **MIRACL-th** ([card](https://huggingface.co/datasets/miracl/miracl)) | train **2,972 queries / 21,293 judgments**; dev 733 / 7,573; corpus **542,166 passages** ([miracl-corpus th](https://huggingface.co/datasets/miracl/miracl-corpus)) | **~2,972 queries** (positives เป็น subset ของ 21,293 judgments — judgments รวม label 0 ด้วย) | Wikipedia ไทย, คำถามเปิดโดเมนที่ native speaker เขียน | **Apache-2.0** |
| **mteb/MIRACLRetrieval (th)** ([card](https://huggingface.co/datasets/mteb/MIRACLRetrieval)) | th-queries **733** / th-qrels **7,573** / th-corpus **542,166** — **มีแต่ split `dev`** | **0 (eval-only)** | เหมือนบน | CC-BY-SA-4.0 |
| **mteb/XQuADRetrieval (th)** ([card](https://huggingface.co/datasets/mteb/XQuADRetrieval)) | th-queries **1,180** / th-qrels 1,180 / **th-corpus เพียง 240 ย่อหน้า** — split `validation` เท่านั้น | **0 (eval-only, corpus เล็กเกินจะเป็น retrieval จริง)** | SQuAD แปล → Wikipedia | CC-BY-SA-4.0 |
| **google/xquad** (`xquad.th`) ([card](https://huggingface.co/datasets/google/xquad)) | validation **1,190** rows, ไม่มี train | 0 | เหมือนบน | CC-BY-SA-4.0 |
| **TyDiQA** ([card](https://huggingface.co/datasets/google-research-datasets/tydiqa)) | `primary_task` ภาษาไทย: train **10,362** / validation **2,245** (นับจาก `/filter?where="language"='thai'`) | ~ครึ่งหนึ่งของ 10,362 (หลายข้อ label = ไม่มีคำตอบ) → ประมาณ **5k–7k** | Wikipedia ไทย | **Apache-2.0** |
| ↳ **TyDiQA-GoldP (`secondary_task`)** | train 49,881 / val 5,077 — **แต่ไม่มีภาษาไทย** | **0** | — | — |
| **iapp_wiki_qa_squad** ([card](https://huggingface.co/datasets/iapp/iapp_wiki_qa_squad)) | **5,761 / 742 / 739** จาก 1,529 / 191 / 192 บทความ | **5,761** (มีเวอร์ชันจัดรูป retrieval แล้ว: [`kornwtp/iapp-wikiqa-tha-qaretrieval`](https://huggingface.co/datasets/kornwtp/iapp-wikiqa-tha-qaretrieval) — 5,761/742/739) | Wikipedia ไทย | **MIT** |
| **pythainlp/thaiqa_squad** ([card](https://huggingface.co/datasets/pythainlp/thaiqa_squad)) | train **4,000** / dev **74** | 4,000 | Wikipedia ไทย (NECTEC) | **CC-BY-NC-SA-3.0 → ห้ามใช้เชิงพาณิชย์** ⚠️ |
| **scb-mt-en-th-2020** ([card](https://huggingface.co/datasets/airesearch/scb_mt_enth_2020)) | train **801,402** / val 100,173 / test 100,177 = **1,001,752 คู่** | **0 คู่ query→passage** — เป็นคู่แปล EN↔TH ไม่ใช่ถาม-ตอบ | generated reviews 40.95%, task dialogs 23.17%, web 11.67%, **เอกสารราชการเพียง 2.46% (~24,600 คู่)**, Wikipedia 3.26% | CC-BY-SA-4.0 |
| **Thai-Sentence-Vector-Benchmark** ([repo](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark)) | **เป็น benchmark ไม่ใช่ training set** — eval ด้วย Thai STS-B, Wisesight/Wongnai, XNLI, และ retrieval = XQuAD/MIRACL/TyDiQA | 0 (ให้ notebook เทรน แต่ไม่แจก data) | — | ไม่ระบุใน README |

### ชุดที่ไม่ได้อยู่ในลิสต์ตั้งต้น แต่ **ใกล้โดเมนที่สุดที่หาเจอ**

| ชุด | ขนาด | ทำไมสำคัญ | ใบอนุญาต |
|---|---|---|---|
| **`airesearch/WangchanX-Legal-ThaiCCL-RAG`** ([card](https://huggingface.co/datasets/airesearch/WangchanX-Legal-ThaiCCL-RAG)) | train **8,211** / test **3,742** | schema = `question` / `positive_contexts` / `hard_negative_contexts` / `positive_answer` — **เป็นรูปแบบ retrieval training พร้อมใช้** และเนื้อหาเป็น **พระราชบัญญัติ/กฎหมายไทย** คือภาษาราชการเชิงนิติกรรม ซึ่ง**ใกล้ทะเบียนภาษาของมติที่ประชุมมากกว่า Wikipedia อย่างเทียบไม่ติด** (ประโยคยาว, นามนัย, "ตามที่...จึงเห็นชอบ...", เลขมาตรา/ข้อ) | **MIT** |
| `iapp/rag_thai_laws` ([card](https://huggingface.co/datasets/iapp/rag_thai_laws)) | **42,755** rows | corpus กฎหมายไทยสำหรับ RAG (ไม่ใช่คู่ query-positive สำเร็จรูป) ใช้เป็นแหล่ง passage ราชการเพิ่มได้ | MIT |

### "ใกล้ภาษามติที่ประชุมแค่ไหน" — คำตอบตรง ๆ

**ไกลมาก และเป็นความไกลสามชั้นซ้อน ไม่ใช่ชั้นเดียว**:

1. **ชั้นทะเบียนภาษา (register)** — ทุกชุดยกเว้น WangchanX-Legal เป็นภาษา Wikipedia/บทวิจารณ์/บทสนทนา
   มติสภาวิชาการเป็นภาษาราชการเชิงพิธีการ: "เรื่อง ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีไม่กระทบกระเทือนโครงสร้าง)"
   — โครงสร้างประโยคแบบนี้ไม่ปรากฏใน corpus สาธารณะชุดใดเลย
2. **ชั้นชนิดคำถาม** — MIRACL/TyDiQA/XQuAD เป็น *factoid open-domain* ("X เกิดที่ไหน")
   ในขณะที่คำถาม Gold ของโปรเจกต์เป็น **entity-anchored + aggregate/temporal**
   ("หลักสูตร X ปรับปรุงกี่ครั้ง แต่ละครั้งมีรายละเอียดอย่างไร") — คนละงานเชิงความหมาย
3. **ชั้นโครงสร้าง corpus** — Wikipedia ย่อหน้าหนึ่งพูดเรื่องหนึ่ง; corpus นี้มีมติหลายพันฉบับที่
   **แทบเหมือนกันเชิงโครงสร้าง** ต่างกันแค่ entity/ปี — เป็นการกระจายตัวที่ Wikipedia ไม่มี

**หลักฐานในบ้านที่ยืนยัน** (แข็งกว่า literature): `congen` และ `sct` ทั้งคู่มาจากสาย
Thai-Sentence-Vector-Benchmark และเทรนบน scb-mt-en-th-2020 / Thai Wikipedia — **ทั้งคู่แพ้ BM25
อย่างมีนัยสำคัญบน Gold 73** (`RAG/docs/paper-results-summary.md` §BM25 lexical baseline)
นี่คือการทดลองที่โปรเจกต์นี้รันไปแล้วโดยไม่ตั้งใจ: *เทรนบนข้อมูลไทยทั่วไป → ได้ retriever ที่แพ้ BM25 บนโดเมนนี้*

### แล้วมันยังมีประโยชน์ในฐานะอะไร — และ "two-stage" มีหลักฐานจริงหรือไม่

**มี — แต่หลักฐานสนับสนุน "stage-2 ต้องเป็นข้อมูลโดเมน" มากกว่าสนับสนุน "stage-1 คุ้มค่า"**

- **GPL (Wang et al., NAACL 2022)** — <https://arxiv.org/abs/2112.07577> / <https://aclanthology.org/2022.naacl-main.168/>:
  ปรับ dense retriever เข้าโดเมนใหม่ด้วย **query generator + pseudo-labeling จาก cross-encoder**
  บน corpus โดเมนที่ไม่มี label เลย ได้ดีกว่า SOTA dense retrieval แบบ out-of-the-box **สูงสุด +9.3 nDCG@10**
  บน 6 โดเมนเฉพาะทาง และ "requires less unlabeled data and is more robust in its training than previous methods"
  → **นี่คือหลักฐานที่ตรงกับทาง 3 ไม่ใช่ทาง 1**
- **Contriever (Izacard et al. 2021)** — <https://arxiv.org/abs/2112.09118>: สุ่ม span สองอันจากเอกสารเดียวกัน
  เป็นคู่ positive → เป็น **pre-training/warm-up** ที่ยังต้อง fine-tune ต่อ ไม่ใช่ตัวจบ
- **Chang et al., ICLR 2020, "Pre-training Tasks for Embedding-based Large-scale Retrieval"** —
  <https://research.google/pubs/pre-training-tasks-for-embedding-based-large-scale-retrieval/>:
  งานเทียบ ICT / Body-First-Selection / Wiki-Link-Prediction (คือคู่ pseudo แบบ "ส่วนหนึ่งของเอกสาร → เอกสารนั้น")
  สรุปว่า **"the key ingredient of learning a strong embedding-based Transformer model is the set of
  pre-training tasks"** และ pre-training ที่ออกแบบดีทำให้ Transformer ชนะ BM25 ได้
  → **สนับสนุน two-stage จริง แต่ระดับ "pre-training หลายล้านคู่" ไม่ใช่ "warm-up 9k คู่"**
- **BEIR (Thakur et al. 2021)** — <https://arxiv.org/abs/2104.08663>: หลักฐานว่า dense retriever
  ที่เทรนบนโดเมนทั่วไป **transfer ข้ามโดเมนไม่สม่ำเสมอ** และหลายโดเมนแพ้ BM25 → เหตุผลว่าทำไม
  stage-1 อย่างเดียวไม่พอ

**คำตัดสินทาง 1**: ใช้ได้เป็น stage-1 อย่างจำกัด แต่โดยรวม **ROI ต่ำ** เพราะ (ก) Qwen3-Embedding-4B
ผ่าน multilingual retrieval pre-training ระดับล้านคู่มาแล้ว การเติม MIRACL-th 3k queries
ไม่น่าเพิ่มอะไรที่โมเดลยังไม่มี (ข) หลักฐานในบ้านชี้ว่า corpus ไทยทั่วไป → retriever อ่อนบนงานนี้
**ข้อยกเว้นเดียวที่ควรพิจารณาจริงคือ WangchanX-Legal-ThaiCCL-RAG (8,211 คู่, MIT, ภาษาราชการ,
มี hard negative ให้แล้ว)** — เป็น warm-up ที่มีเหตุผลเชิงทะเบียนภาษา ไม่ใช่แค่ "มีข้อมูลไทย"

---

## ทาง 2 — Silver set ของตัวเอง (~2,853 คู่ title→ตัวเอง)

### ความเสี่ยงที่ตั้งไว้ในตั๋วนั้น **ถูกต้อง และรุนแรงกว่าที่คิด**

Silver pair คือ (เรื่อง/title ของมติ) → (ตัวมติเอง) ซึ่ง title **เป็น substring หรือเกือบ substring
ของเอกสาร** ดังนั้นสัญญาณที่ contrastive loss จะเรียนได้ง่ายที่สุดคือ **"หาเอกสารที่มีคำเหล่านี้"**
— นี่คือฟังก์ชันของ BM25 เป๊ะ ๆ ซึ่งบน corpus นี้ทำได้ recall@10 = 0.5676 อยู่แล้ว ฟรี ไม่ใช้ GPU
และ **เสมอทางสถิติกับ embedder ที่ดีที่สุด 3 ตัว**

ปัญหาคือ **ทิศทางของการปรับ**: ระบบที่ดีที่สุดตอนนี้คือ hybrid (BM25+dense, RRF) และ finding
ที่แข็งที่สุดในงานคือ *hybrid ชนะ dense-alone อย่างมีนัยสำคัญทุกโมเดล* — ซึ่งแปลว่า
**มูลค่าของ dense อยู่ที่การ "ไม่เหมือน" BM25** การเทรนบน Silver จะดัน dense ให้เข้าไป
**ทับพื้นที่ที่ BM25 ครองอยู่แล้ว** = ลด complementarity = อาจทำให้ hybrid **แย่ลง**
ทั้งที่ตัวเลข dense-alone ดูดีขึ้น (ตัววัดหลอก)

และ Gold 73 **ถูกเขียนใหม่ให้หลีกจากถ้อยคำ title โดยเจตนา** (`paper-results-summary.md`:
"entity-anchored, hand-rephrased away from document title wording") → **การกระจายตัวของ query
ตอนเทรน ≠ ตอนวัด** อย่างจงใจ นี่คือ train/test distribution mismatch ที่เห็นตัวได้

### วรรณกรรมที่ตรงประเด็น

- **Sciavolino, Zhong, Lee, Chen — "Simple Entity-Centric Questions Challenge Dense Retrievers",
  EMNLP 2021** — <https://aclanthology.org/2021.emnlp-main.496/> / <https://arxiv.org/abs/2109.08535>:
  dense retriever **"drastically underperform sparse methods"** บนคำถามที่ยึด entity และ
  **"can only generalize to common entities unless the question pattern is explicitly observed
  during training"** ที่สำคัญที่สุดสำหรับใบนี้: **"data augmentation is unable to fix the
  generalization problem"** → เตือนตรง ๆ ว่าการเพิ่มคู่เทรนแบบง่าย ๆ (ซึ่ง Silver คือแบบนั้น)
  ไม่แก้ปัญหา entity generalization Gold 73 เป็น entity-anchored 100% (30 program / 30 person /
  13 faculty_adjunct) → เปเปอร์นี้คือเปเปอร์ที่ตรงกับสถานการณ์นี้ที่สุด
- **ICT / Lee et al. 2019, Chang et al. ICLR 2020** (อ้างข้างบน): คู่ pseudo แบบ
  "ชิ้นส่วนของเอกสาร → เอกสาร" ถูกจัดเป็น **pre-training task** ไม่เคยถูกเสนอเป็น fine-tuning
  signal ตัวสุดท้าย — Silver คือ ICT/BFS เวอร์ชันหนึ่งพอดี
- **Contriever** (อ้างข้างบน): same-document crops = warm-up
- **Shortcut / spurious-correlation learning**: Du et al., "Shortcut Learning of Large Language
  Models in NLU", CACM 2023 — <https://cacm.acm.org/research/shortcut-learning-of-large-language-models-in-natural-language-understanding/>
  ระบุ **"overlap bias"** ว่าเป็น shortcut ประเภทหนึ่งโดยเฉพาะในงานที่มี input สองฝั่ง
  (NLI, QA, reading comprehension) ที่โมเดลใช้ lexical overlap เป็น spurious correlation;
  และ **"Less Learn Shortcut" (Yuan et al., IJCAI 2023)** — <https://arxiv.org/abs/2205.12593>
  เสนอวิธี *วัดระดับ bias ของแต่ละตัวอย่างแล้วลดน้ำหนัก* ซึ่งเป็น mitigation ที่โอนมาใช้กับ
  Silver ได้ตรง ๆ (ถ่วงน้ำหนักคู่ที่ overlap สูงให้ต่ำลง)

### คำตัดสินเรื่องการใช้งาน + วิธีลดความเสี่ยง

**ห้ามใช้ Silver เป็นสัญญาณเทรนหลักหรือสัญญาณเดียว** — มีเหตุผลเชิงวรรณกรรม (ICT = pre-training task)
+ เชิงโครงสร้าง (สอนสิ่งที่ BM25 ทำอยู่แล้ว = ลด complementarity ที่เป็นแหล่งกำไรจริง)
+ เชิงการวัด (mismatch กับ Gold โดยเจตนา)

**ใช้ได้ในเงื่อนไขต่อไปนี้เท่านั้น** (เรียงตามความแข็งของหลักฐาน):

1. **เป็น stage-1 warm-up สั้น ๆ (≤1 epoch, lr ต่ำ) แล้วตามด้วย stage-2 ข้อมูลสังเคราะห์ตรงโดเมน**
   — มีหลักฐาน (ICT/Contriever/Chang) แต่ต้องวัด hybrid recall ก่อน/หลัง ไม่ใช่ dense-alone
2. **ตัด token ของ title ออกจากฝั่ง passage** ก่อนสร้างคู่ — บังคับให้โมเดลจับคู่จากเนื้อความ
   ไม่ใช่จากบรรทัดหัวเรื่อง (mitigation ที่ตรงกับกลไก overlap bias ที่สุด และทำได้ทันที)
   ⚠️ ผลข้างเคียง: บริบทที่เหลืออาจไม่พอระบุ entity ในบางมติ — ต้อง spot-check
3. **ถ่วงน้ำหนักตาม lexical overlap** (Less-Learn-Shortcut style) — คู่ที่ Jaccard(title, chunk) สูง
   ให้ weight ต่ำ
4. **เขียน query ใหม่ด้วย LLM ในเครื่อง** — ได้ผลดีที่สุด **แต่ต้องเข้าใจว่า ณ จุดนั้นมันกลายเป็น
   ทาง 3 ไปแล้ว ไม่ใช่ Silver ฟรี** ต้นทุนเท่ากับการสังเคราะห์ ไม่ใช่ mitigation ราคาถูก

**สิ่งที่ Silver ยังใช้ได้ฟรีจริง ๆ โดยไม่มีความเสี่ยง**: เป็นแหล่ง **positive สำหรับขุด hard negative**
(รู้ว่ามติไหนคือคำตอบของ title ไหน) และเป็น **sanity check** ระหว่างเทรน (ถ้า silver recall ตก = พัง)

---

## ทาง 3 — สังเคราะห์จาก corpus ด้วย LLM ในเครื่อง (Ollama)

### 3.1 โมเดลไทยในเครื่องที่แต่งคำถามภาษาไทยได้จริง

ทั้งหมดดึงผ่าน Ollama ได้ ([ollama.com/scb10x](https://ollama.com/scb10x)) — **ไม่มีข้อมูลออกนอกเครื่อง**

| โมเดล | ขนาด | หมายเหตุ / หลักฐาน |
|---|---|---|
| `scb10x/typhoon2.1-gemma3-12b` | 12B | สาย Gemma3 ล่าสุดของ SCB10X — คุณภาพไทยสูงสุดในกลุ่มที่ยังพอลง 12GB ได้ที่ q4 (~8GB) แต่ **เบียดกับงานอื่น** |
| `scb10x/llama3.1-typhoon2-8b-instruct` | 8B | [model page](https://ollama.com/scb10x/llama3.1-typhoon2-8b-instruct) — Typhoon 2 ชนะ Typhoon-1.5 บน ThaiExam / M3Exam / IFEval ([SCB10X blog](https://www.scb10x.com/en/blog/introducing-typhoon-2-thai-llm)); context 128k; **จุดสมดุลที่แนะนำให้ทดลองก่อน** (q4 ~5GB เหลือที่ให้ embedder) |
| `scb10x/typhoon2.5-qwen3-4b` | 4B | ฐาน Qwen3 ใหม่กว่า เร็วกว่ามาก เหมาะกับการรัน 3 คำถาม × 2,853 มติในคืนเดียว |
| `scb10x/typhoon2.1-gemma3-4b` | 4B | ทางเลือกเทียบ |
| Gemma 3 12B / Qwen3 8B (generic) | 12B/8B | บน SEA-HELM (<https://arxiv.org/pdf/2502.14301>) โมเดลที่ปรับสำหรับ SEA โดยเฉพาะ (เช่น gemma2-9b-cpt-sea-lion-v3-instruct, Thai 59.9) **ชนะโมเดลทั่วไปขนาดเดียวกัน** (Qwen2.5-7B-Instruct, Thai 55.2) → มีเหตุผลเชิงหลักฐานที่จะเลือก Typhoon มากกว่า generic |

**หมายเหตุนอกขอบเขตแต่ควรรู้**: `scb10x/typhoon-ocr-7b` / `typhoon-ocr1.5-3b` มีอยู่ — ถ้าใบ 01
สรุปว่า OCR พังเป็นเพดาน recall จริง นี่คือทางเลือกที่รันในเครื่องได้ ไม่ต้องไปขอต้นฉบับใหม่

**ข้อจำกัดฮาร์ดแวร์ที่ต้องวางแผน**: RTX 3060 12GB — **ห้ามรัน LLM generator พร้อม fine-tune
embedder** ต้องทำเป็น 2 เฟสแยกกันตามเวลา ไม่ใช่พร้อมกัน

### 3.2 สูตรขุด hard negative ที่เป็นมาตรฐาน

**เครื่องมือพร้อมใช้**: `sentence_transformers.util.mine_hard_negatives`
([docs](https://sbert.net/docs/package_reference/util/hard_negatives.html)) — พารามิเตอร์จริงและค่า default:

```
num_negatives: int = 3
range_min: int = 0            # ข้าม top-N ที่คล้ายที่สุด (กัน positive หลุดมาเป็น negative)
range_max: int | None = None
max_score: float | None = None
min_score: float | None = None
absolute_margin: float | None = None   # negative ต้องมีคะแนน < positive − margin
relative_margin: float | None = None   # negative ต้องมีคะแนน < positive × (1 − margin)
sampling_strategy: Literal['random','top'] = 'top'
cross_encoder: CrossEncoder | None = None   # ใช้ rescore/denoise ที่ mined negatives
use_faiss: bool = False
```

เอกสารระบุตรง ๆ ว่า `range_min` มีไว้ **"skips the most similar texts to avoid marking texts as
negative that are actually positives"** — คือ mitigation false negative ชั้นแรกที่ built-in อยู่แล้ว

**ที่มาของค่า margin ที่ควรใช้ — NV-Retriever (NVIDIA, 2024)** — <https://arxiv.org/abs/2407.15831>
/ <https://arxiv.org/html/2407.15831>:
- **TopK-PercPos**: ตัด negative ที่คะแนน > **95%** ของคะแนน positive → **ค่าที่ดีที่สุดในการ ablation**
  (avg nDCG@10 = 0.5856) และ *"training on high-scoring negatives with respect to positives is
  detrimental to model accuracy"*
- **TopK-MarginPos**: ตัด negative ที่คะแนน > positive − **0.05** — margin ใหญ่กว่านี้ทำให้แย่ลง
- แปลงเป็นพารามิเตอร์ sentence-transformers ได้ตรง ๆ: `relative_margin=0.05` (≈ PercPos 95%)
  หรือ `absolute_margin=0.05` (≈ MarginPos)

**จำนวน negative ต่อ positive**:
- sentence-transformers default = **3**
- NV-Retriever ใช้ **4** ในการทดลองหลัก (e5-large-unsupervised), **1** เมื่อ Mistral-7B เพราะ VRAM,
  และ **1 (stage 1) → 5 (stage 2)** ตอนเทรน NV-Retriever-v1 จริง
- **ไม่มี ablation ตรง ๆ ว่าเท่าไรดีที่สุด** — NV-Retriever เองไม่ได้ทำ ค่า 3–5 คือช่วงที่ practice ใช้
- **บน 12GB ตัวจำกัดคือ VRAM ไม่ใช่ทฤษฎี** — ดู §3.4

### 3.3 ปัญหา false negative — และมันรุนแรงเป็นพิเศษที่นี่

**ขนาดของปัญหาที่มีการวัดจริงในวรรณกรรม**:
- **RocketQA (Qu et al., NAACL 2021)** — <https://arxiv.org/abs/2010.08191>: ตรวจมือ top-retrieved
  ของ 100 คำถามที่ไม่ได้ label เป็น positive พบว่า **~70% จริง ๆ แล้วเป็น positive หรือเกี่ยวข้องสูง**
- **NV-Retriever** วัดด้วย LLM-as-judge (Llama 3.1 70B): naive mining ให้ false negative
  **38.8% บน Natural Questions** และ **47% บน StackExchange** — วิธี positive-aware ลดลงได้ **57% / 50%**

**ทำไม corpus นี้แย่กว่านั้นอีก**: มติหลายฉบับ**เหมือนกันเชิงโครงสร้าง** — "อาจารย์พิเศษเกิน 50%"
มีทุกคณะทุกปี, "ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีไม่กระทบกระเทือนโครงสร้าง)" มีเป็นร้อยฉบับ
ต่างกันแค่ชื่อหลักสูตร/ปี → top-k ที่ขุดมาจะ**เต็มไปด้วยมติพี่น้องที่ตอบคำถามคนละใบแต่หน้าตาเหมือนกัน 95%**
และบางกรณี **ตอบคำถามได้จริง** (เห็นได้จาก Gold เอง: หลายคำถามมี `relevant_resolution_ids` หลายใบ
เพราะถามแบบ "กี่ครั้ง/แต่ละครั้ง") — นั่นคือ false negative แบบที่ label ของ Silver ไม่มีทางรู้

**วิธีลดที่มีตีพิมพ์รองรับ** (เรียงจากลงมือได้ง่ายที่สุด):

| วิธี | กลไก | อ้างอิง |
|---|---|---|
| **positive-aware threshold** | ตัด negative ที่คะแนน ≥ 95% ของ positive (หรือ positive−0.05) | NV-Retriever <https://arxiv.org/abs/2407.15831> |
| **`range_min` > 0** | ข้าม top-N แรกทิ้งไปเลย | [sbert docs](https://sbert.net/docs/package_reference/util/hard_negatives.html) |
| **cross-encoder denoising** | เทรน/ใช้ cross-encoder แล้วทิ้ง negative ที่มันทำนายว่าเป็น positive ด้วยความมั่นใจสูง | RocketQA <https://arxiv.org/abs/2010.08191> (ทำได้ด้วย `cross_encoder=` ใน `mine_hard_negatives` เลย) |
| **`GISTEmbedLoss` / `CachedGISTEmbedLoss`** ⭐ | ใช้ guide model กรอง **in-batch negative** ระหว่างเทรน: ถ้า guide บอกว่า negative ตัวไหนคล้าย anchor มากกว่า positive ของมัน → mask ออกจาก loss เลย | GISTEmbed <https://arxiv.org/abs/2402.16829>; implementation: [`GISTEmbedLoss.py`](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/GISTEmbedLoss.py), [`CachedGISTEmbedLoss.py`](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/CachedGISTEmbedLoss.py) — **นี่คือวิธีที่ actionable ที่สุด: อยู่ใน library แล้ว และ Cached เวอร์ชันแก้ปัญหา VRAM ไปพร้อมกัน** |
| **soft label แทน hard 0/1** | GPL ใช้ MarginMSE จาก cross-encoder → false negative ไม่กลายเป็น 0 แข็ง ๆ | GPL <https://arxiv.org/abs/2112.07577> |
| **SimANS** | สุ่ม negative จากโซน "กำกวมปานกลาง" แทน top-k สุด | <https://arxiv.org/abs/2210.11773> |
| **หลีกเลี่ยงเชิงโครงสร้าง** | ห้ามหยิบมติที่ `entity` ตรงกัน หรืออยู่ในตระกูล title เดียวกัน มาเป็น negative (ใช้ `meeting_manifest.json` + NER ที่มีอยู่แล้ว) | ไม่มีเปเปอร์ — เป็น domain heuristic ที่โปรเจกต์ทำได้เพราะมี metadata |

### 3.4 ข้อจำกัด VRAM ที่กำหนดว่า "กี่ negative" ได้จริง

Contrastive learning ต้องการ batch ใหญ่ (in-batch negatives เยอะ = สัญญาณดี) แต่ 12GB ไม่ให้
→ ใช้ **`CachedMultipleNegativesRankingLoss` / `CachedGISTEmbedLoss`** ซึ่งใช้ **GradCache**
(<https://arxiv.org/abs/2101.06983>) แยก effective batch ออกจาก memory: embed แบบไม่เก็บ graph
→ คำนวณ loss → cache gradient → embed รอบสองต่อ backward chain
ทำให้ batch หลักพันได้ด้วย memory คงที่ **แลกกับความเร็ว ~2–2.4×**
([release notes](https://github.com/huggingface/sentence-transformers/releases/tag/v5.3.0))

**ผลต่อการตัดสินใจ**: negatives/positive = 3–4 เป็นค่าที่ควรตั้ง; ถ้าอยากได้ 5+ ต้องยอมช้าลง
และถ้า fine-tune Qwen3-Embedding-4B (2560 dim) เต็มตัวไม่พอ 12GB → ต้อง **LoRA/PEFT**
ซึ่งเป็น confound ใหม่ที่โปรเจกต์เคยปฏิเสธไปแล้วในเรื่อง quantization — **ต้องตัดสินใจโดยรู้ตัว**
(ทางเลือก: fine-tune `qwen3_0.6b` แทน ซึ่งบน hybrid ทำได้ 0.6543 สูงกว่า 4B อยู่แล้ว และเทรนเต็มตัวได้สบาย)

---

## ข้อเสนอที่ลงมือได้

### การกัน Gold 73 ไม่ให้ปนเปื้อน — ตัวเลขจริง

นับจาก `config/eval/gold_query_set_73det.yaml`: **73 คำถาม → 481 `resolution_id` ที่ไม่ซ้ำกัน
= 16.86% ของ corpus 2,853 ฉบับ**

กฎที่ต้องบังคับ:
1. **481 ฉบับนั้นห้ามเป็น positive ในชุดเทรนเด็ดขาด** (เหลือ **2,372 ฉบับ** ให้สังเคราะห์)
2. **เป็น negative ที่ขุดมาได้** (ไม่รั่ว label เพราะไม่ได้บอกว่ามันคือคำตอบของอะไร)
   — แต่ถ้าจะให้ conservative ที่สุดคือกันออกทั้งหมด แลกกับ negative pool เล็กลง 17%
3. ห้ามใช้ข้อความ query ของ Gold เป็น seed/few-shot example ให้ LLM

### สูตรที่เสนอ (ตัวเลข = ข้อเสนอ ไม่ใช่คำสั่ง)

| เฟส | แหล่ง | จำนวนคู่ | เหตุผล |
|---|---|---|---|
| **0. Baseline ที่ต้องมีก่อนทุกอย่าง** | — | 0 | วัด hybrid recall@10 ปัจจุบันซ้ำ + ตั้ง paired-bootstrap harness ให้พร้อม **ห้ามเทรนก่อนมีเส้นฐานที่วัดซ้ำได้** |
| **1. (ทางเลือก) warm-up ราชการ** | `WangchanX-Legal-ThaiCCL-RAG` train | 8,211 | MIT, ภาษาราชการไทย, มี `hard_negative_contexts` มาแล้ว — warm-up ที่มีเหตุผลเชิงทะเบียนภาษา ไม่ใช่แค่ "ภาษาไทย" |
| **2. สังเคราะห์ตรงโดเมน (ทางหลัก)** | 2,372 มติ (หัก Gold) × **3 คำถาม** | **~7,100** | 3 แบบต่อมติ: (ก) entity-anchored เลียนสไตล์ Gold (ข) เชิงเวลา/รวบยอด ("กี่ครั้ง/ปีไหน") (ค) เชิงเนื้อหาที่ **ห้ามใช้คำใน title** |
| **3. Silver แบบดัดแปลง (ทางเลือก, stage-1 เท่านั้น)** | 2,372 title (ตัด title ออกจาก passage) | ~2,372 | ≤1 epoch, lr ต่ำ, ถ่วงน้ำหนักตาม overlap; **วัด hybrid ก่อน/หลัง ถ้าไม่ขึ้น → ทิ้ง** |
| **4. Hard negatives** | ขุดจาก index `semantic × qwen3` ที่มีอยู่ | 3–4 ตัว/positive → ~21k–28k triplet | `mine_hard_negatives(range_min=5, num_negatives=4, relative_margin=0.05, cross_encoder=BAAI/bge-reranker-v2-m3, use_faiss=True)` + กฎ domain: ห้ามหยิบมติที่ `entity` ตรงกัน |
| **5. เทรน** | `qwen3_0.6b` ก่อน (ไม่ใช่ 4B) | — | `CachedGISTEmbedLoss` (guide = `bge-m3`) กัน false negative + แก้ VRAM พร้อมกัน; 0.6B เทรนเต็มตัวได้ไม่ต้อง LoRA และบน hybrid ทำได้ 0.6543 สูงกว่า 4B (0.6235) อยู่แล้ว |

### ต้นทุนเวลาโดยประมาณ (คิดเลขให้เห็น ไม่ใช่เดา)

- **สังเคราะห์คำถาม**: 2,372 มติ × 3 คำถาม ≈ 7,116 คำถาม
  สมมติ prompt ~1,500 tok เข้า / ~180 tok ออก ต่อมติ
  - `typhoon2.5-qwen3-4b` q4 บน 3060 ≈ 40–50 tok/s → ~6–8 วิ/มติ → **~4–5 ชม.**
  - `llama3.1-typhoon2-8b-instruct` q4 ≈ 25–30 tok/s → ~10–12 วิ/มติ → **~8–10 ชม.** (รันข้ามคืน)
  - `typhoon2.1-gemma3-12b` q4 ≈ 15–18 tok/s → **~15–18 ชม.**
- **ขุด hard negative**: ไม่ต้อง re-embed corpus (index มีแล้ว) — encode 7,116 query
  ที่ ~187 ms (qwen3_0.6b) ≈ 22 นาที + คูณเมทริกซ์/เรียง ≈ **~1 ชม.**
- **cross-encoder denoising**: 7,116 × 50 candidates ≈ 356k คู่ ที่ ~60–100 คู่/วิ (bge-reranker-v2-m3 fp16)
  → **~1–1.5 ชม.**
- **เทรน `qwen3_0.6b`**: ~9.5k คู่ × 3–4 epoch ด้วย CachedGIST (ช้ากว่าปกติ ~2×) → **~3–6 ชม./รอบ**
- **eval + paired bootstrap + Holm**: มีสคริปต์อยู่แล้ว → **~1–2 ชม.**
- **ตรวจคุณภาพคำถามสังเคราะห์ด้วยมือ** (สุ่ม 100 ข้อ): **~2 ชม. แรงคน** — **ห้ามข้าม**

**รวม ~2–3 วันเครื่อง + ~half day แรงคน สำหรับรอบแรก**

### ความเสี่ยงที่ยัง**ลดไม่ได้**

1. **มติพี่น้องที่เหมือนกันเชิงโครงสร้าง** ⚠️ **ไม่มีทางแก้สะอาด** — การกัน 481 `resolution_id`
   ของ Gold ออก **ไม่ได้กันมติ "อาจารย์พิเศษเกิน 50%" ของคณะอื่น/ปีอื่นที่หน้าตาแทบเหมือนกัน**
   ออกจากชุดเทรน โมเดลอาจเห็นรูปแบบมติแบบนั้นมาแล้วนับร้อยครั้ง → **Gold 73 หลัง fine-tune
   จะประเมินสูงเกินจริงโดยที่ตรวจไม่ได้** margin threshold ช่วยได้แค่ตอนขุด negative
   ไม่ได้ช่วยเรื่องการรั่วของ *รูปแบบ*
2. **Gold 73 เล็กเกินไปสำหรับการตัดสินความต่างเล็ก ๆ** — ระบบท็อป 5 ต่างกันแค่ 0.014
   (`paper-results-summary.md`) ถ้า fine-tune ได้กำไร <0.03 จะพิสูจน์ไม่ได้ว่าไม่ใช่ noise
   (เชื่อมกับใบ 05)
3. **คุณภาพคำถามสังเคราะห์ = เพดานของทั้งวิธี** — ถ้า LLM ในเครื่องแต่งคำถามที่ลอกคำใน chunk
   ก็จะได้ Silver อีกรอบในกระดาษห่อใหม่ **ต้องบังคับใน prompt + ตรวจมือ** และแม้ตรวจแล้ว
   ก็ยังไม่ใช่คำถามที่**คนจริงจากสภาวิชาการ**จะถาม — ไม่มีทางแก้จนกว่าจะได้ query log จริง
4. **Sciavolino et al. บอกไว้ล่วงหน้าว่า data augmentation แก้ปัญหา entity generalization ไม่ได้**
   — ทาง 3 ก็คือ data augmentation รูปแบบหนึ่ง **นี่คือหลักฐานตีพิมพ์ที่ค้านความคุ้มค่าของทั้งใบนี้**
   และควรถูกยกขึ้นในการตัดสิน go/no-go ของแผนที่
5. **เพดาน OCR** — ถ้าใบ 01 พบว่าที่พลาดคือเอกสารที่ OCR พัง **ไม่มี training data ใดแก้ได้**
6. **LoRA confound ถ้ายืนยันจะ fine-tune 4B** — 12GB ไม่พอเทรนเต็มตัว → ต้อง PEFT ซึ่งเป็น
   confound แบบเดียวกับที่โปรเจกต์เคยปฏิเสธ quantization มาแล้ว

### สิ่งที่ควรทำก่อนลงทุน 2–3 วัน (ราคาถูกที่สุด)

รัน **pilot 200 มติ** (สังเคราะห์ ~600 คำถาม, ~1 ชม.) แล้ววัด 2 อย่าง:
(ก) คำถามที่ได้ **ลอกคำจาก title/chunk เกิน X%** หรือไม่ (วัดด้วย Jaccard) —
ถ้าลอก แปลว่าจะได้ Silver ซ้ำ ไม่ต้องรันต่อ
(ข) จาก 600 คำถามนั้น มีกี่ % ที่ **BM25 เปล่า ๆ ตอบถูกอยู่แล้ว** — ถ้าสูงมาก
แปลว่าชุดเทรนนี้จะไม่สอนอะไรที่ BM25 ยังไม่มี ซึ่งเป็นความเสี่ยงเดียวกับ Silver เป๊ะ ๆ

---

## แหล่งอ้างอิงทั้งหมด

**Dataset cards (ตรวจ 2026-07-31)**
- MIRACL — <https://huggingface.co/datasets/miracl/miracl> / corpus <https://huggingface.co/datasets/miracl/miracl-corpus>
- MTEB MIRACLRetrieval — <https://huggingface.co/datasets/mteb/MIRACLRetrieval>
- MTEB XQuADRetrieval — <https://huggingface.co/datasets/mteb/XQuADRetrieval>
- google/xquad — <https://huggingface.co/datasets/google/xquad>
- TyDiQA — <https://huggingface.co/datasets/google-research-datasets/tydiqa> / GoldP ไม่รวมไทย: <https://github.com/google-research-datasets/tydiqa/blob/master/gold_passage_baseline/README.md>
- iapp_wiki_qa_squad — <https://huggingface.co/datasets/iapp/iapp_wiki_qa_squad> / เวอร์ชัน retrieval <https://huggingface.co/datasets/kornwtp/iapp-wikiqa-tha-qaretrieval>
- pythainlp/thaiqa_squad — <https://huggingface.co/datasets/pythainlp/thaiqa_squad>
- scb-mt-en-th-2020 — <https://huggingface.co/datasets/airesearch/scb_mt_enth_2020>
- WangchanX-Legal-ThaiCCL-RAG — <https://huggingface.co/datasets/airesearch/WangchanX-Legal-ThaiCCL-RAG>
- iapp/rag_thai_laws — <https://huggingface.co/datasets/iapp/rag_thai_laws>
- Thai-Sentence-Vector-Benchmark — <https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark>

**วรรณกรรม**
- Sciavolino et al., EMNLP 2021 — <https://aclanthology.org/2021.emnlp-main.496/>
- Izacard et al. (Contriever) — <https://arxiv.org/abs/2112.09118>
- Chang et al., ICLR 2020 — <https://research.google/pubs/pre-training-tasks-for-embedding-based-large-scale-retrieval/>
- Thakur et al. (BEIR) — <https://arxiv.org/abs/2104.08663>
- Wang et al. (GPL), NAACL 2022 — <https://arxiv.org/abs/2112.07577>
- Qu et al. (RocketQA), NAACL 2021 — <https://arxiv.org/abs/2010.08191>
- Moreira et al. (NV-Retriever), 2024 — <https://arxiv.org/abs/2407.15831>
- Solatorio (GISTEmbed), 2024 — <https://arxiv.org/abs/2402.16829>
- Zhou et al. (SimANS), EMNLP 2022 — <https://arxiv.org/abs/2210.11773>
- Gao et al. (GradCache) — <https://arxiv.org/abs/2101.06983>
- Du et al., CACM 2023 (shortcut learning / overlap bias) — <https://cacm.acm.org/research/shortcut-learning-of-large-language-models-in-natural-language-understanding/>
- Yuan et al. (Less Learn Shortcut), IJCAI 2023 — <https://arxiv.org/abs/2205.12593>

**เครื่องมือ / โมเดล**
- `mine_hard_negatives` — <https://sbert.net/docs/package_reference/util/hard_negatives.html>
- `GISTEmbedLoss` / `CachedGISTEmbedLoss` — <https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/CachedGISTEmbedLoss.py>
- sentence-transformers v5.3.0 release notes — <https://github.com/huggingface/sentence-transformers/releases/tag/v5.3.0>
- Typhoon บน Ollama — <https://ollama.com/scb10x> / <https://ollama.com/scb10x/llama3.1-typhoon2-8b-instruct>
- Typhoon 2 (ThaiExam/M3Exam/IFEval) — <https://www.scb10x.com/en/blog/introducing-typhoon-2-thai-llm>
- SEA-HELM — <https://arxiv.org/abs/2502.14301>
