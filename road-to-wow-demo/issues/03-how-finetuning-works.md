# fine-tune embedding model เขาทำกันยังไง + RTX 3060 12GB ทำได้แค่ไหน

Part of: [map.md](../map.md)
Type: research
Status: resolved
Blocked by: —

## Question

ตอบสองข้อจากคำถามตั้งต้นของผู้ใช้: **"มีขั้นตอนอย่างไร"** และ **"จะยากเกินไปมั้ย"**

### สิ่งที่ต้องหาคำตอบ

1. **วิธีมาตรฐานปัจจุบัน** ของการ fine-tune embedding model สำหรับ retrieval —
   contrastive learning กับ in-batch negatives, hard negative mining,
   loss ที่ใช้จริง (MultipleNegativesRankingLoss / InfoNCE), บทบาทของ instruction prefix
   ใน Qwen3-Embedding โดยเฉพาะ
2. **LoRA vs full fine-tune** สำหรับ embedding model — ต่างจากกรณี generative ยังไง
   มีหลักฐานว่าอันไหนดีกว่าในงาน retrieval domain adaptation
3. **คณิตศาสตร์ VRAM สำหรับ Qwen3-Embedding-4B บน 12GB** — นี่คือหัวใจของ "ยากเกินไปมั้ย":
   - น้ำหนัก fp16 ~8GB แล้ว เหลือที่ให้ activation/optimizer เท่าไหร่
   - contrastive learning กินแรมตาม **batch size** (in-batch negatives) ไม่ใช่ตามน้ำหนัก —
     batch เล็กทำให้คุณภาพตก ตรวจว่า **GradCache / `CachedMultipleNegativesRankingLoss`**
     (sentence-transformers) แก้ปัญหานี้ได้จริงมั้ย และแลกกับอะไร
   - ถ้าต้อง QLoRA 4-bit จริง ๆ นั่นสร้าง **confound เรื่อง precision** ซึ่ง repo นี้เคยปฏิเสธ
     มาแล้วตอนตัด Qwen3-8B ออก — ต้องระบุให้ชัดว่าเลี่ยงได้หรือไม่
   - เวลาเทรนโดยประมาณบน 3060 สำหรับข้อมูลระดับไม่กี่พันคู่
4. **ผลที่คาดหวังได้จริงจากวรรณกรรม** — domain adaptation ของ embedding model
   บนคลังเฉพาะทางขนาดกลาง โดยทั่วไปได้ recall เพิ่มขึ้นเท่าไหร่ (ช่วงตัวเลข ไม่ใช่คำคุณศัพท์)
   และมีรายงานกรณีที่ **แย่ลง** (catastrophic forgetting) มั้ย
5. **ทางเลือกที่ถูกกว่าและให้ผลใกล้เคียง** — linear adapter / matryoshka truncation /
   การเทรนเฉพาะ projection layer แทนตัวโมเดล

### แหล่งที่ควรอ่าน

เอกสาร sentence-transformers (training overview + losses + GradCache),
model card และ technical report ของ Qwen3-Embedding, เอกสาร PEFT/LoRA,
งานวิจัย domain adaptation ของ dense retriever

## ข้อจำกัดที่ห้ามลืม

RTX 3060 **12GB** — เป็นข้อจำกัดจริงที่แก้ไม่ได้ ผู้ใช้เคยบอกไว้ว่า "ถ้าไม่พอ ก็ไม่เอาครับ"

## Answer

> ตั๋วชนิด **research** — หน้าที่คือ *วางข้อเท็จจริงบนโต๊ะ* ไม่ใช่ตัดสินใจแทน
> ทุกตัวเลขที่อ้างระบุ (โมเดลฐาน / เดลต้าสัมบูรณ์ / ชุดวัด / in-domain หรือไม่) เท่าที่แหล่งให้มา
> ตัวเลขที่แหล่งให้มาเป็น "เปอร์เซ็นต์สัมพัทธ์บนฐานที่ไม่บอก" ถูกทิ้งหรือติดป้ายกำกับไว้ชัดเจน

---

## 1. contrastive retrieval fine-tuning ทำกันยังไงจริง ๆ (pipeline sentence-transformers v3+/v5)

### 1.1 โครงประกอบ 5 ชิ้น

`SentenceTransformerTrainer` (v3 ขึ้นไป, ปัจจุบัน v5.3) ต้องการ 5 อย่าง — dataset, loss,
`SentenceTransformerTrainingArguments`, evaluator (ไม่บังคับ), และตัวโมเดล
รูปแบบ dataset สำคัญที่ **ลำดับคอลัมน์** ไม่ใช่ชื่อคอลัมน์ และจำนวนคอลัมน์ต้องตรงกับที่ loss ต้องการ
(https://huggingface.co/blog/train-sentence-transformers)

### 1.2 loss ที่ใช้จริงสำหรับ retrieval

เอกสาร sbert ติดดาว (★) ให้ 3 ตัวสำหรับงาน retrieval:
`MultipleNegativesRankingLoss` (MNRL), `CachedMultipleNegativesRankingLoss` (CachedMNRL),
`CachedGISTEmbedLoss` (https://sbert.net/docs/sentence_transformer/loss_overview.html)

- **MNRL = InfoNCE = in-batch negatives** โดยตรง: ข้อมูลเป็นคู่ `(a_i, b_i)` ที่ถือว่าเข้าคู่กัน
  แล้ว **สมมติว่า `(a_i, b_j)` ที่ `i≠j` ทั้งหมดเป็น negative** — negative จึงมาฟรีจาก batch
  รับได้ทั้ง `(anchor, positive)`, `(anchor, positive, negative)`, และ
  `(anchor, positive, neg_1..neg_N)`
- ต้องตั้ง `batch_sampler=BatchSamplers.NO_DUPLICATES` เพราะ loss ตระกูลนี้พังถ้ามีข้อความซ้ำใน batch
- **batch ใหญ่ = negative เยอะ = คุณภาพดีขึ้น** นี่คือเหตุผลเดียวที่ batch size เป็นตัวแปรคุณภาพ
  ไม่ใช่แค่ตัวแปรความเร็ว
- ⚠️ **จุดเชื่อมที่ต้องอ่านคู่กับข้อ 2.3**: MNRL forward ทั้ง query และ positive (และ hard negative
  ถ้ามี) ในก้าวเดียว → memory ของ activation โตตาม batch size โดยตรง แปลว่า batch size
  เป็น **ตัวแปรคุณภาพ และ ข้อจำกัด memory ที่ผูกมัด พร้อมกัน** — สองอย่างนี้ดึงกันคนละทาง
  และเป็นเหตุผลทั้งหมดที่ GradCache (ข้อ 2.4) มีอยู่

### 1.3 hard negatives — และกับดัก false negative ที่เกี่ยวกับคลังนี้โดยตรง

`sentence_transformers.util.mine_hard_negatives()` มีพารามิเตอร์
`range_min/range_max`, `max_score/min_score`, `absolute_margin`, `relative_margin`,
`num_negatives` (default 3), `sampling_strategy` (`top`/`random`), `use_faiss`
(https://sbert.net/docs/package_reference/util/hard_negatives.html)

**หลักฐานว่าการขุด negative แบบไร้เดียงสาทำร้ายโมเดล** — NV-Retriever (arXiv 2407.15831):
ในบรรดา passage ที่คล้าย query ของ MS-MARCO มากที่สุด **~70% จริง ๆ แล้วควรถูกติดป้ายว่า positive**
วัดผลบน 3 ชุด QA (NQ, HotpotQA, FiQA), NDCG@10:

| วิธีขุด negative | NDCG@10 |
|---|---|
| naive top-k | 0.5407 |
| **BM25-mined negatives** | **0.5002** (แย่ที่สุด) |
| random negatives | 0.5248 |
| TopK-PercPos (ตัด negative ที่คะแนน > 95% ของ positive) | **0.5856** |

(https://arxiv.org/html/2407.15831v1)

> ⚠️ **เกี่ยวกับคลังนี้โดยตรง**: วิธีขุด hard negative ที่คนแนะนำกันบ่อยที่สุดคือ "ใช้ BM25 ขุด"
> ซึ่งในการทดลองนี้เป็นวิธีที่ **แย่กว่าการสุ่มเสียอีก** และคลังมติสภาวิชาการเป็น
> entity-anchored ซ้ำซาก (มติเรื่องเดียวกัน คนเดียวกัน หลักสูตรเดียวกัน คนละครั้งประชุม)
> → อัตรา false negative น่าจะสูงกว่า MS-MARCO ไม่ใช่ต่ำกว่า

เคสวัดผลจริงอีกอัน (ภาษาเกาหลี, dragonkue): **MNRL เปล่า ๆ ได้ NDCG@10 = 0.626 ซึ่ง
*ต่ำกว่า* โมเดลฐาน −0.045** เพราะ false negative ล้วน ๆ — วิธีแก้ที่แนะนำคือ
`mine_hard_negatives(relative_margin=0.05)` และ/หรือ `CachedGISTEmbedLoss` ที่ใช้
guide model มาปิด (mask) negative ที่คล้าย anchor มากกว่า positive
(https://huggingface.co/blog/dragonkue/mitigating-false-negatives-in-retriever-training)

### 1.4 instruction prefix เฉพาะของ Qwen3-Embedding — ห้ามทำพัง

จาก model card + repo ทางการ
(https://huggingface.co/Qwen/Qwen3-Embedding-4B , https://github.com/QwenLM/Qwen3-Embedding):

- รูปแบบ query: `Instruct: {task_description}\nQuery:{query}`
- **document ไม่ใส่ prefix ใด ๆ เลย** — เป็น asymmetric encoding
- **pooling = last-token** และ **tokenizer ต้อง `padding_side='left'`**
- ใช้ instruction ได้ผลดีกว่าไม่ใช้ **1%–5%** ในงาน downstream ส่วนใหญ่
- **instruction ควรเขียนเป็นภาษาอังกฤษ** แม้ query จะเป็นภาษาอื่น เพราะ instruction
  ตอนเทรนต้นฉบับเป็นอังกฤษเกือบทั้งหมด (สำคัญมากสำหรับเคสภาษาไทย)
- สเปก: 4B = 36 layers, embedding dim 2560, context 32K, bf16, รองรับ MRL 32–2560 มิติ

**ผลต่อการ fine-tune**: ถ้าเทรนโดยไม่รักษา 3 อย่างนี้ (instruction เฉพาะ query, last-token pooling,
left padding) จะไม่ใช่การปรับโมเดล แต่เป็นการ *ทำลาย* convention ที่โมเดลถูกเทรนมา
และ instruction ที่ใช้ตอนเทรนต้องเป็นตัวเดียวกับที่ใช้ตอน inference ใน repo `RAG` เป๊ะ ๆ

### 1.5 PEFT/LoRA ใน sentence-transformers

รองรับ native ผ่าน `model.add_adapter(LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=..., lora_alpha=...))`
พร้อม `load_adapter/set_adapter/get_adapter_state_dict/delete_adapter`
(https://sbert.net/examples/sentence_transformer/training/peft/README.html)
Qwen3-Embedding เองก็ถูกเทรนด้วย LoRA จาก backbone Qwen3 (arXiv 2506.05176)
ค่า config ที่พบบ่อยในสาย Qwen3 คือ `r=64, alpha=128` บน attention+MLP projection ทุกตัว

---

## 2. คณิตศาสตร์ VRAM: Qwen3-Embedding-4B บน RTX 3060 12GB (แสดงเลขทุกบรรทัด)

### 2.0 งบจริงบนการ์ด ≠ 12GB

RTX 3060 12GB = 12 GiB. บน **Windows 11 + WDDM + จอต่ออยู่**:
desktop/compositor กิน ~0.5–1.0 GiB, CUDA context + cuBLAS/cuDNN workspace อีก ~0.3–0.6 GiB
→ **งบที่ใช้ได้จริง ≈ 10.5–11.2 GiB** ไม่ใช่ 12 — ที่ระยะขอบแคบขนาดนี้ตัวเลขนี้เป็นตัวชี้ขาด

### 2.1 น้ำหนัก

4.02e9 params × 2 bytes (bf16) = **8.04 GB (decimal) = 7.49 GiB**
→ เหลืองบ ≈ **3.0–3.7 GiB** สำหรับทุกอย่างที่เหลือ

### 2.2 สามทางเลือก — เลขล้วน

| | grad | Adam m,v (fp32) | fp32 master | รวม trainable state | + weights | ผล |
|---|---|---|---|---|---|---|
| **Full FT** | 7.49 GiB | 4.02e9×8B = **29.95 GiB** | 14.97 GiB | **52.4 GiB** | **59.9 GiB** | ❌ เกินงบ 10.8 GiB ไป **~5.5 เท่า** ไม่ใช่ "ตึง ๆ" แต่ *เป็นไปไม่ได้* |
| **LoRA r=16** (~33M trainable) | 63 MiB | 252 MiB | 126 MiB | **0.43 GiB** | **7.92 GiB** | ✅ เหลือ ~2.6–3.3 GiB |
| **LoRA r=32** (~66M) | 126 MiB | 504 MiB | 252 MiB | **0.86 GiB** | **8.35 GiB** | ✅ เหลือ ~2.1–2.8 GiB |
| **LoRA r=64** (~132M) | 252 MiB | 1,008 MiB | 504 MiB | **1.72 GiB** | **9.21 GiB** | ⚠️ เหลือ ~1.3–2.0 GiB (ตึงมาก) |

*(ทุกช่องเป็นหน่วยฐานสอง GiB/MiB สม่ำเสมอ; คอลัมน์ "+ weights" รวมน้ำหนัก 7.49 GiB แล้ว)*

(นับ LoRA params จาก Qwen3-4B: 36 layers × {q 2560×4096, k/v 2560×1024, o 4096×2560,
gate/up 2560×9728, down 9728×2560}; LoRA param ต่อเมทริกซ์ = r×(fan_in+fan_out);
สูตร state ต่อ trainable param = 2B grad + 8B Adam + 4B master = **14 bytes/param**)

### 2.3 activation — ตัวแปรอิสระตัวเดียวที่เหลือ และจุดที่ contrastive ต่างจาก LLM SFT

**เอกลักษณ์ที่บล็อกส่วนใหญ่มองข้ามและตั๋วนี้ขึ้นอยู่กับมัน**: MNRL ต้อง forward
**2× batch** (query + positive) หรือ **3× batch** ถ้ามี hard negative ชัดเจน
→ "batch 16" = 32–48 sequence ที่ต้องเก็บ activation จริง ๆ
นี่คือเหตุผลที่ contrastive training บนโมเดล 4B ตึงกว่า causal-LM SFT ที่ batch เท่ากันมาก

**ไม่มี gradient checkpointing**: ต่อ token ต่อ layer เก็บ intermediate ~15 ตัว ขนาด hidden
→ 2560 × 2B × 15 ≈ 76.8 KB/token/layer × 36 layers ≈ **2.76 MB ต่อ token**
→ 4,096 token (เช่น 8 seq × 512) = **~11.3 GiB** → ❌ ล้นทันที

**เปิด gradient checkpointing**: เก็บแค่ boundary ของแต่ละ layer
→ 36 × 2560 × 2B = **184 KB ต่อ token** + working set ของ layer ที่กำลัง recompute
→ งบ ~2.4 GiB ⇒ **~13,000 token ในหน่วยความจำ**; กันครึ่งไว้ให้ recompute/attention workspace
⇒ ปลอดภัยที่ **~6,000–7,000 token ต่อ micro-batch**
เช่น 16 sequence × 384 token หรือ 24 sequence × 256 token

> **สรุปข้อ 2.3: gradient checkpointing ไม่ใช่ทางเลือก แต่เป็นข้อบังคับ** — และมันแลกด้วย
> wall-clock ประมาณ **+30–40%** (recompute forward ทุก layer ตอน backward)

### 2.4 CachedMNRL / GradCache ตัดขาด batch size ออกจาก memory ได้จริงหรือไม่ — **จริง**

กลไก (Gao et al. 2021, GradCache):
1. forward ทั้ง batch แบบ `no_grad` เป็น chunk ย่อย → ได้เมทริกซ์ embedding เต็ม
2. คำนวณ loss + gradient เทียบกับ **embedding** แล้ว cache ไว้
3. forward ซ้ำทีละ chunk แบบ **มี** gradient เพื่อต่อ computational graph เข้ากับ cache

**เมทริกซ์ที่ cache ไว้เล็กมาก**: batch 256 × 2560 dim × 4B = **2.6 MB** — memory จึงถูกกำหนดโดย
`mini_batch_size` เท่านั้น ไม่ใช่ `per_device_train_batch_size`
เอกสาร sbert: "often used to increase the batch size, resulting in superior performance";
`mini_batch_size` คือ "how much memory is actually used during training"

**ราคาที่จ่าย**: forward ทำสองรอบ → **ช้าลง ~2–2.4×** เทียบ MNRL ธรรมดา
และ **ทบกับ** ค่า recompute ของ gradient checkpointing อีกชั้น
(https://sbert.net/docs/package_reference/sentence_transformer/losses.html ,
https://deepwiki.com/huggingface/sentence-transformers/5.3-loss-functions)

**⚠️ ความเสี่ยงเชิงวิศวกรรมที่ต้องเช็คก่อน commit**: การจับคู่ CachedMNRL + PEFT/gradient
checkpointing มี issue เปิดอยู่หลายใบใน upstream —
sentence-transformers #3173 (DeepSpeed + CachedMNRL: "Gradient computed twice for this partition"),
#3434 / #3701 (resume จาก LoRA checkpoint พัง),
transformers #42947 ("Gradient Checkpointing Ineffective with PEFT LoRA Despite Proper Configuration"),
peft #2826 — **ต้องทดสอบ 1 step จริงก่อน ไม่ใช่สมมติว่าใช้ได้**
มี example ทางการที่คู่กันสำเร็จ: `examples/sentence_transformer/training/unsloth/training_gooaq_unsloth.py`
(LoRA + CachedMNRL) และ `training_gooaq_lora.py` (LoRA + MNRL)

### 2.5 คำตอบตรง ๆ: **12GB พอ โดย *ไม่* ต้อง quantize**

**ไม่จำเป็นต้องใช้ QLoRA / 4-bit** สูตรที่พอดี:

```
bf16 base (ไม่ quantize) + LoRA r=16–32
+ gradient_checkpointing=True                        ← บังคับ
+ CachedMultipleNegativesRankingLoss(mini_batch_size=8–16)
+ max_seq_length = 256–512                           ← จุดเจ็บ ดูข้อ 2.6
+ per_device_train_batch_size = 128–256 (effective)  ← memory ไม่ขึ้นกับตัวนี้
```

สิ่งที่ **ไม่** พอ: full fine-tune (เกิน ~5.5 เท่า), MNRL ธรรมดาที่ batch ใหญ่พอจะมีคุณภาพ,
LoRA r=64 พร้อม seq ยาว

**บันทึกเรื่อง confound ของ 4-bit ไว้เป็นหลักฐาน** (แม้จะไม่ต้องใช้): repo นี้ตัด Qwen3-8B ออก
ด้วยเหตุผลว่า quantization จะเพิ่มตัวแปร precision ทับตัวแปร size
(`paper-results-summary.md` §Group C). เหตุผลเดียวกันใช้กับที่นี่: ถ้า fine-tune ด้วย 4-bit
แล้วเทียบกับ baseline fp16 เดิม ผลต่างจะแยกไม่ออกว่ามาจาก "การ fine-tune" หรือ
"การสูญเสีย precision" — และเนื่องจากไม่จำเป็นต้องใช้ ก็ไม่มีเหตุผลจะรับ confound นี้
**ข้อสรุปเชิงบวก: เส้นทาง bf16 LoRA มีอยู่จริง confound นี้เลี่ยงได้**

### 2.6 ⚠️ ข้อจำกัดที่อาจสำคัญกว่าเลข VRAM: train/inference length mismatch

VRAM บังคับให้เทรนที่ **256–512 token** แต่ semantic chunk ของโปรเจกต์นี้
**ยาวถึง 3,116 token** (`paper-results-summary.md` §Resolved 2026-07-21)

**repo นี้มีหลักฐานในบ้านแล้วว่านี่คือ failure mode จริง ไม่ใช่การคาดเดา**:
`congen` เทรนบนอินพุตสั้น → พอป้อน 510 token ตอน inference **recall@10 ตก −0.0298, p=0.0016**
สาเหตุที่บันทึกไว้คือ train/test input-length distribution mismatch ล้วน ๆ

การ fine-tune ที่ 384 token แล้วเสิร์ฟ chunk 3,000 token คือการสร้าง mismatch
ในทิศทางเดียวกันด้วยมือ — **ควรถือเป็นความเสี่ยงระดับเดียวกับเรื่อง VRAM**
ทางเลี่ยงที่เป็นไปได้: เทรนบน chunk สั้นแล้วเสิร์ฟ chunk สั้น (แต่ semantic chunking คือ
ตัวชนะปัจจุบัน) หรือ fine-tune ตัว 0.6B ที่ seq ยาวกว่าได้ (ดูข้อ 6.5)

---

## 3. ผลที่คาดหวังได้จริง — เฉพาะตัวเลขที่มีเดลต้าสัมบูรณ์

> **คำเตือนเรื่องคุณภาพแหล่ง**: การค้นครั้งแรกคืนตัวเลขแบบ "+33% recall", "+95% context recall",
> "median 700% improvement" ทั้งหมดเป็น **เปอร์เซ็นต์สัมพัทธ์บนฐานที่ไม่ระบุ จากบล็อกการตลาด**
> — **ทิ้งทั้งหมด** ไม่นำมาอ้าง วรรณกรรมสาย applied ส่วนใหญ่รายงานในรูปแบบที่เทียบกับ
> วินัย paired-bootstrap ของ repo นี้ไม่ได้ ด้านล่างคือเฉพาะที่มีเลขสัมบูรณ์

### 3.1 กรณีที่ช่วย (มีเดลต้าสัมบูรณ์)

| แหล่ง | โมเดลฐาน | ก่อน → หลัง | ข้อมูลเทรน | ฮาร์ดแวร์ | in-domain เท่านั้น? |
|---|---|---|---|---|---|
| NVIDIA, NVDocs | Llama-Nemotron-Embed-1B-v2 | **Recall@10 0.6298 → 0.6930 (+0.063)**<br>nDCG@10 0.5551 → 0.6156 (+0.061)<br>Recall@5 0.5449 → 0.6029 | "พัน ๆ คู่" สังเคราะห์ | A100 80GB | ใช่ ไม่รายงาน OOD |
| NVIDIA, Atlassian JIRA | เดียวกัน | **Recall@60 0.751 → 0.951 (+0.200)** | เดียวกัน | A100 80GB | ใช่ |
| Cisco + Nemotron recipe | NV-EmbedQA 1B | **Recall@10 +6.8 pts (+8.5%)**<br>nDCG@1 +7.1–7.3, MAP@10 +6.5 | ~7,800 ตัวอย่าง / ~9,200 QA pair จาก ~925 เอกสาร | H200 143GB | ใช่ ไม่รายงาน OOD |
| REFINE (arXiv 2410.12890) | BGE | TOURISM +5.76% R@3, SQUAD +6.58% R@3, **RAG-12000 +0.32% R@3** | 493 / 2,550 / 850 ตัวอย่าง | — | รายงาน OOD ด้วย (ดู §4) |
| HTX DSAI (สิงคโปร์, multilingual) | multilingual-E5-large | **MRR 0.7736 → 0.8062 (+0.033, +4.21%)** | 4 ชุดผสม, GPT-4o สังเคราะห์ | A100 80GB ×N | รายงาน MTEB +4.23% ด้วย |
| ช่วงที่ practitioner รายงานทั่วไป | — | **Recall@k +5–15% (สัมพัทธ์)** | 500–1k ทดลอง / 5k–10k จริงจัง / 50k+ production | — | — |

แหล่ง: https://huggingface.co/blog/nvidia/domain-specific-embedding-finetune ,
https://blogs.cisco.com/ai/fine-tuning-embedding-models-for-enterprise-retrieval-a-practical-guide-with-nvidia-nemotron-recipe ,
https://arxiv.org/html/2410.12890 ,
https://medium.com/htx-dsai/fine-tuning-multilingual-embedding-models-for-a-singaporean-context-1a9efd395e1f ,
https://vivedhaelango.substack.com/p/lesson-45-should-i-fine-tune-my-embedding

**ตัวเลขที่น่าจะเป็นตัวยึดที่สุดสำหรับที่นี่**: **+0.03 ถึง +0.07 recall@10 สัมบูรณ์**
(NVIDIA NVDocs +0.063, Cisco +0.068, HTX MRR +0.033) — ไม่ใช่ +0.20 แบบ Atlassian
ซึ่งเป็น Recall@**60** (k กว้างมาก ตัวเลขพองง่าย) และไม่ใช่เปอร์เซ็นต์สามหลักจากบล็อก

### 3.2 กรณีที่ไม่ช่วยหรือทำร้าย

- **MNRL เปล่า ๆ ทำให้แย่ลง**: NDCG@10 = 0.626, **ต่ำกว่าโมเดลฐาน −0.045** เพราะ
  false negative ใน in-batch negatives
  (https://huggingface.co/blog/dragonkue/mitigating-false-negatives-in-retriever-training)
- **BM25-mined negatives แย่กว่าการสุ่ม**: 0.5002 vs 0.5248 NDCG@10 (NV-Retriever)
- **REFINE บน RAG-12000: +0.32% เท่านั้น** — คือ "ไม่ต่างจากศูนย์" ในทางปฏิบัติ
  โดเมนที่โมเดลฐานเข้าใจอยู่แล้วให้ผลตอบแทนเกือบศูนย์
- **BM25 ชนะ embedding model โดยไม่สนขนาด** ในบางโดเมน และ "การขยายขนาดโมเดลจาก small → large
  ไม่ให้ผลดีขึ้นเลย บางกรณีคะแนนต่ำลง" — สอดคล้องกับที่ repo นี้วัดเองเป๊ะ ๆ
  (BM25 0.5676 tie กับ 3 embedder ท็อป; qwen3_0.6b tie กับ qwen3-4B)
- **reranker สำเร็จรูปทำร้ายโดเมนเฉพาะทางได้**: บน patent citation retrieval
  Jina-reranker-v2 ทำ nDCG@10 ตกจาก 0.197 → 0.174 (−12%),
  BGE-reranker-v2-m3 ตกไป 0.130 (**−34%**)
  (https://arxiv.org/pdf/2605.24297)

### 3.3 การเจือจางจาก hybrid — **เป็นการอนุมาน ไม่ใช่การอ้างอิง**

ตัวเลข lift ทั้งหมดข้างบนเป็น **dense-alone → dense-alone** แต่ระบบ production ของโปรเจกต์นี้คือ
RRF(BM25 + dense) ซึ่ง BM25 เดี่ยว ๆ (0.5902) เกือบเท่า dense เดี่ยว ๆ (0.6581)
และ hybrid ได้ 0.6797 → **การเพิ่ม dense-alone +0.06 ไม่โอนเข้าระบบ fused แบบ 1:1**
ส่วนที่ fine-tune ปรับปรุงได้อาจเป็นส่วนที่ BM25 คลุมอยู่แล้ว
**ยังไม่มีวรรณกรรมที่วัดเรื่องนี้ให้ — ต้องวัดเองถ้าจะรู้**

---

## 4. Catastrophic forgetting / การเสื่อมนอกโดเมน + ความสามารถหลายภาษา

### 4.1 ที่รายงานไว้

- นิยาม: fine-tune ดีขึ้น in-domain แต่เสีย generalization นอกโดเมน เพราะโมเดลเลื่อนไปหา
  distribution ของ downstream task ออกจาก pre-training distribution
- ตัวอย่างเชิงปริมาณนอกสาย text: full fine-tune CLIP บน ImageNet-1K →
  **zero-shot retrieval ตก 28%** (https://arxiv.org/html/2501.15377v1)
- **ข่าวดีเชิงประจักษ์สำหรับ embedding โดยเฉพาะ**: REFINE วัด OOD ตรง ๆ แล้ว
  **แทบไม่พบการเสื่อม** — เทรนบน SQUAD ทดสอบบน RAG: BGE fine-tuned = **0.932** recall
  vs vanilla BGE = **0.937** (ตกแค่ 0.005); REFINE (model fusion) = 0.938 (ดีกว่า vanilla)
  → ที่ **ขนาดข้อมูลเล็ก (493–2,550 ตัวอย่าง) และ 1 epoch การเสื่อมนอกโดเมนมีน้อยกว่าที่กลัวกัน**
  (https://arxiv.org/html/2410.12890)

### 4.2 มิติหลายภาษา — จุดที่เกี่ยวกับภาษาไทยโดยตรง

- "การ fine-tune multilingual foundation model บนภาษาเฉพาะมักเหนี่ยวนำ catastrophic
  forgetting ทำให้ภาษาที่ไม่ได้อยู่ใน fine-tuning เสื่อมลง"
- "gradient descent บางทิศทางทำให้โมเดล fit ภาษาต้นทางมากเกินไป แล้วลืมความรู้ cross-lingual
  จาก pre-training → ทำลาย zero-shot"
- **ตัวกำหนดหลักคืออัตราส่วน model scale : data size** — โมเดล 4B + ข้อมูลไม่กี่พันคู่
  อยู่ในโซนที่ forgetting **ต่ำ** (ข้อมูลน้อยเกินกว่าจะย้าย 4B params ได้มาก) ซึ่งสอดคล้องกับผล REFINE
- (https://arxiv.org/html/2309.06089v2 , https://arxiv.org/pdf/2510.19546 ,
  https://proceedings.neurips.cc/paper_files/paper/2022/file/5f9f9e4da57a94547491a39dc18f1696-Paper-Conference.pdf)

**ข้อสังเกตเฉพาะเคสนี้**: ถ้าเทรนไทยล้วน ความสามารถ multilingual ของ Qwen3 (100+ ภาษา)
เป็นสิ่งที่เสี่ยงจะเสีย — แต่โปรเจกต์นี้ **ไม่ได้ใช้ความสามารถนั้น** คลังเป็นไทยล้วน
คำถามเป็นไทยล้วน → ความเสียหายจาก multilingual forgetting อาจ **ไม่ใช่ต้นทุนจริง**
สิ่งที่เป็นต้นทุนจริงกว่าคือการเสียความสามารถ **generalist ภายในภาษาไทย** —
ซึ่งคือคุณสมบัติที่ repo นี้วัดมาแล้วว่า qwen3-4B มีอยู่คนเดียว (ไม่มีจุดอ่อนพิสูจน์ได้ทั้ง person และ program)

### 4.3 มาตรการบรรเทา (ที่มีในเอกสาร)

| มาตรการ | รายละเอียด | แหล่ง |
|---|---|---|
| **LoRA/PEFT** | base ถูกแช่แข็ง เทรนเฉพาะ adapter — ระบุว่าเป็น "the strongest mitigation in practice"; ปิด adapter ได้ทันทีเพื่อกลับเป็นโมเดลเดิม | zeroentropy |
| **1 epoch + clip gradient norm** | "จำกัดที่ 1 epoch และคุม gradient norm ให้สมดุลดีกว่า" | zeroentropy / lit. |
| **LR เล็กลง 10×** | ลด weight drift | zeroentropy |
| **Rehearsal/replay 5–20%** | ผสมข้อมูลสไตล์ pre-training | zeroentropy |
| **Model fusion / interpolation** | REFINE: `E = λ·E_frozen + (1−λ)·E_finetuned`, λ=0.35 → รักษา generalization พร้อมได้ in-domain | arXiv 2410.12890 |
| **Early stopping บน OOD holdout** | เฝ้าดู benchmark นอกโดเมนระหว่างเทรน | zeroentropy |
| **เกณฑ์ธงแดง** | "ดีขึ้น in-domain แต่ตก >5–10% บน general benchmark = over-specialization" | substack |

(https://zeroentropy.dev/concepts/catastrophic-forgetting/)

> **หมายเหตุเข้ากับ repo นี้ได้ดี**: LoRA adapter เปิด/ปิดได้ → สามารถวัด
> paired bootstrap ระหว่าง "adapter on" vs "adapter off" **บนโมเดลตัวเดียวกัน น้ำหนักฐานเดียวกัน**
> เป็นการทดลองที่ควบคุมตัวแปรได้สะอาดกว่าการเทียบข้ามโมเดลที่ repo นี้ทำมาทั้งหมด

---

## 5. ทางเลือกที่ถูกกว่า — cost/benefit คร่าว ๆ

### 5.1 Linear adapter บน query embedding (frozen model) — **ตัวที่คุ้มที่สุดในรายการ**

Chroma Research (https://www.trychroma.com/research/embedding-adapters):
- สถาปัตยกรรม: **เมทริกซ์เชิงเส้นตัวเดียวคูณกับ query embedding เท่านั้น**
  → **ไม่ต้อง re-embed คลัง 81,489 chunks เลย** (นี่คือประโยชน์ใหญ่ที่สุด)
- ข้อมูลที่ต้องใช้: **น้อยถึง 1,500 คู่ที่ติดป้ายแล้ว**; ablation พบจุดคุ้มทุนที่ 20–35% ของข้อมูล
- ผล: CQADupstackEnglishRetrieval ปรับปรุง NDCG **สูงสุด 70%** และ **ชนะโมเดลที่ fine-tune เต็ม**;
  SpanishPassageRetrievalS2S "แข่งได้กับการ fine-tune"
- ⚠️ **อ่านเลข 70% นี้ด้วยความระวังแบบเดียวกับคำเตือนหัวข้อ 3**: มันเป็น **เปอร์เซ็นต์สัมพัทธ์
  บน baseline ที่ไม่ระบุค่าสัมบูรณ์ จาก benchmark ภาษาอังกฤษชุดเดียวที่ผลดีที่สุด** รายงานต้นทาง
  เองระบุว่าผลบน held-out "น่าประทับใจน้อยกว่า" และ gain ภาษาเกาหลีเล็กกว่ามาก
  → **ห้ามเอา "70%" ไปเทียบตรง ๆ กับ "+0.03–0.07 สัมบูรณ์" ของข้อ 3** มันคนละหน่วยคนละฐาน
  ตัวเลขสัมบูรณ์ของ adapter บนคลังไทยชุดนี้ **ไม่มีใครวัดไว้** — ต้องวัดเอง
- VRAM: ~0 (เทรนเมทริกซ์ 2560×2560 = 6.5M params); เวลา: นาที ไม่ใช่ชั่วโมง
- **ข้อควรระวังที่รายงานเอง**: ผลบน held-out ด้อยกว่าบน training data ชัดเจน (overfit ง่าย);
  **ผลข้ามภาษาอ่อน** — บน Ko-miracl (เกาหลี) ยังชนะ baseline แต่ gain "น้อยกว่ามาก"
  → ⚠️ นี่เป็นสัญญาณเตือนตรงสำหรับเคสภาษาไทย
- **สำคัญ**: เข้ากันได้กับ hybrid ทันที เพราะไม่แตะ index

### 5.2 Matryoshka truncation — **ไม่ใช่คู่แข่งของ fine-tune**

Qwen3-Embedding รองรับ MRL 32–2560 มิติ native (ไม่ต้องเทรนอะไรเลย)
แต่ MRL คือ **คันโยกด้าน cost/latency/storage ไม่ใช่คันโยกด้านคุณภาพ** —
มันทำให้ truncate แล้วไม่พังเท่าไร ไม่ได้เพิ่มความรู้โดเมน
(https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html ,
https://arxiv.org/pdf/2205.13147)
เกี่ยวข้องกับโปรเจกต์นี้ในแง่เดียว: ถ้าจะลดต้นทุน qwen3-4B (dim 2560 แพงที่สุดในตาราง
cost/latency) การ truncate เป็น 1024 มิติเป็นทางลดที่ "ฟรี" กว่าการเปลี่ยนโมเดล
— **แต่ยังไม่มีใครวัดบนคลังนี้**

### 5.3 เทรนเฉพาะ N layer สุดท้าย

- ประหยัด optimizer state ตามสัดส่วน layer แต่ **ไม่ประหยัด activation** (ยังต้อง forward ทั้ง 36 layer)
  และ backward ผ่านเฉพาะ N layer บน → ประหยัด compute ~ (36−N)/36 ของ backward
- vs LoRA: LoRA ประหยัดกว่าและมีหลักฐานเรื่อง forgetting ดีกว่า — **ไม่มีเหตุผลชัดที่จะเลือกทางนี้แทน LoRA**
- หลักฐานเปรียบเทียบ LoRA vs full FT ในสาย dense retrieval **ขัดกันเอง**:
  บางแหล่งบอก LoRA "เกือบเท่า full FT และเลี่ยง catastrophic forgetting";
  บางแหล่งบอก "LoRA ให้ผลด้อยกว่า full parameter tuning สำหรับ dense retrieval
  แม้อาจเป็นเพราะ capacity ของ LoRA ไม่พอ" — และ arXiv 2410.21228 แสดงว่า
  SVD ของเมทริกซ์ที่ได้จาก LoRA มี singular vector ใหม่อันดับสูงที่ full FT ไม่มี
  → **กลไกการปรับตัวต่างกันจริง ไม่ใช่แค่ประหยัดกว่า**
  (https://arxiv.org/html/2410.21228v2)
  **สำหรับเคสนี้เรื่องนี้เป็นประเด็นเชิงวิชาการล้วน เพราะ full FT ไม่ใช่ทางเลือกบน 12GB อยู่แล้ว**

### 5.4 Reranker (cross-encoder) แทนการแตะ embedder

*(map บล็อกการตัดสินใจข้อนี้ไว้กับตั๋ว 01 — ที่นี่ให้แต่ข้อเท็จจริงต้นทุน/ผล)*
- BGE-reranker-v2-m3: **278M params, multilingual**, nDCG@10 51.8 บน BEIR;
  bge-reranker-large (560M) เพิ่ม ~2 nDCG@10 แต่ latency เท่าตัว
- **ต้นทุน**: ไม่ต้องเทรนอะไรเลย, ไม่ต้อง re-index, VRAM ตอน inference ~0.6GB, รันบน 3060 สบาย
- **ความเสี่ยงที่วัดได้แล้ว**: บนโดเมนเฉพาะทาง (patent) reranker สำเร็จรูป **ทำให้แย่ลง**
  −12% ถึง −34% nDCG@10 → "reranker ช่วยเสมอ" ไม่จริง
- **เงื่อนไขที่ใช้ได้**: ช่วยเฉพาะเมื่อเอกสารที่ถูกต้อง **ติด top-k แล้วแต่อันดับแย่**
  ถ้าไม่ติด top-k เลย reranker ช่วยไม่ได้ → รอตั๋ว 01
- (https://huggingface.co/BAAI/bge-reranker-base , https://arxiv.org/pdf/2605.24297)

### 5.5 สรุปตารางเทียบ

| ทาง | VRAM เทรน | เวลา | ต้อง re-index? | lift ที่มีหลักฐาน | ความเสี่ยง |
|---|---|---|---|---|---|
| Linear query adapter | ~0 | นาที | **ไม่** | *สัมพัทธ์* สูงสุด +70% NDCG — benchmark อังกฤษชุดเดียว, held-out อ่อนกว่า, ข้ามภาษาน้อยกว่ามาก; **ไม่มีเลขสัมบูรณ์ ห้ามเทียบตรงกับแถวอื่น** | overfit; ผลข้ามภาษาอ่อน |
| Reranker สำเร็จรูป | ไม่ต้องเทรน | ชั่วโมง (setup) | ไม่ | ขึ้นกับโดเมน | ทำให้แย่ลงได้ −12%..−34% |
| LoRA fine-tune 0.6B | ~3–4 GiB | <1 ชม./epoch | **ใช่ 81k chunks** | คาด +0.03..+0.07 R@10 | mismatch ความยาว |
| LoRA fine-tune 4B | ~10 GiB (ตึง) | 4–10 ชม./epoch | **ใช่ 81k chunks** | เดียวกัน | mismatch + engineering risk |
| MRL truncation | 0 | นาที | ใช่ (แค่ตัด) | ลดต้นทุน ไม่เพิ่มคุณภาพ | — |
| Full FT 4B | ~60 GiB | — | — | — | **เป็นไปไม่ได้บน 3060** |

---

## 6. เวลาเทรนจริงบน 3060 12GB สำหรับข้อมูลไม่กี่พันคู่

### 6.1 วิธีประมาณ (steps × sec/step ไม่ใช่ FLOPs)

**จุดยึดที่หาได้จริง** (ไม่มี benchmark ของ 3060 + 4B embedder ตรง ๆ — ไม่มีใครเผยแพร่):
arXiv 2509.12229 "Profiling LoRA/QLoRA Fine-Tuning Efficiency on Consumer GPUs: An RTX 4060 Case Study" —
Qwen2.5-**1.5B**, LoRA, gradient checkpointing เปิด, seq 512/1024/2048:
**360–628 tokens/s**, peak VRAM 6.2–8.06 GiB
(https://arxiv.org/html/2509.12229v1)

การปรับสเกล (ระบุว่าเป็นการอนุมาน):
1. 4.0B / 1.5B ≈ **2.7×** FLOPs ต่อ token → ~185 tok/s
2. 3060 vs 4060: bandwidth 360 vs 272 GB/s (3060 ดีกว่า), compute ต่ำกว่าเล็กน้อย → **~0.9–1.1×** → ~165–205 tok/s
3. **CachedMNRL forward สองรอบ → ÷1.8–2.2** → **~75–115 tokens/s ที่มีผลจริง**

### 6.2 ตัวเลขที่ได้

สมมติ 3,000 triplet (query, positive, hard negative), `max_seq_length=384`,
query ~48 token, doc ชนเพดาน:
- token ต่อ triplet ≈ 48 + 384 + 384 = **816**
- ต่อ epoch ≈ 3,000 × 816 = **2.45M tokens**
- ที่ 75–115 tok/s → **21,000–33,000 วินาที = ~6–9 ชั่วโมงต่อ epoch**

ตัวแปรอื่น:
| config | tokens/epoch | เวลา/epoch |
|---|---|---|
| 3,000 triplet, seq 256 | 1.68M | **~4–6 ชม.** |
| 3,000 **pair** (ไม่มี explicit hard neg), seq 384 | 1.30M | **~3–5 ชม.** |
| 3,000 triplet, seq 384 | 2.45M | **~6–9 ชม.** |
| 3,000 triplet, seq 512 | 3.2M | **~8–12 ชม.** |
| เดียวกันบน **Qwen3-Embedding-0.6B** | เดียวกัน | **~1–1.5 ชม.** (÷6.7) |

**เนื่องจากวรรณกรรมแนะนำ 1 epoch เพื่อเลี่ยง forgetting → งานเทรนจริงคือ "รันข้ามคืน 1 ครั้ง"
สำหรับตัว 4B และ "รันตอนพักเที่ยง" สำหรับตัว 0.6B**

> ⚠️ **ตัวเลขนี้เป็นการอนุมานสามชั้นจากจุดยึดคนละโมเดลคนละการ์ด**
> ต้องยืนยันด้วยการวัด **1 training step จริง** ก่อนเชื่อ — ค่าที่ผิดได้ง่ายที่สุดคือ
> ค่าปรับของ CachedMNRL และ overhead ของ gradient checkpointing บน PEFT
> (ซึ่งมี bug report ว่าอาจไม่ทำงานตามคาด ดู §2.4)

### 6.3 ต้นทุนที่ตั๋วไม่ได้ระบุแต่ใหญ่กว่าเวลา GPU: **ข้อมูลเทรนยังไม่มีอยู่**

- ปัจจุบันมี **Gold set 73 คำถามเท่านั้น** — "ไม่กี่พันคู่" หมายถึงต้อง**สังเคราะห์**
- ข้อจำกัด "ห้ามให้เนื้อหาคลังออกจากเครื่อง" ⇒ **generator ต้องรันในเครื่อง** ⇒
  ต้องโหลด LLM ไทยตัวหนึ่งลง 3060 ตัวเดียวกัน (คนละเวลากับการเทรน) — เป็นงานวิศวกรรมจริงอีกก้อน
- **73 Gold queries ต้อง held out ทั้งหมด** ไม่งั้น eval ปนเปื้อนและตัวเลขทั้งหมดใน
  `paper-results-summary.md` เทียบกันไม่ได้อีกต่อไป
- แนวทางที่ 3 แหล่งตรงกัน: LLM สร้าง query 2–3 ข้อจากแต่ละ chunk + ขุด hard negative
  **โดยไม่ใช้ BM25 เปล่า ๆ** (ดู §1.3) + กรอง false negative ด้วย `relative_margin=0.05`
- ปริมาณอ้างอิง: 500–1,000 คู่ = "ทดสอบว่าช่วยไหม", 5,000–10,000 = "ได้ผลจริงจัง",
  50,000+ = production (https://vivedhaelango.substack.com/p/lesson-45-should-i-fine-tune-my-embedding)
  **→ "ไม่กี่พันคู่" อยู่ปลายล่างของช่วง "ได้ผลจริงจัง"**

### 6.4 checklist ที่วรรณกรรมแนะนำให้ทำ *ก่อน* fine-tune

จาก substack + NVIDIA + practitioner consensus — "fine-tuning เป็นคันโยกสุดท้าย ไม่ใช่คันแรก":
1. เช็ค chunking strategy ✅ *(repo ทำแล้ว — semantic ชนะ)*
2. ลองโมเดลฐานที่ใหญ่กว่า/เฉพาะโดเมนกว่า ✅ *(repo ทำแล้ว — 9 embedder)*
3. ตรวจว่า **metadata filtering** แก้ปัญหาได้ไหม ⬜ *(= ตั๋ว 02, ยังไม่ทำ)*
4. ตรวจ chunk ที่ retrieve มาด้วยมือ เพื่อยืนยันว่า embedder คือคอขวดจริง ⬜ *(= ตั๋ว 01, ยังไม่ทำ)*
5. **empirical test**: encode query-doc pair จริง 5–10 คู่ วัด cosine —
   >0.8 = โมเดลเข้าใจโดเมนแล้ว gain น้อย; 0.5–0.8 = ลองอย่างอื่นก่อน; <0.5 = fine-tune มีเหตุผล
   ⬜ **ยังไม่เคยทำ และเป็นการทดลอง 20 นาทีที่ให้สัญญาณตรงที่สุด**

### 6.5 ข้อเท็จจริงที่ต้องวางบนโต๊ะ (ไม่ตัดสินให้)

จาก `paper-results-summary.md` เอง:
- `qwen3` (4B) เป็น embedder **แพงที่สุด**ในตาราง cost/latency (encode 264ms, dim 2560)
  และ **ไม่ใช่** ผู้นำ hybrid — `qwen3_0.6b × semantic × hybrid` = **0.6935** สูงกว่า
  `qwen3 × semantic × hybrid` = 0.6797
- `qwen3_0.6b`: น้ำหนัก bf16 ≈ **1.2 GiB** → คำถาม VRAM ทั้งหมดในข้อ 2 **หายไปเลย**
  เทรนได้ที่ seq ยาวกว่า batch ใหญ่กว่า เร็วกว่า ~6.7× และเปิดทางให้ทำ ablation
  หลาย config ในเวลาเดียวกับ 4B รอบเดียว
- เหตุผลเดียวที่ยึด 4B คือคุณสมบัติ **"generalist ไม่มีจุดอ่อนพิสูจน์ได้ทั้ง person และ program"**
  ซึ่ง 0.6B ไม่มี (แพ้ bge-m3 บน person, Holm-adj p<0.0001, −0.14)
  → **และคุณสมบัตินี้คือสิ่งที่ catastrophic forgetting จะทำลายเป็นอย่างแรก** (§4.2)

---

## แหล่งอ้างอิงทั้งหมด

**เอกสารเครื่องมือ**
- https://huggingface.co/blog/train-sentence-transformers
- https://sbert.net/docs/sentence_transformer/loss_overview.html
- https://sbert.net/docs/package_reference/sentence_transformer/losses.html
- https://sbert.net/docs/package_reference/util/hard_negatives.html
- https://sbert.net/examples/sentence_transformer/training/peft/README.html
- https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html
- https://deepwiki.com/huggingface/sentence-transformers/5.3-loss-functions
- https://github.com/huggingface/sentence-transformers/issues/3173 (CachedMNRL gradient)
- https://github.com/huggingface/sentence-transformers/issues/3434 , /3701 (PEFT checkpoint)
- https://github.com/huggingface/transformers/issues/42947 (gradient checkpointing + PEFT)
- https://unsloth.ai/docs/basics/embedding-finetuning (มี notebook Qwen3-Embedding 4B และ 0.6B)

**Qwen3-Embedding**
- https://huggingface.co/Qwen/Qwen3-Embedding-4B
- https://github.com/QwenLM/Qwen3-Embedding
- https://qwenlm.github.io/blog/qwen3-embedding/
- https://arxiv.org/abs/2506.05176 (technical report)
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/discussions/29 (prefix best practices)

**ผลเชิงปริมาณ**
- https://huggingface.co/blog/nvidia/domain-specific-embedding-finetune
- https://blogs.cisco.com/ai/fine-tuning-embedding-models-for-enterprise-retrieval-a-practical-guide-with-nvidia-nemotron-recipe
- https://arxiv.org/html/2407.15831v1 (NV-Retriever, hard negatives)
- https://huggingface.co/blog/dragonkue/mitigating-false-negatives-in-retriever-training
- https://arxiv.org/html/2410.12890 (REFINE, OOD retention)
- https://medium.com/htx-dsai/fine-tuning-multilingual-embedding-models-for-a-singaporean-context-1a9efd395e1f
- https://www.trychroma.com/research/embedding-adapters
- https://arxiv.org/pdf/2605.24297 (patent benchmark — reranker ทำร้ายโดเมนเฉพาะทาง)
- https://arxiv.org/html/2509.12229v1 (consumer-GPU LoRA profiling)

**Forgetting / LoRA vs full FT**
- https://zeroentropy.dev/concepts/catastrophic-forgetting/
- https://arxiv.org/html/2501.15377v1 (CLIP zero-shot −28%)
- https://arxiv.org/html/2410.21228v2 (LoRA vs full FT: an illusion of equivalence)
- https://arxiv.org/html/2309.06089v2 (cross-lingual forgetting)
- https://arxiv.org/pdf/2510.19546 (conditions for catastrophic forgetting, multilingual)
- https://vivedhaelango.substack.com/p/lesson-45-should-i-fine-tune-my-embedding

**ในบ้าน (ไม่ใช่ web)**
- `RAG/docs/paper-results-summary.md` — ConGen −0.0298 (p=0.0016) จาก length mismatch;
  ตาราง cost/latency; ผล hybrid; เหตุผลปฏิเสธ Qwen3-8B เพราะ quantization confound
