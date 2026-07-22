# Gap Analysis: กรอบงานวิจัย Embedding เทียบกับสิ่งที่ทำแล้ว (20 ก.ค. 2569)

เทียบโน้ตใน `Embedding โมเดล.docx` (กรอบเปเปอร์ที่ผู้ใช้ร่างไว้) กับ codebase จริง
เพื่อตอบ 2 คำถาม: **(1) อะไรทำแล้ว (2) อะไรน่าทำเพิ่ม** — จัดลำดับตาม
effort × คุณค่าเชิงวิจัย และกรองด้วยข้อจำกัดฮาร์ดแวร์ (RTX 3060 12GB → เพดาน ~4-5B
params fp16)

> **อัปเดตสถานะ 2026-07-21: Tier 1 (§8) ปิดครบทั้ง 4 ข้อแล้ว.** ตารางสถานะใน §3-6
> ด้านล่างเป็น snapshot ตอนวันที่วิเคราะห์ (20 ก.ค.) — **ไม่ได้อัปเดตย้อนหลัง**
> (คงไว้เป็นบันทึกจุดเริ่มต้น) ดูตัวเลข/ผลลัพธ์ล่าสุดที่ `docs/paper-results-summary.md`
> แทน สรุปย่อ 4 ข้อ:
>
> 1. **MAP + Precision@k + multi-k** — เพิ่มใน `src/rag_lab/metrics.py` แล้ว
>    (`evaluate()` รับ `k` เป็น int หรือ list) และรันซ้ำด้วย multi-k จริงแล้ว
>    (2026-07-22, `tools/eval/multi_k_report.py` — pure recompute จากผลที่
>    persist ไว้แล้ว top_k=10 อยู่แล้ว ไม่ต้อง retrieve ใหม่) เจอ nuance ใหม่:
>    ที่ MAP/precision@1 `bge_m3` นำ `qwen3_0.6b` สวนทางกับที่เสมอกันบน recall@10
>    — ยังไม่ทดสอบนัยสำคัญ เป็น open item ใหม่
> 2. **BM25 standalone baseline** — รันแล้ว + sig-test กับ embedder ทั้ง 9 ตัว: BM25
>    เฉยๆ ผูกสถิติเสมอกับ top tier (bge-m3/Qwen3-4B/Qwen3-0.6B) และชนะ embedder ที่
>    อ่อนกว่าอย่างมีนัยสำคัญทุกตัว
> 3. **Bootstrap + Holm stats** — เปลี่ยนจาก paired t-test เป็น paired bootstrap
>    (n=10000) + Holm correction แล้ว ยืนยัน hybrid (RRF) ชนะ dense-alone อย่างมี
>    นัยสำคัญทุก embedder ทุก metric — ผลที่แข็งแรงที่สุดของโครงการ
> 4. **ตารางระบบ + Pareto (cost/latency)** — `tools/eval/cost_latency_pareto.py`
>    พบว่า implementation ปัจจุบันของ hybrid เพิ่ม overhead คงที่ ~2.1-2.3 วินาที
>    ต่อ query แทบไม่ขึ้นกับ embedder (BM25Okapi rebuild ทุก query + over-fetch ทั้ง
>    corpus ก่อน fuse ไม่ใช่ต้นทุนของ RRF เอง) — รายงานไว้ ไม่ได้แก้โค้ด
>
> ระหว่างทางพบว่า `qwen3_0.6b × semantic × hybrid` (recall@10=0.6935) เป็นตัวเลข
> สูงสุดในทั้ง study แต่ยังไม่ผ่าน significance test เทียบกับ combo อื่นในกลุ่มบนสุด
> จึง **ยังไม่ยกแชมป์ให้ embedder ตัวใดตัวหนึ่ง** — headline ของเปเปอร์อยู่ที่ระดับ
> ระบบ (semantic chunking + hybrid retrieval) ไม่ใช่ embedder ตัวเดียว รายละเอียด
> เต็ม: `docs/paper-results-summary.md`, narrative: `docs/chunker-embedder-comparison-log.md`.
>
> **Tier 2 (§8) ก็ปิดครบทั้ง 2 ข้อแล้ว** ระหว่างขยายเป็น 9 embedders: ข้อ 5
> (โมเดลไทยกลุ่ม A เพิ่ม) คือ `sct` (SCT-KD-BGE-M3-model-phayathaibert — training
> method อื่นบน backbone เดียวกับ ConGen ที่มีอยู่แล้ว) และข้อ 6 (Qwen3 scaling)
> คือ `qwen3_0.6b` — รายละเอียดที่ `[[project_embedder_models_to_add]]` (memory)
> **Tier 3 ยังไม่เริ่มเลย**: RQ3 (normalization/segmentation ablation), cross-encoder
> reranker, RQ4 (end-to-end RAG + RAGAS/LLM-judge) — ยังเป็นเฟสใหม่ทั้งก้อน

**แกนหลักของเปเปอร์ (ข้อเสนอ)**: RQ1/RQ2 (embedder comparison) — ตรงกับสิ่งที่
build ไว้แล้วมากที่สุด; RQ4 (end-to-end RAG) โน้ตเองจัดเป็น "เสริม" และเป็น subsystem
ใหม่ทั้งก้อน จึงควรเป็น *เฟสถัดไป* ไม่ใช่ส่วนหนึ่งของรอบแรก

---

## 1. Research Questions — สถานะ

| RQ | ใจความ | สถานะ |
|---|---|---|
| RQ1 | embedder ใดค้นคืนไทยดีสุดใน RAG + ต่างกันอย่างมีนัยสำคัญไหม | **ทำเกือบครบ** — มี 6 embedder × 4 chunker + t-test แล้ว; ขาด stats ที่แข็งกว่า (ดู §5) |
| RQ2 | โมเดลไทยเฉพาะทาง เหนือกว่า multilingual/LLM-based เมื่อคิดทั้งคุณภาพ+ต้นทุน? | **ทำบางส่วน** — มีคุณภาพแล้ว, ขาดมิติ "ต้นทุน" เชิงระบบ (§4) และโมเดลไทยกลุ่ม A ยังมีตัวเดียว |
| RQ3 | preprocessing (chunk size, normalize ไทย) ส่งผลแค่ไหน? | **แทบยังไม่เริ่ม** — chunk_size=512 คงที่ทุกที่, ไม่มี normalization ablation |
| RQ4 | retrieval สัมพันธ์กับคุณภาพคำตอบ end-to-end แค่ไหน? | **ยังไม่มี** — ไม่มี generation stage เลย = subsystem ใหม่ |

---

## 2. กลุ่มโมเดล (4 กลุ่มตามโน้ต) — สถานะ

| กลุ่ม | โน้ตเสนอ | เรามี | ช่องว่าง |
|---|---|---|---|
| **A. ไทยเฉพาะทาง** | SimCSE-WangchanBERTa, SCT-PhayaThaiBERT, ConGen-XLMR-Thai | **ConGen-PhayaThaiBERT** (1 ตัว) | เพิ่มได้อีกหลายตัว — near-zero code, ตรงกับ RQ2 โดยตรง (memory `[[project_embedder_models_to_add]]` มีคิวไว้แล้ว) |
| **B. Multilingual เปิด** | BGE-M3, mE5-large/base, gte-multilingual, jina-v3 | **bge-m3, e5-large, jina_v5, m2v** (jina-v3 distilled) | ครอบคลุมดีแล้ว; gte-multilingual-base เพิ่มได้ถ้าอยากครบ |
| **C. LLM-based** | Qwen3 0.6B/4B/8B, e5-mistral/NV-Embed 7-8B | **Qwen3-Embedding-4B** | 7-8B **เกินเพดาน 12GB** — out of scope บนเครื่องนี้ (Qwen3-8B ปฏิเสธไปแล้ว). Qwen3-0.6B เพิ่มได้ (เล็ก) เพื่อดู scaling ภายในตระกูลเดียว |
| **D. API เชิงพาณิชย์** | OpenAI-3-large, Cohere v4, Gemini | `api_embedder.py` เป็น **stub** (ยังไม่ต่อ provider) | optional — โน้ตเองบอกเป็น "upper-bound อ้างอิง"; ต้องมี key + ค่าใช้จ่าย + ส่งข้อมูลออกนอกเครื่อง |

**สรุปกลุ่มโมเดล**: กลุ่ม B ครบ, กลุ่ม C เต็มเพดานฮาร์ดแวร์แล้ว, **กลุ่ม A คือช่องว่าง
ที่คุ้มสุด** (ถูก+ตรง RQ2), กลุ่ม D เลื่อนได้

---

## 3. Retrieval Metrics (§5.1) — สถานะ

| เมตริก | สถานะ | หมายเหตุ |
|---|---|---|
| Recall@k | ✅ มี (`recall_at_k`) | แต่ **hardcode k=10 ตัวเดียว** |
| MRR | ✅ มี (`reciprocal_rank`) | |
| nDCG@k | ✅ มี (`ndcg_at_k`) | binary relevance (ไม่ graded) |
| **MAP** | ❌ ไม่มี | โน้ตขอ — ขยาย `evaluate()` ง่าย |
| **Precision@k** | ❌ ไม่มี | โน้ตขอ |
| **k ∈ {1,3,5,10}** | ❌ มีแค่ 10 | โน้ตขอหลาย k เพื่อดู trade-off |

**ทำได้ทันที**: เพิ่ม MAP, P@k, และ loop หลาย k ใน `src/rag_lab/metrics.py` — โค้ดน้อย
มี unit test รองรับอยู่แล้ว คุณค่าเชิงเปเปอร์สูง (reviewer คาดหวังชุดนี้)

---

## 4. ประสิทธิภาพเชิงระบบ + ต้นทุน (§5.2) — สถานะ

| มิติ | สถานะ |
|---|---|
| Encoding throughput | ⚠️ มีบางส่วน — log `embed_seconds`/`chunk_seconds` ต่อ combo อยู่แล้ว (ดู `chunker-embedder-comparison-log.md`) แต่ไม่ได้ทำเป็นตารางเทียบเป็นระบบ |
| Query latency p50/p95 | ❌ ไม่ได้วัด |
| ขนาดเวกเตอร์ / index size | ❌ ไม่ได้รวบรวมเป็นตาราง (มิติ: e5=1024, bge-m3=1024, qwen3=2560, jina_v5=1024, m2v=1024) |
| ต้นทุน (GPU/API) | ❌ ยังไม่ทำ |
| **Pareto frontier plot** | ❌ ไม่มี | คุณภาพ (nDCG@10) vs ต้นทุน/latency — จุดขายของ RQ2 |

**คุ้มค่า**: รวบรวมตารางระบบ (dim, index size, embed throughput, query latency) + พล็อต
Pareto — ข้อมูล throughput ครึ่งหนึ่งมีใน meta.json อยู่แล้ว เหลือเก็บ latency + dim/size

---

## 5. การทดสอบนัยสำคัญทางสถิติ (§5.4) — สถานะ

| วิธี | สถานะ |
|---|---|
| Paired t-test รายคำถาม | ✅ ทำแล้ว (fixed_size vs semantic, `gold_eval_breakdown.py`) |
| **Paired bootstrap / permutation test** | ❌ โน้ตแนะนำ — robust กว่า t-test เมื่อ n น้อย/ไม่ normal |
| **Confidence intervals** | ❌ ไม่ได้รายงาน |
| **Multiple-comparison correction (Holm/Bonferroni)** | ❌ สำคัญ — ตอนนี้เทียบ 6 embedder = หลายคู่พร้อมกัน ต้องคุม false positive |

**คุ้มค่าสูง**: มี t-test เป็นฐานแล้ว การอัปเกรดเป็น bootstrap + Holm เป็น
methodological upgrade ที่ reviewer มองหา — effort ปานกลาง เครดิตสูง

---

## 6. Baselines & preprocessing — ช่องว่างที่โน้ตชี้เฉพาะ

- **BM25 lexical baseline**: `retrievers/bm25.py` **มีโค้ดแล้ว** และถูกใช้ใน
  `HybridRetriever` — **แต่ eval ที่รันอยู่เป็น `dense` ล้วน ยังไม่มีแถว BM25 เดี่ยว
  ในตารางเทียบ** โน้ตขอ "≥1 lexical baseline" ชัดเจน → ปิดช่องว่างถูกมาก (รัน retriever
  ที่มีอยู่บน Gold set) และเป็นสิ่งที่ reviewer มองหาแรกๆ
- **RQ3 preprocessing ablation**: `strip_mapping_tables()` เป็นการ *clean* ไม่ใช่
  normalization ablation; chunk_size คงที่ 512 ทุกที่ → **RQ3 แทบยังไม่เริ่ม** อย่าให้
  chunker comparison สวมรอยเป็น RQ3 โน้ตต้องการ normalize (เลขไทย↔อารบิก, วรรณยุกต์ซ้ำ,
  ช่องว่าง) + newmm segmentation เป็น *ตัวแปรทดลอง*

---

## 7. หัวข้อเสริมท้ายโน้ต

- **Cross-Encoder reranking**: ยังไม่มี (`retrievers/` ไม่มี reranker) — optional ที่
  scope ชัด คุ้ม ทำ 2-stage retrieval (dense/hybrid → rerank top-k) ยกคุณภาพได้จริง
- **Late chunking**: ยังไม่มี — เชิงทดลองกว่า flag เป็น "ถ้ามีเวลา"

---

## 8. แผนจัดลำดับ (effort × คุณค่าเชิงวิจัย)

**Tier 1 — ถูก + คุณค่าสูง + ตรงโน้ต (ทำก่อน):**
1. ขยาย `metrics.py`: MAP, Precision@k, k∈{1,3,5,10}
2. รัน **BM25 standalone baseline** บน Gold set (โค้ดมีแล้ว) → เพิ่มแถว lexical ในตาราง
3. Stats upgrade: paired bootstrap + CI + Holm correction
4. ตารางระบบ + Pareto: dim, index size, throughput (มีบางส่วน), query latency p50/p95

**Tier 2 — ขยายความครอบคลุม ตรง RQ2 (คุ้ม, code น้อย):**
5. เพิ่มโมเดลไทยกลุ่ม A อีก 1-2 ตัว (คิวใน `[[project_embedder_models_to_add]]`)
6. Qwen3-0.6B เพื่อดู scaling ภายในตระกูล Qwen3 (เล็ก, พอดีฮาร์ดแวร์)

**Tier 3 — เฟสใหม่ / งานใหญ่ (ตัดสินใจเชิงกลยุทธ์):**
7. **RQ3**: normalization + segmentation ablation (ตัวแปรทดลองใหม่)
8. Cross-encoder reranker (optional, scope ชัด)
9. **RQ4 / End-to-end RAG**: generation stage + RAGAS/LLM-judge — subsystem ใหม่ทั้งก้อน

**นอกขอบเขตฮาร์ดแวร์นี้:**
- LLM-based 7-8B (e5-mistral, NV-Embed, Qwen3-8B) — เกิน 12GB
- API เชิงพาณิชย์กลุ่ม D — ต้อง key/ค่าใช้จ่าย/ส่งข้อมูลออก; โน้ตจัดเป็นอ้างอิง upper-bound
