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
> **แก้ไข 2026-07-29 — ตัวเลขข้างบน 2 จุดใน blockquote นี้เลิกใช้ได้แล้ว, อย่าอ้างอิงต่อ**:
> (1) "0.6935" (และเลขที่พัฒนาต่อมาเป็น 0.7048 ในรอบ refresh 2026-07-25) ค้างมาจาก
> ก่อน OCR-remediation rebuild (28 ก.ค.) — ค่าสดตอนนี้คือ 0.6152 และไม่ใช่เลขสูงสุด
> ของ `qwen3_0.6b` เองด้วยซ้ำ (`sentence`/`fixed_size` สูงกว่า) claim "สูงสุดในทั้ง
> study" **ถูกถอนแล้ว**; (2) "BM25 ผูกสถิติเสมอกับ top tier (bge-m3/Qwen3-4B/
> Qwen3-0.6B)" ก็ไม่จริงอีกต่อไป — BM25 ตอนนี้**ชนะ bge-m3 อย่างมีนัยสำคัญ** ผูกแค่
> qwen3/qwen3_0.6b เท่านั้น ทั้งคู่เป็นผลจาก retrieval cache ของ BM25/hybrid ที่ค้าง
> 3 วันหลัง rebuild แล้วเพิ่งแก้ ดูตัวเลข/บทสรุปปัจจุบันที่
> `docs/paper-results-summary.md`, narrative เต็มที่
> `docs/chunker-embedder-comparison-log.md` ("Re-eval หลัง OCR-remediation rebuild"),
> memory `[[project_eval_refresh_2026_07_29]]`.
>
> **แก้ไขเพิ่ม 2026-07-29 (บ่ายวันเดียวกัน) — "headline ของเปเปอร์อยู่ที่ระดับระบบ
> (semantic chunking + hybrid retrieval)" ในบรรทัดข้างบนก็ต้องแก้ด้วย**: สร้าง
> chunker-vs-chunker significance test ตัวแรกของโปรเจกต์
> (`tools/eval/hybrid_chunker_significance_test.py`) แล้วพบว่า **`semantic`
> ไม่เคยชนะ chunker ตัวไหนอย่างมีนัยสำคัญเลย ทั้งต่อ embedder เดียวและรวมทุก
> embedder** — claim "semantic ชนะ" เป็นแค่ค่าเฉลี่ยดิบที่ไม่เคยผ่าน
> significance test มาก่อน กรอบใหม่: `recursive`/`semantic`/`sentence` ผูกกัน
> เป็นกลุ่มบน ไม่มีตัวชนะที่พิสูจน์ได้ มีแค่ `fixed_size` ที่พิสูจน์แล้วว่าด้อย
> กว่า `recursive` รายละเอียดที่ `docs/paper-results-summary.md` § "Chunkers
> compared"
>
> **แก้ไขเพิ่ม 2026-07-29 (เย็นวันเดียวกัน) — จุด #4 (cost/latency Pareto) ข้าง
> บนก็ต้อง refresh**: `cost_latency_pareto.py` รันสด (background task
> `bd9g6naw7`) ยืนยัน mechanics แทบไม่เปลี่ยน — overhead ปรับจาก ~2.1-2.3s เหลือ
> ~1.9-2.0s (การขยับเล็กน้อยจากขนาด corpus 74,819 chunks ไม่ใช่ effect ใหม่) แต่
> quality column ที่รายงานคู่กันตกลงทุกตัวเหมือนตารางอื่น (`qwen3 × semantic`
> dense 0.6581→0.5382 เป็นต้น) — เลข `qwen3_0.6b` ที่สูงสุดในตารางนั้น**ไม่ใช่
> claim optimal-chunker** ตามข้อแก้ไขข้างบน รายละเอียดที่
> `docs/paper-results-summary.md` § "Cost / latency characterization"
> **Tier 2 (§8) ก็ปิดครบทั้ง 2 ข้อแล้ว** ระหว่างขยายเป็น 9 embedders: ข้อ 5
> (โมเดลไทยกลุ่ม A เพิ่ม) คือ `sct` (SCT-KD-BGE-M3-model-phayathaibert — training
> method อื่นบน backbone เดียวกับ ConGen ที่มีอยู่แล้ว) และข้อ 6 (Qwen3 scaling)
> คือ `qwen3_0.6b` — รายละเอียดที่ `[[project_embedder_models_to_add]]` (memory)
> **Tier 3 อัปเดต 2026-07-23: RQ3 ตอนนี้ปิดแล้ว** — build จริงเต็มคอร์ปัสรันครบทั้ง
> 3 ablation (normalization, word-aware segmentation, chunk-size sweep 256/512/1024)
> ผลสรุป: **มีแค่ chunk_size ที่ส่งผลอย่างมีนัยสำคัญ** (เล็กกว่าดีกว่าสำหรับ
> recall@10) ส่วน normalization กับ segmentation ไม่มีผลนัยสำคัญเลยสักตัว — ตัวเลข
> เต็มที่ `docs/paper-results-summary.md` § "RQ3 ablation results" และ narrative ที่
> `docs/chunker-embedder-comparison-log.md`. **cross-encoder reranker (ข้อ 8) ก็ปิด
> แล้วเช่นกัน 2026-07-23** — สร้างจริง+รัน sig-test จริง ผลเป็น **ลบอย่างมีนัยสำคัญ
> สำหรับ hybrid** (MRR 0.848→0.760 p=0.006, nDCG@10 0.675→0.617 p=0.030) ไม่มีผลกับ
> dense เลย มี literature review primary-source รองรับคำอธิบายที่
> `docs/reranker-hybrid-interaction-research.md` รายละเอียดตัวเลขเต็มที่
> `docs/paper-results-summary.md` § "Cross-encoder reranker results" — **เหลือแค่
> RQ4 (end-to-end RAG + RAGAS/LLM-judge) ที่ยังไม่เริ่มใน Tier 3**
>
> **อัปเดต 2026-07-23 (ท้ายวัน) — methodology caveat สำคัญ**: พบ+แก้บั๊ก
> corpus-discovery ที่ทำให้ full-corpus index ทุกอันที่เคยสร้างมา (รวมถึง
> index เบื้องหลังตัวเลขทั้งหมดข้างบนนี้) ปนเปื้อนไฟล์ที่ไม่ใช่มติจริง
> ~6.87-8.25% ของ chunk ทั้งหมด (รายละเอียดเต็ม:
> `docs/chunker-embedder-comparison-log.md` § "บั๊ก corpus discovery ปนเปื้อน
> ทุก full-corpus index ที่เคยสร้างมา") อัตราใกล้เคียงกันในทุก chunker เลยไม่
> น่าพลิกผลสรุปเชิงคุณภาพ แต่ตัวเลขที่แม่นยำควรถือว่ามี noise แฝงอยู่ในระดับนี้
> จนกว่าจะ rebuild index ให้สะอาด
>
> **อัปเดต 2026-07-25**: ผู้ใช้เปลี่ยนใจจากการเลื่อนด้านบน — สั่ง rebuild
> `chunker_compare_full` ทั้งชุด (4 chunker × 9 embedder) แบ่งเป็น 4 batch
> ตาม chunker รันเสร็จสมบูรณ์และตรวจสอบไม่มีการปนเปื้อนแล้ว (0 chunk ปลอมใน
> ทุก 36 combo) รายละเอียดเต็ม:
> `docs/chunker-embedder-comparison-log.md` § "Rebuild index ประวัติศาสตร์
> ทั้งชุดเสร็จสมบูรณ์ (24-25 ก.ค. 2569)" — **แต่ตัวเลขในเอกสารนี้และใน
> `docs/paper-results-summary.md` ยังไม่ได้ regenerate จาก index ใหม่**
> ยังเป็นงานค้างแยกต่างหาก

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

- **Cross-Encoder reranking**: ✅ ปิด 2026-07-23 (ดู §8 ข้อ 8) — สร้างจริง รันจริง
  ผลเป็น**ลบ**อย่างมีนัยสำคัญสำหรับ hybrid ไม่ใช่บวกตามที่โน้ตคาดไว้เดิม
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
   **โค้ดเขียนเสร็จแล้ว 2026-07-22, ยังไม่ได้รัน** (ผู้ใช้ขอ "เขียนก่อน รันทีหลัง"):
   3 ablation แยกกัน, ตัวแทน 1 chunker + 1 embedder (bge-m3) ต่อ ablation,
   ใช้ baseline ที่ persist ไว้แล้ว (fixed_size-512/semantic × bge-m3 ใน
   `chunker_compare_full`) แทนการ build baseline ใหม่ ลดงานที่ต้องรันจริงเหลือ
   แค่ 4 combo ใหม่ (ไม่ใช่ full matrix ซ้ำ):
   - **Normalize**: `rag_lab.text_normalize.normalize_thai_text()` (เลขไทย→อารบิก
     + `pythainlp.util.normalize()`) ผ่าน loader ใหม่ `normalized`
     (`src/rag_lab/loaders/normalized.py`) จับคู่กับ `semantic × bge-m3`
     (ระบบที่ชนะอยู่แล้ว) — **normalize คำถามด้วยฟังก์ชันเดียวกันตอน eval ด้วย**
     (symmetric, ไม่งั้น BM25/hybrid จะดูแย่ลงลมๆ แล้งๆ จาก mismatch เลขไทย/อารบิก
     ระหว่าง corpus กับ query ไม่ใช่จาก normalization จริงๆ)
     config: `config/experiments/rq3_normalize_ablation.yaml`,
     eval: `tools/eval/rq3_normalize_significance_test.py`
   - **Segmentation**: chunker ใหม่ `fixed_size_wordaware`
     (`src/rag_lab/chunkers/fixed_size_wordaware.py`) ตัด chunk ตามขอบคำ
     (pythainlp newmm) แทนตัดตามจำนวนตัวอักษรดิบ (ซึ่งตัดกลางคำไทยได้ เพราะไทยไม่มี
     เว้นวรรคระหว่างคำ) จับคู่กับ `fixed_size(512) × bge-m3` เดิมเป็น baseline
     config: `config/experiments/rq3_segmentation_ablation.yaml`,
     eval: `tools/eval/rq3_segmentation_significance_test.py`
   - **Chunk-size sweep**: `fixed_size` เดิม แปร chunk_size = 256/1024 (512 ใช้ของเดิม)
     config: `config/experiments/rq3_chunksize_sweep.yaml`,
     eval: `tools/eval/rq3_chunksize_sweep_report.py`

   ทุก eval script ทดสอบทั้ง dense และ hybrid retrieval (ต้นทุนเพิ่มแค่เวลา query
   ไม่ต้อง index ใหม่ เพราะ hybrid สร้าง BM25 จาก chunks ที่มีอยู่แล้วตอน retrieve),
   paired bootstrap + Holm correction แบบเดียวกับที่ใช้ทั่วทั้ง study นี้ ยังไม่ได้รัน
   build จริง (ต้องใช้ GPU embed คอร์ปัสเต็ม 4 combo ใหม่) — รอคำสั่งรันจากผู้ใช้

   **Smoke-test ผ่านแล้ว 2026-07-22** (ก่อนรันจริง): build ทั้ง 3 ablation ด้วย
   `subset: dev, limit: 10` (10 เอกสาร แทนคอร์ปัสเต็ม) แล้วรัน eval script ทั้ง 3
   จริงจนจบ (ไม่ใช่แค่ `--help`) — ยืนยันเส้นทาง retrieve → join → bootstrap →
   Holm-correct → เขียนรายงาน ทำงานได้ไม่ error ทั้ง dense/hybrid รวมถึง
   chunk-size-drift table และ 3-way pairwise sweep ตัวเลขที่ได้เป็นขยะ (10 เอกสาร
   เทียบ 73 query จริง) — ทดสอบ code path เท่านั้น ไม่ใช่ผลจริง แล้วลบ artifact
   สำหรับ smoke ทั้งหมด (`data/index/rq3_*`, `data/results/rq3_*`, configs
   `_smoke_rq3_*.yaml`) ออกหมดแล้ว ไม่เหลือของปลอมปนกับของจริง ระหว่างทาง
   เพิ่ม guard สั้นๆ ใน eval script ทั้ง 3 (เช็ค index dir มีอยู่จริงก่อนเรียก
   `discover_indices`, error message บอกให้ build ก่อน แทน raw `FileNotFoundError`)
   และยืนยัน (ไม่ใช่แค่สมมติ) ว่า hybrid arm ของ ablation ใช้ `rrf_k=60`/`method=rrf`
   เดียวกับที่ใช้ตอน generate baseline `gold_hybrid_73det` จริง (เทียบ
   `StrategySpec(type="hybrid")` ไม่มี params ทั้งสองฝั่ง) ไม่มี confound เรื่อง
   fusion param mismatch โค้ดทั้งหมด commit แล้ว (`5a06c5b`) — **ยังไม่ push**

   **อัปเดต 2026-07-23 — รัน build จริงเต็มคอร์ปัสครบทั้ง 3 ablation แล้ว ✅ ปิดข้อ 7**
   (segmentation → chunksize sweep → normalize, background sequential, exit
   code 0 ทุกขั้นตอน, เร็วกว่าที่ประมาณไว้มาก): มีแค่ **chunk_size** ที่ส่งผล
   อย่างมีนัยสำคัญ (256/512 ชนะ 1024 บน recall@10 ทั้ง dense/hybrid, Holm-adj
   p≤0.0132) ส่วน normalization และ word-aware segmentation **ไม่มีผลนัยสำคัญ
   เลยสักตัว** (Holm-adj p≥0.414 และ =1.0 ตามลำดับ) รายละเอียดเต็มที่
   `docs/paper-results-summary.md` § "RQ3 ablation results (2026-07-23)"

   > **แก้ไข 2026-07-29 — rebuild treatment index ทั้ง 3 ตัวเพื่อล้าง confound
   > clean-vs-dirty**: index ฝั่ง treatment build ไว้ 23 ก.ค. (ก่อน kernel-A OCR
   > remediation และก่อน rebuild `chunker_compare_full`) ขณะที่ฝั่ง baseline
   > reuse combo จาก `chunker_compare_full` ที่ถูก rebuild บนข้อความสะอาดไปแล้ว
   > → เป็น confound เชิงระเบียบวิธีจริง ไม่ใช่แค่ตัวเลขเก่า และแก้ด้วยการ rerun
   > eval เฉยๆ ไม่ได้ ต้อง build index ใหม่บน GPU ทั้ง 3 ตัว **ทำเสร็จ 29 ก.ค.**
   > ผลหลัง refresh: **normalization/segmentation ข้อสรุปเดิมยืนทั้งคู่** (ยังไม่มี
   > นัยสำคัญ, Holm-adj p≥0.42 / ≥0.4524 — แต่ segmentation ไม่ได้ p=1.0 ทั้งกระดาน
   > อีกแล้ว dense MRR ขยับเป็น +0.0398 raw p=0.0754) ส่วน **chunk_size มีข้อสรุป
   > เปลี่ยนจริง**: สิ่งที่ replicate คือ **โทษของ 1024** (แพ้ทั้ง 256 และ 512
   > อย่างมีนัยสำคัญบน recall@10 ทั้ง dense/hybrid + hybrid nDCG@10) แต่
   > **"เล็กกว่าดีกว่าแบบ monotonic" ไม่ replicate** — dense recall@10 ตอนนี้
   > 512 (0.4146) **นำ** 256 (0.4103) แบบเสมอทางสถิติ (p=0.8802) กลับทิศจากเดิม
   > 256 ชนะ 512 เฉพาะ hybrid recall@10 เท่านั้น (Holm-adj p=0.0112)
   > **ให้อ้างอิงว่า "1024 แย่กว่า 512/256 อย่างมีนัยสำคัญ" ห้ามอ้างว่า "256 ดีที่สุด"
   > หรือ "recall ลดลงตามขนาดแบบ monotonic"** — ค่า default 512 ของโปรเจกต์ไม่ได้ถูก
   > พิสูจน์ว่าด้อยกว่า มีแค่ 1024 ที่พิสูจน์ว่าเป็นทางเลือกที่ผิด
8. ~~Cross-encoder reranker (optional, scope ชัด)~~ ✅ **ปิด 2026-07-23**: สร้าง
   `CrossEncoderReranker` (`BAAI/bge-reranker-v2-m3`) เป็น query-time stage เต็มรูปแบบ
   (registry pattern เดิม, ต่อจาก `pipeline.retrieve()`, ไม่แตะ runner/combos) + unit
   test 16 ตัว รันจริงบน Gold 73-det ผ่าน `tools/eval/reranker_significance_test.py`
   (paired bootstrap, Holm-corrected) — **ผลลบมีนัยสำคัญสำหรับ hybrid** (MRR
   0.848→0.760 Holm-adj p=0.006, nDCG@10 0.675→0.617 p=0.030) recall@10 แย่ลงแต่ไม่
   นัยสำคัญ, ไม่มีผลกับ dense เลย ยืนยันแล้วไม่ใช่บั๊ก (smoke test คะแนนสมเหตุสมผล)
   ส่ง `/research` ไปหา literature primary-source มารองรับ: paper "Drowning in
   Documents" (arXiv:2411.11767) ทดสอบ **โมเดล reranker ตัวเดียวกับที่เราใช้** พบ
   failure mode "phantom hits" ตรง fingerprint ผลเรา (MRR/nDCG พังแต่ recall รอด) +
   RRF ต้นฉบับ (Cormack et al. 2009) + HYRR (arXiv:2212.10528) อธิบายกลไกเพิ่มเติม
   เต็มที่ `docs/reranker-hybrid-interaction-research.md` และ
   `docs/paper-results-summary.md` § "Cross-encoder reranker results" — สรุป: **ไม่
   ควร rerank เส้นทาง hybrid ด้วยการต่อสายแบบปัจจุบัน**

   **แก้ไข 2026-07-29 (พบระหว่างกวาดเช็ก `data/results/*` ทั้งหมดหลัง
   OCR-remediation rebuild)**: สคริปต์นี้ retrieve สดทุกครั้ง ไม่ใช้ persisted
   results จึงค้างอยู่กับ index ก่อน rebuild — รันใหม่แล้ว ผลลบของ **MRR ยัง
   มีนัยสำคัญเหมือนเดิม** (0.7775→0.6775, Holm-adj p=0.0048) แต่ **nDCG@10
   ไม่ผ่านนัยสำคัญอีกต่อไป** (0.6193→0.5908, Holm-adj p=0.5676, จากเดิม 0.030)
   — เป็นการเปลี่ยนแปลงระดับข้อสรุปจริง ไม่ใช่แค่ตัวเลขขยับ กรอบที่ถูกต้องตอนนี้:
   reranker ทำร้าย **MRR เท่านั้น** ไม่ใช่ MRR+nDCG@10 เหมือนที่เคยสรุปไว้
9. **RQ4 / End-to-end RAG**: generation stage + RAGAS/LLM-judge — subsystem ใหม่ทั้งก้อน

**นอกขอบเขตฮาร์ดแวร์นี้:**
- LLM-based 7-8B (e5-mistral, NV-Embed, Qwen3-8B) — เกิน 12GB
- API เชิงพาณิชย์กลุ่ม D — ต้อง key/ค่าใช้จ่าย/ส่งข้อมูลออก; โน้ตจัดเป็นอ้างอิง upper-bound
