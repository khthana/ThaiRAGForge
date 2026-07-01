# Spec: RAG Indexing Phase Experimentation Framework

## 1. Project Overview

**Goal**: พัฒนาโปรแกรม/framework สำหรับทดลอง (experiment) องค์ประกอบต่างๆ ใน **Indexing Phase** ของระบบ RAG โดยให้ผู้ใช้สามารถสลับ (swap) และเปรียบเทียบวิธีการทำงานในแต่ละขั้นตอนได้อย่างอิสระ ได้แก่:

1. **Document Loading** — โหลดเอกสารด้วยหลายกลยุทธ์
2. **Chunking** — แบ่งเอกสารเป็นชิ้นย่อยด้วยหลายกลยุทธ์
3. **Embedding** — แปลง chunks เป็น vector ด้วยหลาย embedding model

**Scope ปัจจุบัน**: ครอบคลุมเฉพาะ 3 ขั้นตอนข้างต้น **ยังไม่รวม** การเก็บลง Vector Database — ผลลัพธ์สุดท้ายของ pipeline คือ chunks พร้อม embeddings และ metadata ที่ถูก serialize เก็บไว้ในไฟล์ (เช่น JSON/Parquet) เพื่อนำไปวิเคราะห์หรือใช้ในขั้นตอนถัดไป

**Design Philosophy**: ทุกขั้นตอนต้องเป็น **modular** และ **pluggable** — สามารถเพิ่มวิธีการใหม่ (new strategy) โดยไม่ต้องแก้โค้ดเดิม (Open/Closed Principle) และสามารถรัน**หลาย combination** ของ (Loader × Chunker × Embedder) แบบ batch เพื่อเปรียบเทียบผลลัพธ์ได้

---

## 2. Architecture Overview

```
project/
├── config/
│   └── experiments/*.yaml       # นิยาม experiment combinations
├── src/
│   ├── loaders/                 # Document Loading strategies
│   │   ├── base.py              # abstract interface
│   │   ├── plain_loader.py
│   │   ├── metadata_loader.py
│   │   └── ner_loader.py
│   ├── chunkers/                # Chunking strategies
│   │   ├── base.py
│   │   ├── fixed_size_chunker.py
│   │   ├── recursive_chunker.py
│   │   ├── sentence_chunker.py
│   │   └── semantic_chunker.py
│   ├── embedders/                # Embedding strategies
│   │   ├── base.py
│   │   ├── local_embedder.py     # e.g. sentence-transformers
│   │   ├── openai_embedder.py
│   │   └── api_embedder.py       # generic API-based (Voyage, Cohere ฯลฯ)
│   ├── pipeline/
│   │   ├── runner.py             # orchestrator: รัน combination ทั้งหมด
│   │   └── registry.py           # factory/registry pattern สำหรับ plug-in
│   ├── schema/
│   │   └── models.py             # Pydantic models: Document, Chunk, EmbeddedChunk
│   └── utils/
│       ├── logging_utils.py
│       └── io_utils.py           # save/load ผลลัพธ์
├── data/
│   ├── raw/                      # เอกสารต้นฉบับ
│   └── output/                   # ผลลัพธ์แต่ละ experiment run
├── tests/
└── requirements.txt
```

**เหตุผลของโครงสร้างนี้**: ใช้ **Strategy Pattern + Registry/Factory Pattern** เพื่อให้ agent (หรือผู้ใช้) เพิ่มวิธีการใหม่ได้โดยแค่สร้างไฟล์ใหม่ที่ implement abstract interface แล้ว register เข้าระบบ ไม่ต้องแตะ pipeline หลัก

---

## 3. Data Schema

กำหนด schema กลางที่ทุกโมดูลต้อง conform ตาม (แนะนำใช้ Pydantic):

```python
class Document(BaseModel):
    doc_id: str
    source_path: str
    raw_text: str
    metadata: dict = {}          # เช่น title, author, created_date, ner_entities

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    metadata: dict = {}          # เช่น start_char, end_char, chunking_strategy

class EmbeddedChunk(BaseModel):
    chunk_id: str
    embedding: list[float]
    embedding_model: str
    embedding_dim: int
```

---

## 4. Module Specifications

### 4.1 Document Loading

Interface กลาง (`base.py`):

```python
class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> Document:
        ...
```

ต้อง implement อย่างน้อย 3 กลยุทธ์:

| กลยุทธ์ | รายละเอียด |
|---|---|
| `PlainLoader` | โหลดข้อความดิบตรงๆ ไม่มีการประมวลผลเพิ่ม รองรับ .txt, .pdf, .docx (ใช้ library เช่น `pypdf`, `python-docx`) |
| `MetadataLoader` | สกัด metadata เชิงโครงสร้าง เช่น title, author, created_date, page_count, section headers |
| `NERLoader` | รัน Named Entity Recognition (เช่น spaCy หรือ transformers NER model — ควรรองรับภาษาไทยด้วย เช่น `pythainlp`) แล้วเก็บ entities ที่พบไว้ใน `metadata['entities']` |

**Requirement เพิ่มเติม**:
- รองรับไฟล์หลายประเภท: `.txt`, `.pdf`, `.docx`, `.md`, `.html`
- รองรับภาษาไทยและอังกฤษ (ระบุ language detection หรือ config)
- แต่ละ Loader ต้อง log เวลาที่ใช้ (สำหรับ benchmark)

### 4.2 Chunking

Interface กลาง:

```python
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        ...
```

ต้อง implement อย่างน้อย 4 กลยุทธ์:

| กลยุทธ์ | รายละเอียด |
|---|---|
| `FixedSizeChunker` | แบ่งตามจำนวน character/token คงที่ พร้อม overlap ที่ config ได้ |
| `RecursiveChunker` | แบ่งตามลำดับ separator (paragraph → sentence → word) แบบ LangChain's RecursiveCharacterTextSplitter |
| `SentenceChunker` | แบ่งตามประโยค แล้วรวมกลุ่มประโยคจนถึงขนาดที่กำหนด (รองรับ sentence tokenizer ภาษาไทย) |
| `SemanticChunker` | ใช้ embedding similarity ระหว่างประโยค/ย่อหน้าเพื่อหาจุดตัดที่ semantic เปลี่ยน (semantic breakpoint) |

**Parameter ที่ควร config ได้ต่อกลยุทธ์**: `chunk_size`, `chunk_overlap`, `min_chunk_size`, `separator_priority`

### 4.3 Embedding

Interface กลาง:

```python
class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        ...
```

ต้อง implement อย่างน้อย 3 กลยุทธ์:

| กลยุทธ์ | รายละเอียด |
|---|---|
| `LocalEmbedder` | ใช้ open-source model ผ่าน `sentence-transformers` (เช่น `intfloat/multilingual-e5-large`, BGE-M3 — เลือกที่รองรับภาษาไทย) รันบน local GPU/CPU |
| `APIEmbedder` | เรียก embedding API ภายนอก (OpenAI, Voyage, Cohere) ผ่าน config-able provider |
| `HybridEmbedder` (optional) | รวมผลจากหลาย embedding model หรือทำ dimensionality reduction |

**Requirement เพิ่มเติม**:
- ต้อง batch request เพื่อประสิทธิภาพ
- ต้อง cache ผลลัพธ์ (ป้องกันเรียก API ซ้ำ / คำนวณซ้ำ)
- บันทึก cost/latency ต่อ request (สำหรับ API-based)

---

## 5. Experiment Configuration

ใช้ YAML config เพื่อนิยาม combination ที่จะรัน โดยไม่ต้องแก้โค้ด:

```yaml
experiment_name: "thai_academic_docs_v1"
input_dir: "data/raw/"
output_dir: "data/output/thai_academic_docs_v1/"

loaders:
  - type: plain
  - type: metadata
  - type: ner
    params:
      ner_model: "pythainlp"

chunkers:
  - type: fixed_size
    params: { chunk_size: 512, chunk_overlap: 50 }
  - type: recursive
    params: { chunk_size: 512, chunk_overlap: 50 }
  - type: semantic
    params: { breakpoint_threshold: 0.75 }

embedders:
  - type: local
    params: { model_name: "intfloat/multilingual-e5-large" }
  - type: api
    params: { provider: "openai", model_name: "text-embedding-3-large" }

run_mode: "cartesian"   # รันทุก combination ของ loader × chunker × embedder
```

**Pipeline Runner ต้องรองรับ**:
- `run_mode: cartesian` — รันทุก combination (loader × chunker × embedder)
- `run_mode: paired` — รันเฉพาะคู่ที่ระบุไว้ตรงๆ (สำหรับกรณีไม่ต้องการทุก combination)

---

## 6. Output & Evaluation Artifacts

แต่ละ experiment run ต้องบันทึก:

1. **ผลลัพธ์หลัก**: ไฟล์ JSON หรือ Parquet เก็บ `EmbeddedChunk` list พร้อม metadata ครบ (loader/chunker/embedder ที่ใช้)
2. **Run manifest**: ไฟล์ config + timestamp + git commit hash (ถ้ามี) เพื่อ reproducibility
3. **Metrics summary** (ต่อ combination):
   - จำนวน chunks ที่ได้
   - ค่าเฉลี่ย/การกระจายตัวของขนาด chunk (character/token)
   - เวลาที่ใช้ในแต่ละ phase (load / chunk / embed)
   - (ถ้าเป็น API) ค่าใช้จ่ายโดยประมาณ
4. **Log file** แยกต่อ run สำหรับ debug

โครงสร้างผลลัพธ์แนะนำ:
```
data/output/{experiment_name}/{loader}_{chunker}_{embedder}/
├── chunks.parquet
├── manifest.yaml
├── metrics.json
└── run.log
```

---

## 7. Non-Functional Requirements

- **Extensibility**: เพิ่ม strategy ใหม่ต้องทำได้โดยสร้างไฟล์ใหม่ + register ใน registry เท่านั้น (ห้ามแก้ pipeline runner)
- **Reproducibility**: fix random seed, บันทึก config ทุกครั้งที่รัน
- **Error handling**: ถ้า combination ใด fail (เช่น API timeout) ต้อง log error และรัน combination อื่นต่อ ไม่ crash ทั้ง batch
- **Progress visibility**: แสดง progress bar (เช่น `tqdm`) ระหว่างรัน batch ที่มีหลาย combination
- **Language support**: ทุกโมดูลต้องรองรับข้อความภาษาไทยและอังกฤษปนกันได้อย่างถูกต้อง (โดยเฉพาะ sentence tokenization และ NER)
- **Testability**: แต่ละ Loader/Chunker/Embedder ต้องมี unit test พร้อม sample document ขนาดเล็ก

---

## 8. Suggested Tech Stack

| ส่วนประกอบ | เทคโนโลยีที่แนะนำ |
|---|---|
| ภาษาโปรแกรม | Python 3.11+ |
| Data validation | Pydantic v2 |
| Document parsing | `pypdf`, `python-docx`, `unstructured` (optional) |
| Thai NLP | `pythainlp` (tokenization, NER) |
| Chunking (recursive) | `langchain-text-splitters` (ใช้เฉพาะ splitter ไม่จำเป็นต้องใช้ LangChain ทั้ง framework) |
| Local embedding | `sentence-transformers` |
| Config management | `pydantic-settings` หรือ `omegaconf` + YAML |
| Storage format | Parquet (`pyarrow`) สำหรับ embeddings, JSON สำหรับ manifest/metrics |
| CLI | `typer` หรือ `click` |
| Progress/Logging | `tqdm`, `loguru` |

---

## 9. Deliverables (สำหรับ AI Agent)

1. โครงสร้างโปรเจกต์ตาม Section 2
2. Implementation ของทุก module ตาม Section 4 (ครบทุกกลยุทธ์ที่ระบุเป็นขั้นต่ำ)
3. Pipeline runner ที่อ่าน YAML config และรันได้ตาม Section 5
4. CLI command เช่น `python run_experiment.py --config config/experiments/thai_academic_docs_v1.yaml`
5. ตัวอย่าง config file อย่างน้อย 1 ไฟล์พร้อม sample documents สำหรับทดสอบ
6. Unit tests ครอบคลุมทุก strategy (อย่างน้อย happy-path)
7. `README.md` อธิบายวิธีเพิ่ม strategy ใหม่ (สำหรับ extensibility ในอนาคต)

---

## 10. Out of Scope (ยังไม่ทำในเฟสนี้)

- การเก็บลง Vector Database (Milvus, Qdrant, Chroma ฯลฯ)
- Retrieval logic / similarity search
- Re-ranking / query transformation
- Evaluation ด้วย RAGAS หรือ metric อื่นที่ต้องมี retrieval ก่อน
- UI/Dashboard (เฟสนี้เป็น CLI/script-based เท่านั้น)