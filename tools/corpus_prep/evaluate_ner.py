"""Evaluate PyThaiNLP NER on KMITL Academic Resolutions.

This script runs the codebase's NERLoader over 2-3 months of meeting resolutions,
aggregates extraction statistics, generates a human-readable Markdown evaluation report,
and saves the detailed results in JSON format.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rag_lab.loaders.ner_loader import NERLoader
from tqdm import tqdm


def get_snippet(text: str, entity: str, window: int = 50) -> str:
    """Finds the entity in text and returns a snippet with surrounding context."""
    idx = text.find(entity)
    if idx == -1:
        return ""
    start = max(0, idx - window)
    end = min(len(text), idx + len(entity) + window)
    snippet = text[start:end]
    
    # Add ellipsis if truncated
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    
    # Highlight entity text using markdown bold
    highlighted = snippet.replace(entity, f"**{entity}**")
    return f"{prefix}{highlighted}{suffix}".replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PyThaiNLP NER on KMITL Academic Resolutions."
    )
    parser.add_argument(
        "--year",
        type=str,
        default="2569",
        help="Year of the meetings (default: 2569)"
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=["ครั้งที่ 1", "ครั้งที่ 2", "ครั้งที่ 3"],
        help="Sessions of the meetings (default: ครั้งที่ 1, 2, 3)"
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="academic_resolutions",
        help="Path to the academic resolutions corpus"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="academic_resolutions/entity_tags/ner_eval",
        help="Directory to save evaluation reports"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process total (optional)"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="thainer",
        choices=["thainer", "thainer-v2", "wangchanberta-thainer", "phayathaibert-thainer"],
        help=(
            "NER engine: 'thainer' (PyThaiNLP CRF, default), 'thainer-v2' "
            "(PyThaiNLP's own wangchanberta checkpoint), 'wangchanberta-thainer' "
            "(Porameht/wangchanberta-thainer-corpus-v2-2, GPU), or "
            "'phayathaibert-thainer' (Pavarissy/phayathaibert-thainer, GPU)"
        ),
    )

    args = parser.parse_args()
    
    corpus_base = Path(args.corpus_dir) / args.year
    if not corpus_base.exists():
        print(f"Error: Corpus path {corpus_base} does not exist.")
        sys.exit(1)
        
    # Discover files
    files: list[Path] = []
    for session in args.sessions:
        session_dir = corpus_base / session
        if session_dir.exists():
            # Only match standard .md files
            session_files = sorted(session_dir.glob("*.md"))
            files.extend(session_files)
        else:
            print(f"Warning: Session folder {session_dir} not found.")

    if not files:
        print("No markdown resolutions found.")
        sys.exit(1)
        
    if args.limit:
        files = files[:args.limit]
        print(f"Limited processing to first {args.limit} files.")

    print(f"Found {len(files)} markdown resolution documents to process.")

    device = "cpu"
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    print(f"Using NER engine: {args.engine} (device: {device})")

    # Initialize NER loader
    loader = NERLoader(engine=args.engine)
    
    results = []
    all_entities = []
    entity_counts = Counter()
    entity_by_type = defaultdict(Counter)
    
    # Track performance
    start_time = time.perf_counter()
    total_chars = 0
    
    print("Running Named Entity Recognition (NER) tagger...")
    for file_path in tqdm(files, desc="Processing files"):
        try:
            doc_start = time.perf_counter()
            resolution = loader.load(str(file_path))
            doc_duration = time.perf_counter() - doc_start
            
            entities = resolution.metadata.get("entities", [])
            raw_text = resolution.raw_text
            total_chars += len(raw_text)
            
            # Record per-document result
            doc_info = {
                "file_name": file_path.name,
                "title": resolution.title,
                "session": resolution.session,
                "char_count": len(raw_text),
                "processing_time_sec": doc_duration,
                "entities": entities
            }
            results.append(doc_info)
            
            for ent in entities:
                text = ent["text"]
                tag = ent["tag"]
                all_entities.append(ent)
                entity_counts[tag] += 1
                entity_by_type[tag][text] += 1
                
        except Exception as e:
            print(f"\nError processing {file_path.name}: {e}")
            
    total_duration = time.perf_counter() - start_time
    print(f"\nCompleted processing in {total_duration:.2f} seconds.")
    
    # Build reports
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    json_report_path = output_path / f"ner_report_{args.year}_{args.engine}.json"
    report_data = {
        "summary": {
            "year": args.year,
            "engine": args.engine,
            "sessions": args.sessions,
            "total_documents": len(files),
            "total_characters": total_chars,
            "total_entities_extracted": len(all_entities),
            "total_processing_time_sec": total_duration,
            "avg_chars_per_doc": total_chars / len(files) if files else 0,
            "avg_time_per_doc_sec": total_duration / len(files) if files else 0,
            "avg_chars_per_sec": total_chars / total_duration if total_duration > 0 else 0
        },
        "entity_counts_by_type": dict(entity_counts),
        "documents": results
    }
    
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON report to {json_report_path}")
    
    # Generate human-readable Markdown report
    markdown_report_path = output_path / f"ner_report_{args.year}_{args.engine}.md"

    md_lines = []
    md_lines.append(f"# รายงานการทดสอบประสิทธิภาพของ Thai NER")
    md_lines.append(f"**วันที่รันการทดสอบ:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append(f"**Engine:** {args.engine} (device: {device})")
    md_lines.append(f"**ขอบเขตข้อมูล:** ปี {args.year} ({', '.join(args.sessions)})")
    md_lines.append("")
    
    md_lines.append("## 1. สรุปภาพรวมเชิงปริมาณ (Quantitative Summary)")
    md_lines.append("| หัวข้อ | ค่าสถิติ |")
    md_lines.append("| :--- | :--- |")
    md_lines.append(f"| จำนวนเอกสารทั้งหมดที่ทดสอบ | {len(files)} ไฟล์ |")
    md_lines.append(f"| จำนวนตัวอักษรทั้งหมดที่ประมวลผล | {total_chars:,} ตัวอักษร |")
    md_lines.append(f"| จำนวน Named Entity ทั้งหมดที่สกัดได้ | {len(all_entities):,} รายการ |")
    md_lines.append(f"| เวลาที่ใช้ประมวลผลทั้งหมด | {total_duration:.2f} วินาที |")
    md_lines.append(f"| ความเร็วเฉลี่ยต่อไฟล์ | {total_duration / len(files):.3f} วินาที/ไฟล์ |")
    md_lines.append(f"| ความเร็วประมวลผลเฉลี่ย | {total_chars / total_duration:,.2f} ตัวอักษร/วินาที |")
    md_lines.append("")
    
    md_lines.append("### จำนวน Entity แยกตามประเภท (Entity Type Counts)")
    md_lines.append("| ประเภท (Tag) | คำอธิบาย (Description) | จำนวนที่พบ (Counts) |")
    md_lines.append("| :--- | :--- | :--- |")
    
    tag_descriptions = {
        "PERSON": "ชื่อบุคคล (เช่น อธิการบดี, อาจารย์)",
        "ORGANIZATION": "ชื่อหน่วยงาน/องค์กร (เช่น สจล., คณะวิทยาศาสตร์)",
        "LOCATION": "สถานที่ (เช่น กรุงเทพฯ, วิทยาเขตชุมพร)",
        "DATE": "วันที่ (เช่น 2568, ภาคเรียนที่ 2)",
        "TIME": "เวลา",
        "MONEY": "จำนวนเงิน/งบประมาณ",
        "PERCENT": "ร้อยละ / เปอร์เซ็นต์",
        "ZIP": "รหัสไปรษณีย์",
        "EMAIL": "อีเมล",
        "URL": "ลิงก์เว็บไซต์",
        "PHONE": "เบอร์โทรศัพท์",
        "LAW": "กฎหมาย/ข้อบังคับ/ประกาศ",
    }
    
    for tag, count in entity_counts.most_common():
        desc = tag_descriptions.get(tag, "ประเภทอื่นๆ")
        md_lines.append(f"| **{tag}** | {desc} | {count:,} |")
    md_lines.append("")
    
    md_lines.append("## 2. ตัวอย่างข้อมูลและวิเคราะห์ความแม่นยำรายประเภท (Top Entities with Context)")
    md_lines.append("ส่วนนี้แสดงคำสำคัญที่พบบ่อยที่สุด พร้อมตัวอย่างบริบทแวดล้อม (Snippet Context) จากเอกสารเพื่อช่วยประเมินความแม่นยำ:")
    md_lines.append("")
    
    # Let's show top 10 entities for major tags: PERSON, ORGANIZATION, LOCATION, LAW
    major_tags = ["PERSON", "ORGANIZATION", "LOCATION", "LAW"]
    for tag in major_tags:
        if tag not in entity_by_type or not entity_by_type[tag]:
            continue
            
        md_lines.append(f"### ประเภท {tag} ({tag_descriptions.get(tag, '')})")
        md_lines.append("| อันดับ | คำสำคัญ (Entity) | ความถี่ | ตัวอย่างบริบทแวดล้อมในเอกสาร (Context Snippet) |")
        md_lines.append("| :---: | :--- | :---: | :--- |")
        
        for rank, (entity, freq) in enumerate(entity_by_type[tag].most_common(12), 1):
            # Find a context snippet from the files that contain this entity
            snippet = ""
            for doc in results:
                # Search in original doc raw text if we find it
                for file_p in files:
                    if file_p.name == doc["file_name"]:
                        try:
                            # Read file content to search
                            text_content = file_p.read_text(encoding="utf-8-sig")
                            snippet = get_snippet(text_content, entity)
                            if snippet:
                                break
                        except Exception:
                            pass
                if snippet:
                    break
            
            md_lines.append(f"| {rank} | `{entity}` | {freq} | {snippet} |")
        md_lines.append("")
        
    md_lines.append("## 3. รายละเอียดผลการวิเคราะห์แยกรายไฟล์ (Detailed Per-File Analysis)")
    md_lines.append("กดปุ่มเพื่อขยายดูรายการเอกสารทั้งหมดและ Entity ที่พบหลักๆ ของแต่ละหัวข้อ:")
    md_lines.append("")
    md_lines.append("<details>")
    md_lines.append("<summary><b>คลิกเพื่อแสดงตารางรายไฟล์ทั้งหมด (คลิกเพื่อขยาย)</b></summary>")
    md_lines.append("")
    md_lines.append("| ชื่อไฟล์ / หัวข้อการประชุม | จำนวนตัวอักษร | วินาที | สรุป Entities เด่นที่พบ |")
    md_lines.append("| :--- | :---: | :---: | :--- |")
    
    for doc in results:
        # Group entities for this doc
        doc_ent_counts = Counter(e["text"] for e in doc["entities"])
        ent_by_t = defaultdict(list)
        for e in doc["entities"]:
            ent_by_t[e["tag"]].append(e["text"])
            
        summary_parts = []
        for tag in ["PERSON", "ORGANIZATION", "LOCATION"]:
            if ent_by_t[tag]:
                unique_ents = list(dict.fromkeys(ent_by_t[tag]))[:3]
                summary_parts.append(f"*{tag}*: {', '.join(unique_ents)}")
                
        summary_str = " | ".join(summary_parts) if summary_parts else "ไม่พบ Entity สำคัญ"
        md_lines.append(
            f"| **{doc['title']}**<br><small>{doc['file_name']}</small> | "
            f"{doc['char_count']:,} | {doc['processing_time_sec']:.3f} | {summary_str} |"
        )
        
    md_lines.append("")
    md_lines.append("</details>")
    md_lines.append("")
    
    md_lines.append("## 4. ข้อสังเกตและข้อแนะนำในการนำไปใช้ (Observations & Recommendations)")
    md_lines.append("- **การสะกดคำผิดจาก OCR:** เนื่องจากระบบใช้ข้อมูลที่มาจากการทำ OCR ความแม่นยำของสะกดคำจึงส่งผลต่อ NER ค่อนข้างมาก เช่น มีเครื่องหมายวรรคตอนเกิน หรือเว้นวรรคไม่ถูกต้อง ซึ่งส่งผลต่อการจับคู่ IOB แท็ก")
    md_lines.append("- **การแบ่งกลุ่ม Entity:** โมเดล CRF (`thainer`) ค่อนข้างเร็ว แต่อาจเกิดปัญหา Boundary Detection ผิดพลาดในภาษาไทย เช่น การสกัดเอาคำนำหน้าชื่อ (รองศาสตราจารย์ ดร.) ปนรวมเข้าไปเป็นส่วนหนึ่งของชื่อ หรือสกัดสถาบันปลายทางไม่ครบถ้วน")
    md_lines.append("- **การกรองคำรบกวน:** สามารถนำผลลัพธ์จากไฟล์รายงาน JSON นี้ ไปพิจารณาสร้าง Blacklist/Whitelist หรือทำ Entity Linkage ในเฟสถัดไปเพื่อนำไปใช้กรองใน Metadata Filter ของ RAG Pipeline ต่อไป")
    
    # Write MD file
    with open(markdown_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    print(f"Saved Markdown report to {markdown_report_path}")
    print("\nEvaluation successfully completed.")


if __name__ == "__main__":
    main()
