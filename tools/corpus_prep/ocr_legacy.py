import os
import ollama
import time
from pathlib import Path
from pdf2image import convert_from_path

# =================================================================
# CONFIGURATION SECTION (ตั้งค่าตรงนี้)
# =================================================================
TARGET_SUBFOLDER = "ครั้งที่ 5"  # แก้เป็น ครั้งที่ 2, ครั้งที่ 3 ตามต้องการ
BASE_DIR = r"C:\Users\Terry\Desktop\Code\RAG\academic_resolutions\2567"
MODEL_NAME = "scb10x/typhoon-ocr1.5-3b:latest"

# Path ของ Poppler ตามที่แจ้งม
POPPLER_PATH = r"C:\poppler\Library\bin" 

# Official Prompt ที่โมเดลตัวนี้ต้องการเพื่อเปิดโหมด OCR
OFFICIAL_PROMPT = (
    "Extract all text from the image.\n\n"
    "Instructions:\n"
    "- Only return the clean Markdown.\n"
    "- Do not include any explanation or extra text.\n"
    "- You must include all information on the page."
)

# =================================================================
# CORE FUNCTIONS
# =================================================================

def ocr_process(image_path):
    """ส่งภาพให้ Typhoon OCR 1.5-3b พร้อม Force Prompt"""
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    'role': 'user',
                    'content': OFFICIAL_PROMPT,
                    'images': [image_path]
                }
            ],
            options={
                'temperature': 0.0,
                'num_predict': 4096,
                'num_ctx': 8192
            }
        )
        content = response['message']['content'].strip()
        
        # ป้องกันกรณีโมเดลคายแต่คำสั่งกลับมา (ถ้าสั้นเกินไปและมีแต่ Instruction)
        if "Extract all text" in content and len(content) < 500:
            return "--- [OCR Failed: Model returned instructions instead of content] ---"
            
        return content
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return f"--- [Error processing image: {e}] ---"

def main():
    # 1. เตรียม Path
    target_path = Path(BASE_DIR) / TARGET_SUBFOLDER
    if not target_path.exists():
        print(f"❌ ไม่พบโฟลเดอร์: {target_path}")
        return

    print(f"🚀 เริ่มต้นโปรแกรม OCR (Model: {MODEL_NAME})")
    print(f"📂 โฟลเดอร์เป้าหมาย: {target_path}")

    # 2. ค้นหาไฟล์ (PDF, JPG, PNG)
    extensions = ('*.pdf', '*.jpg', '*.jpeg', '*.png')
    files = []
    for ext in extensions:
        files.extend(list(target_path.glob(ext)))

    if not files:
        print("⚠️ ไม่พบไฟล์เอกสารในโฟลเดอร์นี้")
        return

    # 3. เริ่มประมวลผลทีละไฟล์
    for file_path in files:
        output_file = file_path.with_suffix('.md')
        
        # ข้ามถ้ามีไฟล์ .md อยู่แล้ว
        if output_file.exists():
            print(f"⏩ ข้าม: {file_path.name} (มีไฟล์ .md แล้ว)")
            continue

        print(f"📄 กำลังจัดการ: {file_path.name}")
        final_markdown = f"# Document: {file_path.name}\n\n"

        # กรณีไฟล์ PDF
        if file_path.suffix.lower() == '.pdf':
            try:
                # แปลง PDF เป็นรูปภาพ
                images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
                for i, img in enumerate(images):
                    temp_img = f"temp_page_{i}.png"
                    img.save(temp_img, "PNG")
                    
                    print(f"   - กำลังอ่านหน้าที่ {i+1}/{len(images)}...")
                    page_text = ocr_process(temp_img)
                    final_markdown += f"## Page {i+1}\n\n{page_text}\n\n---\n\n"
                    
                    # ลบไฟล์ภาพชั่วคราวทันที
                    if os.path.exists(temp_img):
                        os.remove(temp_img)
            except Exception as e:
                print(f"   ❌ PDF Error: {e}")
                continue
        
        # กรณีไฟล์รูปภาพตรงๆ
        else:
            final_markdown += ocr_process(str(file_path))

        # 4. บันทึกผลลัพธ์
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_markdown)
        
        print(f"✅ บันทึกสำเร็จ: {output_file.name}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\n✨ ทำงานเสร็จสิ้นในเวลา {time.time() - start_time:.2f} วินาที")