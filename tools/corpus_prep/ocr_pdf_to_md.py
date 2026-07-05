import sys
import os
import time
import ollama
from pathlib import Path
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image, ImageEnhance

sys.stdout.reconfigure(encoding='utf-8')

# =================================================================
# CONFIGURATION
# =================================================================
# Default target when run without arguments. To OCR other folders, pass them on
# the command line instead (one or more; processed sequentially), e.g.:
#   python tools/corpus_prep/ocr_pdf_to_md.py "academic_resolutions/missing" "academic_resolutions/md-not-found"
TARGET_SUBFOLDER = "ครั้งที่ 5"
BASE_DIR = r"C:\Users\Terry\Desktop\Code\RAG\academic_resolutions\2567"
MODEL_NAME = "scb10x/typhoon-ocr1.5-3b:latest"
POPPLER_PATH = r"C:\poppler\Library\bin"

PDF_DPI = 300
MAX_RETRIES = 3
RETRY_DELAY = 3
RESET_EVERY_N = 10

OFFICIAL_PROMPT = (
    "Extract all text from the image.\n\n"
    "Instructions:\n"
    "- Only return the clean Markdown.\n"
    "- Do not include any explanation or extra text.\n"
    "- You must include all information on the page."
)

# =================================================================
# MODEL MANAGEMENT
# =================================================================

def reset_model():
    print("   [RESET] Resetting model...")
    try:
        ollama.generate(model=MODEL_NAME, prompt="", keep_alive=0)
        time.sleep(2)
    except Exception as e:
        print(f"   [WARN] Reset warning: {e}")


def is_bad_output(text: str) -> bool:
    if not text or len(text) < 5:
        return True
    most_common_ratio = max(text.count(c) for c in set(text)) / len(text)
    if most_common_ratio > 0.5:
        return True
    if "Extract all text" in text and len(text) < 500:
        return True
    return False

# =================================================================
# IMAGE PREPROCESSING
# =================================================================

def preprocess_image(image_path: str) -> str:
    p = Path(image_path)
    preprocessed_path = str(p.with_stem(p.stem + "_processed").with_suffix(".png"))
    
    img = Image.open(image_path).convert("RGB")
    
    # ทำพื้นขาวขึ้น — ลายน้ำสีอ่อนจะหายไป
    img = ImageEnhance.Brightness(img).enhance(1.3)
    
    # เพิ่ม contrast หลังจาก brighten — ตัวอักษรเข้มจะยังอยู่ ลายน้ำจางหาย
    img = ImageEnhance.Contrast(img).enhance(2.0)
    
    # sharpness เบาลง เพราะ contrast สูงแล้ว
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    
    img.save(preprocessed_path, "PNG")
    return preprocessed_path

# =================================================================
# OCR CORE
# =================================================================

def ocr_image(image_path: str) -> str:
    processed_path = preprocess_image(image_path)
    try:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=[{
                        'role': 'user',
                        'content': OFFICIAL_PROMPT,
                        'images': [processed_path]
                    }],
                    options={
                        'temperature': 0.0,
                        'num_predict': 4096,
                        'num_ctx': 8192
                    }
                )
                content = response['message']['content'].strip()

                if is_bad_output(content):
                    print(f"   [WARN] Bad output (attempt {attempt}/{MAX_RETRIES})")
                    reset_model()
                    time.sleep(RETRY_DELAY)
                    continue

                return content

            except Exception as e:
                print(f"   [ERROR] attempt {attempt}/{MAX_RETRIES}: {e}")
                reset_model()
                time.sleep(RETRY_DELAY)

    finally:
        if os.path.exists(processed_path):
            os.remove(processed_path)

    return "--- [OCR Failed: Bad output after all retries] ---"


def process_pdf(file_path: Path) -> str:
    # แปลงทีละหน้า — ไฟล์ใหญ่ (100+ หน้า) ถ้าแปลงทั้งเล่มทีเดียวจะกินแรมหลาย GB
    markdown = ""
    try:
        total = pdfinfo_from_path(file_path, poppler_path=POPPLER_PATH)["Pages"]
        for i in range(1, total + 1):
            images = convert_from_path(
                file_path,
                poppler_path=POPPLER_PATH,
                dpi=PDF_DPI,
                first_page=i,
                last_page=i,
            )
            temp_img = f"temp_page_{i}.png"
            images[0].save(temp_img, "PNG")

            print(f"   - Page {i}/{total}...")
            page_text = ocr_image(temp_img)
            markdown += f"## Page {i}\n\n{page_text}\n\n---\n\n"

            if os.path.exists(temp_img):
                os.remove(temp_img)

    except Exception as e:
        print(f"   [ERROR] PDF: {e}")

    return markdown


def process_image(file_path: Path) -> str:
    return ocr_image(str(file_path))

# =================================================================
# MAIN
# =================================================================

def run_folder(target_path: Path):
    if not target_path.exists():
        print(f"[ERROR] Folder not found: {target_path}")
        return

    print(f"[START] OCR - Model: {MODEL_NAME}")
    print(f"[INFO] Target: {target_path}")

    files = []
    for ext in ('*.pdf', '*.jpg', '*.jpeg', '*.png'):
        files.extend(target_path.glob(ext))

    if not files:
        print("[WARN] No files found")
        return

    print(f"[INFO] Found {len(files)} files\n")

    for idx, file_path in enumerate(files):
        output_file = file_path.with_suffix('.md')

        if output_file.exists():
            print(f"[SKIP] {file_path.name}")
            continue

        if idx > 0 and idx % RESET_EVERY_N == 0:
            print(f"\n[RESET] Periodic reset...")
            reset_model()
            time.sleep(5)

        print(f"\n[{idx+1}/{len(files)}] {file_path.name}")
        reset_model()

        if file_path.suffix.lower() == '.pdf':
            content = process_pdf(file_path)
        else:
            content = process_image(file_path)

        markdown = f"# Document: {file_path.name}\n\n{content}"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"[DONE] {output_file.name}")

    print("\n[FINISH]")


def main():
    targets = [Path(a) for a in sys.argv[1:]] or [Path(BASE_DIR) / TARGET_SUBFOLDER]
    for target_path in targets:
        run_folder(target_path)


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"[TIME] {elapsed/60:.1f} min")