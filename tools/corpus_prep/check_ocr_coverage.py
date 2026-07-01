from pathlib import Path

TARGET_SUBFOLDER = "ครั้งที่ 5"
BASE_DIR = r"C:\Users\Terry\Desktop\Code\RAG\academic_resolutions\2567"

target_path = Path(BASE_DIR) / TARGET_SUBFOLDER

files = []
for ext in ('*.pdf', '*.jpg', '*.jpeg', '*.png'):
    found = list(target_path.glob(ext))
    print(f"{ext}: {len(found)} files")
    files.extend(found)

print(f"\nTotal: {len(files)}")
for f in files:
    md_exists = f.with_suffix('.md').exists()
    print(f"  {f.name} | .md exists: {md_exists}")