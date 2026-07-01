import os

TARGET_SUBFOLDER = "ครั้งที่ 2"
BASE_DIR = r"C:\Users\Terry\Desktop\Code\RAG\academic_resolutions\2566"
ERROR_PATTERN = "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

DRY_RUN = False  # เปลี่ยนเป็น False เมื่อต้องการลบจริง


def scan_and_delete(base_dir, subfolder, pattern, dry_run=True):
    target_dir = os.path.join(base_dir, subfolder)

    if not os.path.isdir(target_dir):
        print(f"[ERROR] ไม่พบโฟลเดอร์: {target_dir}")
        return

    mode_label = "DRY-RUN (ยังไม่ลบจริง)" if dry_run else "DELETE MODE (ลบจริง)"
    print(f"{'='*55}")
    print(f"  โหมด : {mode_label}")
    print(f"  โฟลเดอร์ : {target_dir}")
    print(f"  Pattern : {pattern}")
    print(f"{'='*55}\n")

    found_files = []

    for filename in os.listdir(target_dir):
        if not filename.endswith(".md"):
            continue

        filepath = os.path.join(target_dir, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, "r", encoding="utf-8-sig") as f:
                content = f.read()

        if pattern in content:
            found_files.append(filepath)
            print(f"  [พบ] {filename}")

            if not dry_run:
                os.remove(filepath)
                print(f"       -> ลบแล้ว")

    print(f"\n{'='*55}")
    print(f"  พบไฟล์ที่มี pattern ทั้งหมด : {len(found_files)} ไฟล์")

    if dry_run and found_files:
        print(f"\n  หากต้องการลบจริง ให้เปลี่ยน DRY_RUN = False แล้วรันใหม่")

    print(f"{'='*55}")


if __name__ == "__main__":
    scan_and_delete(BASE_DIR, TARGET_SUBFOLDER, ERROR_PATTERN, dry_run=DRY_RUN)