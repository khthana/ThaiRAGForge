# เส้นทางโปรเจกต์ทั้งหมด — ตัวชี้ทาง (pointer)

> **ไฟล์นี้ไม่มีเนื้อหา** — เก็บไว้เป็นตัวชี้ทางเท่านั้น
>
> เนื้อหาฉบับเต็มอยู่ที่ **[`docs/project-journey.html`](project-journey.html)**
> (แหล่งความจริงเดียว, tracked ใน git) และฉบับ PDF ที่ render จากไฟล์นั้น
>
> เหตุผลที่ไม่เก็บเนื้อหาซ้ำไว้ที่นี่: โปรเจกต์นี้เจอบั๊ก "เอกสาร/ผลลัพธ์ 2 ชุด
> แตกกันเงียบๆ" มาแล้ว **5 ครั้ง** (ดูบทเรียนข้อ 1 ในเอกสารฉบับเต็ม)
> จึงยึดหลักเดียวกับที่ `README.md` ใช้กับคำสั่ง dev — **เก็บไว้ที่เดียว
> ไม่เขียนซ้ำสองที่**

## เอกสารนี้ครอบคลุมอะไร

สรุปเชิงเล่าเรื่องของงานทั้งโปรเจกต์ (1–29 ก.ค. 2569, 100 commit) — 13 บท 36 หน้า:
ทำอะไรไปตามลำดับเวลา · แต่ละขั้นได้ผลอย่างไร · **ข้อสรุปไหนถูกถอนไปแล้วและเพราะอะไร** ·
บทเรียนเชิงกระบวนการ · ภาคผนวกอธิบาย metric และวิธีการทางสถิติที่ใช้

เหมาะกับการอ่านเพื่อ **เข้าใจภาพรวม** — ถ้าต้องการตัวเลขพร้อมอ้างอิงในเปเปอร์
ให้ไปที่ `docs/paper-results-summary.md` แทน

## วิธี render PDF ใหม่

ไฟล์ `.pdf` เป็นของ generated (gitignored) — สร้างใหม่จาก `.html` ได้ด้วย:

```bash
"/c/Program Files/Google/Chrome/Application/chrome.exe" \
  --headless --disable-gpu --no-pdf-header-footer \
  --virtual-time-budget=20000 \
  --print-to-pdf="C:\Users\Terry\Desktop\Code\RAG\docs\project-journey.pdf" \
  "file:///C:/Users/Terry/Desktop/Code/RAG/docs/project-journey.html"
```

`--virtual-time-budget` จำเป็นเพื่อรอให้ฟอนต์ Sarabun จาก Google Fonts โหลดเสร็จ
ก่อน render (ถ้าไม่ใส่ ข้อความไทยจะ fallback ไปฟอนต์ระบบ)
