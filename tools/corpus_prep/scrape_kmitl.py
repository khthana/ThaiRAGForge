import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class KmitlFinalScraper:
    def __init__(self, target_url, base_dir="academic_resolutions"):
        self.target_url = target_url
        self.base_dir = base_dir
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def _clean_name(self, text):
        return re.sub(r'[\\/*?:"<>|]', '_', text).strip()

    def scrape_by_year(self, target_year):
        print(f"=== [Index Mode] คัดกรองข้อมูลปี {target_year} แบบลำดับขั้นตอน ===")
        try:
            res = requests.get(self.target_url, headers=self.headers, timeout=30)
            res.encoding = 'utf-8'
            soup = BeautifulSoup(res.text, 'html.parser')

            # 1. ดึงเนื้อหาหลัก และลิสต์ "ทุก Element" ออกมาตามลำดับที่ปรากฏใน HTML
            main_content = soup.find('article') or soup.find('main') or soup.find('div', class_='entry-content') or soup.body
            all_elements = main_content.find_all(True) # ดึงทุกลำดับ

            # 2. หา Index ของ "จุดเริ่ม" และ "จุดจบ"
            start_index = -1
            end_index = len(all_elements)

            for i, el in enumerate(all_elements):
                text = el.get_text(strip=True)
                
                # หาจุดเริ่มต้น: ต้องมีคำว่า "2567" หรือ "๒๕๖๗" และอยู่ใน Tag หัวข้อ
                if start_index == -1 and el.name in ['h1', 'h2', 'h3', 'h4', 'strong']:
                    if str(target_year) in text or self._to_thai_num(target_year) in text:
                        start_index = i
                        print(f"[*] พบจุดเริ่มที่ลำดับ {i}: '{text}'")
                
                # หาจุดสิ้นสุด: ถ้าเจอปีอื่น (เช่น 2566) หลังจากที่เจอจุดเริ่มแล้ว
                elif start_index != -1 and el.name in ['h1', 'h2', 'h3', 'h4', 'strong']:
                    # เช็คว่ามีปี พ.ศ. อื่นที่ไม่ใช่ปีเป้าหมายไหม
                    found_years = re.findall(r'25\d{2}', text)
                    if found_years and str(target_year) not in found_years:
                        end_index = i
                        print(f"[*] พบจุดสิ้นสุดที่ลำดับ {i} (หัวข้อปีอื่น): '{text}'")
                        break

            if start_index == -1:
                print(f"[!] ไม่พบหัวข้อปี {target_year}")
                return

            # 3. ดึงเฉพาะ vc_toggle ที่อยู่ในช่วง Index ที่เราตัดไว้
            target_elements = all_elements[start_index:end_index]
            found_toggles = []
            for el in target_elements:
                if el.name == 'div' and 'vc_toggle' in el.get('class', []):
                    # ป้องกันการเก็บกล่องซ้ำ (เพราะ find_all True อาจดึงมาทั้ง parent/child)
                    if el not in found_toggles:
                        found_toggles.append(el)

            print(f"[*] พบกล่องมติการประชุมทั้งหมด {len(found_toggles)} กล่อง")

            # 4. ประมวลผลแต่ละกล่อง
            for toggle in found_toggles:
                title_div = toggle.find('div', class_='vc_toggle_title')
                content_div = toggle.find('div', class_='vc_toggle_content')
                
                if title_div and content_div:
                    title_text = title_div.get_text(strip=True)
                    if "ครั้งที่" not in title_text: continue
                    
                    print(f"\n[Processing] {title_text}")
                    folder_path = os.path.join(self.base_dir, target_year, self._clean_name(title_text))
                    os.makedirs(folder_path, exist_ok=True)

                    for link in content_div.find_all('a', href=True):
                        url = urljoin(self.target_url, link['href'])
                        link_text = link.get_text(strip=True)
                        if len(link_text) < 2: continue
                        
                        safe_fn = self._clean_name(link_text)[:100]
                        # บันทึกไฟล์ LINK.txt
                        with open(os.path.join(folder_path, f"{safe_fn}_LINK.txt"), "w", encoding="utf-8") as f:
                            f.write(url)
                        
                        # ดาวน์โหลด PDF
                        self._download_pdf(url, folder_path, safe_fn)

        except Exception as e:
            print(f"[Error] {e}")

    def _to_thai_num(self, n):
        return str(n).translate(str.maketrans("0123456789", "๐๑๒๓๔๕๖๗๘๙"))

    def _download_pdf(self, url, folder, filename):
        try:
            if 'drive.google.com' in url:
                file_id = re.search(r'd/([\w-]+)', url) or re.search(r'id=([\w-]+)', url)
                if not file_id: return
                url = f"https://docs.google.com/uc?export=download&id={file_id.group(1)}"

            r = requests.get(url, stream=True, timeout=10)
            if r.status_code == 200 and 'text/html' not in r.headers.get('Content-Type', ''):
                with open(os.path.join(folder, f"{filename}.pdf"), "wb") as f:
                    for chunk in r.iter_content(32768): f.write(chunk)
                print(f"      - {filename[:40]}... [OK]")
        except: pass

if __name__ == "__main__":
    scraper = KmitlFinalScraper("https://office.kmitl.ac.th/oaq/academic/")
    scraper.scrape_by_year("2563")