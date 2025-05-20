import os
import time
import json
from dotenv import load_dotenv
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc

# ========== CONFIG ==========
HANDLE = "jiangly"
SAVE_DIR = f"codeforces_{HANDLE}_cpp"
CHECKPOINT_FILE = f"{SAVE_DIR}/checkpoint.txt"
MAX_SUBMISSIONS = 5000  # 可調整爬幾筆

# ========== ENV & LOGIN ==========
load_dotenv()
USERNAME = os.getenv("CF_USERNAME")
PASSWORD = os.getenv("CF_PASSWORD")
os.makedirs(SAVE_DIR, exist_ok=True)

def setup_driver():
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1200,800")
    driver = uc.Chrome(options=options)
    return driver

def wait_manual_login(driver):
    driver.get("https://codeforces.com/enter")
    print("🔒 請在瀏覽器中手動通過 Cloudflare 並登入帳號，登入成功後按 Enter 繼續...")
    input("✅ 已登入後按 Enter 開始抓取...")

def get_submissions():
    submissions = []
    page = 1
    print("📥 開始抓取 jiangly 的提交紀錄...")
    while len(submissions) < MAX_SUBMISSIONS:
        url = f"https://codeforces.com/api/user.status?handle={HANDLE}&from={(page - 1) * 100 + 1}&count=100"
        print(f" - 抓第 {page} 頁...")
        try:
            import requests
            res = requests.get(url, timeout=10)
            data = res.json()
            if data["status"] != "OK":
                break
            batch = data["result"]
            if not batch:
                break
            submissions += batch
            if len(batch) < 100:
                break
        except Exception as e:
            print("⚠️ 抓取錯誤：", e)
            break
        page += 1
        time.sleep(0.5)
    print(f"✅ 共抓到 {len(submissions)} 筆提交")
    return submissions

def filter_cpp_ac(submissions):
    seen = set()
    selected = []
    for sub in submissions:
        if sub.get("verdict") != "OK":
            continue
        lang = sub.get("programmingLanguage", "")
        if "C++" not in lang:
            continue
        if "contestId" not in sub or "problem" not in sub:
            continue
        pid = f"{sub['contestId']}{sub['problem']['index']}"
        if pid in seen:
            continue
        seen.add(pid)
        selected.append({
            "contestId": sub['contestId'],
            "submissionId": sub['id'],
            "problemId": pid
        })
    print(f"✅ 篩選後留下 {len(selected)} 筆 C++ AC 提交")
    return selected

def wait_if_cloudflare(driver):
    time.sleep(10)
    title = driver.title.lower()
    source = driver.page_source.lower()
    if ("just a moment" in title or "attention required" in title or "verify" in title or
        "驗證您是人類" in source or "complete verification" in source or "連線安全性" in source):
        print("⚠️ 偵測到 Cloudflare 驗證頁，請手動完成後按 Enter 繼續...")
        input()

def get_code(driver, contest_id, submission_id):
    url = f"https://codeforces.com/contest/{contest_id}/submission/{submission_id}"
    driver.get(url)
    wait_if_cloudflare(driver)
    time.sleep(1)
    if "temporarily blocked" in driver.page_source:
        print("⚠️ 被封鎖了，暫停 60 秒再試...")
        time.sleep(60)
        driver.get(url)
        wait_if_cloudflare(driver)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "program-source-text"))
        )
        return driver.find_element(By.ID, "program-source-text").text
    except:
        return "[未找到程式碼]"

def save_code(pid, code):
    path = os.path.join(SAVE_DIR, f"{pid}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return int(f.read())
    return 0

def save_checkpoint(index):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(index))

def main():
    driver = setup_driver()
    try:
        wait_manual_login(driver)
        subs = get_submissions()
        targets = filter_cpp_ac(subs)
        start = load_checkpoint()
        print(f"▶️ 從第 {start} 筆開始續爬...")
        for i in range(start, len(targets)):
            item = targets[i]
            cid, sid, pid = item["contestId"], item["submissionId"], item["problemId"]
            if cid > 5000:
                continue
            try:
                print(f"[{i+1}/{len(targets)}] 抓取 {pid}...")
                code = get_code(driver, cid, sid)
                save_code(pid, code)
                save_checkpoint(i + 1)
            except Exception as e:
                print(f"⚠️ 第 {i+1} 筆抓取失敗: {e}")
            print("🕒 等待 1又1/6 分鐘再抓下一筆...")
            time.sleep(70)
        print("✅ 所有程式碼已下載完畢！")
    finally:
        try:
            driver.quit()
        except:
            pass

if __name__ == "__main__":
    main()
