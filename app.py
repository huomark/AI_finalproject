import os
import requests
from bs4 import BeautifulSoup
import time


def fetch_with_retry(url, retries=3, delay=3):
    for attempt in range(retries):
        try:
            return requests.get(url, timeout=10)
        except Exception as e:
            print(f"第 {attempt+1} 次嘗試失敗：{e}")
            time.sleep(delay)
    raise RuntimeError(f"多次重試失敗：{url}")

HANDLE = "jiangly"
LANG_KEYWORDS = ["C++"]
SAVE_DIR = f"codeforces_{HANDLE}_cpp"

os.makedirs(SAVE_DIR, exist_ok=True)

def get_submissions():
    submissions = []
    page = 1
    print(f"開始抓取使用者 {HANDLE} 的提交...")
    while True:
        url = f"https://codeforces.com/api/user.status?handle={HANDLE}&from={(page-1)*100+1}&count=100"
        try:
            res = fetch_with_retry(url)
            print(f"正在抓第 {page} 頁... ({url})")
        except Exception as e:
            print(f"抓第 {page} 頁時發生錯誤: {e}")
            break

        if res.status_code != 200:
            print(f"API 錯誤：狀態碼 {res.status_code}，訊息：{res.text[:200]}")
            break

        data = res.json()
        if data["status"] != "OK":
            print("API 回傳非 OK 狀態")
            break

        result = data["result"]
        if not result:
            print("沒有更多提交了")
            break

        submissions += result
        print(f"目前總共抓到 {len(submissions)} 筆提交")

        if len(result) < 100:
            print("已經到最後一頁")
            break

        page += 1
        time.sleep(0.5)
    return submissions

def filter_ac_cpp(submissions):
    print("開始篩選 C++ 的 AC 提交...")
    seen = set()
    selected = []
    for sub in submissions:
        if sub.get("verdict") != "OK":
            continue
        lang = sub.get("programmingLanguage", "")
        if not any(kw in lang for kw in LANG_KEYWORDS):
            continue
        if "contestId" not in sub or "problem" not in sub or "index" not in sub["problem"]:
            continue  # 跳過非正式題目或格式錯誤

        contest_id = sub["contestId"]
        index = sub["problem"]["index"]
        problem_id = f"{contest_id}{index}"
        if problem_id in seen:
            continue
        seen.add(problem_id)
        selected.append((contest_id, sub["id"], problem_id))
    print(f"總共找到 {len(selected)} 筆 C++ AC 提交（每題一筆）")
    return selected

def download_code(contest_id, submission_id):
    url = f"https://codeforces.com/contest/{contest_id}/submission/{submission_id}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Cookie": "JSESSIONID=D4B14E5FC1E30A13CC3775DAFC3E7A4D; 39ce7=CFrCAzkj;"
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        pre = soup.find("pre", {"id": "program-source-text"})
        return pre.text if pre else "[未找到程式碼]"
    except Exception as e:
        print(f"下載 {submission_id} 時錯誤: {e}")
        return "[下載失敗]"

def save_code(problem_id, code):
    filename = os.path.join(SAVE_DIR, f"{problem_id}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)

def main():
    submissions = get_submissions()
    if not submissions:
        print("沒有抓到任何提交，程式結束")
        return
    selected = filter_ac_cpp(submissions)
    for i, (contest_id, sub_id, prob_id) in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] 下載 {prob_id} 的程式碼中...")
        code = download_code(contest_id, sub_id)
        save_code(prob_id, code)
        time.sleep(0.5)
    print("下載完成！")

if __name__ == "__main__":
    main()
