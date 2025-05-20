import os
import time
import json
import requests

# ========== CONFIG ==========
HANDLE = "jiangly"
TAG_DIR = f"codeforces_{HANDLE}_tags"
CHECKPOINT_FILE = f"{TAG_DIR}/checkpoint.txt"
MAX_SUBMISSIONS = 5000

os.makedirs(TAG_DIR, exist_ok=True)

def get_submissions():
    submissions = []
    page = 1
    print("📥 開始抓取 jiangly 的提交紀錄...")
    while len(submissions) < MAX_SUBMISSIONS:
        url = f"https://codeforces.com/api/user.status?handle={HANDLE}&from={(page - 1) * 100 + 1}&count=100"
        print(f" - 抓第 {page} 頁...")
        try:
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
            "problemId": pid,
            "index": sub['problem']['index']
        })
    print(f"✅ 篩選後留下 {len(selected)} 筆 C++ AC 提交")
    return selected

def get_problem_tags():
    print("📦 抓取所有題目資料中...")
    res = requests.get("https://codeforces.com/api/problemset.problems")
    data = res.json()
    if data["status"] != "OK":
        raise Exception("API 失敗")
    problem_tags = {}
    for item in data["result"]["problems"]:
        pid = f"{item['contestId']}{item['index']}"
        tags = item.get("tags", [])
        problem_tags[pid] = tags
    return problem_tags

def save_tags(pid, tags):
    path = os.path.join(TAG_DIR, f"{pid}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(", ".join(tags))

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return int(f.read())
    return 0

def save_checkpoint(index):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(index))

def main():
    subs = get_submissions()
    targets = filter_cpp_ac(subs)
    tags_map = get_problem_tags()
    start = load_checkpoint()
    print(f"▶️ 從第 {start} 筆開始續爬 Tag...")
    for i in range(start, len(targets)):
        item = targets[i]
        pid = item["problemId"]
        try:
            print(f"[{i+1}/{len(targets)}] 儲存 Tag {pid}...")
            tags = tags_map.get(pid, [])
            save_tags(pid, tags)
            save_checkpoint(i + 1)
        except Exception as e:
            print(f"⚠️ 第 {i+1} 筆失敗: {e}")
        time.sleep(0.1)
    print("✅ 所有 Tag 已下載完畢！")

if __name__ == "__main__":
    main()
