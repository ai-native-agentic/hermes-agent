"""50 hermes use case matrix — categorized, model-optimized.

10 categories × 5 scenarios. Each scenario specifies the most appropriate
lunark model based on V3 variance data:
  - Qwen3-32B   → tool-using agent work (10/10 tool, 98.4% mean)
  - Qwen2.5-32B → fast instruct + research (96.8%, 3.3s)
  - Qwen3.5-27B → single-shot QA, writing, learning (99.8%, 4.2s)
  - Gemma       → reserved (smallest, used as control if needed)

Workspace at /tmp/hermes_uc preseeded with sample.py + numbers.txt.
"""
from __future__ import annotations
import json, os, re, sys, time, threading, subprocess
from concurrent.futures import ThreadPoolExecutor

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PATH"] = f"{os.environ['HOME']}/.local/bin:" + os.environ.get("PATH", "")

from examples.acp_client import HermesACPClient

# Make sure workspace files exist
os.makedirs("/tmp/hermes_uc", exist_ok=True)
if not os.path.exists("/tmp/hermes_uc/sample.py"):
    with open("/tmp/hermes_uc/sample.py", "w") as f:
        f.write('''"""Sample module."""

def fibonacci(n):
    """Return first n Fib numbers."""
    seq, a, b = [], 0, 1
    for _ in range(n):
        seq.append(a); a, b = b, a + b
    return seq

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0: return False
    return True

class Calculator:
    def __init__(self): self.history = []
    def add(self, a, b): return a + b
    def multiply(self, a, b): return a * b
''')

with open("/tmp/hermes_uc/data.csv", "w") as f:
    f.write("name,score\nalice,92\nbob,87\ncarol,95\ndave,71\nerin,88\n")

# Each use case: (id, category, model, prompt, check_callable)
# check_callable returns True if response indicates success.

def has(*words, ignore_case=True):
    def chk(r):
        s = r.lower() if ignore_case else r
        return all(w.lower() in s for w in words)
    return chk

def has_any(*words):
    def chk(r):
        s = r.lower()
        return any(w.lower() in s for w in words)
    return chk

def has_num(*nums):
    def chk(r):
        return any(str(n) in r for n in nums)
    return chk

# Models
M3 = "Qwen3-32B"          # heavy agent (tools)
M25 = "Qwen2.5-32B-Instruct"  # fast instruct
M35 = "Qwen3.5-27B"        # single-shot champion

UC = []

# A. Code (5) — Qwen3-32B
UC += [
    ("A1", "code", M3, "Read /tmp/hermes_uc/sample.py and tell me how many functions and how many classes are defined. Reply with just two numbers separated by a space (functions classes).", has("2", "1")),
    ("A2", "code", M3, "Read /tmp/hermes_uc/sample.py. What does fibonacci(7) return? Reply with just the list.", has("0", "1", "1", "2", "3", "5", "8")),
    ("A3", "code", M3, "Write a Python script /tmp/hermes_uc/fizzbuzz.py that prints 'FizzBuzz' for multiples of 15, 'Fizz' for 3, 'Buzz' for 5, otherwise the number — for 1..15. Run it. Reply with the last line of output.", has("FizzBuzz")),
    ("A4", "code", M3, "Read /tmp/hermes_uc/sample.py and find the line number where 'class Calculator' is defined. Reply with just the line number.", has_num(15, 16, 17, 18, 19, 20, 21)),
    ("A5", "code", M3, "Use execute_code to import the sample module from /tmp/hermes_uc/sample.py and check if is_prime(17) returns True. Reply yes or no.", has_any("yes", "true")),
]

# B. DevOps / sysadmin (5) — Qwen3-32B
UC += [
    ("B1", "devops", M3, "Use the terminal tool to count files in /tmp/hermes_uc (only files, not directories). Reply with just the number.", has_num(2, 3, 4, 5, 6)),
    ("B2", "devops", M3, "Use terminal to run 'uname -s' and reply with just the OS name.", has_any("linux", "darwin")),
    ("B3", "devops", M3, "Use terminal to run 'wc -l /tmp/hermes_uc/sample.py' and reply with just the line count number.", has_num(*range(15, 30))),
    ("B4", "devops", M3, "Use terminal to find the largest file in /tmp/hermes_uc by bytes. Reply with just the filename (basename, no path).", has_any("sample.py", "data.csv")),
    ("B5", "devops", M3, "Use terminal to list .py files in /tmp/hermes_uc. Reply with comma-separated filenames (basename only).", has("sample.py")),
]

# C. Data processing (5) — Qwen3-32B
UC += [
    ("C1", "data", M3, "Use execute_code to compute the mean of [10, 20, 30, 40, 50]. Reply with just the number.", has_num(30)),
    ("C2", "data", M3, "Read /tmp/hermes_uc/data.csv and tell me the highest score. Reply with just the number.", has_num(95)),
    ("C3", "data", M3, "Read /tmp/hermes_uc/data.csv and tell me whose score is highest. Reply with just the name (lowercase).", has("carol")),
    ("C4", "data", M3, "Use execute_code to compute the standard deviation of [2, 4, 4, 4, 5, 5, 7, 9]. Reply with just the number rounded to 1 decimal.", has_num(2.0)),
    ("C5", "data", M3, "Use execute_code to count how many primes are between 1 and 30. Reply with just the number.", has_num(10)),
]

# D. Research / general knowledge (5) — Qwen2.5-32B
UC += [
    ("D1", "research", M25, "What is the speed of light in vacuum, in m/s? Reply with just the number (no commas).", has_num(299792458)),
    ("D2", "research", M25, "Who is credited with inventing the telephone? Reply with just the last name.", has("bell")),
    ("D3", "research", M25, "In what year did the Western Roman Empire fall? Reply with just the year.", has_num(476)),
    ("D4", "research", M25, "What is the chemical formula of water? Reply with just the formula.", has("h2o")),
    ("D5", "research", M25, "What does TCP stand for? Reply with just the expansion.", has("transmission", "control", "protocol")),
]

# E. Automation (5) — Qwen3-32B
UC += [
    ("E1", "automation", M3, "Create a directory /tmp/hermes_uc/auto/ and put 3 empty files named a.txt b.txt c.txt inside. Then list its contents. Reply with comma-separated filenames.", has("a.txt", "b.txt", "c.txt")),
    ("E2", "automation", M3, "Use terminal to count how many lines in /tmp/hermes_uc/data.csv (including header). Reply with just the number.", has_num(6)),
    ("E3", "automation", M3, "Use terminal to extract just the names from /tmp/hermes_uc/data.csv (skip header, just first column). Reply with comma-separated names.", has("alice", "bob", "carol", "dave", "erin")),
    ("E4", "automation", M3, "Use execute_code in Python to base64-encode the string 'hermes' and reply with just the encoded value.", has("aGVybWVz")),
    ("E5", "automation", M3, "Use terminal to compute the SHA256 of the string 'hermes' (echo -n 'hermes' | sha256sum). Reply with just the first 12 hex characters.", has("e2adf67e87aa")),
]

# F. Writing (5) — Qwen3.5-27B
UC += [
    ("F1", "writing", M35, "Write a haiku about Python programming. Three lines, follow 5-7-5 syllable pattern roughly. Just the haiku, no explanation.", lambda r: r.count("\n") >= 2),
    ("F2", "writing", M35, "Translate 'Good morning' to Korean. One word/phrase only.", has("좋은", "아침")),
    ("F3", "writing", M35, "Write a one-sentence tagline for a calculator app. Just the tagline.", lambda r: 5 < len(r.strip()) < 200),
    ("F4", "writing", M35, "Summarize this in one sentence: 'The mitochondria is the powerhouse of the cell, responsible for producing ATP through cellular respiration.' Just the summary.", has_any("mitochondria", "energy", "atp", "powerhouse")),
    ("F5", "writing", M35, "Translate 'Hello, how are you?' to Japanese (Hiragana or Romaji). Just the translation.", has_any("こんにちは", "konnichiwa", "ohayou", "genki")),
]

# G. Learning / explanation (5) — Qwen3.5-27B
UC += [
    ("G1", "learning", M35, "Explain recursion in one sentence (no longer than 30 words). Just the sentence.", has_any("recursion", "function", "itself", "calls")),
    ("G2", "learning", M35, "What does Big-O O(1) mean in one sentence? Just the sentence.", has_any("constant", "time", "input", "size")),
    ("G3", "learning", M35, "In Python, what is the main difference between a list and a tuple? One sentence.", has_any("mutable", "immutable", "change")),
    ("G4", "learning", M35, "What does REST stand for in REST API? Just the expansion.", has("representational", "state", "transfer")),
    ("G5", "learning", M35, "In one sentence, what is a linked list?", has_any("nodes", "node", "pointer", "next", "linked")),
]

# H. Math / science calculations (5) — Qwen3-32B
UC += [
    ("H1", "math", M3, "Use execute_code to solve x^2 - 5x + 6 = 0. Reply with the two solutions as 'a, b' (smaller first).", has("2", "3")),
    ("H2", "math", M3, "Use execute_code to compute pi to 10 decimal digits. Reply with just the number.", has("3.1415926535")),
    ("H3", "math", M3, "Use execute_code to compute the probability of rolling a sum of 7 with two fair six-sided dice as a fraction. Reply with just the fraction (e.g. 1/6).", has("1/6")),
    ("H4", "math", M3, "Convert 100 degrees Fahrenheit to Celsius. Use execute_code if helpful. Reply with just the number rounded to 2 decimals.", has("37.78")),
    ("H5", "math", M3, "Use execute_code to compute the factorial of 10. Reply with just the number.", has("3628800")),
]

# I. Agent meta — introspection (5) — Qwen3-32B
UC += [
    ("I1", "meta", M3, "How many tools do you currently have access to? Reply with just the number.", has_num(*range(20, 50))),
    ("I2", "meta", M3, "Name 3 tools you have access to. Reply with comma-separated tool names only.", lambda r: r.count(",") >= 1),
    ("I3", "meta", M3, "What model are you running on? Reply with just the model name.", has_any("qwen", "qwen3")),
    ("I4", "meta", M3, "Can you read files? Reply yes or no.", has_any("yes")),
    ("I5", "meta", M3, "What is the current working directory? Use terminal to find out and reply with just the path.", has("/tmp")),
]

# J. Personal assistant / advisory (5) — Qwen2.5-32B
UC += [
    ("J1", "assistant", M25, "Suggest 3 quick personal productivity tips. Reply as a numbered list, one per line.", lambda r: r.count("\n") >= 2),
    ("J2", "assistant", M25, "Recommend a healthy lunch idea under 500 calories. One sentence.", has_any("calorie", "calories", "salad", "lean", "vegetable", "protein", "grain")),
    ("J3", "assistant", M25, "Suggest a 20-minute beginner workout routine. Brief outline only, 3-5 lines.", has_any("squat", "pushup", "push-up", "jog", "stretch", "plank", "exercise")),
    ("J4", "assistant", M25, "Recommend one book about software engineering. Reply with just the title.", has_any("clean", "code", "pragmatic", "mythical", "design", "patterns", "refactoring")),
    ("J5", "assistant", M25, "Suggest a simple morning routine in 3 steps. Numbered list, one per line.", lambda r: r.count("\n") >= 2),
]

print(f"[uc50] {len(UC)} use cases queued", file=sys.stderr)

results = []
results_lock = threading.Lock()
RESULT_FILE = "/tmp/uc50_results.jsonl"
open(RESULT_FILE, "w").close()


def run_one(uc):
    uc_id, category, model, prompt, check = uc
    home = f"/tmp/hermes_{model}"
    env = os.environ.copy()
    env["HERMES_HOME"] = home
    env["LUNARK_API_KEY"] = "vllm-local"
    t0 = time.time()
    msg = ""
    err = ""
    try:
        client = HermesACPClient(cwd="/tmp/hermes_uc", env=env)
        try:
            r = client.prompt(prompt, timeout=180)
            msg = r.message
        finally:
            client.close()
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
    elapsed = round(time.time() - t0, 1)
    passed = False
    try:
        passed = bool(check(msg)) if msg else False
    except Exception:
        passed = False
    rec = {
        "id": uc_id, "cat": category, "model": model,
        "passed": passed, "elapsed": elapsed, "err": err,
        "msg_tail": msg.strip()[-200:],
    }
    with results_lock:
        results.append(rec)
        with open(RESULT_FILE, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n = len(results)
    print(f"  [{n:2d}/{len(UC)}] {uc_id:4s} {category:12s} {model[:18]:18s} {'PASS' if passed else 'FAIL':4s} {elapsed:6.1f}s", file=sys.stderr)


t_start = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(run_one, UC))
t_total = time.time() - t_start

print(f"\n[uc50] DONE in {t_total:.1f}s", file=sys.stderr)


# === Summary ===
from collections import defaultdict

print("\n=== BY CATEGORY ===")
cats = defaultdict(list)
for r in results:
    cats[r["cat"]].append(r)
for cat in ["code","devops","data","research","automation","writing","learning","math","meta","assistant"]:
    rs = cats.get(cat, [])
    p = sum(1 for r in rs if r["passed"])
    avg = sum(r["elapsed"] for r in rs)/len(rs) if rs else 0
    print(f"  {cat:12s}  {p}/{len(rs)}  avg {avg:5.1f}s")

print("\n=== BY MODEL ===")
mods = defaultdict(list)
for r in results:
    mods[r["model"]].append(r)
for m, rs in mods.items():
    p = sum(1 for r in rs if r["passed"])
    avg = sum(r["elapsed"] for r in rs)/len(rs)
    print(f"  {m:24s}  {p}/{len(rs)}  avg {avg:5.1f}s")

total_pass = sum(1 for r in results if r["passed"])
print(f"\n=== TOTAL ===")
print(f"  pass: {total_pass}/{len(results)}  ({100*total_pass/len(results):.1f}%)")
print(f"  errors: {sum(1 for r in results if r['err'])}")
print(f"  wall: {t_total:.0f}s")

# Failures detail
fails = [r for r in results if not r["passed"]]
print(f"\n=== FAILURES ({len(fails)}) ===")
for r in fails:
    print(f"  {r['id']:4s} {r['cat']:12s} {r['model'][:18]:18s}  tail={r['msg_tail'][-100:]!r}")
