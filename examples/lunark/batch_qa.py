"""Parallel ACP batch runner — 100 prompts across 4 workers."""
from __future__ import annotations
import json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
os.environ["PATH"] = f"{os.environ['HOME']}/.local/bin:" + os.environ.get("PATH", "")
from examples.acp_client import HermesACPClient

# 100 prompts across 10 categories. Each entry: (id, category, prompt, check)
# check: substring that must appear in lowercased response, or callable
P = []

# 1. Arithmetic / math (10)
math_qs = [
    ("17*23", "391"), ("144/12", "12"), ("2^10", "1024"), ("sqrt(169)", "13"),
    ("sum 1 to 100", "5050"), ("factorial of 6", "720"), ("13*17", "221"),
    ("999+1", "1000"), ("256/8", "32"), ("gcd(48,36)", "12"),
]
for i, (q, ans) in enumerate(math_qs):
    P.append((f"math-{i:02d}", "math", f"What is {q}? Reply with just the number.", ans))

# 2. Logic / puzzles (10)
logic_qs = [
    ("If all bloops are razzles and all razzles are lazzles, are all bloops lazzles? Yes or no.", "yes"),
    ("A is taller than B. B is taller than C. Who is shortest? Reply with just the letter.", "c"),
    ("True or false: 7 is a prime number.", "true"),
    ("How many days in a leap February? Reply with just the number.", "29"),
    ("Which is heavier: 1kg of feathers or 1kg of steel? Reply 'same' or which one.", "same"),
    ("If today is Monday, what day is it 10 days from now?", "thursday"),
    ("How many sides does a hexagon have? Reply with just the number.", "6"),
    ("The day before yesterday was Wednesday. What day is tomorrow?", "saturday"),
    ("True or false: a square is always a rectangle.", "true"),
    ("If x+5=12, what is x? Reply with just the number.", "7"),
]
for i, (q, ans) in enumerate(logic_qs):
    P.append((f"logic-{i:02d}", "logic", q, ans))

# 3. Knowledge facts (10)
kn_qs = [
    ("What is the capital of France? One word.", "paris"),
    ("Who wrote Romeo and Juliet? Last name only.", "shakespeare"),
    ("What is the chemical symbol for gold? Two letters.", "au"),
    ("How many planets in our solar system? Reply with just the number.", "8"),
    ("What year did WW2 end? Reply with just the year.", "1945"),
    ("What is the largest ocean? One word.", "pacific"),
    ("Speed of light in m/s, order of magnitude as 3*10^N. What is N?", "8"),
    ("Element with atomic number 1? One word.", "hydrogen"),
    ("Who painted the Mona Lisa? Last name only.", "vinci"),
    ("Tallest mountain on Earth? One word.", "everest"),
]
for i, (q, ans) in enumerate(kn_qs):
    P.append((f"know-{i:02d}", "knowledge", q, ans))

# 4. Code reasoning (10) — predict output
code_qs = [
    ("In Python, what does len('hello') return? Just the number.", "5"),
    ("In Python, what does list(range(3)) print? Just the list.", "[0, 1, 2]"),
    ("What does 'abc'[::-1] return in Python? Just the string.", "cba"),
    ("In Python, what is 7 // 2? Just the number.", "3"),
    ("What does True and False evaluate to in Python? Just the value.", "false"),
    ("In Python, what does 'hello'.upper() return? Just the string.", "hello"),  # check 'HELLO'
    ("What is the result of 2 ** 8 in Python? Just the number.", "256"),
    ("In Python, what does {1,2,2,3}.__len__() return? Just the number.", "3"),
    ("What does 'a,b,c'.split(',') return in Python? Just the list.", "['a', 'b', 'c']"),
    ("In Python, what does bool([]) return? Just the value.", "false"),
]
for i, (q, ans) in enumerate(code_qs):
    P.append((f"code-{i:02d}", "code", q, ans))

# 5. Translation (10)
tr_qs = [
    ("Translate '안녕' to English. One word.", "hello"),
    ("Translate 'thank you' to Korean. One word in hangul.", "감사"),
    ("Translate 'water' to Japanese romaji. One word.", "mizu"),
    ("Translate 'cat' to French. One word.", "chat"),
    ("Translate 'dog' to Spanish. One word.", "perro"),
    ("Translate 'book' to German. One word.", "buch"),
    ("Translate '사과' to English. One word.", "apple"),
    ("Translate 'red' to Italian. One word.", "rosso"),
    ("Translate 'sun' to Latin. One word.", "sol"),
    ("Translate '산' (Korean) to English. One word.", "mountain"),
]
for i, (q, ans) in enumerate(tr_qs):
    P.append((f"tran-{i:02d}", "translate", q, ans))

# 6. Format / structured (10)
fmt_qs = [
    ('Convert {"a":1,"b":2} to YAML key-value lines. Just the YAML.', "a: 1"),
    ("Convert the list [1,2,3,4,5] to a comma-separated string. Just the result.", "1,2,3,4,5"),
    ("How many key-value pairs in {'x':1,'y':2,'z':3}? Just the number.", "3"),
    ("What is the JSON for an empty object? Just the JSON.", "{}"),
    ("What is the JSON for an empty array? Just the JSON.", "[]"),
    ('Extract the value of "name" from {"name":"alice","age":30}. Just the value.', "alice"),
    ("How many fields in CSV row 'a,b,c,d,e'? Just the number.", "5"),
    ('Convert "hello world" to title case. Just the result.', "hello world"),  # 'Hello World'
    ("What is the 3rd element of [10,20,30,40]? Just the number.", "30"),
    ("How many characters in 'abcdef'? Just the number.", "6"),
]
for i, (q, ans) in enumerate(fmt_qs):
    P.append((f"fmt-{i:02d}", "format", q, ans))

# 7. Text manipulation (10)
txt_qs = [
    ("How many 'a' in 'banana'? Just the number.", "3"),
    ("How many words in 'the quick brown fox'? Just the number.", "4"),
    ("Reverse 'stressed'. Just the result.", "desserts"),
    ("What is 'hello' in uppercase? Just the result.", "hello"),  # 'HELLO'
    ("How many vowels in 'sequoia'? Just the number.", "5"),
    ("Replace 'cat' with 'dog' in 'the cat sat'. Just the result.", "dog sat"),
    ("How many letters in the alphabet? Just the number.", "26"),
    ("First letter of 'umbrella'? Just the letter.", "u"),
    ("Last letter of 'umbrella'? Just the letter.", "a"),
    ("Length of 'antidisestablishmentarianism'? Just the number.", "28"),
]
for i, (q, ans) in enumerate(txt_qs):
    P.append((f"text-{i:02d}", "text", q, ans))

# 8. Step-by-step reasoning (10)
sbs_qs = [
    ("Alice has 5 apples. She gives 2 to Bob and eats 1. How many left? Just the number.", "2"),
    ("A train leaves at 9am at 60mph. At 11am, how far has it traveled in miles? Just the number.", "120"),
    ("If 3 workers build a wall in 6 days, how many days for 6 workers? Just the number.", "3"),
    ("A box has 10 red and 5 blue balls. Probability of red as a fraction? Just the fraction.", "2/3"),
    ("If a triangle has angles 30 and 60 degrees, what is the third angle? Just the number.", "90"),
    ("A rectangle is 4x5. What is its area? Just the number.", "20"),
    ("Sum of even numbers from 2 to 10? Just the number.", "30"),
    ("How many minutes in 2.5 hours? Just the number.", "150"),
    ("If a shirt costs $20 with 10% discount, what is the price? Just the number.", "18"),
    ("Average of 10, 20, 30, 40? Just the number.", "25"),
]
for i, (q, ans) in enumerate(sbs_qs):
    P.append((f"step-{i:02d}", "stepwise", q, ans))

# 9. Comparison / classification (10)
cmp_qs = [
    ("Is 17 prime? Yes or no.", "yes"),
    ("Which is bigger: 7/8 or 8/9? Just the fraction.", "8/9"),
    ("Is 'apple' a fruit or vegetable? Just one word.", "fruit"),
    ("Is Python compiled or interpreted? Just one word.", "interpreted"),
    ("Is the Earth flat? Yes or no.", "no"),
    ("Is 0 positive, negative, or neither? Just one word.", "neither"),
    ("Is HTML a programming language? Yes or no.", "no"),
    ("Is Mercury closer to the sun than Venus? Yes or no.", "yes"),
    ("Is JSON a data format or a programming language? Just one word.", "format"),
    ("Is 100 divisible by 7? Yes or no.", "no"),
]
for i, (q, ans) in enumerate(cmp_qs):
    P.append((f"comp-{i:02d}", "compare", q, ans))

# 10. Short creative + constrained (10)
cre_qs = [
    ("Give me one synonym for 'happy'. Just the word.", "joy"),  # 'joyful' etc
    ("Give me one antonym for 'big'. Just the word.", "small"),
    ("Name one primary color. Just the word.", "red"),  # red/blue/yellow
    ("Name one noble gas. Just the element name.", "helium"),
    ("Name one Shakespearean play. Just the title.", "hamlet"),
    ("Name one Greek god. Just the name.", "zeus"),
    ("Name one programming language starting with 'P'. Just the word.", "python"),
    ("Name one continent. Just the word.", "asia"),
    ("Name one prime number between 20 and 30. Just the number.", "23"),  # 23 or 29
    ("What rhymes with 'cat'? One word.", "bat"),  # any -at word
]
for i, (q, ans) in enumerate(cre_qs):
    P.append((f"creat-{i:02d}", "creative", q, ans))


# Lower-case checks with special-case overrides
SPECIAL_CHECKS = {
    "code-05": lambda r: "hello" in r.lower(),  # HELLO upper
    "fmt-07":  lambda r: "hello world" in r.lower(),  # title case
    "text-03": lambda r: "hello" in r.lower(),  # uppercase
    "creat-00": lambda r: any(w in r.lower() for w in ["joy","joyful","glad","cheerful","content","elated","pleased","merry"]),
    "creat-02": lambda r: any(w in r.lower() for w in ["red","blue","yellow"]),
    "creat-03": lambda r: any(w in r.lower() for w in ["helium","neon","argon","krypton","xenon","radon"]),
    "creat-04": lambda r: any(w in r.lower() for w in ["hamlet","macbeth","othello","tempest","romeo","juliet","lear","caesar","midsummer"]),
    "creat-05": lambda r: any(w in r.lower() for w in ["zeus","hera","poseidon","hades","apollo","ares","athena","artemis","hermes","dionysus"]),
    "creat-07": lambda r: any(w in r.lower() for w in ["asia","africa","europe","america","australia","antarctica"]),
    "creat-08": lambda r: any(w in r for w in ["23","29"]),
    "creat-09": lambda r: any(w in r.lower() for w in ["bat","hat","mat","rat","fat","sat","pat","that","flat"]),
    "logic-05": lambda r: "thursday" in r.lower(),
    "logic-07": lambda r: "saturday" in r.lower(),
    # know-08: "Da Vinci" — accept either "vinci" or full "leonardo da vinci"
    "know-08": lambda r: "vinci" in r.lower() or "da vinci" in r.lower() or "leonardo" in r.lower(),
    # comp-01: 8/9 — accept either fraction form
    "comp-01": lambda r: any(p in _normalize_for_check(r) for p in ["8/9", "8/9"]),
    # comp-07: Mercury closer to sun than Venus → yes (or any affirmation that
    # Mercury is the closest/innermost/first planet)
    "comp-07": lambda r: (
        "yes" in r.lower()
        or ("mercury" in r.lower() and any(w in r.lower() for w in ["closest","closer","innermost","first","nearest"]))
    ),
}

import re as _re

def _normalize_for_check(s: str) -> str:
    """Strip LaTeX/markdown wrappers so substring checks find the answer.

    Handles: \\frac{a}{b} → a/b, \\dfrac{a}{b} → a/b, \\boxed{x} → x,
    `code spans`, **bold**, $...$ math, leading/trailing whitespace.
    """
    out = s
    # \frac{a}{b} and \dfrac{a}{b} → a/b
    out = _re.sub(r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", out)
    # \boxed{x} → x
    out = _re.sub(r"\\boxed\s*\{([^{}]+)\}", r"\1", out)
    # \(...\) and \[...\] math delimiters
    out = _re.sub(r"\\[\(\[\)\]]", "", out)
    # $...$ inline math (keep contents)
    out = _re.sub(r"\$([^$]+)\$", r"\1", out)
    # markdown bold/italic/code
    out = out.replace("**", "").replace("`", "").replace("__", "")
    return out.lower()


def check(pid: str, expected: str, response: str) -> bool:
    if pid in SPECIAL_CHECKS:
        return SPECIAL_CHECKS[pid](response)
    norm = _normalize_for_check(response)
    return expected.lower() in norm


if __name__ == "__main__":
    _run_main = True
else:
    _run_main = False

if not _run_main:
    # Imported as module — expose P, check, SPECIAL_CHECKS only
    pass

print(f"[runner] {len(P)} prompts queued", file=sys.stderr) if _run_main else None
results_lock = threading.Lock()
results: list[dict] = []

def worker(worker_id: int, jobs: list):
    """Each worker spawns its own ACP process and runs sequential prompts.
    A new session is created per prompt to avoid history accumulation."""
    print(f"[w{worker_id}] starting with {len(jobs)} jobs", file=sys.stderr)
    for pid, cat, prompt, expected in jobs:
        t0 = time.time()
        msg, thought, stop, err = "", "", "", ""
        try:
            client = HermesACPClient(cwd="/tmp")
            try:
                r = client.prompt(prompt, timeout=120)
                msg = r.message
                thought = r.thought
                stop = r.stop_reason
            finally:
                client.close()
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        elapsed = time.time() - t0
        passed = check(pid, expected, msg) if msg else False
        rec = {
            "id": pid, "cat": cat, "expected": expected,
            "passed": passed, "elapsed": round(elapsed, 1),
            "stop": stop, "err": err,
            "msg_preview": msg.strip()[:100],
        }
        with results_lock:
            results.append(rec)
            with open("/tmp/acp_results.jsonl", "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[w{worker_id}] {pid:12s} {'PASS' if passed else 'FAIL'} {elapsed:5.1f}s  {msg.strip()[:50]!r}", file=sys.stderr)


if _run_main:
    # Reset results file
    open("/tmp/acp_results.jsonl", "w").close()

    N_WORKERS = 4
    chunks = [P[i::N_WORKERS] for i in range(N_WORKERS)]
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(worker, i, chunks[i]) for i in range(N_WORKERS)]
        for f in as_completed(futures):
            f.result()

    t_total = time.time() - t_start
    print(f"\n[runner] DONE in {t_total:.1f}s — {len(results)} results", file=sys.stderr)

    by_cat = {}
    for r in results:
        by_cat.setdefault(r["cat"], []).append(r)

    print("\n=== SUMMARY BY CATEGORY ===", file=sys.stderr)
    for cat, rs in sorted(by_cat.items()):
        p = sum(1 for r in rs if r["passed"])
        avg = sum(r["elapsed"] for r in rs) / len(rs)
        print(f"  {cat:12s}  {p}/{len(rs)}  avg {avg:.1f}s", file=sys.stderr)

    total_pass = sum(1 for r in results if r["passed"])
    print(f"\nTOTAL: {total_pass}/{len(results)} pass  total wall {t_total:.0f}s", file=sys.stderr)
