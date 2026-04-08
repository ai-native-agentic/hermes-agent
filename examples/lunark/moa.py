"""Mixture-of-Agents (MoA) demo with 4 lunark models.

For each prompt:
  1. Query all 4 models in parallel.
  2. Extract a normalized "answer" from each response.
  3. Vote: pick the most common answer.
  4. Compare MoA accuracy vs each single model.

Hard prompts focus on the stepwise category where individual models had
the most disagreement (Gemma 6/10).
"""
from __future__ import annotations
import json, os, sys, time, threading, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["PATH"] = f"{os.environ['HOME']}/.local/bin:" + os.environ.get("PATH", "")

from batch_qa import _normalize_for_check  # noqa: E402
from examples.acp_client import HermesACPClient  # noqa: E402

MODELS = ["Qwen3-32B", "Gemma-4-E4B-it", "Qwen2.5-32B-Instruct", "Qwen3.5-27B"]

# Hard prompts — focus on stepwise/arithmetic where disagreement is highest.
# Each: (id, prompt, normalize fn, expected canonical answer)

def num_extract(s: str) -> str:
    """Extract last integer from response (often the final answer)."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else ""

def yn_extract(s: str) -> str:
    """yes/no extractor — checks for affirmation in last sentence."""
    s = _normalize_for_check(s)
    last = s.split(".")[-2] if "." in s else s
    if any(w in last for w in ["yes", "true", "correct", "indeed"]):
        return "yes"
    if any(w in last for w in ["no", "false", "not", "isn't", "isnt"]):
        return "no"
    # Fallback: scan whole response
    if "yes" in s and "no" not in s: return "yes"
    if "no" in s and "yes" not in s: return "no"
    return ""

def word_extract(s: str) -> str:
    """Last meaningful word."""
    s = _normalize_for_check(s)
    s = re.sub(r"[^\w\s가-힣]", " ", s)
    words = [w for w in s.split() if w and not w.isdigit()]
    return words[-1] if words else ""


PROMPTS = [
    # Stepwise math (the hardest category yesterday)
    ("step-1", "Alice has 5 apples. She gives 2 to Bob and eats 1. How many left? Just the number.", num_extract, "2"),
    ("step-2", "A train leaves at 9am at 60mph. At 11am, how far has it traveled in miles? Just the number.", num_extract, "120"),
    ("step-3", "If 3 workers build a wall in 6 days, how many days for 6 workers? Just the number.", num_extract, "3"),
    ("step-4", "A box has 10 red and 5 blue balls. Probability of red as a fraction? Just the simplified fraction.", lambda s: re.search(r"\d+/\d+", _normalize_for_check(s)).group() if re.search(r"\d+/\d+", _normalize_for_check(s)) else "", "2/3"),
    ("step-5", "If a triangle has angles 30 and 60 degrees, what is the third angle? Just the number.", num_extract, "90"),
    ("step-6", "A rectangle is 4x5. What is its area? Just the number.", num_extract, "20"),
    ("step-7", "Sum of even numbers from 2 to 10? Just the number.", num_extract, "30"),
    ("step-8", "How many minutes in 2.5 hours? Just the number.", num_extract, "150"),
    ("step-9", "If a shirt costs $20 with 10% discount, what is the price? Just the number.", num_extract, "18"),
    ("step-10", "Average of 10, 20, 30, 40? Just the number.", num_extract, "25"),
    # A few harder ones
    ("hard-1", "What is 12% of 250? Just the number.", num_extract, "30"),
    ("hard-2", "If x^2 = 144, what are the two values of x? Just the two numbers separated by space.", lambda s: " ".join(sorted(re.findall(r"-?\d+", s)[-2:])) if len(re.findall(r"-?\d+", s)) >= 2 else "", "-12 12"),
    ("hard-3", "Sum of digits of 12345? Just the number.", num_extract, "15"),
    ("hard-4", "How many seconds in 1.5 hours? Just the number.", num_extract, "5400"),
    ("hard-5", "What is 7! (factorial)? Just the number.", num_extract, "5040"),
]

print(f"[moa] {len(MODELS)} models × {len(PROMPTS)} prompts", file=sys.stderr)

# Cartesian: ask every model every prompt
JOBS = [(m, pid, prompt, extract, expected) for m in MODELS for pid, prompt, extract, expected in PROMPTS]
results = {}  # (model, pid) → {raw, extracted}
results_lock = threading.Lock()


def run_one(job):
    model, pid, prompt, extract, expected = job
    home = f"/tmp/hermes_{model}"
    env = os.environ.copy()
    env["HERMES_HOME"] = home
    env["LUNARK_API_KEY"] = "vllm-local"
    t0 = time.time()
    msg, err = "", ""
    try:
        client = HermesACPClient(cwd="/tmp", env=env)
        try:
            r = client.prompt(prompt, timeout=120)
            msg = r.message
        finally:
            client.close()
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
    elapsed = round(time.time() - t0, 1)
    extracted = ""
    try:
        extracted = extract(msg) if msg else ""
    except Exception:
        extracted = ""
    correct = extracted.strip().lower() == expected.strip().lower()
    with results_lock:
        results[(model, pid)] = {
            "raw": msg.strip()[-150:],
            "extracted": extracted,
            "expected": expected,
            "correct": correct,
            "elapsed": elapsed,
            "err": err,
        }
    print(f"  {model:22s} {pid:8s} {'✓' if correct else '✗'} extracted={extracted!r:12s} expected={expected!r}", file=sys.stderr)


t_start = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(run_one, JOBS))
t_total = time.time() - t_start

print(f"\n[moa] DONE in {t_total:.1f}s", file=sys.stderr)


# === MoA voting ===
print("\n=== MoA VOTING RESULTS ===\n")
moa_correct = 0
single_correct = {m: 0 for m in MODELS}
detail_rows = []

for pid, prompt, extract, expected in PROMPTS:
    votes = []
    per_model = {}
    for m in MODELS:
        r = results.get((m, pid))
        if r and r["extracted"]:
            votes.append(r["extracted"].strip().lower())
            per_model[m] = r["extracted"]
            if r["correct"]:
                single_correct[m] += 1
        else:
            per_model[m] = "—"
    # Majority vote
    if votes:
        winner, count = Counter(votes).most_common(1)[0]
    else:
        winner, count = "", 0
    moa_pass = winner == expected.strip().lower()
    if moa_pass:
        moa_correct += 1
    detail_rows.append((pid, expected, per_model, winner, count, moa_pass))
    print(f"  {pid:8s}  exp={expected!r:10s}  vote={winner!r:10s}({count}/{len(MODELS)})  {'PASS' if moa_pass else 'FAIL'}")

print()

# === Summary ===
print("\n=== ACCURACY ===")
n = len(PROMPTS)
print(f"  MoA (majority vote)        {moa_correct:2d}/{n}  ({100*moa_correct/n:.0f}%)")
for m in MODELS:
    p = single_correct[m]
    print(f"  {m:22s}    {p:2d}/{n}  ({100*p/n:.0f}%)")

# Cases where MoA WINS over best single model
best_single = max(single_correct.values())
moa_advantage = moa_correct - best_single
print(f"\n  MoA vs best single ({best_single}/{n}): {'+' if moa_advantage > 0 else ''}{moa_advantage}")

# Cases where MoA fixed an individual model error
print("\n=== CASES WHERE MoA FIXED ===")
fixed = 0
for pid, expected, per_model, winner, count, moa_pass in detail_rows:
    if moa_pass:
        wrong_models = [m for m, v in per_model.items() if v.strip().lower() != expected.strip().lower() and v != "—"]
        if wrong_models:
            fixed += 1
            print(f"  {pid:8s}  {wrong_models}  → MoA chose {winner!r}")
print(f"\n  Total cases where MoA recovered from wrong individual answers: {fixed}")

# Save raw
with open("/tmp/acp_moa_results.json", "w") as f:
    json.dump({
        "results": {f"{k[0]}|{k[1]}": v for k, v in results.items()},
        "summary": {
            "moa_correct": moa_correct,
            "single_correct": single_correct,
            "n": n,
            "wall_time": t_total,
        },
    }, f, indent=2, default=str)
