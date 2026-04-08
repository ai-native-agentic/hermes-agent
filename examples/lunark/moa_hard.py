"""MoA hard re-run with robust extractor + weighted voting.

Compares:
  - Old extractor (regex), new extractor (canonical normalization)
  - Simple majority vote
  - Weighted vote based on prior accuracy
  - Each single model
"""
from __future__ import annotations
import json, os, sys, time, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PATH"] = f"{os.environ['HOME']}/.local/bin:" + os.environ.get("PATH", "")

from extractor import extract_number, extract_fraction, extract_yesno, extract_word, normalize_for_compare
from examples.acp_client import HermesACPClient

MODELS = ["Qwen3-32B", "Gemma-4-E4B-it", "Qwen2.5-32B-Instruct", "Qwen3.5-27B"]

# Weights from prior 100-prompt baseline accuracy
WEIGHTS = {
    "Qwen3-32B":            1.5,   # 100% baseline
    "Qwen3.5-27B":          1.5,   # 100% baseline
    "Qwen2.5-32B-Instruct": 1.0,   # 98% baseline
    "Gemma-4-E4B-it":       0.5,   # 95% baseline + tool fail
}

# 100 hard prompts (id, prompt, kind, expected canonical)
P = []

# 25 arithmetic
arith = [
    ("23 * 47", "1081"), ("15% of 240", "36"),
    ("sum of integers from 1 to 50", "1275"),
    ("greatest common divisor of 84 and 126", "42"),
    ("least common multiple of 12 and 18", "36"),
    ("8 factorial", "40320"), ("square root of 729", "27"),
    ("3 cubed plus 4 cubed", "91"),
    ("100 minus 37 plus 18", "81"),
    ("7 * 8 * 9", "504"),
    ("the 10th Fibonacci number (1,1,2,3,5,8...)", "55"),
    ("number of seconds in a day", "86400"),
    ("number of minutes in 3 days", "4320"),
    ("hours in 2 weeks", "336"),
    ("13 squared", "169"), ("2 to the power 11", "2048"),
    ("sum of first 10 odd numbers", "100"),
    ("sum of first 10 even numbers", "110"),
    ("number of diagonals in an octagon", "20"),
    ("perimeter of a square with side 17", "68"),
    ("area of a circle with radius 5 (use pi=3.14159)", "78"),  # accept ~78
    ("volume of a cube with edge 6", "216"),
    ("17 + 28 + 39 + 16", "100"),
    ("250 / 5 + 30 * 2", "110"),
    ("(15 + 25) * (8 - 3)", "200"),
]
for i, (q, ans) in enumerate(arith):
    P.append((f"arith-{i:02d}", f"What is {q}? Reply with just the number.", "number", ans))

# 20 word problems
wp = [
    ("A rectangle is 12cm by 8cm. What is its area in square cm?", "number", "96"),
    ("If a car travels at 60mph for 2.5 hours, how far in miles?", "number", "150"),
    ("Sally has 24 apples. She gives 1/3 to Bob. How many does Bob receive?", "number", "8"),
    ("A pizza is cut into 12 slices. 3 people eat 2 slices each. How many remain?", "number", "6"),
    ("If 5 pencils cost $2.50, how much do 12 pencils cost in dollars?", "number", "6"),
    ("A train leaves at 8:00 and arrives at 10:30. How long was the trip in minutes?", "number", "150"),
    ("If today is Wednesday, what day is it 100 days from now?", "word", "friday"),  # 100 mod 7 = 2 → Fri
    ("A book has 240 pages. Sarah reads 30 pages per day. How many days to finish?", "number", "8"),
    ("If a tank holds 500 liters and is 60% full, how many liters are in it?", "number", "300"),
    ("A worker earns $15 per hour for 40 hours. What is the weekly pay in dollars?", "number", "600"),
    ("If a recipe calls for 2 cups of flour for 12 cookies, how many cups for 30 cookies?", "number", "5"),
    ("A box has 6 red, 4 blue, and 5 green balls. How many balls total?", "number", "15"),
    ("If x + 7 = 20, what is x?", "number", "13"),
    ("If 3x = 24, what is x?", "number", "8"),
    ("A bag has 8 marbles. You add 5 more and remove 3. How many marbles now?", "number", "10"),
    ("Mary is 12 and her brother is twice her age. How old is the brother?", "number", "24"),
    ("If a clock shows 3:00, what is the angle between hour and minute hand in degrees?", "number", "90"),
    ("A discount of 25% off a $80 jacket. What is the discounted price in dollars?", "number", "60"),
    ("If 4 workers paint a room in 6 hours, how long for 2 workers (in hours)?", "number", "12"),
    ("A train is 8 cars long. Each car holds 60 people. Total capacity?", "number", "480"),
]
for i, (q, kind, ans) in enumerate(wp):
    suffix = " Just the day name." if kind == "word" else " Just the number."
    P.append((f"wp-{i:02d}", q + suffix, kind, ans))

# 15 yes/no logic
logic = [
    ("Is 91 a prime number?", "no"),
    ("Is the sum of 17 and 23 a prime number?", "no"),
    ("Is 1 a prime number?", "no"),
    ("Are all squares rectangles?", "yes"),
    ("Is the cube root of 343 equal to 7?", "yes"),
    ("Does the equation x^2 = -4 have real solutions?", "no"),
    ("Is 2026 a leap year?", "no"),
    ("Is 0 an even number?", "yes"),
    ("Is the sum of two odd numbers always even?", "yes"),
    ("Is the product of two negative numbers positive?", "yes"),
    ("Does a triangle's interior angles always sum to 180 degrees?", "yes"),
    ("Is pi a rational number?", "no"),
    ("Is the square root of 2 rational?", "no"),
    ("Is e (Euler's number) less than 3?", "yes"),
    ("Is the Earth's circumference greater than 30000 kilometers?", "yes"),
]
for i, (q, ans) in enumerate(logic):
    P.append((f"log-{i:02d}", q + " Reply yes or no.", "yesno", ans))

# 20 trivia
trivia = [
    ("How many bones in the adult human body?", "number", "206"),
    ("How many continents?", "number", "7"),
    ("How many oceans on Earth?", "number", "5"),
    ("Speed of sound in m/s in air at sea level (just integer)?", "number", "343"),
    ("Year the Berlin Wall fell?", "number", "1989"),
    ("Year humans first landed on the Moon?", "number", "1969"),
    ("Atomic number of carbon?", "number", "6"),
    ("Atomic number of oxygen?", "number", "8"),
    ("How many keys on a standard piano?", "number", "88"),
    ("How many planets in our solar system (excluding Pluto)?", "number", "8"),
    ("How many degrees in a full circle?", "number", "360"),
    ("Boiling point of water in Celsius at sea level?", "number", "100"),
    ("Freezing point of water in Fahrenheit?", "number", "32"),
    ("Number of US states?", "number", "50"),
    ("Number of players on a soccer team on the field?", "number", "11"),
    ("How many sides on a dodecagon?", "number", "12"),
    ("How many letters in the English alphabet?", "number", "26"),
    ("How many seconds in a minute?", "number", "60"),
    ("Roman numeral for 100?", "word", "c"),
    ("Roman numeral for 50?", "word", "l"),
]
for i, (q, kind, ans) in enumerate(trivia):
    P.append((f"trv-{i:02d}", q + " Just the answer.", kind, ans))

# 10 fractions
frac = [
    ("Simplify 12/16 to lowest terms.", "fraction", "3/4"),
    ("Simplify 15/25 to lowest terms.", "fraction", "3/5"),
    ("What is 1/2 + 1/3? Single fraction in lowest terms.", "fraction", "5/6"),
    ("What is 3/4 - 1/4? Single fraction in lowest terms.", "fraction", "1/2"),
    ("What is 1/2 * 2/3? Single fraction in lowest terms.", "fraction", "1/3"),
    ("Simplify 6/9.", "fraction", "2/3"),
    ("What is 1/4 of 100? Just the number.", "number", "25"),
    ("What is 3/5 of 50? Just the number.", "number", "30"),
    ("If a pie is cut into 8 slices and you eat 3, what fraction remains in lowest terms?", "fraction", "5/8"),
    ("Convert 0.75 to a fraction in lowest terms.", "fraction", "3/4"),
]
for i, (q, kind, ans) in enumerate(frac):
    P.append((f"frc-{i:02d}", q, kind, ans))

# 10 sequences
seqs = [
    ("Next in sequence 2, 4, 8, 16, ?", "32"),
    ("Next in 1, 4, 9, 16, ?", "25"),
    ("Next in 3, 6, 12, 24, ?", "48"),
    ("Next in 1, 1, 2, 3, 5, 8, ?", "13"),
    ("Next in 100, 90, 80, 70, ?", "60"),
    ("Next in 5, 11, 17, 23, ?", "29"),
    ("7th term of arithmetic sequence 3, 7, 11, ...?", "27"),
    ("Next in 2, 6, 12, 20, 30, ?", "42"),
    ("Next in 81, 27, 9, 3, ?", "1"),
    ("Sum 1+2+3+...+20?", "210"),
]
for i, (q, ans) in enumerate(seqs):
    P.append((f"seq-{i:02d}", q + " Just the number.", "number", ans))

print(f"[v2] {len(P)} prompts × {len(MODELS)} models = {len(P)*len(MODELS)} jobs", file=sys.stderr)

EXTRACTORS = {
    "number": extract_number,
    "fraction": extract_fraction,
    "yesno": extract_yesno,
    "word": extract_word,
}

JOBS = [(m, pid, prompt, kind, exp) for m in MODELS for pid, prompt, kind, exp in P]
results = {}
results_lock = threading.Lock()


def run_one(job):
    model, pid, prompt, kind, exp = job
    home = f"/tmp/hermes_{model}"
    env = os.environ.copy()
    env["HERMES_HOME"] = home
    env["LUNARK_API_KEY"] = "vllm-local"
    t0 = time.time()
    msg = ""
    try:
        client = HermesACPClient(cwd="/tmp", env=env)
        try:
            r = client.prompt(prompt, timeout=120)
            msg = r.message
        finally:
            client.close()
    except Exception:
        msg = ""
    elapsed = round(time.time() - t0, 1)
    extractor = EXTRACTORS.get(kind, extract_number)
    extracted = ""
    try:
        extracted = extractor(msg) if msg else ""
    except Exception:
        extracted = ""
    # Normalize comparison
    norm_exp = normalize_for_compare(exp, kind)
    norm_got = normalize_for_compare(extracted, kind)
    correct = norm_got == norm_exp
    with results_lock:
        results[(model, pid)] = {
            "extracted": extracted,
            "norm": norm_got,
            "expected": exp,
            "correct": correct,
            "elapsed": elapsed,
            "raw": msg.strip()[-100:],
        }
    n = len(results)
    if n % 40 == 0:
        print(f"  [{n}/{len(JOBS)}]", file=sys.stderr)


t_start = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(run_one, JOBS))
t_total = time.time() - t_start

print(f"\n[v2] DONE in {t_total:.1f}s", file=sys.stderr)


# === Voting ===
def vote_simple(votes):
    """Plain majority. Returns winner string or empty."""
    if not votes:
        return ""
    counts = defaultdict(int)
    for v in votes:
        counts[v] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def vote_weighted(per_model_extracted, weights):
    """Weighted vote: each model contributes its weight to its extracted answer."""
    if not per_model_extracted:
        return ""
    scores = defaultdict(float)
    for model, val in per_model_extracted.items():
        if val:
            scores[val] += weights.get(model, 1.0)
    if not scores:
        return ""
    return max(scores.items(), key=lambda kv: kv[1])[0]


single_correct = {m: 0 for m in MODELS}
moa_simple_correct = 0
moa_weighted_correct = 0
moa_recovered = []

for pid, prompt, kind, exp in P:
    norm_exp = normalize_for_compare(exp, kind)
    per_model = {}
    votes = []
    for m in MODELS:
        r = results.get((m, pid))
        if r:
            per_model[m] = r["norm"]
            if r["norm"]:
                votes.append(r["norm"])
            if r["correct"]:
                single_correct[m] += 1

    simple = vote_simple(votes)
    weighted = vote_weighted(per_model, WEIGHTS)
    if simple == norm_exp:
        moa_simple_correct += 1
    if weighted == norm_exp:
        moa_weighted_correct += 1
    if weighted == norm_exp:
        wrong_models = [m for m, v in per_model.items() if v != norm_exp and v]
        if wrong_models:
            moa_recovered.append((pid, exp, wrong_models, "weighted"))


# === Summary ===
n = len(P)
print("\n=== ACCURACY (100 hard prompts, robust extractor) ===")
print(f"  MoA weighted          {moa_weighted_correct:3d}/{n}  ({100*moa_weighted_correct/n:5.1f}%)")
print(f"  MoA simple            {moa_simple_correct:3d}/{n}  ({100*moa_simple_correct/n:5.1f}%)")
for m in MODELS:
    p = single_correct[m]
    print(f"  {m:22s}  {p:3d}/{n}  ({100*p/n:5.1f}%)")

best_single = max(single_correct.values())
print(f"\n  best single: {best_single}/{n}")
print(f"  MoA weighted vs best:  {'+' if moa_weighted_correct >= best_single else ''}{moa_weighted_correct - best_single}")
print(f"  MoA simple   vs best:  {'+' if moa_simple_correct   >= best_single else ''}{moa_simple_correct   - best_single}")

print(f"\n=== Per-category ===")
cats = defaultdict(list)
for pid, _, _, _ in P:
    cats[pid.split("-")[0]].append(pid)
print(f"  {'cat':6s}  {'WMoA':>5s} {'sMoA':>5s}  " + " ".join(f"{m[:8]:>8s}" for m in MODELS))
for cat in sorted(cats):
    pids = cats[cat]
    w = sum(1 for pid in pids if vote_weighted({m:results.get((m,pid),{}).get("norm","") for m in MODELS}, WEIGHTS) == normalize_for_compare(next(p[3] for p in P if p[0]==pid), next(p[2] for p in P if p[0]==pid)))
    s = sum(1 for pid in pids if vote_simple([results.get((m,pid),{}).get("norm","") for m in MODELS if results.get((m,pid),{}).get("norm")]) == normalize_for_compare(next(p[3] for p in P if p[0]==pid), next(p[2] for p in P if p[0]==pid)))
    line = f"  {cat:6s}  {w:>2d}/{len(pids):<2d} {s:>2d}/{len(pids):<2d}"
    for m in MODELS:
        sp = sum(1 for pid in pids if results.get((m,pid),{}).get("correct"))
        line += f"  {sp:>3d}/{len(pids):<3d}"
    print(line)

print(f"\n=== {len(moa_recovered)} cases where weighted MoA fixed individual errors ===")
for pid, exp, wrong, _ in moa_recovered[:15]:
    print(f"  {pid:10s}  exp={exp!r:8s}  fixed: {wrong}")
if len(moa_recovered) > 15:
    print(f"  ... +{len(moa_recovered)-15} more")

with open("/tmp/acp_moa_v2_results.json", "w") as f:
    json.dump({
        "n": n,
        "moa_weighted": moa_weighted_correct,
        "moa_simple": moa_simple_correct,
        "single": single_correct,
        "weights": WEIGHTS,
        "wall_time": t_total,
    }, f, indent=2)
