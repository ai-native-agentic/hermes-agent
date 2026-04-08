"""Variance measurement: run the 100 hard prompts 5 times against each model.

Goal: quantify how much each model's accuracy varies run-to-run on the
same prompts. Sampling temperature in the lunark vLLM defaults can flip
borderline answers, and earlier runs of moa_hard.py showed 1 vs 2
failures from "the same" 100 prompts.

Output: per-model accuracy distribution (mean ± stdev), which prompts
flip across runs (= unstable), and which prompts every model gets
right/wrong every time (= stable).
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

# Reuse moa_hard's prompt list via the same exec trick
_moa_hard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moa_hard.py")
with open(_moa_hard_path) as _f:
    _src = _f.read()
_cut = _src.index('print(f"[v2]')
_ns = {"__file__": _moa_hard_path, "__name__": "moa_hard_loader"}
exec(compile(_src[:_cut], _moa_hard_path, "exec"), _ns)
P = _ns["P"]

MODELS = ["Qwen3-32B", "Gemma-4-E4B-it", "Qwen2.5-32B-Instruct", "Qwen3.5-27B"]
N_RUNS = 5

EXTRACTORS = {
    "number": extract_number,
    "fraction": extract_fraction,
    "yesno": extract_yesno,
    "word": extract_word,
}

print(f"[var] {len(P)} prompts × {len(MODELS)} models × {N_RUNS} runs = {len(P)*len(MODELS)*N_RUNS} jobs", file=sys.stderr)

JOBS = []
for run in range(N_RUNS):
    for m in MODELS:
        for pid, prompt, kind, exp in P:
            JOBS.append((run, m, pid, prompt, kind, exp))

results = {}  # (run, model, pid) → bool correct
results_lock = threading.Lock()
done = 0
RESULT_FILE = "/tmp/acp_variance_results.jsonl"
open(RESULT_FILE, "w").close()


def run_one(job):
    global done
    run, model, pid, prompt, kind, exp = job
    home = f"/tmp/hermes_{model}"
    env = os.environ.copy()
    env["HERMES_HOME"] = home
    env["LUNARK_API_KEY"] = "vllm-local"
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
    extractor = EXTRACTORS.get(kind, extract_number)
    extracted = ""
    try:
        extracted = extractor(msg) if msg else ""
    except Exception:
        extracted = ""
    norm_exp = normalize_for_compare(exp, kind)
    norm_got = normalize_for_compare(extracted, kind)
    correct = norm_got == norm_exp
    rec = {"run": run, "model": model, "id": pid, "correct": correct,
           "extracted": extracted, "expected": exp}
    with results_lock:
        results[(run, model, pid)] = correct
        done += 1
        with open(RESULT_FILE, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if done % 100 == 0:
        print(f"  [{done}/{len(JOBS)}]", file=sys.stderr)


t_start = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(run_one, JOBS))
t_total = time.time() - t_start

print(f"\n[var] DONE in {t_total:.1f}s", file=sys.stderr)


# === Analysis ===
import statistics

# Per (model, run) accuracy
print("\n=== ACCURACY PER RUN ===")
print(f"  {'model':24s}  " + "  ".join(f"r{r}" for r in range(N_RUNS)) + "    mean   stdev")
per_model_runs = {}
for m in MODELS:
    accs = []
    for r in range(N_RUNS):
        c = sum(1 for pid, _, _, _ in P if results.get((r, m, pid)))
        accs.append(c)
    per_model_runs[m] = accs
    mean = statistics.mean(accs)
    stdev = statistics.stdev(accs) if len(accs) > 1 else 0
    print(f"  {m:24s}  " + "  ".join(f"{a:3d}" for a in accs) + f"   {mean:5.1f}   {stdev:5.2f}")

# Stable vs flippy prompts
stable_correct = []  # always right across all runs and all models
stable_wrong = []    # always wrong across all runs and all models
flippy = defaultdict(int)  # (model, pid) → number of distinct outcomes (1=stable, 2=flips)

for pid, _, _, _ in P:
    all_outcomes = []
    for m in MODELS:
        for r in range(N_RUNS):
            all_outcomes.append(results.get((r, m, pid), False))
    if all(all_outcomes):
        stable_correct.append(pid)
    elif not any(all_outcomes):
        stable_wrong.append(pid)

# Per-prompt model flip counts
prompt_flips = []  # (pid, total flips across models)
for pid, _, _, _ in P:
    flips = 0
    for m in MODELS:
        runs = [results.get((r, m, pid), False) for r in range(N_RUNS)]
        if len(set(runs)) > 1:
            flips += 1
    if flips:
        prompt_flips.append((pid, flips))

prompt_flips.sort(key=lambda x: -x[1])

print(f"\n=== STABILITY ===")
print(f"  prompts always right (all 4 models, all {N_RUNS} runs): {len(stable_correct)}/{len(P)}")
print(f"  prompts always wrong (all 4 models, all {N_RUNS} runs): {len(stable_wrong)}")
print(f"  prompts where ≥1 model flipped between runs: {len(prompt_flips)}")

print(f"\n=== TOP UNSTABLE PROMPTS (most models flipping) ===")
for pid, flips in prompt_flips[:15]:
    line = f"  {pid:10s}  flips={flips}/{len(MODELS)}  "
    for m in MODELS:
        runs = [results.get((r, m, pid), False) for r in range(N_RUNS)]
        marks = "".join("✓" if c else "✗" for c in runs)
        line += f"  {m[:8]}={marks}"
    print(line)

print(f"\n  wall: {t_total:.0f}s")

with open("/tmp/acp_variance_summary.json", "w") as f:
    json.dump({
        "n_prompts": len(P),
        "n_runs": N_RUNS,
        "n_models": len(MODELS),
        "wall_time": round(t_total, 1),
        "per_model_runs": per_model_runs,
        "per_model_mean": {m: statistics.mean(per_model_runs[m]) for m in MODELS},
        "per_model_stdev": {m: (statistics.stdev(per_model_runs[m]) if len(per_model_runs[m]) > 1 else 0) for m in MODELS},
        "stable_correct_count": len(stable_correct),
        "stable_wrong_count": len(stable_wrong),
        "unstable_count": len(prompt_flips),
        "top_unstable": prompt_flips[:20],
    }, f, indent=2)
