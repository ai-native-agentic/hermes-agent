"""4 models × 100 scenarios = 400 ACP calls.

Reuses the 100 scenarios from acp_batch.py and the per-model isolated
HERMES_HOME profiles. Worker pool of 8 threads spreads calls across all
4 models in parallel.
"""
from __future__ import annotations
import json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["PATH"] = f"{os.environ['HOME']}/.local/bin:" + os.environ.get("PATH", "")

# Reuse the 100 prompts + check function
from batch_qa import P, check  # noqa: E402
from examples.acp_client import HermesACPClient  # noqa: E402

MODELS = ["Qwen3-32B", "Gemma-4-E4B-it", "Qwen2.5-32B-Instruct", "Qwen3.5-27B"]

# Cartesian product: each (model, prompt) is one job
JOBS = [(m, pid, cat, prompt, expected) for m in MODELS for pid, cat, prompt, expected in P]
print(f"[matrix] {len(JOBS)} jobs ({len(MODELS)} models × {len(P)} prompts)", file=sys.stderr)

results: list[dict] = []
results_lock = threading.Lock()
RESULT_FILE = "/tmp/acp_matrix_results.jsonl"
open(RESULT_FILE, "w").close()


def run_one(job):
    model, pid, cat, prompt, expected = job
    home = f"/tmp/hermes_{model}"
    env = os.environ.copy()
    env["HERMES_HOME"] = home
    env["LUNARK_API_KEY"] = "vllm-local"
    t0 = time.time()
    msg, err, stop = "", "", ""
    try:
        client = HermesACPClient(cwd="/tmp", env=env)
        try:
            r = client.prompt(prompt, timeout=120)
            msg = r.message
            stop = r.stop_reason
        finally:
            client.close()
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
    elapsed = round(time.time() - t0, 1)
    passed = check(pid, expected, msg) if msg else False
    rec = {
        "model": model, "id": pid, "cat": cat,
        "passed": passed, "elapsed": elapsed, "stop": stop,
        "err": err, "expected": expected,
        "msg_tail": msg.strip()[-100:],
    }
    with results_lock:
        results.append(rec)
        with open(RESULT_FILE, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  [{len(results):3d}/{len(JOBS)}] {model:22s} {pid:12s} {'PASS' if passed else 'FAIL'} {elapsed:5.1f}s", file=sys.stderr)


t_start = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(run_one, JOBS))
t_total = time.time() - t_start

print(f"\n[matrix] DONE in {t_total:.1f}s", file=sys.stderr)


# === Summary ===
def aggregate(rs):
    p = sum(1 for r in rs if r["passed"])
    e = sum(1 for r in rs if r["err"])
    avg = sum(r["elapsed"] for r in rs) / len(rs) if rs else 0
    return p, len(rs), e, avg


# Per model
print("\n=== BY MODEL ===")
for m in MODELS:
    rs = [r for r in results if r["model"] == m]
    p, n, e, avg = aggregate(rs)
    print(f"  {m:22s}  {p:3d}/{n}  errors={e}  avg {avg:5.1f}s")

# Per category × model
print("\n=== BY CATEGORY × MODEL ===")
cats = sorted(set(r["cat"] for r in results))
header = f"  {'category':12s}  " + "  ".join(f"{m[:10]:>10s}" for m in MODELS)
print(header)
print("  " + "-" * (len(header) - 2))
for cat in cats:
    line = f"  {cat:12s}  "
    for m in MODELS:
        rs = [r for r in results if r["model"] == m and r["cat"] == cat]
        p, n, _, _ = aggregate(rs)
        line += f"  {p:>2d}/{n:<2d}     "
    print(line)

# Total
total_pass = sum(1 for r in results if r["passed"])
total_err = sum(1 for r in results if r["err"])
print(f"\n=== TOTAL ===")
print(f"  jobs: {len(results)}")
print(f"  pass: {total_pass}/{len(results)}  ({100*total_pass/len(results):.1f}%)")
print(f"  errors: {total_err}")
print(f"  wall time: {t_total:.0f}s")
print(f"  avg per call: {sum(r['elapsed'] for r in results)/len(results):.1f}s")
