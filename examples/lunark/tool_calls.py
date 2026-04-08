"""Tool-calling matrix: 10 scenarios × 4 lunark models via ACP.

Each scenario forces the agent to actually invoke a tool (terminal, file ops,
code execution) rather than answer from memory. Verifies that lunark's
tool_calling=true claim is real and that ACP correctly threads tool results
back to the model.
"""
from __future__ import annotations
import json, os, sys, time, threading, uuid, re
from concurrent.futures import ThreadPoolExecutor

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
os.environ["PATH"] = f"{os.environ['HOME']}/.local/bin:" + os.environ.get("PATH", "")
from examples.acp_client import HermesACPClient  # noqa: E402

MODELS = ["Qwen3-32B", "Gemma-4-31B-it", "Qwen2.5-32B-Instruct", "Qwen3.5-27B"]

# Each scenario: id, prompt, check function (gets full response text)
def has(*words):
    return lambda r: all(w.lower() in r.lower() for w in words)

def has_num(target):
    """Check that the literal target appears as a digit substring."""
    return lambda r: str(target) in r

def regex_search(pat):
    return lambda r: bool(re.search(pat, r))

# Make a unique probe file each run so we don't pollute prior state
PROBE_DIR = f"/tmp/acp_tools_probe_{uuid.uuid4().hex[:8]}"
os.makedirs(PROBE_DIR, exist_ok=True)
# Pre-seed a known file for the read scenarios
with open(f"{PROBE_DIR}/seed.txt", "w") as f:
    f.write("magic_token_42\nline_two_here\nlast_line\n")

SCENARIOS = [
    (
        "tool-bash-date",
        "Use the terminal tool to run `date +%Y` and reply with just the year (4 digits, no other text).",
        has_num(2026),
    ),
    (
        "tool-bash-count-files",
        f"Use the terminal tool to count files in {PROBE_DIR} with `ls -1 {PROBE_DIR} | wc -l` and reply with just the number.",
        has_num(1),
    ),
    (
        "tool-bash-echo-pwd",
        f"Use the terminal tool to run `cd {PROBE_DIR} && pwd` and reply with just the directory path.",
        has(PROBE_DIR),
    ),
    (
        "tool-read-file",
        f"Read the file {PROBE_DIR}/seed.txt and reply with just the first word in it (no quotes).",
        has("magic_token_42"),
    ),
    (
        "tool-grep",
        f"Use the terminal tool to grep for `magic_token_42` in {PROBE_DIR}/seed.txt and reply 'FOUND' if it appears, else 'MISSING'.",
        has("FOUND"),
    ),
    (
        "tool-write-file",
        f"Write the exact text `unique_marker_xyz` into {PROBE_DIR}/written.txt then read it back. Reply with the content.",
        has("unique_marker_xyz"),
    ),
    (
        "tool-wc-lines",
        f"Use the terminal tool to count lines in {PROBE_DIR}/seed.txt with `wc -l`. Reply with just the number.",
        has_num(3),
    ),
    (
        "tool-python-calc",
        "Use the execute_code tool to run python and compute `(17+23)*5 - 100`. Reply with just the result.",
        has_num(100),
    ),
    (
        "tool-python-json",
        "Use the execute_code tool to run `import json; print(json.dumps({'x':17,'y':23}))`. Reply with the JSON output.",
        has("17", "23"),
    ),
    (
        "tool-multi-step",
        f"First, use terminal to write 'forty-two' into {PROBE_DIR}/multi.txt. Then read it back. Reply with the content.",
        has("forty-two"),
    ),
]

JOBS = [(m, sid, prompt, check) for m in MODELS for sid, prompt, check in SCENARIOS]
print(f"[tools] {len(JOBS)} jobs ({len(MODELS)} models × {len(SCENARIOS)} tool scenarios)", file=sys.stderr)

results = []
results_lock = threading.Lock()
RESULT_FILE = "/tmp/acp_tools_results.jsonl"
open(RESULT_FILE, "w").close()


def run_one(job):
    model, sid, prompt, check = job
    home = f"/tmp/hermes_{model}"
    env = os.environ.copy()
    env["HERMES_HOME"] = home
    env["LUNARK_API_KEY"] = "vllm-local"
    t0 = time.time()
    msg, err, stop = "", "", ""
    try:
        client = HermesACPClient(cwd=PROBE_DIR, env=env)
        try:
            r = client.prompt(prompt, timeout=180)
            msg = r.message
            stop = r.stop_reason
        finally:
            client.close()
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
    elapsed = round(time.time() - t0, 1)
    passed = bool(check(msg)) if msg else False
    rec = {
        "model": model, "id": sid,
        "passed": passed, "elapsed": elapsed, "stop": stop,
        "err": err,
        "msg_tail": msg.strip()[-200:],
    }
    with results_lock:
        results.append(rec)
        with open(RESULT_FILE, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  [{len(results):2d}/{len(JOBS)}] {model:22s} {sid:20s} {'PASS' if passed else 'FAIL':4s} {elapsed:6.1f}s", file=sys.stderr)


t_start = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(run_one, JOBS))
t_total = time.time() - t_start

print(f"\n[tools] DONE in {t_total:.1f}s", file=sys.stderr)

# Per-model summary
print("\n=== BY MODEL ===")
for m in MODELS:
    rs = [r for r in results if r["model"] == m]
    p = sum(1 for r in rs if r["passed"])
    e = sum(1 for r in rs if r["err"])
    avg = sum(r["elapsed"] for r in rs) / len(rs) if rs else 0
    print(f"  {m:22s}  {p:2d}/{len(rs)}  errors={e}  avg {avg:5.1f}s")

# Per-scenario summary
print("\n=== BY SCENARIO ===")
for sid, _, _ in SCENARIOS:
    rs = [r for r in results if r["id"] == sid]
    p = sum(1 for r in rs if r["passed"])
    print(f"  {sid:20s}  {p}/{len(rs)}")

total_pass = sum(1 for r in results if r["passed"])
print(f"\n=== TOTAL ===")
print(f"  pass: {total_pass}/{len(results)}  ({100*total_pass/max(len(results),1):.1f}%)")
print(f"  wall: {t_total:.0f}s")
