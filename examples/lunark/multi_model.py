"""Run the same prompts against 4 lunark models in parallel via 4 ACP instances.

Each instance uses an isolated HERMES_HOME so it picks up its own config.yaml
(which pins a different default model). All four hermes acp processes run
concurrently and we collect their responses.
"""
from __future__ import annotations
import os, sys, time, json
from concurrent.futures import ThreadPoolExecutor, as_completed

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
os.environ["PATH"] = f"{os.environ['HOME']}/.local/bin:" + os.environ.get("PATH", "")
from examples.acp_client import HermesACPClient

MODELS = ["Qwen3-32B", "Gemma-4-31B-it", "Qwen2.5-32B-Instruct", "Qwen3.5-27B"]

PROMPTS = [
    ("math",  "What is 17*23? Reply with just the number."),
    ("logic", "If today is Monday, what day is it 10 days from now? Just the day name."),
    ("know",  "Capital of France? One word."),
    ("code",  "In Python, what does len('hello') return? Just the number."),
    ("trans", "Translate '안녕' to English. One word."),
]

def run_model(model: str) -> dict:
    home = f"/tmp/hermes_{model}"
    env = os.environ.copy()
    env["HERMES_HOME"] = home
    env["LUNARK_API_KEY"] = "vllm-local"

    out = {"model": model, "results": [], "errors": []}
    t_model = time.time()
    for tag, prompt in PROMPTS:
        t0 = time.time()
        try:
            client = HermesACPClient(cwd="/tmp", env=env)
            try:
                r = client.prompt(prompt, timeout=120)
                msg_full = r.message.strip()
                out["results"].append({
                    "tag": tag,
                    "prompt": prompt,
                    "msg_len": len(msg_full),
                    "msg_tail": msg_full[-150:],
                    "thought_len": len(r.thought),
                    "stop": r.stop_reason,
                    "elapsed": round(time.time() - t0, 1),
                })
            finally:
                client.close()
        except Exception as e:
            out["errors"].append({"tag": tag, "err": f"{type(e).__name__}: {e}"})
    out["total_elapsed"] = round(time.time() - t_model, 1)
    return out


print(f"[multi] launching {len(MODELS)} parallel hermes acp instances", file=sys.stderr)
t_start = time.time()
all_results = {}
with ThreadPoolExecutor(max_workers=len(MODELS)) as ex:
    futs = {ex.submit(run_model, m): m for m in MODELS}
    for f in as_completed(futs):
        res = f.result()
        all_results[res["model"]] = res
        print(f"[multi] done: {res['model']:20s} {res['total_elapsed']}s", file=sys.stderr)

t_total = time.time() - t_start
print(f"\n[multi] all done in {t_total:.1f}s\n", file=sys.stderr)

with open("/tmp/multi_model_results.json", "w") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

# Pretty table
print("\n=== RESULTS ===\n")
for tag, prompt in PROMPTS:
    print(f"\n[{tag}] {prompt}")
    for m in MODELS:
        rs = all_results.get(m, {}).get("results", [])
        match = next((r for r in rs if r["tag"] == tag), None)
        if match:
            print(f"  {m:20s} {match['elapsed']:5.1f}s  msg_len={match['msg_len']:5d}  tail→ {match['msg_tail'][-100:]!r}")
        else:
            print(f"  {m:20s} ERROR")

print("\n=== TOTAL TIMES ===")
for m in MODELS:
    t = all_results.get(m, {}).get("total_elapsed", "?")
    print(f"  {m:20s} {t}s")
print(f"  WALL TIME (parallel): {t_total:.1f}s")
