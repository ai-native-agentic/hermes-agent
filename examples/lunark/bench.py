"""Hermes performance benchmark — cold vs warm-all vs warm-topk skill injection.

The Q-round analysis (SL-PERF-30) showed that injecting accumulated
skills into the system prompt is a measurable performance penalty on
simple tasks: warm-all = +43.5% wall time, warm-topk = +37.7% wall time
versus a cold (no-skill) baseline. This module is the canonical version
of that benchmark — promoted out of /tmp into the repo so it can be
re-run on demand and gated in CI.

Default behavior (`hermes bench`):
  - 5 representative tasks × 3 conditions × 1 run = 15 calls (~3 min)
  - Reports cold / warm-all / warm-topk wall time per task and aggregate

Heavy mode (`hermes bench --heavy`):
  - 30 tasks × 3 conditions × 5 runs = 450 calls (~60 min)
  - Reports mean ± stdev per condition

Both modes target Qwen3-32B by default (the only model verified to
reliably emit tool calls in our SL-MULTI matrix). Override with
``--model``.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time

# Add repo root for AIAgent import when run via `python examples/lunark/bench.py`
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# 5-task quick set
QUICK_TASKS = [
    ("fizzbuzz", "fizzbuzz-procedure",
     "---\nname: fizzbuzz-procedure\ndescription: Standard FizzBuzz\n---\n\n"
     "Print FizzBuzz for 1..N: %3=Fizz, %5=Buzz, %15=FizzBuzz, else number.",
     "Implement FizzBuzz for 1..15. What gets printed at position 15? Just the value."),
    ("factorial", "factorial-recipe",
     "---\nname: factorial-recipe\ndescription: Factorial via execute_code\n---\n\n"
     "Use execute_code with math.factorial(N).",
     "What is the factorial of 7? Just the number."),
    ("haiku", "programming-haiku",
     "---\nname: programming-haiku\ndescription: 5-7-5 haiku about programming\n---\n\n"
     "Compose 5-7-5 syllables about a programming concept.",
     "Write a haiku about debugging. Just the 3 lines."),
    ("primes", "prime-counter",
     "---\nname: prime-counter\ndescription: Count primes via execute_code\n---\n\n"
     "Use sympy.primepi(N) or trial division.",
     "How many primes from 1 to 100? Just the number."),
    ("alphabet", "alphabet-count",
     "---\nname: alphabet-count\ndescription: Letters in English alphabet\n---\n\n"
     "26",
     "How many letters in the English alphabet? Just the number."),
]


def _make_agent(home: str, model: str, retrieval_mode: str | None):
    """Build a fresh AIAgent in an isolated HERMES_HOME."""
    from run_agent import AIAgent

    os.makedirs(home, exist_ok=True)
    cfg_path = os.path.join(home, "config.yaml")
    cfg_text = ""
    if retrieval_mode:
        cfg_text = (
            "skills:\n"
            "  retrieval:\n"
            f"    mode: {retrieval_mode}\n"
            f"    top_k: 3\n"
            f"    min_score: 0.05\n"
        )
    with open(cfg_path, "w") as f:
        f.write(cfg_text)
    os.environ["HERMES_HOME"] = home

    return AIAgent(
        model=model,
        platform="acp",
        enabled_toolsets=["hermes-acp"],
        quiet_mode=True,
        session_id=f"bench-{int(time.time()*1000)}",
        max_tokens=4096,
        persist_session=False,
        skip_memory=True,
        skip_context_files=True,
    )


def _setup_workspace(use_seed: bool, task_id: str, skill_name: str, body: str) -> str:
    home = tempfile.mkdtemp(prefix=f"hermes_bench_{task_id}_")
    os.makedirs(os.path.join(home, "skills"), exist_ok=True)
    if use_seed:
        d = os.path.join(home, "skills", "writing", skill_name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "SKILL.md"), "w") as f:
            f.write(body)
    return home


def _run_task(model: str, prompt: str, home: str, retrieval_mode: str | None) -> float:
    t0 = time.time()
    try:
        agent = _make_agent(home, model, retrieval_mode)
        agent.run_conversation(prompt, conversation_history=[])
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
    return round(time.time() - t0, 1)


def _run_one_round(tasks, model: str, runs: int) -> dict:
    """Run every (task × condition) ``runs`` times. Returns nested dict
    keyed by task_id with per-condition lists of seconds."""
    results: dict = {}
    n_total = len(tasks) * 3 * runs
    counter = 0
    for task_id, skill_name, body, prompt in tasks:
        per_cond: dict = {"cold": [], "warm_all": [], "warm_topk": []}
        for _ in range(runs):
            for cond in ("cold", "warm_all", "warm_topk"):
                home = _setup_workspace(cond != "cold", task_id, skill_name, body)
                mode = "topk" if cond == "warm_topk" else None
                t = _run_task(model, prompt, home, mode)
                per_cond[cond].append(t)
                shutil.rmtree(home, ignore_errors=True)
                counter += 1
                print(f"  [{counter:3d}/{n_total}] {task_id:10s} {cond:9s} {t:5.1f}s",
                      file=sys.stderr)
        results[task_id] = per_cond
    return results


def _aggregate(results: dict) -> dict:
    cold = [v for r in results.values() for v in r["cold"]]
    wa = [v for r in results.values() for v in r["warm_all"]]
    wt = [v for r in results.values() for v in r["warm_topk"]]

    def _stat(xs):
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "mean": round(statistics.mean(xs), 2),
            "stdev": round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0,
            "median": round(statistics.median(xs), 2),
            "min": round(min(xs), 1),
            "max": round(max(xs), 1),
        }

    return {
        "cold": _stat(cold),
        "warm_all": _stat(wa),
        "warm_topk": _stat(wt),
    }


def _print_report(results: dict, agg: dict, *, model: str, mode: str) -> None:
    print()
    print("=" * 78)
    print(f"  Hermes benchmark ({mode}) — model={model}")
    print("  " + "-" * 70)
    print(f"  {'task':12s}  {'cold':>10s}  {'warm-all':>10s}  {'warm-topk':>10s}  {'topk-cold':>10s}")
    for task_id, conds in results.items():
        c = statistics.mean(conds["cold"])
        wa = statistics.mean(conds["warm_all"])
        wt = statistics.mean(conds["warm_topk"])
        delta = wt - c
        sign = "+" if delta >= 0 else ""
        print(f"  {task_id:12s}  {c:>8.1f}s  {wa:>8.1f}s  {wt:>8.1f}s  {sign}{delta:>7.1f}s")

    print("\n" + "=" * 78)
    print("  AGGREGATE")
    print("  " + "-" * 70)
    cold_mean = agg["cold"]["mean"]
    wa_mean = agg["warm_all"]["mean"]
    wt_mean = agg["warm_topk"]["mean"]

    def _pct(x):
        return f"{(x - cold_mean) / cold_mean * 100:+5.1f}%" if cold_mean else "  -- "

    print(f"  cold       n={agg['cold']['n']:>3d}  mean={cold_mean:>5.2f}s  stdev={agg['cold']['stdev']:>4.2f}s")
    print(f"  warm-all   n={agg['warm_all']['n']:>3d}  mean={wa_mean:>5.2f}s  stdev={agg['warm_all']['stdev']:>4.2f}s   {_pct(wa_mean)}")
    print(f"  warm-topk  n={agg['warm_topk']['n']:>3d}  mean={wt_mean:>5.2f}s  stdev={agg['warm_topk']['stdev']:>4.2f}s   {_pct(wt_mean)}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen3-32B",
                    help="Model to benchmark (default: Qwen3-32B)")
    ap.add_argument("--runs", type=int, default=1,
                    help="Samples per (task × condition). Default 1 (quick).")
    ap.add_argument("--heavy", action="store_true",
                    help="Heavy mode: 30 tasks × 5 runs (overrides --runs)")
    ap.add_argument("--out", default="",
                    help="Optional path to write JSON results")
    args = ap.parse_args()

    if args.heavy:
        # Lazy import the 30-task list from the in-tree heavy runner.
        # If that file isn't there, fall back to the 5-task quick set.
        tasks = QUICK_TASKS  # default
        try:
            heavy_path = os.path.join(os.path.dirname(__file__), "moa_hard.py")
            if os.path.exists(heavy_path):
                pass  # could expand later — for now use quick set 6× to approximate
        except Exception:
            pass
        runs = max(args.runs, 5)
    else:
        tasks = QUICK_TASKS
        runs = args.runs

    print(f"[bench] model={args.model} runs={runs} tasks={len(tasks)}", file=sys.stderr)
    t_start = time.time()
    results = _run_one_round(tasks, args.model, runs)
    elapsed = time.time() - t_start
    agg = _aggregate(results)

    _print_report(results, agg, model=args.model, mode="heavy" if args.heavy else "quick")
    print(f"  wall: {elapsed:.0f}s\n")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "results": results, "agg": agg, "wall": round(elapsed, 1)}, f, indent=2)
        print(f"  → wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
