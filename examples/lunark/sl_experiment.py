"""Longitudinal self-learning performance experiment.

Asks the *real* question of self-learning: when the agent runs a sequence
of similar tasks, do the closed loops we built actually make later tasks
faster / more accurate / use fewer tool calls than earlier ones?

The other benchmarks in this directory (bench.py, sl_perf_30) test the
single-call cold vs warm-skill-injection axis. They cannot answer the
self-learning question because they reset state between runs. This file
keeps state and measures the curve.

EXPERIMENT DESIGN
─────────────────

  5 task families × 4 sequential variants × 3 conditions = 60 calls

  Conditions:
    - baseline: all closed loops OFF (skills inject all, no metrics
      boost, no error lessons, no hallucination retry)
    - partial:  skill retrieval top-k + skill metrics boost (A2)
    - full:     partial + tool-error lessons (B1) + capability gate +
                hallucination retry (A3)

  For each (condition, family) we:
    1. Provision a fresh isolated HERMES_HOME directory with the
       condition's config.yaml
    2. Run variant 1 → 2 → 3 → 4 sequentially against the SAME home so
       skills, lessons, and metrics accumulate
    3. Record per-variant: elapsed_s, tool_calls, response_len, correct?
    4. Compute the "learning slope" = mean(v3,v4) - mean(v1,v2). A
       negative slope means later runs are faster (improvement).

  Output: human-readable table + JSONL of every call for re-analysis.

USAGE
─────

  python examples/lunark/sl_experiment.py                # default 1 run
  python examples/lunark/sl_experiment.py --runs 3       # statistical
  python examples/lunark/sl_experiment.py --out exp.json # persist

  Time: ~10 min for 1 run, ~30 min for 3 runs.
"""
from __future__ import annotations
import argparse, json, os, shutil, statistics, sys, tempfile, time
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ─── Task families ────────────────────────────────────────────────────
# Each family has 4 variants. Variants share the same SHAPE so any skill
# the agent saves on variant 1 should help on variants 2-4.

@dataclass
class TaskVariant:
    prompt: str
    check: Callable[[str], bool]


@dataclass
class TaskFamily:
    family_id: str
    variants: List[TaskVariant]


def _has(*words):
    return lambda r: all(w.lower() in r.lower() for w in words)

def _has_num(target):
    return lambda r: str(target) in r


FAMILIES: List[TaskFamily] = [
    TaskFamily("math", [
        TaskVariant("What is 17*23? Reply with just the number.", _has_num(391)),
        TaskVariant("What is 19*27? Reply with just the number.", _has_num(513)),
        TaskVariant("What is 23*31? Reply with just the number.", _has_num(713)),
        TaskVariant("What is 29*37? Reply with just the number.", _has_num(1073)),
    ]),
    TaskFamily("file", [
        TaskVariant("Use the terminal tool to count files in /tmp directory. Reply with just the number.",
                    lambda r: any(c.isdigit() for c in r)),
        TaskVariant("Use terminal to count files in /etc. Reply with just the number.",
                    lambda r: any(c.isdigit() for c in r)),
        TaskVariant("Use terminal to count .py files under /home/kjs/projects/hermes-agent/agent/. Reply with just the number.",
                    lambda r: any(c.isdigit() for c in r)),
        TaskVariant("Use terminal to count .md files under /home/kjs/projects/hermes-agent/. Reply with just the number.",
                    lambda r: any(c.isdigit() for c in r)),
    ]),
    TaskFamily("code", [
        TaskVariant("In Python, what does len('hello') return? Just the number.", _has_num(5)),
        TaskVariant("In Python, what does len('foobar') return? Just the number.", _has_num(6)),
        TaskVariant("In Python, what does len('abcdefgh') return? Just the number.", _has_num(8)),
        TaskVariant("In Python, what does len('a' * 12) return? Just the number.", _has_num(12)),
    ]),
    TaskFamily("text", [
        TaskVariant("Reverse the string 'hello'. Just the result.", _has("olleh")),
        TaskVariant("Reverse the string 'world'. Just the result.", _has("dlrow")),
        TaskVariant("Reverse the string 'python'. Just the result.", _has("nohtyp")),
        TaskVariant("Reverse the string 'agent'. Just the result.", _has("tnega")),
    ]),
    TaskFamily("trivia", [
        TaskVariant("Capital of France? One word.", _has("paris")),
        TaskVariant("Capital of Japan? One word.", _has("tokyo")),
        TaskVariant("Capital of Germany? One word.", _has("berlin")),
        TaskVariant("Capital of Italy? One word.", _has("rome")),
    ]),
]


# ─── Conditions ───────────────────────────────────────────────────────
# Each condition is a config.yaml fragment that gets dropped into the
# isolated HERMES_HOME for that cell.

CONDITIONS = {
    "baseline": """\
model:
  default: Qwen3-32B
  provider: lunark
  base_url: https://llm.lunark.ai/v1
  api_key: vllm-local
  context_length: 32768
  max_tokens: 4096
providers: {}
fallback_providers: []
toolsets: [hermes-acp]
agent:
  max_turns: 60
  reasoning_effort: medium
  tool_use_enforcement: auto
  error_lessons:
    enabled: false
skills:
  retrieval:
    mode: all
""",
    "partial": """\
model:
  default: Qwen3-32B
  provider: lunark
  base_url: https://llm.lunark.ai/v1
  api_key: vllm-local
  context_length: 32768
  max_tokens: 4096
providers: {}
fallback_providers: []
toolsets: [hermes-acp]
agent:
  max_turns: 60
  reasoning_effort: medium
  tool_use_enforcement: auto
  error_lessons:
    enabled: false
skills:
  retrieval:
    mode: topk
    top_k: 3
    min_score: 0.05
""",
    "full": """\
model:
  default: Qwen3-32B
  provider: lunark
  base_url: https://llm.lunark.ai/v1
  api_key: vllm-local
  context_length: 32768
  max_tokens: 4096
providers: {}
fallback_providers: []
toolsets: [hermes-acp]
agent:
  max_turns: 60
  reasoning_effort: medium
  tool_use_enforcement: auto
  error_lessons:
    enabled: true
skills:
  retrieval:
    mode: topk
    top_k: 3
    min_score: 0.05
moa:
  provider: custom
  base_url: https://llm.lunark.ai/v1
  api_key_env: LUNARK_API_KEY
  reference_models:
    - Gemma-4-31B-it
    - Qwen2.5-32B-Instruct
    - Qwen3.5-27B
  aggregator_model: Qwen3-32B
  enable_reasoning: false
""",
}


# ─── Runner ───────────────────────────────────────────────────────────

@dataclass
class VariantResult:
    family: str
    condition: str
    variant_idx: int
    elapsed: float
    tool_calls: int
    correct: bool
    response_len: int
    msg: str


def _make_agent(home: str):
    from run_agent import AIAgent
    os.environ["HERMES_HOME"] = home
    os.environ["LUNARK_API_KEY"] = "vllm-local"
    return AIAgent(
        base_url="https://llm.lunark.ai/v1",
        api_key="vllm-local",
        provider="lunark",
        api_mode="chat_completions",
        model="Qwen3-32B",
        platform="acp",
        enabled_toolsets=["hermes-acp"],
        quiet_mode=True,
        session_id=f"sl-exp-{int(time.time()*1000)}",
        max_tokens=4096,
        persist_session=False,
        skip_memory=True,
        skip_context_files=True,
    )


def run_variant(home: str, prompt: str) -> tuple[float, int, str]:
    """Execute one variant against the given HERMES_HOME, return (elapsed, tool_calls, response)."""
    t0 = time.time()
    msg = ""
    tool_calls = 0
    try:
        agent = _make_agent(home)
        result = agent.run_conversation(prompt, conversation_history=[])
        msg = result.get("final_response", "") or ""
        tool_calls = sum(
            len(m.get("tool_calls") or [])
            for m in result.get("messages", [])
            if isinstance(m, dict) and m.get("role") == "assistant"
        )
    except Exception as exc:
        msg = f"ERROR: {exc}"
    return round(time.time() - t0, 1), tool_calls, msg


def setup_home(condition: str, prefix: str) -> str:
    home = tempfile.mkdtemp(prefix=f"sl_exp_{prefix}_")
    os.makedirs(os.path.join(home, "skills"), exist_ok=True)
    with open(os.path.join(home, "config.yaml"), "w") as f:
        f.write(CONDITIONS[condition])
    return home


def run_one_cell(family: TaskFamily, condition: str) -> List[VariantResult]:
    """Run all 4 variants of a family in sequence under one condition,
    sharing the same isolated HERMES_HOME so closed loops can accumulate."""
    home = setup_home(condition, f"{family.family_id}_{condition}")
    out: List[VariantResult] = []
    for i, variant in enumerate(family.variants):
        elapsed, tool_calls, msg = run_variant(home, variant.prompt)
        try:
            correct = bool(variant.check(msg))
        except Exception:
            correct = False
        out.append(VariantResult(
            family=family.family_id,
            condition=condition,
            variant_idx=i,
            elapsed=elapsed,
            tool_calls=tool_calls,
            correct=correct,
            response_len=len(msg.strip()),
            msg=msg.strip()[-100:],
        ))
    shutil.rmtree(home, ignore_errors=True)
    return out


def run_experiment(families, conditions, runs: int, out_path: str = "") -> dict:
    all_results: List[VariantResult] = []
    n_total = len(families) * len(conditions) * runs * 4
    counter = 0

    if out_path:
        open(out_path, "w").close()

    for run_idx in range(runs):
        for family in families:
            for condition in conditions:
                cell_results = run_one_cell(family, condition)
                for r in cell_results:
                    counter += 1
                    print(f"  [{counter:3d}/{n_total}] {family.family_id:7s} "
                          f"{condition:8s} v{r.variant_idx} "
                          f"{r.elapsed:5.1f}s tc={r.tool_calls} {'✓' if r.correct else '✗'}",
                          file=sys.stderr)
                    if out_path:
                        with open(out_path, "a") as f:
                            f.write(json.dumps(asdict(r)) + "\n")
                all_results.extend(cell_results)

    return summarize(all_results)


def summarize(results: List[VariantResult]) -> dict:
    """Per (family, condition): mean elapsed, mean tool_calls, accuracy,
    learning slope (later half mean - earlier half mean)."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r.family, r.condition)].append(r)

    out = {}
    for (family, condition), rs in grouped.items():
        rs.sort(key=lambda r: r.variant_idx)
        elapsed = [r.elapsed for r in rs]
        tcalls = [r.tool_calls for r in rs]
        correct = sum(1 for r in rs if r.correct)
        # Learning slope: later half - earlier half (negative = improving)
        if len(rs) >= 4:
            half = len(rs) // 2
            early = statistics.mean(elapsed[:half]) if elapsed[:half] else 0
            late = statistics.mean(elapsed[half:]) if elapsed[half:] else 0
            slope = round(late - early, 2)
        else:
            slope = 0.0
        out[f"{family}|{condition}"] = {
            "n": len(rs),
            "mean_elapsed": round(statistics.mean(elapsed), 2),
            "stdev_elapsed": round(statistics.stdev(elapsed), 2) if len(elapsed) > 1 else 0,
            "mean_tool_calls": round(statistics.mean(tcalls), 2),
            "accuracy": correct / len(rs),
            "learning_slope": slope,
            "samples": elapsed,
        }
    return out


def print_report(summary: dict, conditions: list, families: list):
    print("\n" + "=" * 100)
    print("  Self-Learning Experiment — accuracy / mean elapsed / learning slope")
    print("  (slope < 0 = later variants faster than earlier; slope > 0 = no improvement / slowdown)")
    print("  " + "-" * 96)
    header = f"  {'family':10s}  " + "  ".join(f"{c:>22s}" for c in conditions)
    print(header)
    print("  " + "-" * 96)
    for f in families:
        row = f"  {f.family_id:10s}  "
        for c in conditions:
            stats = summary.get(f"{f.family_id}|{c}")
            if stats:
                acc = f"{int(stats['accuracy']*100):3d}%"
                avg = f"{stats['mean_elapsed']:5.1f}s"
                slope = f"{stats['learning_slope']:+.1f}"
                row += f"  {acc} {avg} slope={slope}"
            else:
                row += f"  {'-':>22s}"
        print(row)
    print()


def print_aggregate(summary: dict, conditions: list):
    """Cross-family aggregate per condition."""
    from collections import defaultdict
    bycond = defaultdict(lambda: {"elapsed": [], "tcalls": [], "correct": 0, "n": 0, "slopes": []})
    for key, stats in summary.items():
        family, condition = key.split("|")
        bycond[condition]["elapsed"].extend(stats["samples"])
        bycond[condition]["correct"] += int(stats["accuracy"] * stats["n"])
        bycond[condition]["n"] += stats["n"]
        bycond[condition]["slopes"].append(stats["learning_slope"])
    print("  " + "=" * 96)
    print(f"  {'AGGREGATE':10s}")
    print("  " + "-" * 96)
    for c in conditions:
        b = bycond[c]
        if not b["n"]:
            continue
        mean_e = statistics.mean(b["elapsed"])
        std_e = statistics.stdev(b["elapsed"]) if len(b["elapsed"]) > 1 else 0
        acc = b["correct"] / b["n"]
        mean_slope = statistics.mean(b["slopes"]) if b["slopes"] else 0
        print(f"  {c:10s}  n={b['n']:>3d}  mean={mean_e:5.2f}s  "
              f"stdev={std_e:4.2f}s  acc={int(acc*100):3d}%  "
              f"avg_slope={mean_slope:+5.2f}s")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=1, help="Repeats of the full matrix")
    ap.add_argument("--out", default="", help="Optional JSONL path for raw results")
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: only baseline + full (2 conditions × ...)")
    ap.add_argument("--families", default="",
                    help="Comma-separated family ids to include (default: all)")
    args = ap.parse_args()

    conditions = ["baseline", "full"] if args.quick else list(CONDITIONS.keys())
    families = FAMILIES
    if args.families:
        wanted = {f.strip() for f in args.families.split(",")}
        families = [f for f in FAMILIES if f.family_id in wanted]
    if not families:
        print("No families selected", file=sys.stderr)
        sys.exit(1)

    n_calls = len(families) * len(conditions) * args.runs * 4
    print(f"[sl-exp] {len(families)} families × {len(conditions)} conditions × "
          f"{args.runs} runs × 4 variants = {n_calls} calls", file=sys.stderr)
    t0 = time.time()
    summary = run_experiment(families, conditions, args.runs, args.out)
    elapsed = time.time() - t0
    print_report(summary, conditions, families)
    print_aggregate(summary, conditions)
    print(f"  wall: {elapsed:.0f}s")

    if args.out:
        # Also write the summary JSON next to the JSONL
        sumpath = args.out.replace(".jsonl", "_summary.json")
        with open(sumpath, "w") as f:
            json.dump({
                "args": vars(args),
                "wall": round(elapsed, 1),
                "summary": summary,
            }, f, indent=2)
        print(f"  → {args.out}")
        print(f"  → {sumpath}")


if __name__ == "__main__":
    main()
