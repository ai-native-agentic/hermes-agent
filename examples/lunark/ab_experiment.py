"""A/B Experiment Framework — run any UC scenario under multiple conditions.

Takes any test scenario (from uc_test.py, uc_industry.py, uc_domains.py or
inline) and runs it under 4 conditions to compare self-learning features:

  baseline     — no SL features, mode=all
  topk         — skill retrieval top-k only
  full_sl      — topk + error_lessons + hallucination retry
  model_routing — auto-route simple→Qwen3.5-27B, tools→Qwen3-32B

Produces a markdown report with environment fingerprint, per-scenario ×
condition matrix, statistical summary, and recommendation.

USAGE
─────

  python ab_experiment.py --scenario "What is 17*23? Just the number." --check "391" --runs 3
  python ab_experiment.py --from-jsonl /tmp/uc_test.jsonl --runs 2
  python ab_experiment.py --from-jsonl /tmp/uc_domains.jsonl --conditions baseline,full_sl --runs 1
"""
from __future__ import annotations
import argparse, json, os, platform, shutil, statistics, sys, tempfile, time
from dataclasses import dataclass, asdict, field
from typing import Callable, List, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ─── Conditions ───────────────────────────────────────────────────────────────

CONDITIONS: dict[str, str] = {
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
    "topk": """\
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
    "full_sl": """\
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
  hallucination_retry:
    enabled: true
    max_retries: 2
skills:
  retrieval:
    mode: topk
    top_k: 3
    min_score: 0.05
""",
    "model_routing": """\
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
  model_routing:
    enabled: true
    simple_model: Qwen3.5-27B
    tool_model: Qwen3-32B
skills:
  retrieval:
    mode: topk
    top_k: 3
    min_score: 0.05
""",
}

ALL_CONDITION_NAMES = list(CONDITIONS)


# ─── Scenario ─────────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    scenario_id: str
    prompt: str
    check: Callable[[str], bool]


def _check_from_jsonl(row: dict) -> Callable[[str], bool]:
    """Build a checker from a JSONL row.

    Supported fields (in priority order):
      check_contains   — substring(s) that must appear (str or list)
      check_substring  — alias for check_contains
      check            — substring / Python expression string
      passed           — not used (this is a raw result row); fallback: always True
    """
    for key in ("check_contains", "check_substring"):
        val = row.get(key)
        if val is not None:
            targets = [val] if isinstance(val, str) else list(val)
            return lambda r, t=targets: all(x.lower() in r.lower() for x in t)

    chk = row.get("check")
    if chk is not None:
        needle = str(chk)
        return lambda r, n=needle: n.lower() in r.lower()

    return lambda r: True


def load_scenarios_from_jsonl(path: str) -> List[Scenario]:
    """Load scenarios from a JSONL file.

    Each line may be:
      - An existing TestResult row (has 'prompt', optional 'test_id'/'uc')
      - A raw scenario row (has 'prompt', optional check fields)
    """
    scenarios: List[Scenario] = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [ab-exp] JSONL line {lineno} parse error: {exc}", file=sys.stderr)
                continue
            prompt = row.get("prompt", "")
            if not prompt:
                continue
            sid = row.get("test_id") or row.get("scenario_id") or row.get("uc") or f"s{lineno}"
            check = _check_from_jsonl(row)
            scenarios.append(Scenario(scenario_id=str(sid), prompt=prompt, check=check))
    return scenarios


# ─── Runner ───────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    scenario_id: str
    condition: str
    run_idx: int
    elapsed: float
    tool_calls: int
    response_len: int
    passed: bool
    response_snippet: str


def _setup_home(condition: str, tag: str) -> str:
    home = tempfile.mkdtemp(prefix=f"ab_exp_{tag}_")
    os.makedirs(os.path.join(home, "skills"), exist_ok=True)
    with open(os.path.join(home, "config.yaml"), "w") as fh:
        fh.write(CONDITIONS[condition])
    return home


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
        session_id=f"ab-exp-{int(time.time()*1000)}",
        max_tokens=4096,
        persist_session=False,
        skip_memory=True,
        skip_context_files=True,
    )


def _run_once(home: str, prompt: str) -> tuple[float, int, str]:
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


def run_cell(scenario: Scenario, condition: str, run_idx: int) -> RunResult:
    """One (scenario, condition, run) cell — fresh isolated HERMES_HOME per run."""
    tag = f"{scenario.scenario_id}_{condition}_{run_idx}"
    home = _setup_home(condition, tag)
    try:
        elapsed, tool_calls, msg = _run_once(home, scenario.prompt)
    finally:
        shutil.rmtree(home, ignore_errors=True)

    try:
        passed = bool(scenario.check(msg))
    except Exception:
        passed = False

    return RunResult(
        scenario_id=scenario.scenario_id,
        condition=condition,
        run_idx=run_idx,
        elapsed=elapsed,
        tool_calls=tool_calls,
        response_len=len(msg.strip()),
        passed=passed,
        response_snippet=msg.strip()[:120],
    )


def run_experiment(
    scenarios: List[Scenario],
    conditions: List[str],
    runs: int,
    out_path: str = "",
) -> List[RunResult]:
    all_results: List[RunResult] = []
    n_total = len(scenarios) * len(conditions) * runs
    counter = 0

    if out_path:
        open(out_path, "w").close()

    for scenario in scenarios:
        for condition in conditions:
            for run_idx in range(runs):
                counter += 1
                result = run_cell(scenario, condition, run_idx)
                mark = "pass" if result.passed else "FAIL"
                print(
                    f"  [{counter:3d}/{n_total}] {scenario.scenario_id:20s} "
                    f"{condition:15s} run{run_idx} "
                    f"{result.elapsed:5.1f}s tc={result.tool_calls} {mark}",
                    file=sys.stderr,
                )
                all_results.append(result)
                if out_path:
                    with open(out_path, "a") as fh:
                        fh.write(json.dumps(asdict(result)) + "\n")

    return all_results


# ─── Statistics ───────────────────────────────────────────────────────────────

@dataclass
class ConditionStats:
    condition: str
    n: int
    mean_elapsed: float
    stdev_elapsed: float
    pass_rate: float
    avg_tool_calls: float


def compute_stats(results: List[RunResult], condition: str) -> Optional[ConditionStats]:
    rs = [r for r in results if r.condition == condition]
    if not rs:
        return None
    elapsed = [r.elapsed for r in rs]
    return ConditionStats(
        condition=condition,
        n=len(rs),
        mean_elapsed=round(statistics.mean(elapsed), 2),
        stdev_elapsed=round(statistics.stdev(elapsed), 2) if len(elapsed) > 1 else 0.0,
        pass_rate=round(sum(1 for r in rs if r.passed) / len(rs), 3),
        avg_tool_calls=round(statistics.mean(r.tool_calls for r in rs), 2),
    )


@dataclass
class Comparison:
    a: str
    b: str
    delta_elapsed: float   # b.mean - a.mean; negative = b is faster
    delta_pass_rate: float  # b.pass_rate - a.pass_rate; positive = b is better
    a_wins: int            # runs where a was faster
    b_wins: int            # runs where b was faster
    ties: int


def pairwise_compare(results: List[RunResult], a: str, b: str) -> Comparison:
    a_res = {r.scenario_id + str(r.run_idx): r for r in results if r.condition == a}
    b_res = {r.scenario_id + str(r.run_idx): r for r in results if r.condition == b}
    shared = set(a_res) & set(b_res)
    a_wins = sum(1 for k in shared if a_res[k].elapsed < b_res[k].elapsed)
    b_wins = sum(1 for k in shared if b_res[k].elapsed < a_res[k].elapsed)
    ties = len(shared) - a_wins - b_wins

    stats_a = compute_stats(results, a)
    stats_b = compute_stats(results, b)
    delta_e = round((stats_b.mean_elapsed - stats_a.mean_elapsed) if stats_a and stats_b else 0.0, 2)
    delta_p = round((stats_b.pass_rate - stats_a.pass_rate) if stats_a and stats_b else 0.0, 3)
    return Comparison(a=a, b=b, delta_elapsed=delta_e, delta_pass_rate=delta_p,
                      a_wins=a_wins, b_wins=b_wins, ties=ties)


# ─── Report ───────────────────────────────────────────────────────────────────

def _env_fingerprint() -> dict:
    return {
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
    }


def generate_markdown_report(
    results: List[RunResult],
    scenarios: List[Scenario],
    conditions: List[str],
    runs: int,
    wall: float,
) -> str:
    lines: List[str] = []

    # Header
    lines += [
        "# A/B Experiment Report",
        "",
        "## Environment",
        "",
    ]
    fp = _env_fingerprint()
    for k, v in fp.items():
        lines.append(f"- **{k}**: {v}")
    lines += [
        f"- **scenarios**: {len(scenarios)}",
        f"- **conditions**: {', '.join(conditions)}",
        f"- **runs per cell**: {runs}",
        f"- **total calls**: {len(results)}",
        f"- **wall time**: {wall:.1f}s",
        "",
    ]

    # Per-scenario × condition matrix
    lines += ["## Scenario × Condition Matrix", ""]
    col_w = 22
    header = f"| {'Scenario':<22} |" + "".join(f" {c:<{col_w}} |" for c in conditions)
    sep = f"|{'-'*24}|" + "".join(f"{'-'*(col_w+2)}|" for _ in conditions)
    lines += [header, sep]

    for sc in scenarios:
        sc_results = [r for r in results if r.scenario_id == sc.scenario_id]
        row = f"| {sc.scenario_id:<22} |"
        for cond in conditions:
            cell_rs = [r for r in sc_results if r.condition == cond]
            if not cell_rs:
                row += f" {'—':<{col_w}} |"
                continue
            mean_e = statistics.mean(r.elapsed for r in cell_rs)
            pass_rate = sum(1 for r in cell_rs if r.passed) / len(cell_rs)
            avg_tc = statistics.mean(r.tool_calls for r in cell_rs)
            cell = f"{mean_e:.1f}s {int(pass_rate*100)}% tc={avg_tc:.1f}"
            row += f" {cell:<{col_w}} |"
        lines.append(row)
    lines.append("")

    # Per-condition aggregate statistics
    lines += ["## Statistical Summary", ""]
    stats_by_cond: dict[str, ConditionStats] = {}
    for cond in conditions:
        st = compute_stats(results, cond)
        if st:
            stats_by_cond[cond] = st

    tbl_header = "| Condition       | N   | Mean elapsed | Stdev | Pass rate | Avg tool calls |"
    tbl_sep    = "|-----------------|-----|--------------|-------|-----------|----------------|"
    lines += [tbl_header, tbl_sep]
    for cond in conditions:
        st = stats_by_cond.get(cond)
        if not st:
            continue
        lines.append(
            f"| {cond:<15} | {st.n:<3} | {st.mean_elapsed:>8.2f}s    "
            f"| {st.stdev_elapsed:>5.2f} | {st.pass_rate*100:>7.1f}%  "
            f"| {st.avg_tool_calls:>14.2f} |"
        )
    lines.append("")

    # Pairwise comparisons (all pairs)
    if len(conditions) >= 2:
        lines += ["## Pairwise Comparisons", ""]
        tbl2_header = "| A vs B                    | delta elapsed | delta pass% | A wins | B wins | ties |"
        tbl2_sep    = "|---------------------------|---------------|-------------|--------|--------|------|"
        lines += [tbl2_header, tbl2_sep]
        for i, ca in enumerate(conditions):
            for cb in conditions[i+1:]:
                cmp = pairwise_compare(results, ca, cb)
                sign = "+" if cmp.delta_elapsed >= 0 else ""
                dp_sign = "+" if cmp.delta_pass_rate >= 0 else ""
                lines.append(
                    f"| {ca} vs {cb:<{max(0,19-len(ca))}} "
                    f"| {sign}{cmp.delta_elapsed:>8.2f}s    "
                    f"| {dp_sign}{cmp.delta_pass_rate*100:>6.1f}%    "
                    f"| {cmp.a_wins:>6} | {cmp.b_wins:>6} | {cmp.ties:>4} |"
                )
        lines.append("")

    # Recommendation
    lines += ["## Recommendation", ""]
    if stats_by_cond:
        best_pass = max(stats_by_cond.values(), key=lambda s: s.pass_rate)
        best_speed = min(stats_by_cond.values(), key=lambda s: s.mean_elapsed)
        lines.append(
            f"- **Best pass rate**: `{best_pass.condition}` "
            f"({best_pass.pass_rate*100:.1f}%, mean {best_pass.mean_elapsed:.1f}s)"
        )
        lines.append(
            f"- **Fastest**: `{best_speed.condition}` "
            f"(mean {best_speed.mean_elapsed:.1f}s, pass {best_speed.pass_rate*100:.1f}%)"
        )
        if best_pass.condition == best_speed.condition:
            lines.append(
                f"- **Recommendation**: `{best_pass.condition}` dominates on both "
                f"accuracy and speed. Consider adopting it as the default."
            )
        else:
            lines.append(
                f"- **Recommendation**: Use `{best_pass.condition}` when accuracy matters most; "
                f"use `{best_speed.condition}` when latency is the priority."
            )

        # Identify which condition best handles scenarios that require tool calls
        tool_heavy = [r for r in results if r.tool_calls > 0]
        if tool_heavy:
            cond_pass_tool = {}
            for cond in conditions:
                rs = [r for r in tool_heavy if r.condition == cond]
                if rs:
                    cond_pass_tool[cond] = sum(1 for r in rs if r.passed) / len(rs)
            if cond_pass_tool:
                best_tool_cond = max(cond_pass_tool, key=cond_pass_tool.__getitem__)
                lines.append(
                    f"- **Best for tool-heavy tasks**: `{best_tool_cond}` "
                    f"(pass rate {cond_pass_tool[best_tool_cond]*100:.1f}% on runs with tool calls)"
                )
    lines.append("")
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scenario", metavar="PROMPT",
                     help="Single inline prompt to test")
    src.add_argument("--from-jsonl", metavar="PATH",
                     help="Load scenarios from a JSONL file (uc_test/uc_industry/uc_domains format)")
    ap.add_argument("--check", default="",
                    help="Substring that must appear in the response (used with --scenario)")
    ap.add_argument("--runs", type=int, default=3,
                    help="Number of runs per (scenario, condition) cell (default: 3)")
    ap.add_argument("--conditions", default="",
                    help=f"Comma-separated conditions to run (default: all). "
                         f"Available: {', '.join(ALL_CONDITION_NAMES)}")
    ap.add_argument("--out", default="",
                    help="JSONL path for raw per-run results")
    ap.add_argument("--report", default="",
                    help="Markdown report output path (default: print to stdout)")
    args = ap.parse_args()

    # Resolve conditions
    if args.conditions:
        selected_conditions = [c.strip() for c in args.conditions.split(",")]
        unknown = [c for c in selected_conditions if c not in CONDITIONS]
        if unknown:
            ap.error(f"Unknown conditions: {unknown}. Available: {ALL_CONDITION_NAMES}")
    else:
        selected_conditions = ALL_CONDITION_NAMES

    # Resolve scenarios
    if args.scenario:
        needle = args.check
        check_fn: Callable[[str], bool] = (
            (lambda r, n=needle: n.lower() in r.lower()) if needle else (lambda r: True)
        )
        scenarios = [Scenario(scenario_id="inline", prompt=args.scenario, check=check_fn)]
    else:
        scenarios = load_scenarios_from_jsonl(args.from_jsonl)
        if not scenarios:
            ap.error(f"No valid scenarios found in {args.from_jsonl}")

    n_total = len(scenarios) * len(selected_conditions) * args.runs
    print(
        f"[ab-exp] {len(scenarios)} scenario(s) × {len(selected_conditions)} condition(s) "
        f"× {args.runs} run(s) = {n_total} call(s)",
        file=sys.stderr,
    )

    t0 = time.time()
    all_results = run_experiment(scenarios, selected_conditions, args.runs, args.out)
    wall = time.time() - t0

    report_md = generate_markdown_report(all_results, scenarios, selected_conditions, args.runs, wall)

    if args.report:
        with open(args.report, "w") as fh:
            fh.write(report_md)
        print(f"[ab-exp] report -> {args.report}", file=sys.stderr)
    else:
        print(report_md)

    if args.out:
        print(f"[ab-exp] raw results -> {args.out}", file=sys.stderr)
        summary_path = args.out.replace(".jsonl", "_ab_summary.json")
        with open(summary_path, "w") as fh:
            json.dump({
                "args": {
                    "scenarios": len(scenarios),
                    "conditions": selected_conditions,
                    "runs": args.runs,
                },
                "wall": round(wall, 1),
                "stats": {
                    cond: asdict(compute_stats(all_results, cond))
                    for cond in selected_conditions
                    if compute_stats(all_results, cond)
                },
            }, fh, indent=2)
        print(f"[ab-exp] summary -> {summary_path}", file=sys.stderr)

    print(f"[ab-exp] wall: {wall:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
