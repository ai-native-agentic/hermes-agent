"""Use Case Test Framework — 6 service scenarios × 3-4 tests each.

Each UC maps to a real service proposal from the planning team. Tests
exercise the actual Hermes features (browser, delegate, MoA, memory,
cron, sandbox, error_lessons, self-refine) against lunark Qwen3-32B.

Output:
  - Per-UC pass/fail + avg time
  - Cross-UC summary
  - Raw JSONL for re-analysis

Usage:
  python examples/lunark/uc_test.py
  python examples/lunark/uc_test.py --out /tmp/uc_test.jsonl
  python examples/lunark/uc_test.py --uc 1 4    # only UC1 and UC4
"""
from __future__ import annotations
import argparse, json, os, sys, time, subprocess, sqlite3
from dataclasses import dataclass, asdict
from typing import List, Callable, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

HERMES = os.path.expanduser("~/.local/bin/hermes")
NODE_PATH = os.path.expanduser("~/.nvm/versions/node/v22.22.2/bin")


@dataclass
class TestResult:
    uc: str
    test_id: str
    prompt: str
    passed: bool
    elapsed: float
    tool_calls: int
    detail: str
    response: str


def _env(**overrides) -> dict:
    e = os.environ.copy()
    e["PATH"] = f"{NODE_PATH}:{os.path.expanduser('~/.local/bin')}:{e.get('PATH','')}"
    e["LUNARK_API_KEY"] = "vllm-local"
    e.update(overrides)
    return e


def chat(prompt: str, *, model: str = "Qwen3-32B", max_turns: int = 10,
         timeout: int = 300, env_extra: dict = None) -> tuple[str, str, int]:
    """Run hermes chat and return (stdout, stderr, returncode)."""
    cmd = [HERMES, "chat", "--provider", "lunark", "--model", model,
           "-Q", "--max-turns", str(max_turns), "-q", prompt]
    env = _env(**(env_extra or {}))
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    return p.stdout, p.stderr, p.returncode


def extract_response(stdout: str) -> str:
    """Pull the text between Hermes box markers."""
    lines = stdout.strip().split("\n")
    content = []
    in_box = False
    for line in lines:
        if "Hermes" in line and "╭" in line:
            in_box = True
            continue
        if "╰" in line and in_box:
            break
        if in_box:
            content.append(line.strip().strip("│").strip())
    return "\n".join(content).strip()


def count_tool_calls(combined: str) -> int:
    return combined.count("tool calls)")


def run_test(uc: str, test_id: str, prompt: str, check: Callable[[str], bool],
             **kwargs) -> TestResult:
    t0 = time.time()
    try:
        stdout, stderr, rc = chat(prompt, **kwargs)
    except subprocess.TimeoutExpired:
        return TestResult(uc, test_id, prompt[:80], False, 999.0, 0, "TIMEOUT", "")
    elapsed = round(time.time() - t0, 1)
    combined = (stdout or "") + (stderr or "")
    resp = extract_response(stdout)
    tc = count_tool_calls(combined)
    try:
        passed = bool(check(resp or combined))
    except Exception:
        passed = False
    return TestResult(uc, test_id, prompt[:80], passed, elapsed, tc, "" if passed else resp[:100], resp[:300])


# ═══════════════════════════════════════════════════════════════════════
# UC1: DevOps 자율 모니터링 비서
# ═══════════════════════════════════════════════════════════════════════

UC1_TESTS = [
    ("uc1-disk", "Use the terminal tool to check disk usage with 'df -h /' and tell me the used percentage. Just the percentage.",
     lambda r: "%" in r),
    ("uc1-uptime", "Use the terminal tool to run 'uptime' and tell me how long the system has been running. Reply briefly.",
     lambda r: any(w in r.lower() for w in ["up", "day", "hour", "min"])),
    ("uc1-process", "Use the terminal tool to count running processes with 'ps aux | wc -l'. Reply with just the number.",
     lambda r: any(c.isdigit() for c in r)),
    ("uc1-delegate", "Use delegate_task to spawn a subagent with goal: 'Run uname -a and report the kernel version'. Report what the subagent found.",
     lambda r: "linux" in r.lower() or "kernel" in r.lower()),
]

# ═══════════════════════════════════════════════════════════════════════
# UC2: 멀티모델 리서치 에이전트
# ═══════════════════════════════════════════════════════════════════════

UC2_TESTS = [
    ("uc2-browse", "Use browser_navigate to go to https://example.com, then browser_snapshot. What is the page title?",
     lambda r: "example" in r.lower()),
    ("uc2-extract", "Use browser_navigate to go to https://httpbin.org/html, take a snapshot, and tell me the author mentioned. Just the name.",
     lambda r: "melville" in r.lower() or "herman" in r.lower()),
    ("uc2-moa", "**Use the mixture_of_agents tool** to answer: What are 3 benefits of functional programming? List them briefly.",
     lambda r: len(r) > 50 and any(w in r.lower() for w in ["immut", "side effect", "composab", "concurren", "pure", "function"])),
]

# ═══════════════════════════════════════════════════════════════════════
# UC3: 코드 리뷰 + 테스트 자동화
# ═══════════════════════════════════════════════════════════════════════

UC3_TESTS = [
    ("uc3-read", "Read /home/kjs/projects/hermes-agent/agent/skill_retrieval.py and tell me how many functions are defined (def statements). Just the number.",
     lambda r: any(c.isdigit() for c in r)),
    ("uc3-sandbox", "Use execute_code to run: print([x**2 for x in range(1,6)]). Reply with the output.",
     lambda r: "1" in r and "4" in r and "9" in r and "16" in r and "25" in r),
    ("uc3-search", "Use search_files to find all Python files in /home/kjs/projects/hermes-agent/agent/ that contain 'def topk'. List the filenames.",
     lambda r: "skill_retrieval" in r.lower()),
]

# ═══════════════════════════════════════════════════════════════════════
# UC4: 개인 지식 관리 (Second Brain)
# ═══════════════════════════════════════════════════════════════════════

UC4_TESTS = [
    ("uc4-store", "**Use the memory tool** action='add' target='memory' content='UC test: user birthday is March 15'. Emit tool call.",
     lambda r: "added" in r.lower() or "stored" in r.lower() or "memory" in r.lower()),
    ("uc4-recall", "What is my birthday? Check your memory. Just the date.",
     lambda r: "march" in r.lower() or "15" in r or "3월" in r),
    ("uc4-refine", "/refine Explain what a hash table is in one sentence.",
     lambda r: len(r) > 30 and any(w in r.lower() for w in ["key", "value", "hash", "bucket", "lookup"])),
]

# ═══════════════════════════════════════════════════════════════════════
# UC5: 웹 모니터링
# ═══════════════════════════════════════════════════════════════════════

UC5_TESTS = [
    ("uc5-snapshot", "Use browser_navigate to go to https://httpbin.org/ip then browser_snapshot. What is the IP address shown?",
     lambda r: any(c.isdigit() and "." in r for c in r)),
    ("uc5-headers", "Use browser_navigate to go to https://httpbin.org/headers then browser_snapshot. What User-Agent is shown?",
     lambda r: len(r) > 20),
    ("uc5-status", "Use terminal to run 'curl -s -o /dev/null -w \"%{http_code}\" https://example.com'. Reply with just the status code.",
     lambda r: "200" in r),
]

# ═══════════════════════════════════════════════════════════════════════
# UC6: MoA 합의 기반 의사결정
# ═══════════════════════════════════════════════════════════════════════

UC6_TESTS = [
    ("uc6-moa-math", "**Use the mixture_of_agents tool** to compute: What is the sum of the first 20 prime numbers? Reply with just the number.",
     lambda r: "639" in r),  # 2+3+5+7+11+13+17+19+23+29+31+37+41+43+47+53+59+61+67+71=639
    ("uc6-decision", "**Use mixture_of_agents tool**: Should a startup with $500K ARR and 20% monthly growth raise a Series A now or wait 6 months? Give a one-paragraph recommendation.",
     lambda r: len(r) > 100 and any(w in r.lower() for w in ["raise", "wait", "growth", "revenue", "valuation"])),
]


ALL_UCS = {
    "1": ("UC1: DevOps 자율 모니터링", UC1_TESTS),
    "2": ("UC2: 멀티모델 리서치", UC2_TESTS),
    "3": ("UC3: 코드 리뷰 자동화", UC3_TESTS),
    "4": ("UC4: 개인 지식 관리", UC4_TESTS),
    "5": ("UC5: 웹 모니터링", UC5_TESTS),
    "6": ("UC6: MoA 의사결정", UC6_TESTS),
}


def run_uc(uc_num: str, tests: list, out_file: str = "") -> list:
    results = []
    for test_id, prompt, check in tests:
        print(f"  [{test_id}] ...", end=" ", flush=True, file=sys.stderr)
        r = run_test(f"UC{uc_num}", test_id, prompt, check, max_turns=10)
        mark = "✅" if r.passed else "❌"
        print(f"{mark} {r.elapsed}s", file=sys.stderr)
        results.append(r)
        if out_file:
            with open(out_file, "a") as f:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    return results


def print_report(all_results: List[TestResult]):
    from collections import defaultdict
    by_uc = defaultdict(list)
    for r in all_results:
        by_uc[r.uc].append(r)

    print("\n" + "=" * 80)
    print("  USE CASE TEST RESULTS")
    print("  " + "-" * 76)
    for uc in sorted(by_uc):
        rs = by_uc[uc]
        passed = sum(1 for r in rs if r.passed)
        avg = sum(r.elapsed for r in rs) / len(rs)
        uc_name = ALL_UCS.get(uc.replace("UC", ""), ("",))[0] if uc.replace("UC", "") in ALL_UCS else uc
        print(f"  {uc:6s} {uc_name:30s}  {passed}/{len(rs)}  avg {avg:5.1f}s")
        for r in rs:
            mark = "✓" if r.passed else "✗"
            print(f"    [{mark}] {r.test_id:15s}  {r.elapsed:5.1f}s  {r.detail[:50]}")

    total = len(all_results)
    total_pass = sum(1 for r in all_results if r.passed)
    avg_all = sum(r.elapsed for r in all_results) / total if total else 0
    print(f"\n  TOTAL: {total_pass}/{total} ({100*total_pass/total:.0f}%)  avg {avg_all:.1f}s")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uc", nargs="*", default=None, help="UC numbers to run (e.g. 1 4 6)")
    ap.add_argument("--out", default="", help="JSONL output file")
    args = ap.parse_args()

    selected = args.uc or list(ALL_UCS.keys())
    if args.out:
        open(args.out, "w").close()

    total_tests = sum(len(ALL_UCS[u][1]) for u in selected if u in ALL_UCS)
    print(f"[uc-test] {len(selected)} UCs, {total_tests} tests", file=sys.stderr)

    all_results = []
    t_start = time.time()
    for uc_num in selected:
        if uc_num not in ALL_UCS:
            print(f"  ⚠ Unknown UC{uc_num}, skipping", file=sys.stderr)
            continue
        uc_name, tests = ALL_UCS[uc_num]
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  UC{uc_num}: {uc_name}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        results = run_uc(uc_num, tests, args.out)
        all_results.extend(results)

    print_report(all_results)
    print(f"  wall: {time.time()-t_start:.0f}s")

    if args.out:
        print(f"  → {args.out}")


if __name__ == "__main__":
    main()
