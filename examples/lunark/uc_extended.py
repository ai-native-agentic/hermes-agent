"""Extended UC tests (UC7-UC12) — 6 new service use cases.

Supplements the original uc_test.py (UC1-UC6) with 6 more diverse
scenarios. Uses the same framework conventions: hermes chat subprocess,
check function, JSONL output.
"""
from __future__ import annotations
import json, os, sys, time, subprocess
from dataclasses import dataclass, asdict
from typing import List, Callable

HERMES = os.path.expanduser("~/.local/bin/hermes")
NODE_PATH = os.path.expanduser("~/.nvm/versions/node/v22.22.2/bin")


@dataclass
class TestResult:
    uc: str; test_id: str; prompt: str; passed: bool
    elapsed: float; tool_calls: int; detail: str; response: str


def _env():
    e = os.environ.copy()
    e["PATH"] = f"{NODE_PATH}:{os.path.expanduser('~/.local/bin')}:{e.get('PATH','')}"
    e["LUNARK_API_KEY"] = "vllm-local"
    return e


def chat(prompt, *, model="Qwen3-32B", max_turns=10, timeout=300):
    cmd = [HERMES, "chat", "--provider", "lunark", "--model", model,
           "-Q", "--max-turns", str(max_turns), "-q", prompt]
    p = subprocess.run(cmd, env=_env(), capture_output=True, text=True, timeout=timeout)
    return p.stdout, p.stderr, p.returncode


def extract(stdout):
    lines, in_box, content = stdout.strip().split("\n"), False, []
    for line in lines:
        if "Hermes" in line and "╭" in line: in_box = True; continue
        if "╰" in line and in_box: break
        if in_box: content.append(line.strip().strip("│").strip())
    return "\n".join(content).strip()


def run_test(uc, test_id, prompt, check, **kw):
    t0 = time.time()
    try:
        stdout, stderr, rc = chat(prompt, **kw)
    except subprocess.TimeoutExpired:
        return TestResult(uc, test_id, prompt[:80], False, 999, 0, "TIMEOUT", "")
    elapsed = round(time.time() - t0, 1)
    combined = (stdout or "") + (stderr or "")
    resp = extract(stdout)
    tc = combined.count("tool calls)")
    try: passed = bool(check(resp or combined))
    except: passed = False
    return TestResult(uc, test_id, prompt[:80], passed, elapsed, tc,
                      "" if passed else resp[:100], resp[:300])


# ═══════════════════════════════════════════════════════════════
# UC7: 교육 튜터 — 개념 설명 + 코드 예제 + 적응형 학습
# ═══════════════════════════════════════════════════════════════

UC7 = [
    ("uc7-explain",
     "/refine Explain recursion to a beginner programmer in 3 sentences. Include a real-world analogy.",
     lambda r: len(r) > 80 and any(w in r.lower() for w in ["recursion", "itself", "call", "base"])),
    ("uc7-code-example",
     "Use execute_code to run a Python function that computes factorial(5) using recursion. Show the code and output.",
     lambda r: "120" in r),
    ("uc7-quiz",
     "Create a 3-question multiple-choice quiz about Python lists. For each question provide 4 options (A-D) and mark the correct answer.",
     lambda r: r.count("A)") >= 3 or r.count("A.") >= 3 or r.lower().count("correct") >= 1),
]

# ═══════════════════════════════════════════════════════════════
# UC8: 데이터 분석 파이프라인
# ═══════════════════════════════════════════════════════════════

UC8 = [
    ("uc8-csv-stats",
     "Read /tmp/hermes_uc/data.csv using execute_code with pandas. Report: row count, column names, mean score. Reply concisely.",
     lambda r: any(w in r.lower() for w in ["alice", "bob", "carol", "mean", "score", "name"])),
    ("uc8-compute",
     "Use execute_code to compute: generate 100 random numbers (seed=42), find mean, median, std. Reply with 3 numbers.",
     lambda r: any(c.isdigit() for c in r) and ("." in r)),
    ("uc8-json-transform",
     "Use execute_code to convert this dict to YAML-style output: {'name': 'hermes', 'version': '0.7.0', 'tools': 28}. Print the result.",
     lambda r: "hermes" in r.lower() and ("0.7" in r or "version" in r.lower())),
]

# ═══════════════════════════════════════════════════════════════
# UC9: 보안 감사 봇
# ═══════════════════════════════════════════════════════════════

UC9 = [
    ("uc9-permissions",
     "Use terminal to check permissions of /etc/passwd with 'ls -la /etc/passwd'. Is it world-readable? Reply yes or no.",
     lambda r: "yes" in r.lower() or "readable" in r.lower() or "644" in r or "rw-r--r--" in r),
    ("uc9-open-ports",
     "Use terminal to list listening ports with 'ss -tlnp 2>/dev/null | head -10' or 'netstat -tlnp 2>/dev/null | head -10'. Report any ports you see.",
     lambda r: any(c.isdigit() for c in r)),
    ("uc9-sensitive-files",
     "Use search_files to check if any .env files exist under /home/kjs/projects/hermes-agent/ (top level only, not node_modules). List what you find.",
     lambda r: ".env" in r.lower() or "not found" in r.lower() or "no " in r.lower()),
]

# ═══════════════════════════════════════════════════════════════
# UC10: 다국어 번역 + 품질 검증
# ═══════════════════════════════════════════════════════════════

UC10 = [
    ("uc10-translate-ko",
     "Translate 'The quick brown fox jumps over the lazy dog' to Korean. Just the Korean translation.",
     lambda r: any(ord(c) >= 0xAC00 for c in r)),  # contains hangul
    ("uc10-translate-ja",
     "Translate 'Good morning, how are you today?' to Japanese. Just the Japanese.",
     lambda r: any(ord(c) >= 0x3040 for c in r)),  # contains hiragana/katakana/kanji
    ("uc10-refine-translation",
     "/refine Translate this technical sentence to Korean naturally: 'The agent spawns isolated subprocesses with restricted tool access for parallel task execution.'",
     lambda r: any(ord(c) >= 0xAC00 for c in r) and len(r) > 20),
]

# ═══════════════════════════════════════════════════════════════
# UC11: 기술 문서 자동 생성
# ═══════════════════════════════════════════════════════════════

UC11 = [
    ("uc11-docstring",
     "Read /home/kjs/projects/hermes-agent/agent/skill_retrieval.py and generate a one-paragraph module docstring summary (not the existing one — your own summary based on reading the code).",
     lambda r: len(r) > 100 and any(w in r.lower() for w in ["skill", "retrieval", "tfidf", "tf-idf", "search", "query"])),
    ("uc11-api-doc",
     "Read /home/kjs/projects/hermes-agent/agent/feedback.py and list all public functions with their signatures and one-line descriptions. Format as a markdown table.",
     lambda r: "record_rating" in r and ("stats" in r or "load_all" in r)),
    ("uc11-changelog",
     "Read the git log with terminal: 'git -C /home/kjs/projects/hermes-agent log --oneline -5'. Summarize the last 5 commits in a brief changelog format.",
     lambda r: any(w in r.lower() for w in ["feat", "fix", "commit", "change"])),
]

# ═══════════════════════════════════════════════════════════════
# UC12: 일일 브리핑 에이전트
# ═══════════════════════════════════════════════════════════════

UC12 = [
    ("uc12-news",
     "Use browser_navigate to https://news.ycombinator.com and browser_snapshot. What are the top 3 story titles on the front page right now?",
     lambda r: len(r) > 50 and (r.count("\n") >= 2 or r.count(".") >= 2)),
    ("uc12-weather-api",
     "Use terminal to run: curl -s 'https://wttr.in/Seoul?format=3' and tell me the current weather in Seoul.",
     lambda r: "seoul" in r.lower() or "°" in r or "C" in r),
    ("uc12-system-brief",
     "Use delegate_task to spawn 2 parallel subagents: (1) 'Run df -h / and report disk usage percentage' (2) 'Run free -h and report memory usage'. Combine both into a brief system status summary.",
     lambda r: any(w in r.lower() for w in ["disk", "memory", "used", "free", "gb", "%"])),
]


ALL_UCS = {
    "7":  ("UC7: 교육 튜터", UC7),
    "8":  ("UC8: 데이터 분석", UC8),
    "9":  ("UC9: 보안 감사", UC9),
    "10": ("UC10: 다국어 번역", UC10),
    "11": ("UC11: 기술 문서 생성", UC11),
    "12": ("UC12: 일일 브리핑", UC12),
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--uc", nargs="*", default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    selected = args.uc or list(ALL_UCS.keys())
    if args.out: open(args.out, "w").close()

    total = sum(len(ALL_UCS[u][1]) for u in selected if u in ALL_UCS)
    print(f"[uc-ext] {len(selected)} UCs, {total} tests", file=sys.stderr)

    all_results = []
    t0 = time.time()
    counter = 0
    for uc_num in selected:
        if uc_num not in ALL_UCS: continue
        name, tests = ALL_UCS[uc_num]
        print(f"\n{'='*60}\n  UC{uc_num}: {name}\n{'='*60}", file=sys.stderr)
        for test_id, prompt, check in tests:
            counter += 1
            print(f"  [{counter}/{total}] {test_id} ...", end=" ", flush=True, file=sys.stderr)
            r = run_test(f"UC{uc_num}", test_id, prompt, check, max_turns=10)
            mark = "✅" if r.passed else "❌"
            print(f"{mark} {r.elapsed}s", file=sys.stderr)
            all_results.append(r)
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # Report
    from collections import defaultdict
    by_uc = defaultdict(list)
    for r in all_results: by_uc[r.uc].append(r)

    print(f"\n{'='*80}")
    print("  EXTENDED USE CASE TEST RESULTS")
    print(f"  {'-'*76}")
    for uc in sorted(by_uc):
        rs = by_uc[uc]
        p = sum(1 for r in rs if r.passed)
        avg = sum(r.elapsed for r in rs) / len(rs)
        nm = ALL_UCS.get(uc.replace("UC",""), ("",))[0]
        print(f"  {uc:6s} {nm:30s}  {p}/{len(rs)}  avg {avg:5.1f}s")
        for r in rs:
            m = "✓" if r.passed else "✗"
            print(f"    [{m}] {r.test_id:20s}  {r.elapsed:5.1f}s  {r.detail[:50]}")

    tp = sum(1 for r in all_results if r.passed)
    print(f"\n  TOTAL: {tp}/{len(all_results)} ({100*tp/len(all_results):.0f}%)  avg {sum(r.elapsed for r in all_results)/len(all_results):.1f}s")
    print(f"  wall: {time.time()-t0:.0f}s")
    if args.out: print(f"  → {args.out}")


if __name__ == "__main__":
    main()
