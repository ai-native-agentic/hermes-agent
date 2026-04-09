"""UC Test Report Generator — reproducibility + result analysis.

Reads the JSONL output from uc_test.py and produces:
  1. Timestamped markdown report (human-readable)
  2. Per-UC statistics
  3. Run-to-run comparison (if multiple runs exist)
  4. Environment fingerprint for reproducibility

Usage:
  python examples/lunark/uc_report.py /tmp/uc_test.jsonl
  python examples/lunark/uc_report.py /tmp/uc_test.jsonl --compare /tmp/uc_test_prev.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, statistics, subprocess, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def _load_results(path: str) -> list:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _env_fingerprint() -> dict:
    """Capture environment state for reproducibility."""
    fp = {
        "timestamp": datetime.now().isoformat(),
        "hermes_version": "",
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "hermes_home": os.environ.get("HERMES_HOME", "~/.hermes (default)"),
        "lunark_models": [],
    }
    try:
        r = subprocess.run(
            [os.path.expanduser("~/.local/bin/hermes"), "--version"],
            capture_output=True, text=True, timeout=10
        )
        fp["hermes_version"] = r.stdout.strip().split("\n")[0]
    except Exception:
        pass
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://llm.lunark.ai/v1/models",
            headers={"Authorization": "Bearer vllm-local", "User-Agent": "hermes/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            fp["lunark_models"] = [m["id"] for m in data.get("data", [])]
    except Exception:
        pass
    return fp


def generate_report(results: list, compare_results: list = None) -> str:
    """Generate a markdown report from test results."""
    by_uc = defaultdict(list)
    for r in results:
        by_uc[r["uc"]].append(r)

    fp = _env_fingerprint()
    lines = []
    lines.append(f"# UC Test Report — {fp['timestamp'][:19]}")
    lines.append("")
    lines.append("## Environment")
    lines.append(f"- Hermes: {fp['hermes_version']}")
    lines.append(f"- Python: {fp['python_version']}")
    lines.append(f"- Platform: {fp['platform']}")
    lines.append(f"- HERMES_HOME: {fp['hermes_home']}")
    if fp["lunark_models"]:
        lines.append(f"- Lunark models: {', '.join(fp['lunark_models'])}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| UC | Tests | Passed | Avg Time | Status |")
    lines.append("|---|---:|---:|---:|---|")
    total_pass, total_n = 0, 0
    for uc in sorted(by_uc):
        rs = by_uc[uc]
        passed = sum(1 for r in rs if r["passed"])
        avg = statistics.mean([r["elapsed"] for r in rs])
        status = "✅" if passed == len(rs) else f"⚠️ {passed}/{len(rs)}"
        lines.append(f"| {uc} | {len(rs)} | {passed} | {avg:.1f}s | {status} |")
        total_pass += passed
        total_n += len(rs)
    pct = (100 * total_pass / total_n) if total_n else 0
    lines.append(f"| **TOTAL** | **{total_n}** | **{total_pass}** | "
                 f"**{statistics.mean([r['elapsed'] for r in results]):.1f}s** | "
                 f"**{pct:.0f}%** |")
    lines.append("")

    # Per-test detail
    lines.append("## Per-Test Detail")
    lines.append("")
    for uc in sorted(by_uc):
        lines.append(f"### {uc}")
        lines.append("")
        lines.append("| Test | Pass | Time | Tools | Detail |")
        lines.append("|---|:---:|---:|---:|---|")
        for r in by_uc[uc]:
            mark = "✅" if r["passed"] else "❌"
            detail = r.get("detail", "")[:60] or "—"
            lines.append(
                f"| {r['test_id']} | {mark} | {r['elapsed']:.1f}s | "
                f"{r.get('tool_calls', 0)} | {detail} |"
            )
        lines.append("")

    # Comparison with previous run
    if compare_results:
        lines.append("## Comparison with Previous Run")
        lines.append("")
        prev_by_test = {r["test_id"]: r for r in compare_results}
        lines.append("| Test | Prev | Curr | Δ Time | Status Change |")
        lines.append("|---|:---:|:---:|---:|---|")
        for r in results:
            prev = prev_by_test.get(r["test_id"])
            if prev:
                prev_mark = "✅" if prev["passed"] else "❌"
                curr_mark = "✅" if r["passed"] else "❌"
                delta = r["elapsed"] - prev["elapsed"]
                change = ""
                if prev["passed"] != r["passed"]:
                    change = "🔴 REGRESSION" if prev["passed"] else "🟢 FIXED"
                lines.append(
                    f"| {r['test_id']} | {prev_mark} {prev['elapsed']:.1f}s | "
                    f"{curr_mark} {r['elapsed']:.1f}s | {delta:+.1f}s | {change} |"
                )
        lines.append("")

    # Failures section
    failures = [r for r in results if not r["passed"]]
    if failures:
        lines.append("## Failures")
        lines.append("")
        for r in failures:
            lines.append(f"### {r['test_id']} ({r['uc']})")
            lines.append(f"- Prompt: `{r['prompt'][:100]}`")
            lines.append(f"- Response: `{r.get('response', '')[:200]}`")
            lines.append(f"- Detail: {r.get('detail', 'N/A')}")
            lines.append("")

    # Reproducibility
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```bash")
    lines.append("# Re-run this exact test suite:")
    lines.append("python examples/lunark/uc_test.py --out /tmp/uc_test_$(date +%Y%m%d_%H%M).jsonl")
    lines.append("")
    lines.append("# Compare with this run:")
    lines.append(f"python examples/lunark/uc_report.py /tmp/uc_test_NEW.jsonl --compare /tmp/uc_test.jsonl")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", help="JSONL file from uc_test.py")
    ap.add_argument("--compare", default="", help="Previous JSONL for regression comparison")
    ap.add_argument("--out", default="", help="Markdown output file (default: stdout)")
    args = ap.parse_args()

    results = _load_results(args.results)
    compare = _load_results(args.compare) if args.compare else None
    report = generate_report(results, compare)

    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"Report written to {args.out}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
