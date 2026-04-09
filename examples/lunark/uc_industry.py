"""Industry-specific UC tests — Game Dev Studio + Pharma/Food R&D.

Two concrete org personas:
  A) 중소형 게임개발사 (20-50명): Unity/Unreal, 실시간 밸런싱, QA 자동화
  B) 제약/식품 연구조직 (30-100명): 논문 분석, 실험 데이터, 규제 문서

Tests exercise features verified in prior rounds against lunark Qwen3-32B.
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
    try: stdout, stderr, rc = chat(prompt, **kw)
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
# GAME DEV STUDIO (중소형 게임개발사)
# ═══════════════════════════════════════════════════════════════

# UC-G1: 게임 밸런싱 시뮬레이션
UCG1 = [
    ("ucg1-damage-calc",
     "Use execute_code to simulate: A warrior (ATK=50, CRIT=20%) attacks a monster (DEF=30, HP=200). "
     "Run 10 attacks with random crits (seed=42). Report final HP and how many crits occurred.",
     lambda r: any(c.isdigit() for c in r) and any(w in r.lower() for w in ["hp", "crit", "attack", "damage"])),
    ("ucg1-drop-rate",
     "Use execute_code to simulate 10000 item drops with rates: Common=60%, Rare=25%, Epic=10%, Legendary=5% (seed=42). "
     "Report the actual distribution. Are the results within 1% of expected?",
     lambda r: any(w in r.lower() for w in ["common", "rare", "epic", "legendary"]) and any(c.isdigit() for c in r)),
    ("ucg1-economy",
     "Use execute_code: A game economy has 1000 players each earning 100 gold/day and spending 80 gold/day. "
     "After 30 days, what is the total gold in circulation? Also compute inflation rate if gold supply grows 2% daily.",
     lambda r: any(c.isdigit() for c in r) and any(w in r.lower() for w in ["gold", "inflation", "circulation", "total"])),
]

# UC-G2: QA 자동화 / 테스트 생성
UCG2 = [
    ("ucg2-unit-test",
     "Write a Python unit test for this function and run it with execute_code:\n"
     "def calculate_damage(atk, defense, crit_mult=1.0):\n"
     "    return max(0, (atk - defense) * crit_mult)\n"
     "Test normal hit, crit hit, and defense > atk case.",
     lambda r: ("pass" in r.lower() or "ok" in r.lower() or "3 passed" in r.lower()) and "test" in r.lower()),
    ("ucg2-bug-detect",
     "Read this code and find the bug. Use execute_code to verify:\n"
     "def level_up(player):\n"
     "    player['level'] += 1\n"
     "    player['hp'] = player['base_hp'] * player['level']\n"
     "    player['mp'] = player['base_mp'] * player['level']\n"
     "    return player\n"
     "What happens if base_hp or base_mp keys don't exist?",
     lambda r: any(w in r.lower() for w in ["keyerror", "key error", "missing", "not exist", "doesn't have"])),
    ("ucg2-perf-test",
     "Use execute_code to benchmark: Create a list of 100,000 random integers and measure time for: "
     "(1) list.sort() (2) sorted() (3) heapq.nsmallest(100, ...). Which is fastest for top-100?",
     lambda r: any(w in r.lower() for w in ["heapq", "fastest", "sort", "time", "second", "ms"])),
]

# UC-G3: 게임 기획 문서 자동화
UCG3 = [
    ("ucg3-skill-design",
     "Design a balanced skill for an MMORPG mage class: Name, Mana cost (out of 500 max), Cooldown, Damage formula, "
     "and one unique mechanic. Format as a game design spec.",
     lambda r: len(r) > 100 and any(w in r.lower() for w in ["mana", "cooldown", "damage", "skill"])),
    ("ucg3-quest-gen",
     "Generate a side quest for an open-world RPG: Quest name, NPC giver, 3 objectives, reward table (gold + items + XP). "
     "Include a plot twist in the final objective.",
     lambda r: len(r) > 150 and any(w in r.lower() for w in ["quest", "objective", "reward", "npc"])),
    ("ucg3-patch-notes",
     "/refine Write patch notes for a game update that: nerfed Warrior ATK by 10%, buffed Healer MP regen by 15%, "
     "fixed a duplication bug, added a new dungeon 'Shadow Caves'. Format as official patch notes.",
     lambda r: len(r) > 150 and any(w in r.lower() for w in ["warrior", "healer", "shadow", "patch", "fix"])),
]

# ═══════════════════════════════════════════════════════════════
# PHARMA / FOOD R&D (제약/식품 연구조직)
# ═══════════════════════════════════════════════════════════════

# UC-P1: 논문/특허 분석
UCP1 = [
    ("ucp1-paper-summary",
     "Use browser_navigate to https://pubmed.ncbi.nlm.nih.gov/ and browser_snapshot. "
     "What is the search interface like? Describe the main elements you see.",
     lambda r: any(w in r.lower() for w in ["pubmed", "search", "nlm", "ncbi", "input", "field"])),
    ("ucp1-drug-info",
     "**Use mixture_of_agents tool**: What are the main differences between ACE inhibitors and ARBs for hypertension treatment? "
     "Provide a concise 3-point comparison suitable for a clinical review.",
     lambda r: len(r) > 100 and any(w in r.lower() for w in ["ace", "arb", "angiotensin", "hypertension", "blood pressure"])),
    ("ucp1-literature",
     "/refine Summarize the mechanism of action of metformin for Type 2 diabetes in 3 sentences. "
     "Include its effect on hepatic glucose production and insulin sensitivity.",
     lambda r: len(r) > 80 and any(w in r.lower() for w in ["metformin", "glucose", "insulin", "hepatic", "liver"])),
]

# UC-P2: 실험 데이터 분석
UCP2 = [
    ("ucp2-dose-response",
     "Use execute_code: Simulate a dose-response curve. Doses = [0, 0.1, 1, 10, 100, 1000] µM. "
     "Response follows Hill equation: R = Rmax * C^n / (EC50^n + C^n) where Rmax=100, EC50=10, n=1.5. "
     "Print the response at each dose.",
     lambda r: any(c.isdigit() for c in r) and any(w in r.lower() for w in ["dose", "response", "µm", "um", "100"])),
    ("ucp2-stability",
     "Use execute_code: A food product has initial bacterial count of 1000 CFU/g. Growth follows "
     "N(t) = N0 * 2^(t/generation_time) with generation_time=2 hours. "
     "Compute counts at t=0,2,4,6,8,12,24 hours. At what time does it exceed 100,000 CFU/g?",
     lambda r: "100" in r and any(w in r.lower() for w in ["hour", "cfu", "exceed", "time", "growth"])),
    ("ucp2-stats",
     "Use execute_code: Two groups of mice (n=10 each, seed=42). Control weights: mean=25g, std=2g. "
     "Treatment weights: mean=22g, std=2.5g. Run a t-test. Is the difference significant at p<0.05?",
     lambda r: any(w in r.lower() for w in ["p-value", "p value", "significant", "t-test", "reject", "accept"])),
]

# UC-P3: 규제 문서 / SOP 자동화
UCP3 = [
    ("ucp3-sop",
     "Write a Standard Operating Procedure (SOP) for 'HPLC Column Cleaning' with: "
     "Purpose, Scope, Materials needed, Step-by-step procedure (5 steps), Safety precautions.",
     lambda r: len(r) > 200 and any(w in r.lower() for w in ["hplc", "column", "cleaning", "procedure", "step", "safety"])),
    ("ucp3-regulatory",
     "/refine Draft a brief regulatory submission summary for a new food additive: "
     "Substance: Resveratrol extract from grape skin. Intended use: Antioxidant in beverages at 50mg/L. "
     "Include: identity, intended use, safety data requirements.",
     lambda r: len(r) > 150 and any(w in r.lower() for w in ["resveratrol", "antioxidant", "safety", "beverage", "regulatory"])),
    ("ucp3-coa",
     "Generate a Certificate of Analysis (CoA) template for 'Vitamin C (Ascorbic Acid) Raw Material' with "
     "5 test parameters (Assay, Heavy Metals, Moisture, pH, Microbial), specifications, and result fields.",
     lambda r: len(r) > 150 and any(w in r.lower() for w in ["vitamin c", "ascorbic", "assay", "specification", "certificate"])),
]


ALL_UCS = {
    "G1": ("게임 밸런싱 시뮬레이션", UCG1),
    "G2": ("QA 자동화 / 테스트 생성", UCG2),
    "G3": ("게임 기획 문서 자동화", UCG3),
    "P1": ("논문/특허 분석", UCP1),
    "P2": ("실험 데이터 분석", UCP2),
    "P3": ("규제 문서 / SOP", UCP3),
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
    print(f"[uc-industry] {len(selected)} UCs, {total} tests", file=sys.stderr)

    all_results = []
    t0 = time.time()
    counter = 0
    for uc_num in selected:
        if uc_num not in ALL_UCS: continue
        name, tests = ALL_UCS[uc_num]
        print(f"\n{'='*60}\n  {uc_num}: {name}\n{'='*60}", file=sys.stderr)
        for test_id, prompt, check in tests:
            counter += 1
            print(f"  [{counter}/{total}] {test_id} ...", end=" ", flush=True, file=sys.stderr)
            r = run_test(uc_num, test_id, prompt, check, max_turns=10)
            mark = "✅" if r.passed else "❌"
            print(f"{mark} {r.elapsed}s", file=sys.stderr)
            all_results.append(r)
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    from collections import defaultdict
    by_uc = defaultdict(list)
    for r in all_results: by_uc[r.uc].append(r)

    print(f"\n{'='*80}")
    print("  INDUSTRY UC TEST RESULTS")
    print(f"  {'-'*76}")

    # Game dev section
    print("\n  🎮 게임 개발사")
    for uc in ["G1", "G2", "G3"]:
        if uc not in by_uc: continue
        rs = by_uc[uc]
        p = sum(1 for r in rs if r.passed)
        avg = sum(r.elapsed for r in rs) / len(rs)
        nm = ALL_UCS.get(uc, ("",))[0]
        print(f"  {uc:6s} {nm:30s}  {p}/{len(rs)}  avg {avg:5.1f}s")
        for r in rs:
            m = "✓" if r.passed else "✗"
            print(f"    [{m}] {r.test_id:20s}  {r.elapsed:5.1f}s")

    # Pharma section
    print("\n  🧪 제약/식품 연구조직")
    for uc in ["P1", "P2", "P3"]:
        if uc not in by_uc: continue
        rs = by_uc[uc]
        p = sum(1 for r in rs if r.passed)
        avg = sum(r.elapsed for r in rs) / len(rs)
        nm = ALL_UCS.get(uc, ("",))[0]
        print(f"  {uc:6s} {nm:30s}  {p}/{len(rs)}  avg {avg:5.1f}s")
        for r in rs:
            m = "✓" if r.passed else "✗"
            print(f"    [{m}] {r.test_id:20s}  {r.elapsed:5.1f}s")

    tp = sum(1 for r in all_results if r.passed)
    print(f"\n  TOTAL: {tp}/{len(all_results)} ({100*tp/len(all_results):.0f}%)  avg {sum(r.elapsed for r in all_results)/len(all_results):.1f}s")
    print(f"  wall: {time.time()-t0:.0f}s")
    if args.out: print(f"  → {args.out}")


if __name__ == "__main__":
    main()
