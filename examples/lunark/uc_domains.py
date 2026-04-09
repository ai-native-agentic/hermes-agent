"""Industry domain UC tests — 12 domains, 36 total tests (3 per domain).

Domains:
  L  - 법률/법무
  E  - 교육/에듀테크
  R  - 이커머스/리테일
  C  - 건설/엔지니어링
  F  - 금융/핀테크
  H  - 의료/헬스케어
  D  - 물류/유통
  M  - 마케팅/광고
  A  - 농업/스마트팜
  T  - 여행/호스피탈리티
  HR - HR/인사
  MF - 제조업
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
# L: 법률/법무
# ═══════════════════════════════════════════════════════════════

UCL = [
    ("l1-nda",
     "Draft a Non-Disclosure Agreement (NDA) between Company A and Company B. "
     "Include: parties, definition of confidential info, obligations, term (2 years), "
     "and governing law. Format as a legal document.",
     lambda r: len(r) > 200 and any(w in r.lower() for w in ["confidential", "nda", "agreement", "parties", "obligation"])),
    ("l2-clause-review",
     "Review this contract clause and identify potential risks: "
     "'The Contractor shall indemnify the Client against ALL claims, losses, and damages without limitation.' "
     "What are 3 legal risks?",
     lambda r: len(r) > 100 and any(w in r.lower() for w in ["unlimited", "risk", "liability", "indemnif"])),
    ("l3-legal-memo",
     "/refine Write a legal memorandum analyzing whether a software license can be transferred to a third party "
     "without the licensor's consent under general contract law principles.",
     lambda r: len(r) > 150 and any(w in r.lower() for w in ["license", "transfer", "consent", "assign"])),
]

# ═══════════════════════════════════════════════════════════════
# E: 교육/에듀테크
# ═══════════════════════════════════════════════════════════════

UCE = [
    ("e1-quiz",
     "Create a 5-question multiple choice quiz about Python data structures "
     "(lists, dicts, sets, tuples). Each question has 4 options A-D. Mark the correct answer.",
     lambda r: len(r) > 200 and (r.count("A") >= 5 or r.count("A.") >= 3 or r.count("A)") >= 3)),
    ("e2-lesson-plan",
     "Create a 45-minute lesson plan for teaching 'Introduction to Machine Learning' to college freshmen. "
     "Include: objectives, activities, assessment.",
     lambda r: len(r) > 200 and any(w in r.lower() for w in ["objective", "machine learning", "assessment", "activity"])),
    ("e3-explain",
     "/refine Explain the concept of Object-Oriented Programming to someone who only knows procedural programming. "
     "Use a real-world analogy.",
     lambda r: len(r) > 100 and any(w in r.lower() for w in ["class", "object", "inherit", "encapsulat"])),
]

# ═══════════════════════════════════════════════════════════════
# R: 이커머스/리테일
# ═══════════════════════════════════════════════════════════════

UCR = [
    ("r1-product-desc",
     "Write 3 different marketing descriptions for a wireless noise-canceling headphone "
     "(price $79, battery 30hr, ANC, Bluetooth 5.3). Vary the tone: professional, casual, luxury.",
     lambda r: len(r) > 200 and any(w in r.lower() for w in ["noise", "bluetooth", "headphone", "battery"])),
    ("r2-review-analysis",
     "Use execute_code to analyze this review data and compute sentiment distribution:\n"
     "reviews = ['Great product!', 'Terrible quality', 'Love it', 'Waste of money', "
     "'Average, nothing special', 'Best purchase ever', 'Broke after 1 week', 'Decent for the price']\n"
     "Classify each as positive/negative/neutral and show the distribution.",
     lambda r: any(w in r.lower() for w in ["positive", "negative", "neutral"]) and any(c.isdigit() for c in r)),
    ("r3-pricing",
     "Use execute_code to compute: If a product costs $45 wholesale, what should the retail price be at "
     "40% margin, 50% margin, and 60% margin? Also compute the break-even units if fixed costs are $10,000/month.",
     lambda r: any(c.isdigit() for c in r) and any(w in r.lower() for w in ["margin", "price", "break", "unit"])),
]

# ═══════════════════════════════════════════════════════════════
# C: 건설/엔지니어링
# ═══════════════════════════════════════════════════════════════

UCC = [
    ("c1-spec",
     "Write a technical specification for 'Concrete Foundation Pouring' including: "
     "scope, materials (concrete grade, rebar spec), procedure (5 steps), quality criteria, safety requirements.",
     lambda r: len(r) > 200 and any(w in r.lower() for w in ["concrete", "foundation", "rebar", "procedure", "safety"])),
    ("c2-cost-estimate",
     "Use execute_code to estimate foundation cost: Area=200 sqm, concrete depth=0.3m, "
     "concrete price=$120/cubic meter, labor=$50/sqm, rebar=$30/sqm. "
     "Calculate total material + labor cost.",
     lambda r: any(c.isdigit() for c in r) and any(w in r.lower() for w in ["cost", "total", "concrete", "labor"])),
    ("c3-schedule",
     "Use execute_code to create a simple Gantt chart data: 5 construction phases "
     "(Excavation 5d, Foundation 10d, Structure 20d, MEP 15d, Finishing 10d). "
     "Calculate total duration considering Foundation starts after Excavation, Structure after Foundation, "
     "MEP after Structure, Finishing after MEP. What is the critical path duration?",
     lambda r: any(c.isdigit() for c in r) and any(w in r.lower() for w in ["critical", "duration", "day", "path", "total"])),
]

# ═══════════════════════════════════════════════════════════════
# F: 금융/핀테크
# ═══════════════════════════════════════════════════════════════

UCF = [
    ("f1-monte-carlo",
     "Use execute_code: Run a Monte Carlo simulation (seed=42, 10000 iterations) for a portfolio with "
     "initial value $100,000, annual return mean=8%, std=15%. "
     "What is the 95% VaR (Value at Risk) after 1 year?",
     lambda r: any(c.isdigit() for c in r) and any(w in r.lower() for w in ["var", "value at risk", "95%", "confidence", "loss"])),
    ("f2-loan-calc",
     "Use execute_code: Calculate monthly payment for a $300,000 mortgage at 5.5% annual rate for 30 years. "
     "Also compute total interest paid over the loan life.",
     lambda r: any(c.isdigit() for c in r) and any(w in r.lower() for w in ["monthly", "payment", "interest", "mortgage"])),
    ("f3-ratio",
     "Use execute_code: Given financial data - Revenue=$5M, COGS=$3M, Operating Expenses=$1M, "
     "Total Assets=$8M, Total Liabilities=$3M - compute: "
     "Gross Margin, Operating Margin, Net Margin, Debt-to-Equity ratio, ROA.",
     lambda r: any(c.isdigit() for c in r) and any(w in r.lower() for w in ["margin", "ratio", "roe", "roa", "debt"])),
]

# ═══════════════════════════════════════════════════════════════
# H: 의료/헬스케어
# ═══════════════════════════════════════════════════════════════

UCH = [
    ("h1-patient-guide",
     "/refine Write a patient education guide about managing Type 2 Diabetes. "
     "Include: what it is, lifestyle changes (diet, exercise), medication basics, when to see a doctor. "
     "Use simple language.",
     lambda r: len(r) > 200 and any(w in r.lower() for w in ["diabetes", "blood sugar", "insulin", "diet", "exercise"])),
    ("h2-drug-compare",
     "**Use mixture_of_agents tool**: Compare the efficacy and side effects of SSRIs vs SNRIs for treating "
     "major depression. Provide a structured 3-point comparison.",
     lambda r: len(r) > 100 and any(w in r.lower() for w in ["ssri", "snri", "serotonin", "depression", "side effect"])),
    ("h3-clinical-protocol",
     "Write a clinical protocol outline for 'Management of Acute Chest Pain in the Emergency Department'. "
     "Include: initial assessment, diagnostic workup, risk stratification, treatment algorithm.",
     lambda r: len(r) > 200 and any(w in r.lower() for w in ["chest pain", "ecg", "troponin", "protocol", "emergency"])),
]

# ═══════════════════════════════════════════════════════════════
# D: 물류/유통
# ═══════════════════════════════════════════════════════════════

UCD = [
    ("d1-route",
     "Use execute_code: Given 5 delivery locations with distances "
     "[[0,10,15,20,25],[10,0,35,25,30],[15,35,0,30,20],[20,25,30,0,15],[25,30,20,15,0]], "
     "find the shortest route starting and ending at location 0 using nearest-neighbor heuristic. "
     "Report the route and total distance.",
     lambda r: any(c.isdigit() for c in r) and any(w in r.lower() for w in ["route", "distance", "total", "path"])),
    ("d2-inventory",
     "Use execute_code: Product has daily demand mean=50 units, std=10. Lead time=7 days. "
     "Calculate: reorder point at 95% service level, safety stock, and economic order quantity (EOQ) "
     "if ordering cost=$100, holding cost=$2/unit/year.",
     lambda r: any(c.isdigit() for c in r) and any(w in r.lower() for w in ["reorder", "safety", "eoq", "stock", "order"])),
    ("d3-kpi",
     "Use execute_code: A warehouse processed 1500 orders this month, 45 were returned, "
     "average delivery time was 3.2 days. "
     "Compute: order accuracy rate, return rate, and if the target is 98% accuracy and <3.5 day delivery, "
     "are both KPIs met?",
     lambda r: any(c.isdigit() for c in r) and any(w in r.lower() for w in ["accuracy", "return", "kpi", "met", "target"])),
]

# ═══════════════════════════════════════════════════════════════
# M: 마케팅/광고
# ═══════════════════════════════════════════════════════════════

UCM = [
    ("m1-copy-variants",
     "**Use mixture_of_agents tool**: Write 3 different ad headlines for a fitness app targeting young professionals. "
     "Each headline should use a different emotional appeal: fear of missing out, aspiration, social proof.",
     lambda r: len(r) > 100 and any(w in r.lower() for w in ["fitness", "app", "headline"])),
    ("m2-campaign-roi",
     "Use execute_code: Ad campaign spent $5000, generated 50000 impressions, 2500 clicks, "
     "125 conversions at $40 average order value. "
     "Compute: CTR, conversion rate, CPA, ROAS, and net ROI.",
     lambda r: any(c.isdigit() for c in r) and any(w in r.lower() for w in ["ctr", "roas", "roi", "conversion", "cpa"])),
    ("m3-persona",
     "/refine Create a detailed buyer persona for a B2B SaaS product targeting mid-size companies. "
     "Include: demographics, job role, pain points, goals, preferred channels, objections.",
     lambda r: len(r) > 200 and any(w in r.lower() for w in ["persona", "pain point", "saas", "decision", "channel"])),
]

# ═══════════════════════════════════════════════════════════════
# A: 농업/스마트팜
# ═══════════════════════════════════════════════════════════════

UCA = [
    ("a1-growth-model",
     "Use execute_code: Model crop growth using logistic function: "
     "Y(t) = K / (1 + ((K-Y0)/Y0) * exp(-r*t)) "
     "where K=10000 kg/ha (max yield), Y0=100 (initial), r=0.15 (growth rate). "
     "Compute yield at t=10,20,30,40,50 days.",
     lambda r: any(c.isdigit() for c in r) and any(w in r.lower() for w in ["yield", "growth", "day", "kg"])),
    ("a2-irrigation",
     "Use execute_code: A 10-hectare field needs 5mm of water per day. "
     "The irrigation system delivers 50 cubic meters per hour. "
     "Calculate: daily water requirement in cubic meters, hours of irrigation needed per day, "
     "and monthly water cost at $0.50 per cubic meter.",
     lambda r: any(c.isdigit() for c in r) and any(w in r.lower() for w in ["water", "irrigation", "cubic", "cost", "hour"])),
    ("a3-pest-guide",
     "/refine Write a pest management guide for tomato crops covering: "
     "3 common pests (identification, damage symptoms, treatment), "
     "prevention methods, and organic vs chemical control comparison.",
     lambda r: len(r) > 200 and any(w in r.lower() for w in ["tomato", "pest", "organic", "treatment", "control"])),
]

# ═══════════════════════════════════════════════════════════════
# T: 여행/호스피탈리티
# ═══════════════════════════════════════════════════════════════

UCT = [
    ("t1-itinerary",
     "Plan a detailed 3-day Tokyo travel itinerary for a couple. "
     "Include: daily schedule with times, recommended restaurants, transportation tips, budget estimate per day in USD.",
     lambda r: len(r) > 200 and any(w in r.lower() for w in ["tokyo", "day", "restaurant", "budget"])),
    ("t2-hotel-response",
     "Write 3 different hotel review responses: one for a 5-star positive review, "
     "one for a 2-star complaint about cleanliness, and one for a 3-star mixed review. "
     "Professional hospitality tone.",
     lambda r: len(r) > 200 and any(w in r.lower() for w in ["thank", "review", "guest", "experience", "apologize"])),
    ("t3-translate",
     "/refine Translate this hotel welcome message to Korean, Japanese, and Chinese: "
     "'Welcome to Grand Hotel Seoul. We hope you enjoy your stay. "
     "Breakfast is served from 7AM to 10AM on the 3rd floor. "
     "Please contact the front desk for any assistance.'",
     lambda r: any(ord(c) >= 0xAC00 for c in r) and any(ord(c) >= 0x3040 for c in r)),
]

# ═══════════════════════════════════════════════════════════════
# HR: HR/인사
# ═══════════════════════════════════════════════════════════════

UCHR = [
    ("hr1-jd",
     "/refine Write a job description for a Senior Backend Engineer position at a fintech startup. "
     "Include: responsibilities (5), requirements (5), nice-to-haves (3), benefits, and company culture description.",
     lambda r: len(r) > 200 and any(w in r.lower() for w in ["engineer", "backend", "experience", "requirement", "benefit"])),
    ("hr2-interview",
     "Generate a structured interview guide for a Product Manager role: "
     "5 behavioral questions with STAR-format evaluation criteria, 3 case study questions, "
     "and a scoring rubric (1-5 scale).",
     lambda r: len(r) > 200 and any(w in r.lower() for w in ["interview", "question", "star", "behavioral", "score"])),
    ("hr3-onboarding",
     "Create a 30-day onboarding checklist for a new software engineer. "
     "Week 1: orientation, Week 2: codebase, Week 3: first task, Week 4: review. "
     "Include specific action items per week.",
     lambda r: len(r) > 150 and any(w in r.lower() for w in ["week", "onboarding", "codebase", "orientation", "task"])),
]

# ═══════════════════════════════════════════════════════════════
# MF: 제조업
# ═══════════════════════════════════════════════════════════════

UCMF = [
    ("mf1-spc",
     "Use execute_code: Given 20 sample measurements "
     "[10.2,10.1,9.8,10.3,10.0,9.9,10.4,10.1,9.7,10.2,10.5,9.8,10.1,10.3,9.9,10.0,10.2,10.6,9.8,10.1], "
     "compute: mean, std, UCL (mean+3*std), LCL (mean-3*std). Are all points within control limits?",
     lambda r: any(c.isdigit() for c in r) and any(w in r.lower() for w in ["ucl", "lcl", "control", "mean", "limit"])),
    ("mf2-oee",
     "Use execute_code: Machine ran 8 hours, with 30 min planned downtime, 45 min unplanned downtime. "
     "Produced 400 units at 1 unit/min ideal rate. 12 units were defective. "
     "Compute OEE (Overall Equipment Effectiveness) = Availability x Performance x Quality.",
     lambda r: any(c.isdigit() for c in r) and any(w in r.lower() for w in ["oee", "availability", "performance", "quality"])),
    ("mf3-8d",
     "/refine Write an 8D Problem Solving Report for: "
     "'Customer complaint - 5% of delivered circuit boards have solder bridge defects'. "
     "Complete all 8 disciplines: team, problem description, containment, root cause, "
     "corrective action, preventive action, congratulate team.",
     lambda r: len(r) > 200 and any(w in r.lower() for w in ["root cause", "corrective", "solder", "8d", "containment"])),
]


ALL_UCS = {
    "L":  ("법률/법무",          UCL),
    "E":  ("교육/에듀테크",       UCE),
    "R":  ("이커머스/리테일",      UCR),
    "C":  ("건설/엔지니어링",      UCC),
    "F":  ("금융/핀테크",         UCF),
    "H":  ("의료/헬스케어",        UCH),
    "D":  ("물류/유통",           UCD),
    "M":  ("마케팅/광고",          UCM),
    "A":  ("농업/스마트팜",        UCA),
    "T":  ("여행/호스피탈리티",     UCT),
    "HR": ("HR/인사",             UCHR),
    "MF": ("제조업",              UCMF),
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
    print(f"[uc-domains] {len(selected)} UCs, {total} tests", file=sys.stderr)

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
    print("  DOMAIN UC TEST RESULTS")
    print(f"  {'-'*76}")

    domain_groups = [
        ("⚖️  법률/법무",        ["L"]),
        ("📚 교육/에듀테크",     ["E"]),
        ("🛒 이커머스/리테일",   ["R"]),
        ("🏗️  건설/엔지니어링",  ["C"]),
        ("💰 금융/핀테크",       ["F"]),
        ("🏥 의료/헬스케어",     ["H"]),
        ("🚚 물류/유통",         ["D"]),
        ("📣 마케팅/광고",       ["M"]),
        ("🌱 농업/스마트팜",     ["A"]),
        ("✈️  여행/호스피탈리티", ["T"]),
        ("👥 HR/인사",           ["HR"]),
        ("🏭 제조업",            ["MF"]),
    ]

    for label, ucs in domain_groups:
        print(f"\n  {label}")
        for uc in ucs:
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
    if all_results:
        print(f"\n  TOTAL: {tp}/{len(all_results)} ({100*tp/len(all_results):.0f}%)  avg {sum(r.elapsed for r in all_results)/len(all_results):.1f}s")
    print(f"  wall: {time.time()-t0:.0f}s")
    if args.out: print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
