"""Mixture-of-Agents with an aggregator (the real MoA architecture).

The earlier moa.py / moa_hard.py do plain majority voting on extracted
answers. The original MoA paper (Wang et al., arXiv:2406.04692) does
something stronger:

  1. N reference models answer the prompt in parallel.
  2. An aggregator model receives ALL N raw answers (not extracted)
     plus the original prompt and synthesizes a single high-quality reply.

This file implements that against lunark. We use Qwen3-32B as the
aggregator (highest single-model accuracy in prior tests, 99/100) and
the other three (Gemma, Qwen2.5, Qwen3.5-27B) as reference models.

For comparison we also report:
  - majority vote on the extracted answers (the cheap baseline)
  - the aggregator's answer
  - each individual reference model

Runs against the same 100 hard prompts as moa_hard.py.
"""
from __future__ import annotations
import json, os, sys, time, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extractor import (  # noqa: E402
    extract_number, extract_fraction, extract_yesno, extract_word, normalize_for_compare
)
from examples.acp_client import HermesACPClient  # noqa: E402

REFERENCE_MODELS = ["Gemma-4-E4B-it", "Qwen2.5-32B-Instruct", "Qwen3.5-27B"]
AGGREGATOR_MODEL = "Qwen3-32B"

EXTRACTORS = {
    "number": extract_number,
    "fraction": extract_fraction,
    "yesno": extract_yesno,
    "word": extract_word,
}

# Reuse the 100 hard prompts from moa_hard
import importlib.util
spec = importlib.util.spec_from_file_location(
    "moa_hard",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "moa_hard.py"),
)
_moa_hard = importlib.util.module_from_spec(spec)
# Prevent moa_hard from running its main when imported
_moa_hard.__name__ = "moa_hard_imported"
try:
    spec.loader.exec_module(_moa_hard)
except SystemExit:
    pass
P = _moa_hard.P
print(f"[moa-agg] {len(P)} prompts × {len(REFERENCE_MODELS)} refs + 1 aggregator", file=sys.stderr)


# === Phase 1: get reference answers in parallel ==================================

reference_jobs = [(m, pid, prompt, kind, exp) for m in REFERENCE_MODELS for pid, prompt, kind, exp in P]
ref_results = {}  # (model, pid) → {raw, extracted, norm, correct}
ref_lock = threading.Lock()


def run_reference(job):
    model, pid, prompt, kind, exp = job
    home = f"/tmp/hermes_{model}"
    env = os.environ.copy()
    env["HERMES_HOME"] = home
    env["LUNARK_API_KEY"] = "vllm-local"
    msg = ""
    try:
        client = HermesACPClient(cwd="/tmp", env=env)
        try:
            r = client.prompt(prompt, timeout=120)
            msg = r.message
        finally:
            client.close()
    except Exception:
        msg = ""
    extractor = EXTRACTORS.get(kind, extract_number)
    extracted = ""
    try:
        extracted = extractor(msg) if msg else ""
    except Exception:
        extracted = ""
    norm_exp = normalize_for_compare(exp, kind)
    norm_got = normalize_for_compare(extracted, kind)
    with ref_lock:
        ref_results[(model, pid)] = {
            "raw": msg.strip(),
            "extracted": extracted,
            "norm": norm_got,
            "expected": exp,
            "correct": norm_got == norm_exp,
        }


print(f"\n[phase 1] {len(reference_jobs)} reference calls", file=sys.stderr)
t1 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(run_reference, reference_jobs))
t_phase1 = time.time() - t1
print(f"[phase 1] done in {t_phase1:.1f}s", file=sys.stderr)


# === Phase 2: aggregator synthesizes ================================================

AGG_PROMPT_TEMPLATE = """You have been provided with answers from several models to the same question. Your task is to synthesize them into one final answer. Some answers may be wrong; weigh them critically. Reply with ONLY the final answer in the same format the question requests, no preamble.

QUESTION: {question}

MODEL ANSWERS:
{answers}

YOUR FINAL ANSWER:"""


def build_aggregator_prompt(prompt: str, ref_answers: dict[str, str]) -> str:
    lines = []
    for i, (m, raw) in enumerate(ref_answers.items(), 1):
        # Trim each reference to keep aggregator context manageable
        snippet = raw.strip()[:300] if raw else "(empty)"
        lines.append(f"  Model {i} ({m}): {snippet}")
    return AGG_PROMPT_TEMPLATE.format(
        question=prompt,
        answers="\n".join(lines),
    )


agg_results = {}  # pid → {raw, extracted, norm, correct}
agg_lock = threading.Lock()


def run_aggregator(job):
    pid, prompt, kind, exp = job
    ref_answers = {
        m: ref_results.get((m, pid), {}).get("raw", "")
        for m in REFERENCE_MODELS
    }
    agg_prompt = build_aggregator_prompt(prompt, ref_answers)
    home = f"/tmp/hermes_{AGGREGATOR_MODEL}"
    env = os.environ.copy()
    env["HERMES_HOME"] = home
    env["LUNARK_API_KEY"] = "vllm-local"
    msg = ""
    try:
        client = HermesACPClient(cwd="/tmp", env=env)
        try:
            r = client.prompt(agg_prompt, timeout=120)
            msg = r.message
        finally:
            client.close()
    except Exception:
        msg = ""
    extractor = EXTRACTORS.get(kind, extract_number)
    extracted = ""
    try:
        extracted = extractor(msg) if msg else ""
    except Exception:
        extracted = ""
    norm_exp = normalize_for_compare(exp, kind)
    norm_got = normalize_for_compare(extracted, kind)
    with agg_lock:
        agg_results[pid] = {
            "raw": msg.strip()[-200:],
            "extracted": extracted,
            "norm": norm_got,
            "expected": exp,
            "correct": norm_got == norm_exp,
        }


print(f"\n[phase 2] {len(P)} aggregator calls", file=sys.stderr)
agg_jobs = [(pid, prompt, kind, exp) for pid, prompt, kind, exp in P]
t2 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(run_aggregator, agg_jobs))
t_phase2 = time.time() - t2
print(f"[phase 2] done in {t_phase2:.1f}s", file=sys.stderr)


# === Compare strategies ============================================================

def vote_simple(votes):
    if not votes:
        return ""
    counts = defaultdict(int)
    for v in votes:
        counts[v] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


single_correct = {m: 0 for m in REFERENCE_MODELS}
maj_correct = 0  # majority vote of REFERENCES only
agg_correct = 0  # aggregator
references_unanimous_wrong = 0  # all 3 refs wrong (lower bound on aggregator difficulty)

for pid, prompt, kind, exp in P:
    norm_exp = normalize_for_compare(exp, kind)

    refs_norm = {}
    for m in REFERENCE_MODELS:
        r = ref_results.get((m, pid), {})
        refs_norm[m] = r.get("norm", "")
        if r.get("correct"):
            single_correct[m] += 1

    # Majority over references
    votes = [v for v in refs_norm.values() if v]
    maj = vote_simple(votes)
    if maj == norm_exp:
        maj_correct += 1

    # Aggregator
    a = agg_results.get(pid, {})
    if a.get("norm") == norm_exp:
        agg_correct += 1

    if all(refs_norm[m] != norm_exp for m in REFERENCE_MODELS):
        references_unanimous_wrong += 1


# === Summary =======================================================================

n = len(P)
print("\n=== ACCURACY (100 hard prompts) ===")
print(f"  Aggregator ({AGGREGATOR_MODEL})           {agg_correct:3d}/{n}  ({100*agg_correct/n:5.1f}%)")
print(f"  Reference majority vote                  {maj_correct:3d}/{n}  ({100*maj_correct/n:5.1f}%)")
print()
print(f"  Reference models:")
for m in REFERENCE_MODELS:
    p = single_correct[m]
    print(f"    {m:24s}  {p:3d}/{n}  ({100*p/n:5.1f}%)")

print(f"\n  All 3 refs wrong on: {references_unanimous_wrong}/{n} prompts (aggregator's hardest cases)")

# Where did the aggregator help / hurt?
agg_recovered = []  # majority wrong, aggregator right
agg_lost = []      # majority right, aggregator wrong
for pid, prompt, kind, exp in P:
    norm_exp = normalize_for_compare(exp, kind)
    refs_norm = {m: ref_results.get((m, pid), {}).get("norm", "") for m in REFERENCE_MODELS}
    maj = vote_simple([v for v in refs_norm.values() if v])
    agg = agg_results.get(pid, {}).get("norm", "")
    if agg == norm_exp and maj != norm_exp:
        agg_recovered.append((pid, exp, maj, refs_norm))
    elif agg != norm_exp and maj == norm_exp:
        agg_lost.append((pid, exp, agg, refs_norm))

print(f"\n=== {len(agg_recovered)} prompts where aggregator beat majority vote ===")
for pid, exp, maj, refs in agg_recovered[:10]:
    print(f"  {pid:10s}  exp={exp!r:8s}  maj={maj!r:8s}  refs={dict((m[:8],v) for m,v in refs.items())}")

print(f"\n=== {len(agg_lost)} prompts where aggregator was worse than majority ===")
for pid, exp, agg, refs in agg_lost[:10]:
    print(f"  {pid:10s}  exp={exp!r:8s}  agg={agg!r:8s}  refs={dict((m[:8],v) for m,v in refs.items())}")

print(f"\n  wall: phase1={t_phase1:.0f}s + phase2={t_phase2:.0f}s = {t_phase1+t_phase2:.0f}s")

with open("/tmp/acp_moa_aggregator_results.json", "w") as f:
    json.dump({
        "n": n,
        "aggregator_correct": agg_correct,
        "majority_correct": maj_correct,
        "single_correct": single_correct,
        "references_unanimous_wrong": references_unanimous_wrong,
        "agg_recovered_count": len(agg_recovered),
        "agg_lost_count": len(agg_lost),
        "phase1_seconds": round(t_phase1, 1),
        "phase2_seconds": round(t_phase2, 1),
    }, f, indent=2)
