"""Top-k skill retrieval — pure-Python TF-IDF, zero new dependencies.

The default Hermes flow injects every active skill into the system prompt
on every turn. With 100+ skills that bloats the prompt by tens of thousands
of tokens AND empirically slows the agent on simple Q&A (see SL-PERF
measurement: +84% wall time when accumulated skills are present).

This module implements a tiny retrieval helper that scores skills against
the user's current query so the prompt builder can inject only the top-k
most relevant ones. The scoring is intentionally cheap and dependency-free:

  - Tokenize: lowercase, strip punctuation, split on whitespace
  - TF: raw term frequency in each document (skill name + description)
  - IDF: log((N+1) / (df+1)) + 1 (smoothed)
  - Score: dot product of query vector with each skill vector, then
    normalized by the L2 norm of the skill vector (so longer descriptions
    don't dominate)

For Hermes' typical skill index (10s to 100s of short descriptions) this
runs in <1 ms with no model loading and no network calls — strictly better
than full injection on the cost axis. Quality vs. a real sentence-encoder
is lower, but the alternative isn't a sentence encoder; it's "inject
everything". Even a noisy top-k beats injecting an unrelated 30 skills.
"""
from __future__ import annotations

import math
import re
from typing import Iterable, List, Sequence, Tuple


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, drop tokens shorter than 2 chars."""
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2]


def _term_frequencies(tokens: Sequence[str]) -> dict:
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return counts


def _build_idf(docs_tokens: Sequence[Sequence[str]]) -> dict:
    """Smoothed inverse document frequency over all docs."""
    n = len(docs_tokens)
    df: dict[str, int] = {}
    for tokens in docs_tokens:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    return {term: math.log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()}


def _vectorize(tf: dict, idf: dict) -> dict:
    return {term: count * idf.get(term, 0.0) for term, count in tf.items()}


def _l2_norm(vec: dict) -> float:
    return math.sqrt(sum(v * v for v in vec.values())) or 1.0


def _dot(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(weight * b.get(term, 0.0) for term, weight in a.items())


def score_skills(
    query: str,
    skills: Iterable[dict],
) -> List[Tuple[dict, float]]:
    """Score each skill against the query and return them sorted by relevance.

    Args:
        query: The user's current message (or any free text).
        skills: An iterable of dicts. Each dict must have at least one of
            ``name`` or ``description`` (more text fields are concatenated).

    Returns:
        A list of (skill_dict, score) tuples sorted by descending score.
        Skills with no overlapping tokens get a score of 0.0.
    """
    skill_list = list(skills)
    if not skill_list:
        return []
    if not query or not query.strip():
        # No query → can't rank, return original order with score 0
        return [(s, 0.0) for s in skill_list]

    docs_tokens: List[List[str]] = []
    for s in skill_list:
        text_parts: List[str] = []
        for key in ("name", "description", "category", "tags"):
            v = s.get(key) if isinstance(s, dict) else None
            if isinstance(v, str):
                text_parts.append(v)
            elif isinstance(v, (list, tuple)):
                text_parts.extend(str(x) for x in v)
        docs_tokens.append(_tokenize(" ".join(text_parts)))

    idf = _build_idf(docs_tokens)
    query_tokens = _tokenize(query)
    query_vec = _vectorize(_term_frequencies(query_tokens), idf)
    query_norm = _l2_norm(query_vec)

    scored: List[Tuple[dict, float]] = []
    for skill, tokens in zip(skill_list, docs_tokens):
        skill_vec = _vectorize(_term_frequencies(tokens), idf)
        norm = _l2_norm(skill_vec) * query_norm
        score = _dot(query_vec, skill_vec) / norm if norm else 0.0
        scored.append((skill, score))

    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored


def topk_skills(
    query: str,
    skills: Iterable[dict],
    k: int = 3,
    min_score: float = 0.0,
    usage_boost: bool = True,
) -> List[dict]:
    """Convenience wrapper: return up to ``k`` skills with score ≥ ``min_score``.

    When the query is empty, returns the first ``k`` skills in the input
    order (so callers always get *something* if they ask for top-k).

    A2 wire-up — usage_boost (default True): multiply each TF-IDF score by
    ``log(1 + views) + 1`` from agent.skill_metrics, so skills the agent
    has actually used before float to the top of the index. New skills
    with no usage record get the +1 floor so cold-start doesn't punish
    them. Disable explicitly for tests / synthetic queries.
    """
    if k <= 0:
        return []
    scored = score_skills(query, skills)
    if not query or not query.strip():
        return [s for s, _ in scored[:k]]

    if usage_boost:
        try:
            from agent.skill_metrics import load_metrics

            usage = load_metrics()
            boosted: List[Tuple[dict, float]] = []
            for s, base_score in scored:
                name = s.get("name") if isinstance(s, dict) else None
                views = 0
                if name:
                    entry = usage.get(name) or {}
                    views = int(entry.get("views", 0))
                # Multiplier in [1.0, ~5.0] for typical view counts.
                # New skills (views=0) keep their original TF-IDF score.
                multiplier = 1.0 + math.log1p(views)
                boosted.append((s, base_score * multiplier))
            boosted.sort(key=lambda kv: kv[1], reverse=True)
            scored = boosted
        except Exception:
            # Best-effort: any failure (missing metrics file, import
            # error inside a test) falls back to plain TF-IDF.
            pass

    filtered = [s for s, score in scored if score >= min_score]
    return filtered[:k]
