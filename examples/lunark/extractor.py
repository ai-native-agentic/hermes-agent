"""Robust answer extractors for LLM responses.

Designed for verifier-style work: take a free-form model response and
return a canonical normalized answer that can be compared by string
equality. Handles common equivalence classes:

  - Numbers: "0.75" / "3/4" / "75%" / "three quarters" → "3/4"
  - Integers: "1,234" / "1234" / "one thousand two hundred thirty-four" → "1234"
  - Fractions: simplifies to lowest terms (12/16 → 3/4)
  - LaTeX: \\boxed{x}, \\frac{a}{b}, \\dfrac{a}{b}
  - Markdown: removes **bold**, `code`, $math$
  - Words to numbers (en, basic 0-100)
  - Yes/no with negation handling

Each extractor returns a string. Comparison is case-insensitive.
"""
from __future__ import annotations
import re
from fractions import Fraction
from typing import Optional


# ---- text normalization ----------------------------------------------------

_LATEX_PAT = [
    (re.compile(r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}"), r"\1/\2"),
    (re.compile(r"\\boxed\s*\{([^{}]+)\}"), r"\1"),
    (re.compile(r"\\text\s*\{([^{}]+)\}"), r"\1"),
    (re.compile(r"\\[\(\[\)\]]"), ""),
]


def _strip_latex_markdown(s: str) -> str:
    out = s
    for pat, rep in _LATEX_PAT:
        out = pat.sub(rep, out)
    out = re.sub(r"\$([^$]+)\$", r"\1", out)
    out = out.replace("**", "").replace("__", "").replace("`", "")
    out = re.sub(r"\s+", " ", out)
    return out.strip()


# ---- english number words → int -------------------------------------------

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_FRAC_WORDS = {
    "half": Fraction(1, 2), "halves": Fraction(1, 2),
    "third": Fraction(1, 3), "thirds": Fraction(1, 3),
    "quarter": Fraction(1, 4), "quarters": Fraction(1, 4), "fourth": Fraction(1, 4), "fourths": Fraction(1, 4),
    "fifth": Fraction(1, 5), "fifths": Fraction(1, 5),
    "sixth": Fraction(1, 6), "sixths": Fraction(1, 6),
    "seventh": Fraction(1, 7), "sevenths": Fraction(1, 7),
    "eighth": Fraction(1, 8), "eighths": Fraction(1, 8),
    "ninth": Fraction(1, 9), "ninths": Fraction(1, 9),
    "tenth": Fraction(1, 10), "tenths": Fraction(1, 10),
}


def _word_to_int(text: str) -> Optional[int]:
    """Best-effort English word → integer (0..999). Returns None if no match."""
    text = text.lower().strip().replace("-", " ")
    if text in _UNITS:
        return _UNITS[text]
    if text in _TENS:
        return _TENS[text]
    parts = text.split()
    total = 0
    for w in parts:
        if w in _UNITS:
            total += _UNITS[w]
        elif w in _TENS:
            total += _TENS[w]
        elif w == "hundred":
            total = max(total, 1) * 100
        elif w == "thousand":
            total = max(total, 1) * 1000
        elif w in ("and",):
            continue
        else:
            return None
    return total or None


# ---- canonical extractors --------------------------------------------------

def extract_number(text: str) -> str:
    """Extract a numeric answer as a canonical string.

    Strategy: strip LaTeX/markdown, find numeric tokens, prefer the LAST
    one (final answers are usually at the end), normalize formatting.
    Returns "" if no number found.
    """
    s = _strip_latex_markdown(text).lower()
    # Strip thousands separators inside numbers
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)  # twice for 1,234,567
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if nums:
        n = nums[-1]
        # Strip trailing .0 for integers
        if "." in n:
            try:
                f = float(n)
                if f.is_integer():
                    return str(int(f))
            except ValueError:
                pass
        return n
    # Fallback: try word-to-int on the last significant phrase
    last_words = " ".join(s.split()[-6:])
    w = _word_to_int(last_words)
    if w is not None:
        return str(w)
    return ""


def extract_fraction(text: str) -> str:
    """Extract a fraction in lowest terms, e.g. '3/4'.

    Accepts:
      - 3/4, 12/16 → simplifies
      - \\frac{3}{4} via _strip_latex_markdown
      - 0.75 → 3/4
      - "three quarters" → 3/4
      - 75% → 3/4
    Returns "" if no fraction.
    """
    s = _strip_latex_markdown(text).lower()

    # 1) Direct a/b form (last occurrence)
    m = list(re.finditer(r"(-?\d+)\s*/\s*(\d+)", s))
    if m:
        last = m[-1]
        try:
            f = Fraction(int(last.group(1)), int(last.group(2)))
            return f"{f.numerator}/{f.denominator}"
        except (ValueError, ZeroDivisionError):
            pass

    # 2) Percent → fraction
    pct = list(re.finditer(r"(-?\d+(?:\.\d+)?)\s*%", s))
    if pct:
        try:
            v = float(pct[-1].group(1)) / 100
            f = Fraction(v).limit_denominator(1000)
            return f"{f.numerator}/{f.denominator}"
        except ValueError:
            pass

    # 3) Decimal → fraction (only if it's plausibly a "nice" fraction)
    decs = list(re.finditer(r"-?\d+\.\d+", s))
    if decs:
        try:
            v = float(decs[-1].group())
            f = Fraction(v).limit_denominator(100)
            # Sanity: only return if reasonably simple
            if f.denominator <= 100:
                return f"{f.numerator}/{f.denominator}"
        except ValueError:
            pass

    # 4) "three quarters", "two thirds"
    for word_num, val in _UNITS.items():
        for word_frac, frac_val in _FRAC_WORDS.items():
            patt = rf"\b{word_num}\s+{word_frac}\b"
            if re.search(patt, s):
                f = Fraction(val) * frac_val
                return f"{f.numerator}/{f.denominator}"
    # Also "a half" / "one third" handled above. "half" alone:
    if re.search(r"\bhalf\b", s):
        return "1/2"

    return ""


def extract_yesno(text: str) -> str:
    """Extract yes/no with negation handling.

    Looks at the LAST sentence (final commitment usually wins).
    Returns "yes", "no", or "".
    """
    s = _strip_latex_markdown(text).lower()
    # Take last meaningful sentence
    sentences = re.split(r"[.!?\n]", s)
    sentences = [x.strip() for x in sentences if x.strip()]
    target = sentences[-1] if sentences else s

    # Direct yes/no presence
    has_yes = bool(re.search(r"\byes\b", target))
    has_no = bool(re.search(r"\bno\b|\bnot\b|\bfalse\b|\bisn'?t\b|\baren'?t\b|\bdoesn'?t\b", target))

    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    # Tiebreaker: check whole text
    full_yes = len(re.findall(r"\byes\b|\btrue\b|\bcorrect\b|\bindeed\b", s))
    full_no = len(re.findall(r"\bno\b|\bnot\b|\bfalse\b", s))
    if full_yes > full_no:
        return "yes"
    if full_no > full_yes:
        return "no"
    return ""


def extract_word(text: str) -> str:
    """Extract the last meaningful word (skipping numbers, articles)."""
    s = _strip_latex_markdown(text).lower()
    s = re.sub(r"[^\w\s가-힣]", " ", s)
    skip = {"a", "an", "the", "is", "was", "are", "be", "of", "to", "in", "on", "at", "by"}
    words = [w for w in s.split() if w and not w.isdigit() and w not in skip]
    return words[-1] if words else ""


# ---- equivalence helpers ---------------------------------------------------

def normalize_for_compare(value: str, kind: str = "auto") -> str:
    """Canonicalize a value for equality comparison.

    kind: "number", "fraction", "yesno", "word", or "auto" (try in order).
    """
    v = value.strip().lower()
    if kind in ("number", "auto"):
        if re.fullmatch(r"-?\d+(?:\.\d+)?", v):
            if "." in v:
                try:
                    f = float(v)
                    return str(int(f)) if f.is_integer() else v
                except ValueError:
                    return v
            return v
    if kind in ("fraction", "auto"):
        if re.fullmatch(r"-?\d+\s*/\s*\d+", v):
            try:
                num, den = v.split("/")
                f = Fraction(int(num), int(den))
                return f"{f.numerator}/{f.denominator}"
            except (ValueError, ZeroDivisionError):
                return v
    if kind in ("yesno", "auto"):
        if v in ("yes", "no", "true", "false"):
            return "yes" if v in ("yes", "true") else "no"
    return v


# ---- self-test -------------------------------------------------------------

if __name__ == "__main__":
    cases_num = [
        ("The answer is 391", "391"),
        ("**391**", "391"),
        (r"$\boxed{391}$", "391"),
        ("Result: 5040.0", "5040"),
        ("It is 1,234,567.", "1234567"),
        ("seventeen", "17"),
        ("twenty three", "23"),
    ]
    print("=== extract_number ===")
    for inp, exp in cases_num:
        got = extract_number(inp)
        print(f"  {'✓' if got==exp else '✗'} {inp!r:40s} → {got!r:10s} expected {exp!r}")

    cases_frac = [
        ("3/4", "3/4"),
        ("12/16 simplifies to 3/4", "3/4"),
        (r"\frac{2}{3}", "2/3"),
        ("0.75", "3/4"),
        ("75%", "3/4"),
        ("three quarters", "3/4"),
        ("two thirds", "2/3"),
        ("half", "1/2"),
        ("1/2", "1/2"),
    ]
    print("\n=== extract_fraction ===")
    for inp, exp in cases_frac:
        got = extract_fraction(inp)
        print(f"  {'✓' if got==exp else '✗'} {inp!r:40s} → {got!r:10s} expected {exp!r}")

    cases_yn = [
        ("Yes, that is correct.", "yes"),
        ("No, it is not.", "no"),
        ("Indeed, the answer is yes.", "yes"),
        ("It is true.", "yes"),
        ("That's false.", "no"),
        ("The number 91 is not prime, so no.", "no"),
        ("It IS a prime number, so yes.", "yes"),
    ]
    print("\n=== extract_yesno ===")
    for inp, exp in cases_yn:
        got = extract_yesno(inp)
        print(f"  {'✓' if got==exp else '✗'} {inp!r:50s} → {got!r:5s} expected {exp!r}")
