"""Tests for examples/lunark/extractor.py — robust answer extractors.

Pulled out of the example's __main__ self-test so the canonical
normalizers run in CI alongside the rest of the lunark provider work.
"""
import os
import sys

import pytest

# The extractor lives under examples/lunark/, which is not on the default
# import path. Add it explicitly so tests can import directly.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LUNARK = os.path.join(_REPO, "examples", "lunark")
if _LUNARK not in sys.path:
    sys.path.insert(0, _LUNARK)

from extractor import (  # noqa: E402
    extract_fraction,
    extract_number,
    extract_word,
    extract_yesno,
    normalize_for_compare,
)


# ---- extract_number --------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("The answer is 391", "391"),
    ("**391**", "391"),
    (r"$\boxed{391}$", "391"),
    (r"$\boxed{\dfrac{2}{3}}$", "3"),  # last numeric token wins (denominator)
    ("Result: 5040.0", "5040"),
    ("Result: 5040.5", "5040.5"),
    ("It is 1,234,567.", "1234567"),
    ("seventeen", "17"),
    ("twenty three", "23"),
    ("one hundred", "100"),
    ("forty two", "42"),
    ("Negative answer: -17", "-17"),
])
def test_extract_number(text, expected):
    assert extract_number(text) == expected


def test_extract_number_returns_empty_when_no_number():
    assert extract_number("hello world") == ""


# ---- extract_fraction ------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("3/4", "3/4"),
    ("12/16 simplifies to 3/4", "3/4"),
    (r"\frac{2}{3}", "2/3"),
    (r"\dfrac{6}{8}", "3/4"),
    ("0.75", "3/4"),
    ("75%", "3/4"),
    ("50%", "1/2"),
    ("three quarters", "3/4"),
    ("two thirds", "2/3"),
    ("half", "1/2"),
    ("1/2", "1/2"),
    ("answer: 6/9", "2/3"),  # simplified
])
def test_extract_fraction(text, expected):
    assert extract_fraction(text) == expected


def test_extract_fraction_handles_zero_denominator_gracefully():
    # Should not raise; falls through to other strategies or returns ""
    result = extract_fraction("1/0")
    assert isinstance(result, str)


# ---- extract_yesno ---------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Yes, that is correct.", "yes"),
    ("No, it is not.", "no"),
    ("Indeed, the answer is yes.", "yes"),
    ("It is true.", "yes"),
    ("That's false.", "no"),
    ("The number 91 is not prime, so no.", "no"),
    ("It IS a prime number, so yes.", "yes"),
    ("Yes!", "yes"),
    ("No.", "no"),
])
def test_extract_yesno(text, expected):
    assert extract_yesno(text) == expected


def test_extract_yesno_empty_when_unclear():
    # Mixed signals → no clear winner; should return ""
    assert extract_yesno("") == ""


# ---- extract_word ----------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("The capital is Paris", "paris"),
    ("Answer: Tokyo.", "tokyo"),
    ("**Berlin**", "berlin"),
])
def test_extract_word(text, expected):
    assert extract_word(text) == expected


# ---- normalize_for_compare -------------------------------------------------

@pytest.mark.parametrize("value,kind,expected", [
    ("12", "number", "12"),
    ("12.0", "number", "12"),
    ("12.5", "number", "12.5"),
    ("6/8", "fraction", "3/4"),
    ("yes", "yesno", "yes"),
    ("YES", "yesno", "yes"),
    ("True", "yesno", "yes"),
    ("False", "yesno", "no"),
])
def test_normalize_for_compare(value, kind, expected):
    assert normalize_for_compare(value, kind) == expected


def test_normalize_passthrough_for_unknown_kind():
    assert normalize_for_compare("hello", "word") == "hello"
