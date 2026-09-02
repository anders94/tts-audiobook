from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Strict roman numeral (1-4999); the lookahead rejects an empty match.
_ROMAN = r"(?=[IVXLCDM])M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"

# "CHAPTER VII." / "Part II: The Return" / "BOOK III — AT SEA".
# After the numeral there must be end-of-string, whitespace, punctuation,
# or a dash — an attached hyphen is rejected so a heading word that merely
# looks roman ("MIX-UP") is never treated as a numeral.
_LABELED_HEADING_RE = re.compile(
    rf"^(?P<label>CHAPTER|PART|BOOK|VOLUME|SECTION|ACT|SCENE|STAVE|CANTO)"
    rf"\s+(?P<roman>{_ROMAN})(?P<rest>$|[.:;,\s—–].*)",
    re.IGNORECASE | re.DOTALL,
)

# A heading that is nothing but the numeral, e.g. "VII." on its own line.
_BARE_ROMAN_RE = re.compile(rf"^{_ROMAN}\.?$", re.IGNORECASE)

_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def _roman_to_int(s: str) -> int:
    total = 0
    prev = 0
    for c in reversed(s.lower()):
        v = _ROMAN_VALUES[c]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total


def normalize_heading(text: str, *, chapter_number: int | None = None,
                      is_title: bool = False) -> str:
    """Rewrite a chapter-heading roman numeral so TTS reads it as a number.

    "CHAPTER VII." → "CHAPTER 7."; a bare "VII." heading becomes
    "Chapter 7." (only when is_title, so numeral-only narration segments
    elsewhere are never touched). Non-heading text passes through unchanged.
    """
    stripped = text.strip()
    m = _LABELED_HEADING_RE.match(stripped)
    if m:
        return f"{m.group('label')} {_roman_to_int(m.group('roman'))}{m.group('rest')}"
    if is_title and chapter_number is not None and _BARE_ROMAN_RE.match(stripped):
        return f"Chapter {chapter_number}."
    return text


# Pronunciation hints arrive as free-ish text from the upstream LLM. Accepted
# shapes, most specific first: "word (pron: say-it)", "word → say-it",
# "word -> say-it", "word: say-it", "word = say-it". Anything else is ignored
# (v1 appended unparsed hints to the spoken text, which the TTS read aloud).
_HINT_RES = [
    re.compile(r"^\s*(?P<target>.+?)\s*\(\s*pron(?:ounced|unciation)?\s*:?\s*(?P<repl>.+?)\s*\)\s*$",
               re.IGNORECASE),
    re.compile(r"^\s*(?P<target>.+?)\s*(?:→|->)\s*(?P<repl>.+?)\s*$"),
    re.compile(r"^\s*(?P<target>[^:=]+?)\s*[:=]\s*(?P<repl>.+?)\s*$"),
]


def parse_hint(hint: str) -> tuple[str, str] | None:
    for rx in _HINT_RES:
        m = rx.match(hint)
        if m:
            target = m.group("target").strip().strip("\"'“”‘’")
            repl = m.group("repl").strip().strip("\"'“”‘’")
            if target and repl and target.lower() != repl.lower():
                return target, repl
    return None


def apply_pronunciation(text: str, hints: list[str]) -> str:
    """Substitute pronunciation-hint targets in text; never append hints."""
    for hint in hints:
        parsed = parse_hint(hint)
        if parsed is None:
            log.debug("Ignoring unparseable pronunciation hint: %r", hint)
            continue
        target, repl = parsed
        pattern = re.compile(rf"\b{re.escape(target)}\b", re.IGNORECASE)
        text = pattern.sub(repl, text)
    return text
