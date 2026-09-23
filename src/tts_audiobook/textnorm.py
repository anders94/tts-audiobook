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


def is_bare_roman(text: str) -> bool:
    """True for a heading line that is only a roman numeral ("I.", "XLII")."""
    return bool(_BARE_ROMAN_RE.match(text.strip()))


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


_TERMINAL_PUNCT = (".", "!", "?", ":")
_HEADING_KEY_RE = re.compile(r"[\s/]+")


def heading_matches(text: str, title: str) -> bool:
    """True if a segment is the chapter heading. Upstream titles keep source
    line breaks as " / " and stray double spaces ("Chapter I / The Bertolini",
    "A.  SALVIUS OTHO.") that the text segment doesn't, so compare loosely."""
    def key(s: str) -> str:
        return _HEADING_KEY_RE.sub(" ", s).strip().rstrip(".:;,").casefold()
    return bool(title.strip()) and key(text) == key(title)


def spoken_heading(title: str, chapter_number: int) -> str:
    """Heading to narrate when a chapter's text doesn't open with one.

    "Chapter I" → "Chapter 1."; a title with no chapter label, e.g.
    "Loomings", becomes "Chapter 1. Loomings." so the listener always hears
    the chapter number that separates it from what came before.
    """
    heading = normalize_heading(title, chapter_number=chapter_number,
                                is_title=True).strip()
    if not _LABELED_HEADING_RE.match(heading) and \
            not re.match(r"^(?:chapter|part|book|volume|section|act|scene|stave|canto)\b",
                         heading, re.IGNORECASE):
        heading = f"Chapter {chapter_number}. {heading}" if heading else f"Chapter {chapter_number}"
    if not heading.endswith(_TERMINAL_PUNCT):
        heading += "."
    return heading


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
