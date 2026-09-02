from __future__ import annotations

from tts_audiobook.textnorm import apply_pronunciation, normalize_heading, parse_hint


def test_labeled_heading():
    assert normalize_heading("CHAPTER VII.") == "CHAPTER 7."
    assert normalize_heading("Part II: The Return") == "Part 2: The Return"


def test_bare_roman_only_when_title():
    assert normalize_heading("VII.", chapter_number=7, is_title=True) == "Chapter 7."
    assert normalize_heading("VII.", chapter_number=7, is_title=False) == "VII."


def test_non_heading_passthrough():
    assert normalize_heading("MIX-UP AT THE MILL") == "MIX-UP AT THE MILL"


def test_parse_hint_forms():
    assert parse_hint("Lefroy → Leff-roy") == ("Lefroy", "Leff-roy")
    assert parse_hint("Lefroy -> Leff-roy") == ("Lefroy", "Leff-roy")
    assert parse_hint("Lefroy: Leff-roy") == ("Lefroy", "Leff-roy")
    assert parse_hint("Lefroy (pron: Leff-roy)") == ("Lefroy", "Leff-roy")
    assert parse_hint("Lefroy (pronounced Leff-roy)") == ("Lefroy", "Leff-roy")
    assert parse_hint("just a comment about the text") is None


def test_apply_pronunciation_substitutes_not_appends():
    out = apply_pronunciation("Mr. Lefroy arrived. LEFROY!", ["Lefroy → Leff-roy"])
    assert out == "Mr. Leff-roy arrived. Leff-roy!"


def test_unparseable_hint_never_spoken():
    text = "Hello there."
    out = apply_pronunciation(text, ["emphasise the greeting"])
    assert out == text
