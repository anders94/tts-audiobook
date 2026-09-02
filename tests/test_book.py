from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import REAL_JSON, write_enriched

from tts_audiobook.book import load_book, speakers_by_importance
from tts_audiobook.config import NARRATOR_KEY


@pytest.mark.skipif(not REAL_JSON.exists(), reason="real JSON not present")
def test_load_real_book_without_new_blocks():
    book = load_book(REAL_JSON)
    assert book.title == "Pride and Prejudice"
    assert len(book.chapters) == 61
    assert book.production is None
    # Old-style characters parse into specs with no voice blocks.
    assert len(book.characters) == 65
    assert all(c.voice is None for c in book.characters)
    speakers = dict(speakers_by_importance(book))
    assert NARRATOR_KEY in speakers
    # "Citation" must not be an assignable speaker (v1 leaked it).
    assert "Citation" not in speakers


def test_load_enriched_book(tmp_path):
    book = load_book(write_enriched(tmp_path))
    assert book.production is not None
    assert book.production.narration_person == "third_omniscient"
    assert book.production.narrator_voice is not None
    assert book.production.narrator_voice.accent.locale == "en-GB"

    lizzy = next(c for c in book.characters if c.name == "Elizabeth Bennet")
    assert lizzy.voice is not None
    assert lizzy.voice.sex == "female"
    assert lizzy.voice.age_band == "young_adult"
    assert lizzy.voice.accent.origin == "Hertfordshire"
    assert lizzy.dialogue_segments == 240

    # Partial character (no voice block) still parses.
    mrs_long = next(c for c in book.characters if c.name == "Mrs. Long")
    assert mrs_long.voice is None


def test_alias_resolution_and_citation_collapse(tmp_path):
    book = load_book(write_enriched(tmp_path))
    segs = book.chapters[0].segments
    # "Lizzy" resolves to canonical name.
    assert segs[4].speaker_key == "Elizabeth Bennet"
    # Citation segment collapses to narrator.
    assert segs[5].speaker_key == NARRATOR_KEY
    # Offsets survive loading.
    assert segs[2].start == 52 and segs[2].end == 73


def test_missing_blocks_tolerated(tmp_path):
    p = tmp_path / "minimal.json"
    p.write_text('{"metadata": {"title": "X"}, "chapters": []}', encoding="utf-8")
    book = load_book(p)
    assert book.title == "X"
    assert book.production is None
    assert book.characters == []
