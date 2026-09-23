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


def test_title_chapter_mentions_gutenberg_aloud(tmp_path):
    from tts_audiobook.render import ATTRIBUTION_TEXT, title_chapter
    book = load_book(write_enriched(tmp_path))
    ch = title_chapter(book)
    texts = [s.text for s in ch.segments]
    assert texts[0] == "Pride and Prejudice, by Jane Austen."
    assert texts[1] == "Elizabeth Bennet meets the proud Mr Darcy."
    assert texts[-1] == ATTRIBUTION_TEXT
    assert "Gutenberg Aloud" in ATTRIBUTION_TEXT
    assert all(s.speaker_key == NARRATOR_KEY for s in ch.segments)


def test_missing_heading_is_narrated(tmp_path):
    import copy
    import json
    from fixtures import ENRICHED_BOOK
    data = copy.deepcopy(ENRICHED_BOOK)
    data["chapters"][0]["processed"]["segments"].pop(0)   # drop "Chapter I"
    p = tmp_path / "noheading.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    ch = load_book(p).chapters[0]
    assert ch.segments[0].text == "Chapter 1."
    assert ch.segments[0].speaker_key == NARRATOR_KEY
    assert ch.segments[1].text == "It is a truth universally acknowledged."


def test_existing_heading_not_duplicated(tmp_path):
    ch = load_book(write_enriched(tmp_path)).chapters[0]
    assert [s.text for s in ch.segments[:2]] == \
        ["Chapter 1", "It is a truth universally acknowledged."]


def test_numeral_line_before_title_becomes_chapter_number(tmp_path):
    import copy
    import json
    from fixtures import ENRICHED_BOOK
    data = copy.deepcopy(ENRICHED_BOOK)
    ch = data["chapters"][0]
    ch["chapter"]["title"] = "PLAYING PILGRIMS."
    ch["processed"]["chapter_title"] = "PLAYING PILGRIMS."
    ch["processed"]["segments"][0]["text"] = "I."
    ch["processed"]["segments"].insert(1, {"type": "narration", "text": "PLAYING PILGRIMS.",
                                           "speaker": None, "pronunciation_hints": [],
                                           "notes": None, "start": 3, "end": 20})
    p = tmp_path / "lw.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    ch1 = load_book(p).chapters[0]
    assert [s.text for s in ch1.segments[:3]] == \
        ["Chapter 1.", "PLAYING PILGRIMS.", "It is a truth universally acknowledged."]
    assert ch1.segments[0].speaker_key == NARRATOR_KEY


def test_numeral_segment_not_followed_by_title_is_left_alone(tmp_path):
    import copy
    import json
    from fixtures import ENRICHED_BOOK
    data = copy.deepcopy(ENRICHED_BOOK)
    segs = data["chapters"][0]["processed"]["segments"]
    segs.insert(2, {"type": "narration", "text": "I.", "speaker": None,
                    "pronunciation_hints": [], "notes": None, "start": 51, "end": 52})
    p = tmp_path / "x.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    texts = [s.text for s in load_book(p).chapters[0].segments]
    assert texts[2] == "I."      # a mid-text "I." is content, not a heading

