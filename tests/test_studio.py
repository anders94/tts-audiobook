from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import write_enriched

from tts_audiobook.book import load_book
from tts_audiobook.config import NARRATOR_KEY
from tts_audiobook.studio import (DEFAULT_NARRATOR_SPEC, _infer_sex_from_name,
                                  spec_for_speaker)


def test_infer_sex_from_honorifics():
    assert _infer_sex_from_name("Mrs. Bennet") == "female"
    assert _infer_sex_from_name("Miss Elizabeth Bennet") == "female"
    assert _infer_sex_from_name("Lady Catherine de Bourgh") == "female"
    assert _infer_sex_from_name("Mr. Fitzwilliam Darcy") == "male"
    assert _infer_sex_from_name("Sir William Lucas") == "male"
    assert _infer_sex_from_name("Colonel Fitzwilliam") == "male"
    assert _infer_sex_from_name("Nicholls") is None


def test_infer_sex_from_given_names():
    assert _infer_sex_from_name("Jane Bennet") == "female"
    assert _infer_sex_from_name("Kitty Bennet") == "female"
    assert _infer_sex_from_name("Catherine Bennet") == "female"
    assert _infer_sex_from_name("Charles Bingley") == "male"
    assert _infer_sex_from_name("Fitzwilliam Darcy") == "male"
    assert _infer_sex_from_name("young Lucas") is None


def test_apply_speaker_merges(tmp_path):
    from tts_audiobook.book import apply_speaker_merges, load_book
    book = load_book(write_enriched(tmp_path))
    apply_speaker_merges(book, {"Elizabeth Bennet": "Mrs. Long"})
    speakers = {s.speaker_key for ch in book.chapters for s in ch.segments}
    assert "Elizabeth Bennet" not in speakers
    assert "Mrs. Long" in speakers


def test_spec_for_speaker_prefers_real_spec(tmp_path):
    book = load_book(write_enriched(tmp_path))
    spec = spec_for_speaker(book, "Elizabeth Bennet")
    assert spec.age_band == "young_adult"  # from the JSON, not the fallback
    narrator = spec_for_speaker(book, NARRATOR_KEY)
    assert narrator.pace == "measured"  # production.narration.voice


def test_spec_for_speaker_fallbacks(tmp_path):
    book = load_book(write_enriched(tmp_path))
    # Mrs. Long has no voice block: honorific inference kicks in.
    spec = spec_for_speaker(book, "Mrs. Long")
    assert spec.sex == "female"
    # Unknown speaker entirely: neutral default.
    spec = spec_for_speaker(book, "Somebody")
    assert spec.sex is None


def test_default_narrator_without_production(tmp_path):
    p = tmp_path / "min.json"
    p.write_text('{"metadata": {"title": "X"}, "chapters": []}', encoding="utf-8")
    book = load_book(p)
    assert spec_for_speaker(book, NARRATOR_KEY) == DEFAULT_NARRATOR_SPEC


def test_cast_summary(tmp_path):
    from tts_audiobook import db as dbmod
    from tts_audiobook.studio import cast_summary
    book = load_book(write_enriched(tmp_path))
    conn = dbmod.connect(tmp_path / "studio.db")
    book_id = dbmod.book_upsert(conn, book.source_path, title=book.title,
                                author=book.author, gutenberg_id=None,
                                output_dir=tmp_path / "out")
    clip_id = dbmod.clip_add(conn, path=tmp_path / "c.wav", transcript="hi",
                             duration_s=5.0, sex="female", age_band="adult",
                             locale="en-GB", region="Hertfordshire", quality="good",
                             source="custom", license=None, notes="warm", sha256="x")
    dbmod.cast_upsert(conn, book_id, NARRATOR_KEY, library_clip_id=clip_id,
                      ref_path="/ref/n.wav", status="accepted", engine="qwen")
    dbmod.cast_upsert(conn, book_id, "Elizabeth Bennet", library_clip_id=None,
                      ref_path="/ref/e.wav", design_seed=7, status="proposed")
    dbmod.cast_upsert(conn, book_id, "Ghost", library_clip_id=None,
                      ref_path="/ref/g.wav", design_seed=1, status="accepted")

    rows = {r["character"]: r for r in cast_summary(conn, book, book_id)}
    assert rows[NARRATOR_KEY]["voice"] == f"clip #{clip_id}"
    assert rows[NARRATOR_KEY]["detail"] == "female adult en-GB Hertfordshire · warm"
    assert rows[NARRATOR_KEY]["status"] == "accepted"
    assert rows["Elizabeth Bennet"]["voice"] == "designed (seed 7)"
    assert rows["Elizabeth Bennet"]["status"] == "proposed"
    assert rows["Ghost"]["stale"] is True
    order = [r["character"] for r in cast_summary(conn, book, book_id)]
    assert order[-1] == "Ghost"

    dbmod.cast_delete(conn, book_id, "Elizabeth Bennet")
    rows = {r["character"]: r for r in cast_summary(conn, book, book_id)}
    assert rows["Elizabeth Bennet"]["status"] == "uncast"
    assert rows["Elizabeth Bennet"]["voice"] == "—"

