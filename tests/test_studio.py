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
    assert _infer_sex_from_name("Jane Bennet") is None
    assert _infer_sex_from_name("Nicholls") is None


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
