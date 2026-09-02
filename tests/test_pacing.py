from __future__ import annotations

from tts_audiobook import config
from tts_audiobook.book import Segment
from tts_audiobook.config import NARRATOR_KEY
from tts_audiobook.pacing import compute_gap


def seg(speaker: str, text: str, *, notes: str | None = None,
        start: int | None = None, end: int | None = None) -> Segment:
    return Segment(speaker_key=speaker, text=text, raw_speaker=None,
                   notes=notes, start=start, end=end)


def test_chapter_start_has_no_gap():
    assert compute_gap(None, seg("A", "Hello.")) == 0.0


def test_quote_continues_is_tight():
    prev = seg("A", "…and so on,", notes="quote-continues")
    cur = seg("A", "continued the speech.")
    assert compute_gap(prev, cur) == config.GAP_QUOTE_CONTINUES


def test_midsentence_split_tight():
    # "My dear Mr. Bennet," | said his lady — adjacent offsets, prev nonterminal.
    prev = seg("Mrs. Bennet", "“My dear Mr. Bennet,”", start=380, end=401)
    cur = seg(NARRATOR_KEY, "said his lady to him one day,", start=402, end=431)
    assert compute_gap(prev, cur) == config.GAP_MIDSENTENCE_SPLIT


def test_attribution_tag_without_offsets():
    prev = seg("A", "“Well, I never,”")
    cur = seg(NARRATOR_KEY, "she said.")
    assert compute_gap(prev, cur) == config.GAP_ATTRIBUTION_TAG


def test_plain_speaker_change():
    prev = seg("A", "“I quite agree.”", start=0, end=16)
    cur = seg("B", "“As do I.”", start=18, end=28)
    assert compute_gap(prev, cur) == config.GAP_SPEAKER_CHANGE


def test_same_speaker_new_segment():
    prev = seg("A", "First sentence.")
    cur = seg("A", "Second sentence.")
    assert compute_gap(prev, cur) == config.GAP_SAME_SPEAKER


def test_scene_break_from_offsets():
    prev = seg(NARRATOR_KEY, "End of the scene.", start=0, end=17)
    cur = seg(NARRATOR_KEY, "The next morning…", start=40, end=57)
    assert compute_gap(prev, cur) == config.GAP_SCENE_BREAK


def test_subsplit_gap():
    prev = seg("A", "Long text part one.")
    cur = seg("A", "Long text part two.")
    assert compute_gap(prev, cur, is_subsplit=True) == config.GAP_SUBSPLIT


def test_closing_quote_does_not_hide_comma():
    prev = seg("A", "“Indeed,”")  # comma inside closing quote
    cur = seg(NARRATOR_KEY, "he replied.")
    assert compute_gap(prev, cur) == config.GAP_ATTRIBUTION_TAG
