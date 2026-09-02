from __future__ import annotations

from tts_audiobook import config
from tts_audiobook.book import Chapter, Segment
from tts_audiobook.config import NARRATOR_KEY
from tts_audiobook.planning import bucket_by_speaker, plan_chapter, split_long


def seg(speaker: str, text: str, **kw) -> Segment:
    return Segment(speaker_key=speaker, text=text, raw_speaker=None, **kw)


def test_split_long_respects_sentences():
    text = "One sentence here. " * 80  # ~1520 chars
    parts = split_long(text.strip(), limit=800)
    assert len(parts) >= 2
    assert all(len(p) <= 800 for p in parts)
    assert " ".join(parts) == text.strip()


def test_split_long_short_passthrough():
    assert split_long("Short.") == ["Short."]


def test_plan_preserves_order_and_indexes():
    ch = Chapter(number=1, title="Chapter I", segments=[
        seg(NARRATOR_KEY, "Narration one."),
        seg("A", "“Dialogue,”"),
        seg(NARRATOR_KEY, "said A."),
        seg("A", "“More dialogue.”"),
    ])
    items = plan_chapter(ch)
    assert [i.index for i in items] == [0, 1, 2, 3]
    assert [i.speaker_key for i in items] == [NARRATOR_KEY, "A", NARRATOR_KEY, "A"]
    assert items[0].gap_before_s == 0.0


def test_plan_splits_long_segment_with_subsplit_gaps():
    long_text = ("A full sentence right here. " * 60).strip()  # > 800 chars
    ch = Chapter(number=1, title="t", segments=[seg("A", long_text)])
    items = plan_chapter(ch)
    assert len(items) > 1
    assert items[0].gap_before_s == 0.0
    assert all(i.gap_before_s == config.GAP_SUBSPLIT for i in items[1:])
    assert all(i.speaker_key == "A" for i in items)


def test_pronunciation_applied_in_planning():
    ch = Chapter(number=1, title="t", segments=[
        Segment(speaker_key="A", text="Lefroy spoke.", raw_speaker="A",
                pronunciation_hints=["Lefroy → Leff-roy"]),
    ])
    items = plan_chapter(ch)
    assert items[0].text == "Leff-roy spoke."


def test_bucket_by_speaker_keeps_items():
    ch = Chapter(number=1, title="t", segments=[
        seg(NARRATOR_KEY, "n1"), seg("A", "a1"), seg(NARRATOR_KEY, "n2"),
    ])
    items = plan_chapter(ch)
    buckets = bucket_by_speaker(items)
    assert sorted(buckets) == ["A", NARRATOR_KEY]
    assert [i.index for i in buckets[NARRATOR_KEY]] == [0, 2]
