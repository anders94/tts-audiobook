from __future__ import annotations

import re
from dataclasses import dataclass

from . import config
from .book import Chapter, Segment
from .pacing import compute_gap
from .textnorm import apply_pronunciation

# Sentence boundary: terminal punctuation (plus closing quotes) then whitespace
# before a capital or opening quote. Ported from v1.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"'])")


def split_long(text: str, limit: int = config.MAX_ITEM_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts = _SENT_SPLIT.split(text)
    out: list[str] = []
    buf = ""
    for p in parts:
        if not buf:
            buf = p
            continue
        if len(buf) + 1 + len(p) <= limit:
            buf = f"{buf} {p}"
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    if any(len(x) > limit for x in out):
        out = [x[i:i + limit] for x in out for i in range(0, len(x), limit)]
    return out


@dataclass
class RenderItem:
    index: int            # strict render order within the chapter
    speaker_key: str
    text: str
    gap_before_s: float


def plan_chapter(chapter: Chapter) -> list[RenderItem]:
    """Flatten a chapter into ordered RenderItems with pre-computed gaps.

    Rendering may batch items by voice in any order; reassembly must sort by
    `index` and honor `gap_before_s`.
    """
    items: list[RenderItem] = []
    prev: Segment | None = None
    for seg in chapter.segments:
        text = apply_pronunciation(seg.text, seg.pronunciation_hints)
        chunks = split_long(text)
        for i, chunk in enumerate(chunks):
            gap = compute_gap(prev if i == 0 else seg, seg, is_subsplit=(i > 0))
            items.append(RenderItem(
                index=len(items),
                speaker_key=seg.speaker_key,
                text=chunk,
                gap_before_s=gap,
            ))
        prev = seg
    return items


def bucket_by_speaker(items: list[RenderItem]) -> dict[str, list[RenderItem]]:
    buckets: dict[str, list[RenderItem]] = {}
    for item in items:
        buckets.setdefault(item.speaker_key, []).append(item)
    return buckets
