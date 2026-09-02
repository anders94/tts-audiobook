from __future__ import annotations

from . import config
from .book import Segment

# Text that ends mid-thought: an attribution tag follows ("..." said she) or
# the sentence itself continues in the next segment.
_NONTERMINAL_ENDINGS = (",", ";", ":", "—", "–", "-")


def _ends_nonterminal(text: str) -> bool:
    stripped = text.rstrip().rstrip("”’\"'")
    return stripped.endswith(_NONTERMINAL_ENDINGS)


def _offsets_adjacent(prev: Segment, cur: Segment) -> bool:
    if prev.end is None or cur.start is None:
        return False
    return 0 <= cur.start - prev.end <= 2


def _offsets_scene_break(prev: Segment, cur: Segment) -> bool:
    if prev.end is None or cur.start is None:
        return False
    return cur.start - prev.end >= config.SCENE_BREAK_OFFSET_GAP


def compute_gap(prev: Segment | None, cur: Segment, *,
                is_subsplit: bool = False) -> float:
    """Seconds of silence before `cur`.

    `is_subsplit` marks a continuation chunk of one long segment that was
    split at sentence boundaries (prev is then the same segment).
    """
    if prev is None:
        return 0.0
    if is_subsplit:
        return config.GAP_SUBSPLIT

    same_speaker = prev.speaker_key == cur.speaker_key

    if cur.notes == "quote-continues" or prev.notes == "quote-continues":
        # A quotation running across a paragraph break: same voice keeps
        # talking; keep the seam tight regardless of what attribution says.
        return config.GAP_QUOTE_CONTINUES

    if _offsets_scene_break(prev, cur):
        return config.GAP_SCENE_BREAK

    if not same_speaker:
        if _offsets_adjacent(prev, cur) and _ends_nonterminal(prev.text):
            # Mid-sentence split: '"My dear Mr. Bennet," | said his lady | "..."'
            return config.GAP_MIDSENTENCE_SPLIT
        if _ends_nonterminal(prev.text):
            return config.GAP_ATTRIBUTION_TAG
        return config.GAP_SPEAKER_CHANGE

    return config.GAP_SAME_SPEAKER
