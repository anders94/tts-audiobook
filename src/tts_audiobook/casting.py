from __future__ import annotations

import math
from dataclasses import dataclass

from .specs import AGE_BANDS, VoiceSpec

# Map upstream accent origins (counties, cities) to the macro-regions library
# clips are tagged with. Lowercase keys; matching is substring-based both ways.
REGION_MAP = {
    "hertfordshire": "southern england",
    "kent": "southern england",
    "surrey": "southern england",
    "sussex": "southern england",
    "hampshire": "southern england",
    "london": "london",
    "derbyshire": "northern england",
    "yorkshire": "northern england",
    "lancashire": "northern england",
    "northumberland": "northern england",
    "cornwall": "west country",
    "devon": "west country",
    "somerset": "west country",
    "edinburgh": "scotland",
    "glasgow": "scotland",
    "dublin": "ireland",
    "belfast": "northern ireland",
    "cardiff": "wales",
}

SCORE_LOCALE_EXACT = 30.0
SCORE_REGION_MATCH = 15.0
SCORE_LANGUAGE_FAMILY = 5.0
SCORE_AGE_EXACT = 20.0
SCORE_AGE_STEP_PENALTY = 8.0
SCORE_KEYWORD = 2.0
SCORE_QUALITY_GOOD = 3.0
# Must comfortably exceed any single attribute-mismatch penalty, so two
# prominent characters always prefer distinct clips over sharing the best one.
REUSE_PENALTY_CAP = 40.0


@dataclass
class ClipInfo:
    """Casting-relevant view of a library_clips row."""
    clip_id: int
    sex: str | None
    age_band: str | None
    locale: str | None
    region: str | None
    quality: str | None
    notes: str | None


def _macro_region(origin: str | None) -> str | None:
    if not origin:
        return None
    o = origin.strip().lower()
    if o in REGION_MAP:
        return REGION_MAP[o]
    return o


def _region_matches(spec_origin: str | None, clip_region: str | None) -> bool:
    if not spec_origin or not clip_region:
        return False
    a = _macro_region(spec_origin) or ""
    b = clip_region.strip().lower()
    return bool(a) and (a in b or b in a)


def _language(locale: str | None) -> str | None:
    if not locale:
        return None
    return locale.split("-")[0].lower()


def _age_distance(a: str | None, b: str | None) -> int | None:
    if a not in AGE_BANDS or b not in AGE_BANDS:
        return None
    return abs(AGE_BANDS.index(a) - AGE_BANDS.index(b))


def score_clip(spec: VoiceSpec, clip: ClipInfo) -> float:
    """Deterministic fit score; -inf marks a hard sex mismatch."""
    score = 0.0

    if spec.sex and clip.sex and spec.sex != clip.sex:
        return float("-inf")

    spec_locale = (spec.accent.locale or "").lower()
    clip_locale = (clip.locale or "").lower()
    light = (spec.accent.strength or "").lower() == "light"
    if spec_locale and clip_locale:
        if spec_locale == clip_locale:
            score += SCORE_LOCALE_EXACT
        elif _language(spec_locale) == _language(clip_locale):
            score += SCORE_LANGUAGE_FAMILY
            # A "light" accent spec tolerates a same-language locale mismatch.
            if not light:
                score -= SCORE_LANGUAGE_FAMILY
    if _region_matches(spec.accent.origin, clip.region):
        score += SCORE_REGION_MATCH

    dist = _age_distance(spec.age_band, clip.age_band)
    if dist == 0:
        score += SCORE_AGE_EXACT
    elif dist is not None:
        score -= SCORE_AGE_STEP_PENALTY * dist

    haystack = f"{clip.notes or ''} {clip.quality or ''}".lower()
    for kw in filter(None, [spec.pitch, spec.timbre]):
        if kw.lower() in haystack:
            score += SCORE_KEYWORD
    if (clip.quality or "").lower() == "good":
        score += SCORE_QUALITY_GOOD

    return score


def reuse_penalty(dialogue_a: int, dialogue_b: int) -> float:
    """Cost of giving one clip to two characters, sized by their prominence."""
    return min(REUSE_PENALTY_CAP, 0.5 * math.sqrt(max(0, dialogue_a) * max(0, dialogue_b)))


@dataclass
class CastingChoice:
    character: str
    clip_id: int | None
    score: float


def cast_book(specs: list[tuple[str, VoiceSpec, int]],
              clips: list[ClipInfo]) -> list[CastingChoice]:
    """Assign clips to (character, spec, dialogue_count) triples.

    Characters are cast most-prominent first; reusing an already-assigned clip
    costs `reuse_penalty` so leads never share while one-liners may.
    Fully deterministic: ties break on clip_id.
    """
    assigned: dict[int, list[int]] = {}  # clip_id -> dialogue counts of holders
    out: list[CastingChoice] = []
    ordered = sorted(specs, key=lambda t: (-t[2], t[0]))
    for character, spec, dialogue in ordered:
        best: tuple[float, int] | None = None  # (score, clip_id)
        for clip in clips:
            s = score_clip(spec, clip)
            if s == float("-inf"):
                continue
            for other_dialogue in assigned.get(clip.clip_id, []):
                s -= reuse_penalty(dialogue, other_dialogue)
            if best is None or (s, -clip.clip_id) > (best[0], -best[1]):
                best = (s, clip.clip_id)
        if best is None:
            out.append(CastingChoice(character=character, clip_id=None, score=float("-inf")))
            continue
        assigned.setdefault(best[1], []).append(dialogue)
        out.append(CastingChoice(character=character, clip_id=best[1], score=best[0]))
    return out
