from __future__ import annotations

from tts_audiobook.casting import ClipInfo, cast_book, score_clip
from tts_audiobook.specs import AccentSpec, VoiceSpec


def clip(cid: int, sex: str, age: str, locale: str = "en-GB",
         region: str = "Southern England", quality: str = "good",
         notes: str | None = None) -> ClipInfo:
    return ClipInfo(clip_id=cid, sex=sex, age_band=age, locale=locale,
                    region=region, quality=quality, notes=notes)


def spec(sex: str, age: str, locale: str = "en-GB", origin: str = "Hertfordshire",
         strength: str = "light", **kw) -> VoiceSpec:
    return VoiceSpec(sex=sex, age_band=age,
                     accent=AccentSpec(locale=locale, origin=origin, strength=strength),
                     **kw)


LIBRARY = [
    clip(1, "female", "young_adult"),
    clip(2, "female", "adult"),
    clip(3, "female", "elderly", region="Northern England"),
    clip(4, "male", "middle_aged"),
    clip(5, "male", "young_adult"),
    clip(6, "male", "adult", locale="en-US", region="Midwest"),
]


def test_sex_hard_filter():
    s = spec("female", "young_adult")
    assert score_clip(s, clip(9, "male", "young_adult")) == float("-inf")


def test_locale_and_region_scoring():
    s = spec("female", "young_adult")
    gb = score_clip(s, LIBRARY[0])
    us = score_clip(s, ClipInfo(clip_id=9, sex="female", age_band="young_adult",
                                locale="en-US", region="Midwest", quality="good",
                                notes=None))
    assert gb > us


def test_age_distance_penalty():
    s = spec("female", "young_adult")
    young = score_clip(s, LIBRARY[0])
    old = score_clip(s, LIBRARY[2])
    assert young > old


def test_cast_is_deterministic_and_unique_for_leads():
    specs = [
        ("Elizabeth Bennet", spec("female", "young_adult"), 538),
        ("Jane Bennet", spec("female", "young_adult"), 141),
        ("Mr. Bennet", spec("male", "middle_aged"), 135),
    ]
    a = cast_book(specs, LIBRARY)
    b = cast_book(specs, LIBRARY)
    assert [(c.character, c.clip_id) for c in a] == [(c.character, c.clip_id) for c in b]
    lizzy = next(c for c in a if c.character == "Elizabeth Bennet")
    jane = next(c for c in a if c.character == "Jane Bennet")
    # Both want the same clip; the reuse penalty must force different ones.
    assert lizzy.clip_id != jane.clip_id
    assert lizzy.clip_id == 1  # best match goes to the more prominent character


def test_minor_characters_may_share():
    specs = [(f"Extra {i}", spec("male", "middle_aged"), 1) for i in range(4)]
    result = cast_book(specs, [LIBRARY[3]])  # only one matching male clip
    assert all(c.clip_id == 4 for c in result)


def test_no_candidates_yields_none():
    result = cast_book([("Ghost", spec("female", "adult"), 5)],
                       [clip(1, "male", "adult")])
    assert result[0].clip_id is None
