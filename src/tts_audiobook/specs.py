from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Ordered coarse age scale used by the casting scorer. Upstream values that
# don't match are kept verbatim (scored as unknown), so new bands degrade
# gracefully rather than crash.
AGE_BANDS = ["child", "teen", "young_adult", "adult", "middle_aged", "elderly"]

_AGE_SYNONYMS = {
    "boy": "child", "girl": "child", "youth": "teen", "teenager": "teen",
    "young": "young_adult", "young adult": "young_adult",
    "middle-aged": "middle_aged", "middle aged": "middle_aged",
    "old": "elderly", "senior": "elderly", "aged": "elderly",
}


def normalize_age_band(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower().replace("-", "_").replace(" ", "_")
    if v in AGE_BANDS:
        return v
    return _AGE_SYNONYMS.get(value.strip().lower(), v)


def normalize_sex(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower()
    if v in ("f", "female", "woman", "girl"):
        return "female"
    if v in ("m", "male", "man", "boy"):
        return "male"
    return v


@dataclass
class AccentSpec:
    locale: str | None = None      # e.g. "en-GB"
    origin: str | None = None      # e.g. "Hertfordshire"
    strength: str | None = None    # e.g. "light"


@dataclass
class VoiceSpec:
    sex: str | None = None
    age_band: str | None = None
    accent: AccentSpec = field(default_factory=AccentSpec)
    social_rank: str | None = None
    register: str | None = None
    pitch: str | None = None
    pace: str | None = None
    timbre: str | None = None
    distinctive: str | None = None

    def describe(self) -> str:
        """Render a natural-language description (for voice-design prompting)."""
        bits: list[str] = []
        age = (self.age_band or "adult").replace("_", " ")
        sex = self.sex or "neutral"
        bits.append(f"A {age} {sex} voice")
        if self.accent.locale or self.accent.origin:
            acc = self.accent.origin or self.accent.locale
            strength = f"{self.accent.strength} " if self.accent.strength else ""
            bits.append(f"with a {strength}{acc} accent")
        if self.pitch:
            bits.append(f"{self.pitch} pitch")
        if self.pace:
            bits.append(f"{self.pace} pace")
        if self.timbre:
            bits.append(f"{self.timbre} timbre")
        if self.register:
            bits.append(f"{self.register} register")
        if self.distinctive:
            bits.append(self.distinctive)
        return ", ".join(bits) + "."


@dataclass
class CharacterSpec:
    name: str
    aliases: list[str] = field(default_factory=list)
    dialogue_segments: int = 0
    description: str | None = None
    voice: VoiceSpec | None = None
    confidence: str | None = None
    basis: str | None = None


@dataclass
class Production:
    synopsis: str | None = None
    author_name: str | None = None
    author_years: str | None = None
    author_nationality: str | None = None
    author_note: str | None = None
    narration_person: str | None = None
    narrator_character: str | None = None
    narrator_voice: VoiceSpec | None = None
    narration_basis: str | None = None
    casting_notes: str | None = None


def _get(d: Any, key: str) -> Any:
    return d.get(key) if isinstance(d, dict) else None


def parse_voice(data: Any) -> VoiceSpec | None:
    if not isinstance(data, dict) or not data:
        return None
    accent_raw = _get(data, "accent")
    accent = AccentSpec(
        locale=_get(accent_raw, "locale"),
        origin=_get(accent_raw, "origin"),
        strength=_get(accent_raw, "strength"),
    )
    return VoiceSpec(
        sex=normalize_sex(_get(data, "sex")),
        age_band=normalize_age_band(_get(data, "age_band")),
        accent=accent,
        social_rank=_get(data, "social_rank"),
        register=_get(data, "register"),
        pitch=_get(data, "pitch"),
        pace=_get(data, "pace"),
        timbre=_get(data, "timbre"),
        distinctive=_get(data, "distinctive"),
    )


def parse_production(data: Any) -> Production | None:
    prod = _get(data, "production")
    if not isinstance(prod, dict):
        return None
    author = _get(prod, "author") or {}
    narration = _get(prod, "narration") or {}
    return Production(
        synopsis=_get(prod, "synopsis"),
        author_name=_get(author, "name"),
        author_years=_get(author, "years"),
        author_nationality=_get(author, "nationality"),
        author_note=_get(author, "note"),
        narration_person=_get(narration, "person"),
        narrator_character=_get(narration, "narrator_character"),
        narrator_voice=parse_voice(_get(narration, "voice")),
        narration_basis=_get(narration, "basis"),
        casting_notes=_get(prod, "casting_notes"),
    )


def parse_characters(data: Any) -> list[CharacterSpec]:
    out: list[CharacterSpec] = []
    for c in _get(data, "characters") or []:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        out.append(CharacterSpec(
            name=c["name"],
            aliases=list(c.get("aliases") or []),
            dialogue_segments=int(c.get("dialogue_segments") or 0),
            description=c.get("description"),
            voice=parse_voice(c.get("voice")),
            confidence=c.get("confidence"),
            basis=c.get("basis"),
        ))
    return out
