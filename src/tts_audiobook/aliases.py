from __future__ import annotations

from typing import Iterable


def build_alias_map(characters: Iterable[dict]) -> dict[str, str]:
    """alias-or-canonical-name -> canonical-name. Self-mappings included."""
    canonicals: set[str] = {c["name"] for c in characters}
    mapping: dict[str, str] = {name: name for name in canonicals}
    for c in characters:
        canonical = c["name"]
        for alias in c.get("aliases", []) or []:
            if not alias:
                continue
            # If the alias *is* itself a canonical character name, leave it alone —
            # the JSON occasionally lists a sibling character as an alias by mistake.
            if alias in canonicals and alias != canonical:
                continue
            mapping.setdefault(alias, canonical)
    return mapping


def canonicalize(speaker: str | None, alias_map: dict[str, str]) -> str | None:
    """Return the canonical character name for a raw speaker string, or None for narration."""
    if speaker is None:
        return None
    return alias_map.get(speaker, speaker)
