from __future__ import annotations

import json
from pathlib import Path

REAL_JSON = Path(__file__).resolve().parent.parent / "1342-pride-and-prejudice.json"

VOICE_LIZZY = {
    "sex": "female",
    "age_band": "young_adult",
    "accent": {"locale": "en-GB", "origin": "Hertfordshire", "strength": "light"},
    "social_rank": "gentry",
    "register": "educated",
    "pitch": "medium",
    "pace": "brisk",
    "timbre": "warm",
    "distinctive": "arch, teasing; lands the last word",
}

ENRICHED_BOOK = {
    "metadata": {
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "language": "English",
        "gutenberg_id": "1342",
    },
    "production": {
        "synopsis": "Elizabeth Bennet meets the proud Mr Darcy.",
        "author": {
            "name": "Jane Austen",
            "years": "1775-1817",
            "nationality": "English",
            "note": "Hampshire gentry; wry, ironic register.",
        },
        "narration": {
            "person": "third_omniscient",
            "narrator_character": None,
            "voice": {
                "sex": "female",
                "age_band": "adult",
                "accent": {"locale": "en-GB", "origin": "Hampshire", "strength": "light"},
                "register": "educated",
                "pitch": "medium",
                "pace": "measured",
                "timbre": "warm",
            },
            "basis": "author_nationality",
        },
        "casting_notes": "One accent throughout; differentiate by class and age.",
    },
    "characters": [
        {
            "name": "Elizabeth Bennet",
            "aliases": ["Lizzy", "Miss Eliza"],
            "dialogue_segments": 240,
            "description": "Second Bennet daughter; quick, playful.",
            "voice": VOICE_LIZZY,
            "confidence": "high",
            "basis": "known_work",
        },
        {
            "name": "Mr. Bennet",
            "aliases": ["father"],
            "dialogue_segments": 90,
            "voice": {
                "sex": "male",
                "age_band": "middle_aged",
                "accent": {"locale": "en-GB", "origin": "Hertfordshire", "strength": "light"},
                "pitch": "low",
                "timbre": "dry",
            },
        },
        {
            # Deliberately partial: no voice block at all.
            "name": "Mrs. Long",
            "aliases": [],
        },
    ],
    "chapters": [
        {
            "chapter": {"number": 1, "title": "Chapter I"},
            "processed": {
                "chapter_number": 1,
                "chapter_title": "Chapter I",
                "segments": [
                    {"type": "narration", "text": "Chapter I", "speaker": None,
                     "pronunciation_hints": [], "notes": None, "start": 0, "end": 9},
                    {"type": "narration",
                     "text": "It is a truth universally acknowledged.",
                     "speaker": None, "pronunciation_hints": [], "notes": None,
                     "start": 11, "end": 50},
                    {"type": "dialogue", "text": "“My dear Mr. Bennet,”",
                     "speaker": "Elizabeth Bennet", "pronunciation_hints": [],
                     "notes": None, "start": 52, "end": 73},
                    {"type": "narration", "text": "said she,", "speaker": None,
                     "pronunciation_hints": [], "notes": None, "start": 74, "end": 83},
                    {"type": "dialogue", "text": "“have you heard the news?”",
                     "speaker": "Lizzy", "pronunciation_hints": [],
                     "notes": None, "start": 84, "end": 110},
                    {"type": "dialogue", "text": "“Quoted verse.”",
                     "speaker": "Citation", "pronunciation_hints": [],
                     "notes": "citation", "start": 130, "end": 145},
                ],
            },
        }
    ],
}


def write_enriched(tmp_path: Path) -> Path:
    p = tmp_path / "enriched.json"
    p.write_text(json.dumps(ENRICHED_BOOK), encoding="utf-8")
    return p
