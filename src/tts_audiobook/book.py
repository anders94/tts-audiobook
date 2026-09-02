from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .aliases import build_alias_map, canonicalize
from .config import NARRATOR_KEY
from .specs import CharacterSpec, Production, parse_characters, parse_production
from .textnorm import normalize_heading


@dataclass
class Segment:
    speaker_key: str   # NARRATOR_KEY or canonical character name
    text: str
    raw_speaker: str | None
    pronunciation_hints: list[str] = field(default_factory=list)
    seg_type: str = "narration"
    notes: str | None = None
    start: int | None = None   # offsets into the chapter reading_text
    end: int | None = None


@dataclass
class Chapter:
    number: int
    title: str
    segments: list[Segment]


@dataclass
class Book:
    source_path: Path
    title: str
    author: str | None
    gutenberg_id: str | None
    language: str
    chapters: list[Chapter]
    characters: list[CharacterSpec]
    alias_map: dict[str, str]
    production: Production | None


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 60) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return s[:max_len].rstrip("-") or "untitled"


def load_book(path: Path) -> Book:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {}) or {}
    raw_characters = data.get("characters", []) or []
    alias_map = build_alias_map(raw_characters)
    characters = parse_characters(data)
    production = parse_production(data)

    chapters: list[Chapter] = []
    for ch in data.get("chapters", []):
        proc = ch.get("processed") or {}
        ch_meta = ch.get("chapter") or {}
        number = int(proc.get("chapter_number") or ch_meta.get("number") or len(chapters) + 1)
        title = (proc.get("chapter_title") or ch_meta.get("title") or f"Chapter {number}").strip()
        segs: list[Segment] = []
        for s in proc.get("segments", []) or []:
            text = (s.get("text") or "").strip()
            if not text:
                continue
            text = normalize_heading(text, chapter_number=number,
                                     is_title=(text == title))
            raw_speaker = s.get("speaker")
            seg_type = s.get("type") or "narration"
            notes = s.get("notes")
            # Citations are read by the narrator; "Citation" is a reserved
            # upstream label, not a character (v1 leaked it as assignable).
            if (seg_type == "narration" or raw_speaker is None
                    or raw_speaker == "Citation" or notes == "citation"):
                key = NARRATOR_KEY
            else:
                canonical = canonicalize(raw_speaker, alias_map)
                key = canonical if canonical and canonical != "Unknown" else NARRATOR_KEY
            segs.append(Segment(
                speaker_key=key,
                text=text,
                raw_speaker=raw_speaker,
                pronunciation_hints=list(s.get("pronunciation_hints") or []),
                seg_type=seg_type,
                notes=notes,
                start=s.get("start"),
                end=s.get("end"),
            ))
        chapters.append(Chapter(number=number, title=title, segments=segs))

    return Book(
        source_path=path,
        title=meta.get("title") or path.stem,
        author=meta.get("author"),
        gutenberg_id=str(meta.get("gutenberg_id")) if meta.get("gutenberg_id") else None,
        language=meta.get("language") or "English",
        chapters=chapters,
        characters=characters,
        alias_map=alias_map,
        production=production,
    )


def speaker_counts(book: Book) -> Counter[str]:
    """How many segments each speaker_key has, across the whole book."""
    c: Counter[str] = Counter()
    for ch in book.chapters:
        for s in ch.segments:
            c[s.speaker_key] += 1
    return c


def speakers_by_importance(book: Book) -> list[tuple[str, int]]:
    """Narrator first; then characters by segment count desc, name asc."""
    counts = speaker_counts(book)
    narrator = (NARRATOR_KEY, counts.get(NARRATOR_KEY, 0))
    others = sorted(
        ((k, v) for k, v in counts.items() if k != NARRATOR_KEY),
        key=lambda kv: (-kv[1], kv[0]),
    )
    return [narrator, *others]


def character_spec_for(book: Book, speaker_key: str) -> CharacterSpec | None:
    for c in book.characters:
        if c.name == speaker_key:
            return c
    return None


def sample_line_for(book: Book, speaker_key: str, *, min_len: int = 0,
                    max_len: int | None = None) -> str | None:
    """First line for a speaker; prefers one whose length fits [min_len, max_len]."""
    fallback: str | None = None
    for ch in book.chapters:
        for s in ch.segments:
            if s.speaker_key != speaker_key:
                continue
            if fallback is None:
                fallback = s.text
            if len(s.text) >= min_len and (max_len is None or len(s.text) <= max_len):
                return s.text
    return fallback


def book_output_subdir(book: Book) -> str:
    gid = book.gutenberg_id or "x"
    return f"{gid}-{slugify(book.title)}"
