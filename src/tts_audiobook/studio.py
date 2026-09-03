from __future__ import annotations

import json
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path

import soundfile as sf
from rich import print as rprint
from rich.table import Table

from . import db as dbmod
from .book import Book, book_output_subdir, character_spec_for, sample_line_for, speakers_by_importance
from .casting import CastingChoice, cast_book
from .config import NARRATOR_KEY
from .engines.base import Engine
from .library import clip_info, play_sample
from .specs import AccentSpec, VoiceSpec, parse_voice
from .voicebuild import build_designed_reference, build_reference

# Narrator fallback when the JSON has no production block: a neutral,
# educated adult voice in the book's presumed locale.
DEFAULT_NARRATOR_SPEC = VoiceSpec(
    sex=None, age_band="adult",
    accent=AccentSpec(locale="en-GB", origin=None, strength="light"),
    register="educated", pitch="medium", pace="measured", timbre="warm",
)

DEFAULT_CHARACTER_SPEC = VoiceSpec(sex=None, age_band="adult",
                                   accent=AccentSpec(locale="en-GB"))

_FEMALE_TITLES = ("mrs.", "mrs ", "miss ", "lady ", "madam", "mademoiselle",
                  "mme.", "ms.", "aunt ", "queen ")
_MALE_TITLES = ("mr.", "mr ", "sir ", "lord ", "colonel ", "captain ",
                "general ", "major ", "master ", "uncle ", "king ", "count ",
                "duke ", "monsieur", "dr ", "reverend ", "rev.")


def _infer_sex_from_name(name: str) -> str | None:
    """Honorific-based fallback when a character has no voice spec.

    Keeps casting from handing Mrs. Bennet a male clip before the upstream
    JSON gains voice blocks; a real spec always wins over this guess.
    """
    n = name.strip().lower()
    if n.startswith(_FEMALE_TITLES):
        return "female"
    if n.startswith(_MALE_TITLES):
        return "male"
    return None


def spec_for_speaker(book: Book, speaker_key: str) -> VoiceSpec:
    if speaker_key == NARRATOR_KEY:
        prod = book.production
        if prod:
            if prod.narrator_character:
                cs = character_spec_for(book, prod.narrator_character)
                if cs and cs.voice:
                    return cs.voice
            if prod.narrator_voice:
                return prod.narrator_voice
        return DEFAULT_NARRATOR_SPEC
    cs = character_spec_for(book, speaker_key)
    if cs and cs.voice:
        return cs.voice
    sex = _infer_sex_from_name(speaker_key)
    if sex:
        return VoiceSpec(sex=sex, age_band=DEFAULT_CHARACTER_SPEC.age_band,
                         accent=DEFAULT_CHARACTER_SPEC.accent)
    return DEFAULT_CHARACTER_SPEC


def _design_seed_for(character: str) -> int:
    """Stable per-character starting seed so a recast reproduces the voice."""
    import zlib
    return zlib.crc32(character.encode()) & 0xFFFF


def run_casting(conn: sqlite3.Connection, book: Book, book_id: int,
                *, recast: bool = False, design: bool = False) -> None:
    """Freeze a reference voice for every speaker.

    Default: match against the accent clip library, falling back to a
    VoiceDesign-generated voice when no clip qualifies. With design=True,
    every voice is generated from its spec — fully synthetic, no library.
    """
    clips_rows = dbmod.clip_list(conn)
    if not clips_rows and not design:
        raise RuntimeError("Accent library is empty; import clips first "
                           "(tts-audiobook library import …) or cast --design.")
    clips = [clip_info(r) for r in clips_rows]
    rows_by_id = {int(r["id"]): r for r in clips_rows}

    counts = dict(speakers_by_importance(book))
    triples = []
    for key, seg_count in counts.items():
        existing = dbmod.cast_get(conn, book_id, key)
        if existing and existing["ref_path"] and not recast:
            continue
        triples.append((key, spec_for_speaker(book, key), seg_count))
    if not triples:
        rprint("[green]Cast is already complete.[/green] Use --recast to redo.")
        return

    book_key = book_output_subdir(book)
    specs = {t[0]: t[1] for t in triples}
    if design:
        choices = [CastingChoice(character=t[0], clip_id=None, score=0.0)
                   for t in triples]
    else:
        choices = cast_book(triples, clips)

    table = Table(title="Casting")
    table.add_column("Speaker")
    table.add_column("Voice", justify="right")
    table.add_column("Score", justify="right")
    for choice in choices:
        spec = specs[choice.character]
        if choice.clip_id is None:
            if not design:
                rprint(f"[yellow]No library clip matches {choice.character!r}; "
                       "designing a synthetic voice instead.[/yellow]")
            seed = _design_seed_for(choice.character)
            ref = build_designed_reference(
                book_key=book_key, character=choice.character,
                spec=spec, seed=seed)
            label = f"designed (seed {seed})"
        else:
            row = rows_by_id[choice.clip_id]
            ref = build_reference(
                book_key=book_key, character=choice.character,
                clip_path=Path(row["path"]), clip_transcript=row["transcript"])
            label = f"clip #{choice.clip_id}"
        dbmod.cast_upsert(
            conn, book_id, choice.character,
            spec_json=json.dumps(asdict(spec)),
            library_clip_id=choice.clip_id,
            ref_path=str(ref.path),
            ref_transcript=ref.transcript,
            ref_sha256=ref.sha256,
            design_seed=ref.design_seed,
            audition_seed=0,
            status="proposed",
        )
        table.add_row(choice.character, label,
                      "—" if design else f"{choice.score:.0f}")
    rprint(table)


def signature_line(book: Book, speaker_key: str) -> str:
    line = sample_line_for(book, speaker_key, min_len=40, max_len=200)
    return line or "The evening settled softly over the quiet town."


def _play_wav(wav, sr) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf.write(str(tmp_path), wav, sr, subtype="PCM_16")
        play_sample(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def run_audition(conn: sqlite3.Connection, engine: Engine, book: Book,
                 book_id: int, *, only_character: str | None = None,
                 auto_accept: bool = False) -> None:
    import click

    if not auto_accept:
        engine.load()
    for key, seg_count in speakers_by_importance(book):
        if only_character and key != only_character:
            continue
        row = dbmod.cast_get(conn, book_id, key)
        if not row or not row["ref_path"]:
            rprint(f"[yellow]{key}: not cast yet; skipping.[/yellow]")
            continue
        if row["status"] == "accepted" and not only_character:
            continue
        if auto_accept:
            dbmod.cast_upsert(conn, book_id, key, status="accepted",
                              engine=engine.name)
            continue

        line = signature_line(book, key)
        seed = int(row["audition_seed"] or 0)
        name = "(narrator)" if key == NARRATOR_KEY else key
        rprint(f"\n[bold]{name}[/bold] ({seg_count} segments)")
        rprint(f"[dim]{line}[/dim]")
        while True:
            prompt = engine.clone_prompt(Path(row["ref_path"]),
                                         row["ref_transcript"] or "")
            wavs, sr = engine.generate([line], prompt,
                                       language=book.language, seed=seed)
            _play_wav(wavs[0], sr)
            choice = click.prompt(
                "  [a]ccept / [r]eroll take / [d]esign new voice / "
                "[p]lay again / [s]kip",
                default="a", show_default=False).strip().lower()
            if choice == "a":
                dbmod.cast_upsert(conn, book_id, key, status="accepted",
                                  audition_seed=seed, engine=engine.name)
                break
            if choice == "r":
                seed += 1
                continue
            if choice == "d":
                # New synthetic identity from the spec (not just a new take).
                spec_json = row["spec_json"]
                spec = (parse_voice(json.loads(spec_json)) if spec_json
                        else spec_for_speaker(book, key)) or spec_for_speaker(book, key)
                new_seed = int(row["design_seed"] or _design_seed_for(key)) + 1
                rprint(f"[dim]Designing a new voice (seed {new_seed})…[/dim]")
                ref = build_designed_reference(
                    book_key=book_output_subdir(book), character=key,
                    spec=spec, seed=new_seed)
                dbmod.cast_upsert(conn, book_id, key,
                                  library_clip_id=None,
                                  ref_path=str(ref.path),
                                  ref_transcript=ref.transcript,
                                  ref_sha256=ref.sha256,
                                  design_seed=ref.design_seed,
                                  status="proposed")
                row = dbmod.cast_get(conn, book_id, key)
                continue
            if choice == "p":
                continue
            if choice == "s":
                break
