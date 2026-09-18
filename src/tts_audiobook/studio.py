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
from .casting import CastingChoice, cast_book, score_clip
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

# Common English given names for untitled speakers ("Jane Bennet",
# "Charles Bingley"). Skewed toward classic literature; a real voice spec
# always wins over this guess.
_FEMALE_NAMES = {
    "jane", "kitty", "lydia", "mary", "charlotte", "caroline", "maria",
    "elizabeth", "lizzy", "eliza", "georgiana", "anne", "emma", "harriet",
    "fanny", "susan", "sarah", "hannah", "margaret", "catherine", "eleanor",
    "marianne", "elinor", "amy", "beth", "meg", "jo", "alice", "lucy",
    "dorothea", "esther", "agnes", "clara", "helen", "sophia", "julia",
}
_MALE_NAMES = {
    "charles", "william", "george", "fitzwilliam", "james", "john", "henry",
    "edward", "thomas", "richard", "robert", "arthur", "frederick", "francis",
    "walter", "hugh", "philip", "peter", "david", "samuel", "joseph",
    "nicholas", "edmund", "frank", "fred", "tom", "dick", "harry",
}


def _infer_sex_from_name(name: str) -> str | None:
    """Honorific- then given-name-based fallback when there is no voice spec.

    Keeps casting from handing Mrs. Bennet (or Kitty) a male clip before the
    upstream JSON gains voice blocks; a real spec always wins over this guess.
    """
    n = name.strip().lower()
    if n.startswith(_FEMALE_TITLES):
        return "female"
    if n.startswith(_MALE_TITLES):
        return "male"
    first = n.split()[0] if n.split() else ""
    if first in _FEMALE_NAMES:
        return "female"
    if first in _MALE_NAMES:
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


def cast_summary(conn, book: Book, book_id: int) -> list[dict]:
    """One row per speaker (by importance) describing the voice cast for it.

    Speakers with no cast row still appear (voice "—") so gaps are visible;
    stale cast rows for names no longer in the book (e.g. merged away) are
    listed last and flagged.
    """
    counts = dict(speakers_by_importance(book))
    rows = {r["character"]: r for r in dbmod.cast_all(conn, book_id)}
    out: list[dict] = []

    def describe(row) -> dict:
        clip_id = row["library_clip_id"] if row else None
        clip = dbmod.clip_get(conn, int(clip_id)) if clip_id is not None else None
        if row is None or not row["ref_path"]:
            voice = "—"
        elif clip_id is not None:
            voice = f"clip #{clip_id}"
        else:
            voice = f"designed (seed {row['design_seed']})"
        if clip:
            detail = " ".join(x for x in (clip["sex"], clip["age_band"], clip["locale"],
                                          clip["region"]) if x)
            if clip["notes"]:
                detail = f"{detail} · {clip['notes']}" if detail else clip["notes"]
        elif row is not None and row["library_clip_id"] is not None:
            detail = "(clip deleted)"
        else:
            detail = ""
        return {"voice": voice, "clip_id": clip_id, "detail": detail,
                "status": row["status"] if row else "uncast",
                "engine": (row["engine"] if row else None) or ""}

    for key, n in counts.items():
        out.append({"character": key, "segments": n, "stale": False,
                    **describe(rows.pop(key, None))})
    for key, row in rows.items():
        out.append({"character": key, "segments": 0, "stale": True, **describe(row)})
    return out


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


def library_choices(conn: sqlite3.Connection, book: Book, book_id: int,
                    key: str, spec: VoiceSpec | None) -> list[dict]:
    """Every library clip, scored against `key`'s spec, best first, with the
    speakers already using each clip so a reuse is a visible choice."""
    in_use: dict[int, list[str]] = {}
    for r in dbmod.cast_all(conn, book_id):
        if r["library_clip_id"] is not None and r["character"] != key:
            name = "(narrator)" if r["character"] == NARRATOR_KEY else r["character"]
            in_use.setdefault(int(r["library_clip_id"]), []).append(name)
    out = []
    for r in dbmod.clip_list(conn):
        info = clip_info(r)
        score = score_clip(spec, info) if spec else 0.0
        out.append({"row": r, "clip_id": info.clip_id, "score": score,
                    "in_use": in_use.get(info.clip_id, [])})
    out.sort(key=lambda c: (-c["score"], c["clip_id"]))
    return out


def _choose_library_clip(conn: sqlite3.Connection, book: Book, book_id: int,
                         key: str, spec: VoiceSpec | None,
                         current_clip: int | None) -> int | None:
    """Interactive picker over the whole library: shows a scored table, lets
    the user play clips by id, and returns the chosen clip id (None = cancel)."""
    import click

    choices = library_choices(conn, book, book_id, key, spec)
    if not choices:
        rprint("[yellow]Library is empty.[/yellow]")
        return None
    table = Table(title=f"Library voices for {key}")
    for col, just in (("ID", "right"), ("Fit", "right"), ("Sex", "left"),
                      ("Age", "left"), ("Locale", "left"), ("Region", "left"),
                      ("Notes", "left"), ("Used by", "left")):
        table.add_column(col, justify=just)
    for c in choices:
        r = c["row"]
        fit = "✗" if c["score"] == float("-inf") else f"{c['score']:.0f}"
        cid = f"[bold]{c['clip_id']} ◀[/bold]" if c["clip_id"] == current_clip else str(c["clip_id"])
        table.add_row(cid, fit, r["sex"] or "-", r["age_band"] or "-",
                      r["locale"] or "-", r["region"] or "-",
                      (r["notes"] or "")[:40], ", ".join(c["in_use"]))
    rprint(table)
    rprint("[dim]Fit: casting score against this speaker's spec (✗ = sex mismatch). "
           "◀ = current voice.[/dim]")
    while True:
        ans = click.prompt("  clip id to use / [p] ID to play a clip / [c]ancel",
                           default="c", show_default=False).strip().lower()
        if ans in ("c", ""):
            return None
        play = ans.startswith("p")
        num = ans[1:].strip() if play else ans
        if not num.isdigit() or dbmod.clip_get(conn, int(num)) is None:
            rprint(f"[yellow]No clip #{num}[/yellow]")
            continue
        if play:
            play_sample(Path(dbmod.clip_get(conn, int(num))["path"]))
            continue
        return int(num)


def recast_to_clip(conn: sqlite3.Connection, book: Book, book_id: int,
                   key: str, clip_id: int, spec: VoiceSpec | None) -> None:
    """Freeze `key` to library clip `clip_id` (status back to proposed so the
    audition loop plays it before it is accepted)."""
    row = dbmod.clip_get(conn, clip_id)
    ref = build_reference(book_key=book_output_subdir(book), character=key,
                          clip_path=Path(row["path"]), clip_transcript=row["transcript"])
    dbmod.cast_upsert(
        conn, book_id, key,
        spec_json=json.dumps(asdict(spec)) if spec else None,
        library_clip_id=clip_id,
        ref_path=str(ref.path), ref_transcript=ref.transcript,
        ref_sha256=ref.sha256, design_seed=None, audition_seed=0,
        status="proposed",
    )


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
            voice = (f"clip #{row['library_clip_id']}" if row["library_clip_id"] is not None
                     else f"designed (seed {row['design_seed']})")
            choice = click.prompt(
                f"  [{voice}] [a]ccept / [r]eroll take / [l]ibrary: pick another voice / "
                "[d]esign new voice / [p]lay again / [s]kip",
                default="a", show_default=False).strip().lower()
            if choice == "a":
                dbmod.cast_upsert(conn, book_id, key, status="accepted",
                                  audition_seed=seed, engine=engine.name)
                break
            if choice == "r":
                seed += 1
                continue
            if choice == "l":
                # Re-pin to any clip in the library (e.g. a pivotal voice like
                # the narrator deserves a hand-picked choice, not the scorer's).
                spec_json = row["spec_json"]
                spec = (parse_voice(json.loads(spec_json)) if spec_json
                        else spec_for_speaker(book, key)) or spec_for_speaker(book, key)
                clip_id = _choose_library_clip(
                    conn, book, book_id, key, spec,
                    current_clip=row["library_clip_id"])
                if clip_id is None or clip_id == row["library_clip_id"]:
                    continue
                recast_to_clip(conn, book, book_id, key, clip_id, spec)
                row = dbmod.cast_get(conn, book_id, key)
                seed = 0
                rprint(f"[green]{name}[/green] → clip #{clip_id}; listening…")
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
