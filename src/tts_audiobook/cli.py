from __future__ import annotations

from pathlib import Path

import click
from rich import print as rprint
from rich.table import Table

from . import db as dbmod
from .book import (Book, apply_speaker_merges, book_output_subdir,
                   character_spec_for, load_book, speakers_by_importance)
from .config import NARRATOR_KEY, OUTPUT_ROOT, ensure_dirs


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Spec-driven audiobook studio: cast, audition, render, package."""
    ensure_dirs()


def _open_book(book_path: Path) -> Book:
    return load_book(book_path)


def _resolve_output_dir(book: Book, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    return (OUTPUT_ROOT / book_output_subdir(book)).resolve()


def _ensure_book_row(conn, book: Book, output_dir: Path) -> int:
    """Upsert the book row and apply its stored speaker merges to `book`."""
    book_id = dbmod.book_upsert(
        conn, book.source_path,
        title=book.title, author=book.author,
        gutenberg_id=book.gutenberg_id,
        output_dir=output_dir,
    )
    apply_speaker_merges(book, dbmod.merges_get_all(conn, book_id))
    return book_id


def _engine_for(conn, book_id: int, override: str | None):
    from .engines.base import get_engine
    if override:
        return get_engine(override)
    row = dbmod.book_get(conn, book_id)
    return get_engine((row["engine"] if row and row["engine"] else None) or "qwen")


# ---------- library ----------

@main.group()
def library() -> None:
    """Manage the tagged accent clip library."""


@library.command("import")
@click.argument("audio_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--sex", type=click.Choice(["male", "female"]), default=None)
@click.option("--age-band", "age_band",
              type=click.Choice(["child", "teen", "young_adult", "adult",
                                 "middle_aged", "elderly"]), default=None)
@click.option("--locale", default=None, help="e.g. en-GB")
@click.option("--region", default=None, help="e.g. 'Southern England', Yorkshire")
@click.option("--quality", type=click.Choice(["good", "ok", "poor"]), default="good")
@click.option("--source", default=None, help="vctk | librivox | common-voice | custom")
@click.option("--license", "license_", default=None)
@click.option("--notes", default=None, help="free text; casting matches pitch/timbre words here")
@click.option("--transcript", default=None, help="If omitted, Whisper auto-transcribes.")
def library_import(audio_path: Path, sex, age_band, locale, region, quality,
                   source, license_, notes, transcript) -> None:
    """Add a reference clip to the accent library."""
    from .library import LibraryError, import_clip
    with dbmod.db() as conn:
        try:
            row = import_clip(conn, audio_path, sex=sex, age_band=age_band,
                              locale=locale, region=region, quality=quality,
                              source=source, license=license_, notes=notes,
                              transcript=transcript)
        except LibraryError as e:
            raise click.ClickException(str(e)) from e
    rprint(f"[green]Imported clip #{row['id']}[/green] → {row['path']}")
    rprint(f"[dim]Transcript:[/dim] {row['transcript']}")


@library.command("list")
def library_list() -> None:
    """Show all library clips."""
    with dbmod.db() as conn:
        rows = dbmod.clip_list(conn)
    if not rows:
        rprint("[yellow]Library is empty.[/yellow] Try: tts-audiobook library import PATH …")
        return
    table = Table(title="Accent clip library")
    for col in ("ID", "Sex", "Age", "Locale", "Region", "Qual", "Dur", "Source", "Notes"):
        table.add_column(col)
    for r in rows:
        table.add_row(str(r["id"]), r["sex"] or "-", r["age_band"] or "-",
                      r["locale"] or "-", r["region"] or "-", r["quality"] or "-",
                      f"{r['duration_s']:.1f}s", r["source"] or "-",
                      (r["notes"] or "")[:30])
    rprint(table)


@library.command("morph")
@click.argument("src_id", type=int)
@click.option("--preset", "presets", multiple=True,
              type=click.Choice(["deeper", "lighter", "older", "younger"]),
              help="Named morphs; may repeat, applied together.")
@click.option("--pitch", "pitch_semitones", type=float, default=None,
              help="Median pitch shift in semitones (e.g. -2.5).")
@click.option("--formant", "formant_ratio", type=float, default=None,
              help="Formant ratio: <1 deeper/older, >1 lighter/younger.")
@click.option("--range", "pitch_range_factor", type=float, default=None,
              help="Intonation range factor: <1 flatter, >1 livelier.")
@click.option("--tempo", type=float, default=None,
              help="Speech rate factor: >1 faster, <1 slower. Changes rhythm, "
                   "a strong identity cue.")
@click.option("--sex", type=click.Choice(["male", "female"]), default=None,
              help="Override tag (a strong morph can cross it).")
@click.option("--age-band", "age_band",
              type=click.Choice(["child", "teen", "young_adult", "adult",
                                 "middle_aged", "elderly"]), default=None)
@click.option("--notes", default=None)
def library_morph(src_id: int, presets, pitch_semitones, formant_ratio,
                  pitch_range_factor, tempo, sex, age_band, notes) -> None:
    """Derive a new-sounding voice from clip SRC_ID, keeping its accent."""
    from .library import LibraryError, import_clip_array, load_clip_audio
    from .morph import PRESETS, morph

    params: dict[str, float] = {}
    for p in presets:
        params.update(PRESETS[p])
    if pitch_semitones is not None:
        params["pitch_semitones"] = pitch_semitones
    if formant_ratio is not None:
        params["formant_ratio"] = formant_ratio
    if pitch_range_factor is not None:
        params["pitch_range_factor"] = pitch_range_factor
    if tempo is not None:
        params["tempo"] = tempo
    if not params:
        raise click.ClickException("Give at least one --preset or manual knob.")

    with dbmod.db() as conn:
        src = dbmod.clip_get(conn, src_id)
        if not src:
            raise click.ClickException(f"No clip #{src_id}")
        wav, sr = load_clip_audio(src)
        out = morph(wav, sr, **params)
        desc = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
        try:
            row = import_clip_array(
                conn, out, sr,
                transcript=src["transcript"],  # words are unchanged
                sex=sex or src["sex"], age_band=age_band or src["age_band"],
                locale=src["locale"], region=src["region"],
                quality=src["quality"], source=f"morph:{src_id}",
                license=src["license"],
                notes=notes or f"morph of #{src_id} ({desc})")
        except LibraryError as e:
            raise click.ClickException(str(e)) from e
    rprint(f"[green]Morphed clip #{src_id} → #{row['id']}[/green] ({desc})")
    rprint("[dim]Listen:[/dim] tts-audiobook library play " + str(row["id"]))


@library.command("synth")
@click.option("--voice", required=True,
              help="Kokoro British voicepack, e.g. bf_emma, bm_george "
                   "(bf_*=female, bm_*=male).")
@click.option("--blend", default=None,
              help="Second voicepack to mix in for a new identity.")
@click.option("--blend-weight", type=float, default=0.5, show_default=True)
@click.option("--speed", type=float, default=1.0, show_default=True)
@click.option("--text", "text", default=None,
              help="Custom seed text. Kokoro's delivery follows the text, and "
                   "cloning inherits it — exclamatory text yields an excitable "
                   "voice, measured text a calm one.")
@click.option("--sex", type=click.Choice(["male", "female"]), default=None,
              help="Tag override; inferred from the voicepack prefix if omitted.")
@click.option("--age-band", "age_band",
              type=click.Choice(["child", "teen", "young_adult", "adult",
                                 "middle_aged", "elderly"]), default=None)
@click.option("--region", default=None)
@click.option("--notes", default=None)
def library_synth(voice: str, blend: str | None, blend_weight: float,
                  speed: float, text: str | None, sex, age_band, region,
                  notes) -> None:
    """Generate a fully synthetic en-GB seed clip with Kokoro."""
    from .kokoro_seed import BRITISH_VOICES, synthesize
    from .library import LibraryError, import_clip_array
    from .voicebuild import CALIBRATION_TEXT

    seed_text = text or CALIBRATION_TEXT
    inferred = BRITISH_VOICES.get(voice)
    if sex is None:
        sex = (inferred[0] if inferred
               else {"bf": "female", "bm": "male"}.get(voice[:2]))
    if age_band is None and inferred:
        age_band = inferred[1]

    try:
        wav, sr = synthesize(seed_text, voice, blend=blend,
                             blend_weight=blend_weight, speed=speed)
    except Exception as e:
        raise click.ClickException(f"Kokoro synthesis failed: {e}") from e

    source = f"kokoro:{voice}" + (f"+{blend}@{blend_weight}" if blend else "")
    with dbmod.db() as conn:
        try:
            row = import_clip_array(
                conn, wav, sr, transcript=seed_text,
                sex=sex, age_band=age_band, locale="en-GB", region=region,
                quality="good", source=source, license="Apache-2.0",
                notes=notes or f"synthetic seed ({source})")
        except LibraryError as e:
            raise click.ClickException(str(e)) from e
    rprint(f"[green]Synthesized clip #{row['id']}[/green] ({source})")
    rprint("[dim]Listen:[/dim] tts-audiobook library play " + str(row["id"]))


@library.command("play")
@click.argument("clip_id", type=int)
def library_play(clip_id: int) -> None:
    """Play library clip CLIP_ID (requires ffplay)."""
    from .library import play_sample
    with dbmod.db() as conn:
        row = dbmod.clip_get(conn, clip_id)
    if not row:
        raise click.ClickException(f"No clip #{clip_id}")
    play_sample(Path(row["path"]))


@library.command("retag")
@click.argument("clip_id", type=int)
@click.option("--field", required=True,
              type=click.Choice(["sex", "age_band", "locale", "region", "quality",
                                 "source", "license", "notes", "transcript"]))
@click.option("--value", required=True)
def library_retag(clip_id: int, field: str, value: str) -> None:
    """Update one tag on a library clip."""
    with dbmod.db() as conn:
        if not dbmod.clip_get(conn, clip_id):
            raise click.ClickException(f"No clip #{clip_id}")
        dbmod.clip_retag(conn, clip_id, field, value or None)
    rprint(f"[green]Clip #{clip_id}[/green] {field} = {value!r}")


# ---------- inspect ----------

@main.command("inspect")
@click.argument("book_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def inspect_cmd(book_path: Path) -> None:
    """Summarize BOOK_PATH: structure, speakers, and voice-spec coverage."""
    book = _open_book(book_path)
    total_segments = sum(len(c.segments) for c in book.chapters)
    rprint(f"[bold]{book.title}[/bold]"
           + (f" by {book.author}" if book.author else ""))
    rprint(f"Chapters: {len(book.chapters)}   Segments: {total_segments}   "
           f"Language: {book.language}")

    if book.production:
        narrator = book.production.narrator_character or "third-person narrator"
        has_voice = "yes" if book.production.narrator_voice else "no"
        rprint(f"Production block: [green]present[/green] "
               f"(narration: {book.production.narration_person or '?'}, "
               f"narrator: {narrator}, narrator voice spec: {has_voice})")
    else:
        rprint("Production block: [yellow]absent[/yellow] "
               "(narrator will use the default voice spec)")

    speakers = speakers_by_importance(book)
    with_spec = 0
    table = Table(title="Speakers")
    table.add_column("Speaker")
    table.add_column("Segments", justify="right")
    table.add_column("Voice spec")
    for key, count in speakers:
        if key == NARRATOR_KEY:
            spec = book.production.narrator_voice if book.production else None
            name = "(narrator)"
        else:
            cs = character_spec_for(book, key)
            spec = cs.voice if cs else None
            name = key
        if spec:
            with_spec += 1
            desc = spec.describe()
            desc = desc[:70] + ("…" if len(desc) > 70 else "")
        else:
            desc = "[dim]—[/dim]"
        table.add_row(name, str(count), desc)
    rprint(table)

    n = len(speakers)
    pct = (100 * with_spec // n) if n else 0
    rprint(f"Voice-spec coverage: {with_spec}/{n} speakers ({pct}%)"
           + ("" if with_spec else "  [yellow](fallback: default specs)[/yellow]"))


# ---------- cast / audition ----------

@main.command("cast")
@click.argument("book_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--recast", is_flag=True, help="Redo already-cast speakers too.")
@click.option("--design", is_flag=True,
              help="Generate every voice from its spec with Qwen VoiceDesign "
                   "(fully synthetic) instead of matching library clips.")
def cast_cmd(book_path: Path, recast: bool, design: bool) -> None:
    """Freeze a reference voice for every speaker (library match or design)."""
    from .studio import run_casting
    book = _open_book(book_path)
    out_dir = _resolve_output_dir(book, None)
    with dbmod.db() as conn:
        book_id = _ensure_book_row(conn, book, out_dir)
        try:
            run_casting(conn, book, book_id, recast=recast, design=design)
        except RuntimeError as e:
            raise click.ClickException(str(e)) from e


@main.command("merge")
@click.argument("book_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--from", "from_name", default=None,
              help="Duplicate speaker name to fold away.")
@click.option("--into", "into_name", default=None,
              help="Canonical speaker name that keeps the voice.")
@click.option("--list", "list_", is_flag=True, help="Show stored merges.")
def merge_cmd(book_path: Path, from_name: str | None, into_name: str | None,
              list_: bool) -> None:
    """Fold duplicate speaker identities into one (local fix; the real fix
    is upstream aliases in the book JSON)."""
    book = _open_book(book_path)
    out_dir = _resolve_output_dir(book, None)
    with dbmod.db() as conn:
        book_id = _ensure_book_row(conn, book, out_dir)
        if list_:
            merges = dbmod.merges_get_all(conn, book_id)
            if not merges:
                rprint("[dim]No merges stored.[/dim]")
            for f, t in sorted(merges.items()):
                rprint(f"  {f} → {t}")
            return
        if not (from_name and into_name):
            raise click.ClickException("Pass both --from and --into (or --list).")
        # Validate against the raw roster (`book` has merges applied already).
        raw_speakers = {k for k, _ in speakers_by_importance(_open_book(book_path))}
        for name in (from_name, into_name):
            if name not in raw_speakers:
                raise click.ClickException(f"{name!r} is not a speaker in this book.")
        dbmod.merge_set(conn, book_id, from_name, into_name)
        dbmod.cast_delete(conn, book_id, from_name)  # stale row, if any
    rprint(f"[green]Merged:[/green] {from_name} → {into_name}")
    rprint("[dim]Re-run `cast` if the merged speaker had no voice yet.[/dim]")


@main.command("assign")
@click.argument("book_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--character", required=True,
              help="Speaker to reassign (canonical name, or 'narrator').")
@click.option("--clip", "clip_id", type=int, required=True,
              help="Library clip id to freeze as this character's voice.")
def assign_cmd(book_path: Path, character: str, clip_id: int) -> None:
    """Pin one character to a specific library clip, overriding the scorer."""
    import json as jsonlib
    from dataclasses import asdict

    from .book import book_output_subdir
    from .studio import spec_for_speaker
    from .voicebuild import build_reference

    book = _open_book(book_path)
    out_dir = _resolve_output_dir(book, None)
    if character.lower() == "narrator":
        character = NARRATOR_KEY
    with dbmod.db() as conn:
        book_id = _ensure_book_row(conn, book, out_dir)
        row = dbmod.clip_get(conn, clip_id)
        if not row:
            raise click.ClickException(f"No clip #{clip_id}")
        speakers = {k for k, _ in speakers_by_importance(book)}
        if character not in speakers:
            raise click.ClickException(
                f"{character!r} is not a speaker in this book.")
        ref = build_reference(book_key=book_output_subdir(book),
                              character=character,
                              clip_path=Path(row["path"]),
                              clip_transcript=row["transcript"])
        dbmod.cast_upsert(
            conn, book_id, character,
            spec_json=jsonlib.dumps(asdict(spec_for_speaker(book, character))),
            library_clip_id=clip_id,
            ref_path=str(ref.path), ref_transcript=ref.transcript,
            ref_sha256=ref.sha256, design_seed=None, audition_seed=0,
            status="accepted",
        )
    shown = "(narrator)" if character == NARRATOR_KEY else character
    rprint(f"[green]{shown}[/green] → clip #{clip_id} (frozen, accepted)")


@main.command("audition")
@click.argument("book_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--character", default=None, help="Audition just this speaker.")
@click.option("--engine", "engine_name", default=None,
              type=click.Choice(["qwen", "chatterbox"]))
@click.option("--yes", "auto_accept", is_flag=True,
              help="Accept every proposed voice without listening.")
def audition_cmd(book_path: Path, character: str | None,
                 engine_name: str | None, auto_accept: bool) -> None:
    """Play one line per cast voice; accept, reroll, or skip each."""
    from .studio import run_audition
    book = _open_book(book_path)
    out_dir = _resolve_output_dir(book, None)
    with dbmod.db() as conn:
        book_id = _ensure_book_row(conn, book, out_dir)
        engine = _engine_for(conn, book_id, engine_name)
        run_audition(conn, engine, book, book_id, only_character=character,
                     auto_accept=auto_accept)


# ---------- perform / bakeoff ----------

def _parse_chapters(chapters: str | None) -> list[int] | None:
    if not chapters:
        return None
    try:
        return sorted({int(x.strip()) for x in chapters.split(",") if x.strip()})
    except ValueError as e:
        raise click.ClickException(f"Bad --chapters value: {e}") from e


@main.command("perform")
@click.argument("book_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "output_dir", type=click.Path(file_okay=False, path_type=Path),
              default=None, help="Override the output directory.")
@click.option("--chapters", default=None,
              help="Comma-separated chapter numbers (0 = title) for partial renders.")
@click.option("--engine", "engine_name", default=None,
              type=click.Choice(["qwen", "chatterbox"]))
@click.option("--base-url", default=None,
              help="Base URL for MP3s in the RSS feed (default: file:// URL).")
@click.option("--no-qc", is_flag=True, help="Skip the Whisper QC pass.")
@click.option("--force", is_flag=True,
              help="Re-render the requested chapters even if already done.")
@click.option("--yes", "auto_accept", is_flag=True,
              help="Auto-accept any un-auditioned cast voices.")
def perform_cmd(book_path: Path, output_dir: Path | None, chapters: str | None,
                engine_name: str | None, base_url: str | None,
                no_qc: bool, force: bool, auto_accept: bool) -> None:
    """Render BOOK_PATH chapter-by-chapter with the accepted cast."""
    from .render import load_cast, perform
    book = _open_book(book_path)
    out_dir = _resolve_output_dir(book, output_dir)
    chapter_set = _parse_chapters(chapters)

    with dbmod.db() as conn:
        book_id = _ensure_book_row(conn, book, out_dir)
        engine = _engine_for(conn, book_id, engine_name)
        speakers = [k for k, _ in speakers_by_importance(book)]
        try:
            cast = load_cast(conn, book_id, speakers)
        except RuntimeError as e:
            raise click.ClickException(str(e)) from e

        not_accepted = [k for k in speakers
                        if (row := dbmod.cast_get(conn, book_id, k))
                        and row["status"] != "accepted"]
        if not_accepted:
            if auto_accept:
                for k in not_accepted:
                    dbmod.cast_upsert(conn, book_id, k, status="accepted",
                                      engine=engine.name)
            else:
                raise click.ClickException(
                    f"{len(not_accepted)} voices not auditioned "
                    f"(e.g. {', '.join(not_accepted[:3])}). "
                    "Run `audition`, or pass --yes to accept them all.")

        if engine_name:
            dbmod.book_set_engine(conn, book_id, engine.name)
        if force:
            if chapter_set is None:
                dbmod.chapter_status_clear(conn, book_id)
                dbmod.qc_flags_clear(conn, book_id)
            else:
                for n in chapter_set:
                    dbmod.chapter_status_clear(conn, book_id, n)
                    dbmod.qc_flags_clear(conn, book_id, n)
        try:
            perform(conn, engine, book, book_id, out_dir, cast,
                    chapters=chapter_set, base_url=base_url, run_qc=not no_qc)
        except RuntimeError as e:
            raise click.ClickException(str(e)) from e


@main.command("bakeoff")
@click.argument("book_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--chapter", type=int, required=True)
@click.option("--engines", "engines_", default="qwen,chatterbox",
              help="Comma-separated engine names.")
def bakeoff_cmd(book_path: Path, chapter: int, engines_: str) -> None:
    """Render one chapter with each engine (same cast) for A/B listening."""
    from .engines.base import get_engine
    from .render import load_cast, render_chapter
    book = _open_book(book_path)
    out_dir = _resolve_output_dir(book, None)
    target = next((c for c in book.chapters if c.number == chapter), None)
    if target is None:
        raise click.ClickException(f"No chapter {chapter}.")

    with dbmod.db() as conn:
        book_id = _ensure_book_row(conn, book, out_dir)
        speakers = sorted({s.speaker_key for s in target.segments})
        try:
            cast = load_cast(conn, book_id, speakers)
        except RuntimeError as e:
            raise click.ClickException(str(e)) from e
        for name in [e.strip() for e in engines_.split(",") if e.strip()]:
            engine = get_engine(name)
            dest = out_dir / "bakeoff" / name / f"chapter-{chapter:02d}.mp3"
            dest.parent.mkdir(parents=True, exist_ok=True)
            rprint(f"\n[bold]Engine: {name}[/bold]")
            engine.load()
            render_chapter(conn, engine, book, book_id, target, cast, dest)
            rprint(f"[green]→[/green] {dest}")
    rprint("\nListen, then set the winner with: "
           "tts-audiobook perform BOOK --engine NAME")


# ---------- package / qc ----------

@main.command("package")
@click.argument("book_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--cover", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None)
def package_cmd(book_path: Path, cover: Path | None) -> None:
    """ID3-tag the chapter MP3s and build a .m4b with chapter markers."""
    from .package import build_m4b, tag_mp3s
    book = _open_book(book_path)
    out_dir = _resolve_output_dir(book, None)
    with dbmod.db() as conn:
        book_id = _ensure_book_row(conn, book, out_dir)
        n = tag_mp3s(conn, book, book_id)
        rprint(f"[green]Tagged {n} MP3s.[/green]")
        try:
            build_m4b(conn, book, book_id, out_dir, cover)
        except RuntimeError as e:
            raise click.ClickException(str(e)) from e


@main.command("qc-report")
@click.argument("book_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def qc_report_cmd(book_path: Path) -> None:
    """List segments that failed QC after all retries."""
    book = _open_book(book_path)
    out_dir = _resolve_output_dir(book, None)
    with dbmod.db() as conn:
        book_id = _ensure_book_row(conn, book, out_dir)
        flags = dbmod.qc_flags_for(conn, book_id)
    if not flags:
        rprint("[green]No QC flags.[/green]")
        return
    table = Table(title="QC flags")
    for col in ("Ch", "Item", "Speaker", "WER", "Tries", "Text"):
        table.add_column(col)
    for f in flags:
        table.add_row(str(f["chapter_number"]), str(f["item_index"]),
                      f["character"] or "-", f"{f['best_wer']:.2f}",
                      str(f["attempts"]), (f["text"] or "")[:60])
    rprint(table)
    rprint("[dim]Re-render a chapter: clear it with --chapters N after fixing.[/dim]")


if __name__ == "__main__":
    main()
