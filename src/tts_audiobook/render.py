from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich import print as rprint
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

from . import audio as audiomod
from . import config, device
from . import db as dbmod
from . import qc as qcmod
from .book import Book, Chapter, Segment, slugify
from .config import NARRATOR_KEY
from .engines.base import Engine
from .feed import write_feed
from .library import sha256_file
from .planning import RenderItem, bucket_by_speaker, plan_chapter, takes_for


@dataclass
class CastVoice:
    character: str
    ref_path: Path
    ref_transcript: str
    ref_sha256: str | None
    seed: int


def load_cast(conn: sqlite3.Connection, book_id: int,
              speakers: list[str]) -> dict[str, CastVoice]:
    cast: dict[str, CastVoice] = {}
    missing: list[str] = []
    for key in speakers:
        row = dbmod.cast_get(conn, book_id, key)
        if not row or not row["ref_path"]:
            missing.append(key)
            continue
        cast[key] = CastVoice(
            character=key,
            ref_path=Path(row["ref_path"]),
            ref_transcript=row["ref_transcript"] or "",
            ref_sha256=row["ref_sha256"],
            seed=int(row["audition_seed"] or 0),
        )
    if missing:
        raise RuntimeError(
            "No frozen reference for: " + ", ".join(missing[:5])
            + ("…" if len(missing) > 5 else "") + ". Run `cast` (and `audition`).")
    return cast


def verify_cast_refs(cast: dict[str, CastVoice]) -> None:
    for cv in cast.values():
        if not cv.ref_path.exists():
            raise RuntimeError(f"Frozen reference missing on disk: {cv.ref_path}")
        if cv.ref_sha256 and sha256_file(cv.ref_path) != cv.ref_sha256:
            raise RuntimeError(
                f"Frozen reference for {cv.character!r} changed since casting "
                f"({cv.ref_path}). Re-run `cast --recast` if intentional.")


def _proper_noun_allowlist(book: Book) -> set[str]:
    names: set[str] = set()
    for c in book.characters:
        names.add(c.name)
        names.update(c.aliases)
    return names


def _transcribe_or_none(wav: np.ndarray, sample_rate: int) -> str | None:
    try:
        from .asr import transcribe_array
    except Exception:
        return None
    try:
        return transcribe_array(wav, sample_rate)
    except Exception:
        return None


@dataclass
class _Rendered:
    wav: np.ndarray
    qc: qcmod.QCResult
    attempts: int


def _render_one(engine: Engine, prompt, item: RenderItem, *, language: str,
                seed: int, allowlist: set[str], run_qc: bool,
                ref_f0: float | None = None) -> _Rendered:
    best: _Rendered | None = None
    for attempt in range(config.QC_MAX_ATTEMPTS):
        wavs, sr = engine.generate([item.text], prompt, language=language,
                                   seed=seed + attempt)
        wav = audiomod.trim_silence(wavs[0], sr)
        transcript = _transcribe_or_none(wav, sr) if run_qc else None
        result = qcmod.check(item.text, wav, sr, transcript, allowlist,
                             ref_f0=ref_f0)
        cand = _Rendered(wav=wav, qc=result, attempts=attempt + 1)
        if best is None or qcmod.better(cand.qc, best.qc):
            best = cand
        if result.passed:
            return best
    return best  # type: ignore[return-value]


def reference_f0(cv: CastVoice) -> float | None:
    """Median pitch of a cast voice's frozen reference clip (None if unreadable)."""
    try:
        import soundfile as sf
        wav, sr = sf.read(str(cv.ref_path), dtype="float32", always_2d=True)
        return qcmod.median_f0(wav.mean(axis=1), sr)
    except Exception:
        return None


def make_batches(bucket: list[RenderItem], engine_max: int) -> list[list[RenderItem]]:
    """Chunk a voice's items by row count AND total characters — batch memory
    scales with the longest text times batch size, and a run of long items
    (a letter read aloud) OOMs if batched by count alone.

    An item occupies `takes_for(item)` rows (short lines are rendered several
    times and the best take kept); all of an item's takes ride in one batch."""
    max_count = max(1, min(config.BATCH_SIZE, engine_max))
    batches: list[list[RenderItem]] = []
    cur: list[RenderItem] = []
    cur_rows = 0
    cur_chars = 0
    for item in bucket:
        rows = takes_for(item)
        chars = rows * len(item.text)
        if cur and (cur_rows + rows > max_count
                    or cur_chars + chars > config.BATCH_MAX_CHARS):
            batches.append(cur)
            cur, cur_rows, cur_chars = [], 0, 0
        cur.append(item)
        cur_rows += rows
        cur_chars += chars
    if cur:
        batches.append(cur)
    return batches


def batch_texts(batch: list[RenderItem]) -> list[str]:
    """Engine input rows for a batch: each item's text repeated takes_for(item) times."""
    return [item.text for item in batch for _ in range(takes_for(item))]


def regroup_takes(batch: list[RenderItem], wavs: list[np.ndarray]
                  ) -> list[tuple[RenderItem, list[np.ndarray]]]:
    """Inverse of batch_texts: pair each item with its takes, in order."""
    out: list[tuple[RenderItem, list[np.ndarray]]] = []
    pos = 0
    for item in batch:
        n = takes_for(item)
        out.append((item, wavs[pos:pos + n]))
        pos += n
    if pos != len(wavs):
        raise RuntimeError(f"Engine returned {len(wavs)} clips for {pos} rows.")
    return out


def _pick_take(takes: list[np.ndarray], sr: int, item: RenderItem, *,
               allowlist: set[str], ref_f0: float | None, run_qc: bool,
               transcribe=None) -> tuple[np.ndarray, qcmod.QCResult, int]:
    """Choose among independent takes of one item: the take closest to the
    reference pitch that also passes QC; later takes win ties. Takes are
    silence-trimmed here. Pitch is measured for every take (cheap); Whisper
    runs only down the pitch-sorted list until a take passes, so a short line
    usually costs one transcription, not one per take.

    Returns (wav, qc_result, take_number) with take_number 1-based."""
    transcribe = transcribe or _transcribe_or_none
    trimmed = [audiomod.trim_silence(w, sr) for w in takes]
    devs: list[float | None] = []
    for w in trimmed:
        f0 = qcmod.median_f0(w, sr) if ref_f0 else None
        devs.append(qcmod.semitones(f0, ref_f0) if (f0 and ref_f0) else None)

    def sort_key(i: int) -> tuple[int, float, int]:
        d = devs[i]
        return (0 if d is not None else 1, abs(d) if d is not None else 0.0, -i)

    order = sorted(range(len(trimmed)), key=sort_key)
    best: tuple[np.ndarray, qcmod.QCResult, int] | None = None
    for i in order:
        wav = trimmed[i]
        transcript = transcribe(wav, sr) if run_qc else None
        result = qcmod.check(item.text, wav, sr, transcript, allowlist, ref_f0=ref_f0)
        if best is None or qcmod.better(result, best[1]):
            best = (wav, result, i + 1)
        if result.passed:
            break
    assert best is not None
    return best


def render_chapter(conn: sqlite3.Connection, engine: Engine, book: Book,
                   book_id: int, chapter: Chapter, cast: dict[str, CastVoice],
                   output_path: Path, *, run_qc: bool = True) -> None:
    items = plan_chapter(chapter)
    if not items:
        rprint(f"[yellow]Chapter {chapter.number} has no segments; skipping.[/yellow]")
        return
    dbmod.qc_flags_clear(conn, book_id, chapter.number)

    allowlist = _proper_noun_allowlist(book)
    rendered: dict[int, np.ndarray] = {}
    sample_rate: int | None = None

    for speaker_key, bucket in bucket_by_speaker(items).items():
        cv = cast[speaker_key]
        prompt = engine.clone_prompt(cv.ref_path, cv.ref_transcript)
        # The clone drifts in register on short lines (a male voice can come
        # out in a female range); judge every take against the reference pitch.
        ref_f0 = reference_f0(cv) if run_qc else None
        for batch in make_batches(bucket, engine.max_batch):
            wavs, sr = engine.generate(batch_texts(batch), prompt,
                                       language=book.language, seed=cv.seed)
            sample_rate = sr
            for item, takes in regroup_takes(batch, wavs):
                if len(takes) > 1:
                    wav, result, kept = _pick_take(
                        takes, sr, item, allowlist=allowlist, ref_f0=ref_f0,
                        run_qc=run_qc)
                    rprint(f"[dim]ch{chapter.number} item {item.index}: take "
                           f"{kept}/{len(takes)} kept ({result.describe()})[/dim]")
                else:
                    wav = audiomod.trim_silence(takes[0], sr)
                    transcript = _transcribe_or_none(wav, sr) if run_qc else None
                    result = qcmod.check(item.text, wav, sr, transcript, allowlist,
                                         ref_f0=ref_f0)
                if result.passed:
                    rendered[item.index] = audiomod.normalize_loudness(wav, sr)
                    continue
                # Retry this item alone with varied seeds; keep the best take.
                best = _render_one(engine, prompt, item, language=book.language,
                                   seed=cv.seed + 1000, allowlist=allowlist,
                                   run_qc=run_qc, ref_f0=ref_f0)
                if qcmod.better(best.qc, result):
                    keep, keep_qc, attempts = best.wav, best.qc, 1 + best.attempts
                else:
                    keep, keep_qc, attempts = wav, result, 1 + best.attempts
                rendered[item.index] = audiomod.normalize_loudness(keep, sr)
                if not keep_qc.passed:
                    dbmod.qc_flag_add(conn, book_id, chapter.number, item.index,
                                      speaker_key, item.text, keep_qc.wer, attempts,
                                      reason=keep_qc.reason)
                    rprint(f"[yellow]QC flag[/yellow] ch{chapter.number} "
                           f"item {item.index} ({speaker_key}): {keep_qc.reason} "
                           f"{keep_qc.describe()} after {attempts} attempts")

    assert sample_rate is not None
    pieces: list[np.ndarray] = []
    for item in items:
        if item.gap_before_s > 0 and pieces:
            pieces.append(audiomod.silence(sample_rate, item.gap_before_s))
        pieces.append(rendered[item.index])
    if chapter.number == 0:
        # Leave a clear beat between the title/attribution and chapter one.
        pieces.append(audiomod.silence(sample_rate, config.TITLE_TAIL_SILENCE_S))
    full = audiomod.concat(pieces)
    audiomod.encode_mp3(full, sample_rate, output_path)

    device.empty_cache()


# Spoken before chapter one so listeners can find the project. The URL is
# spelled out for the TTS engine; the sentence is kept long enough that
# Whisper collapsing it to "gutenbergaloud.org" stays under QC_WER_THRESHOLD.
ATTRIBUTION_TEXT = ("This recording is made available by Gutenberg Aloud, a project "
                    "that turns public domain books into free audio recordings. "
                    "You can find this book and many more at gutenberg aloud dot org.")


def title_chapter(book: Book) -> Chapter:
    text = f"{book.title}, by {book.author}." if book.author else f"{book.title}."
    segments = [Segment(speaker_key=NARRATOR_KEY, text=text, raw_speaker=None)]
    if book.production and book.production.synopsis:
        segments.append(Segment(speaker_key=NARRATOR_KEY,
                                text=book.production.synopsis, raw_speaker=None))
    segments.append(Segment(speaker_key=NARRATOR_KEY, text=ATTRIBUTION_TEXT,
                            raw_speaker=None))
    return Chapter(number=0, title="title", segments=segments)


def perform(conn: sqlite3.Connection, engine: Engine, book: Book, book_id: int,
            output_dir: Path, cast: dict[str, CastVoice],
            chapters: list[int] | None = None, base_url: str | None = None,
            run_qc: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if book.language and book.language.lower() != "english":
        raise RuntimeError(f"Only English books are supported; got {book.language!r}.")

    verify_cast_refs(cast)
    pad = max(2, len(str(max(c.number for c in book.chapters))))

    title_ch = title_chapter(book)
    title_mp3 = output_dir / f"{0:0{pad}d}_title.mp3"
    needs_title = (chapters is None or 0 in chapters) and \
        dbmod.chapter_is_done(conn, book_id, 0) is None

    chapter_todo: list[Chapter] = []
    for ch in book.chapters:
        if chapters is not None and ch.number not in chapters:
            continue
        existing = dbmod.chapter_is_done(conn, book_id, ch.number)
        if existing:
            rprint(f"[dim]Chapter {ch.number} already rendered → {existing}[/dim]")
            continue
        chapter_todo.append(ch)

    if not needs_title and not chapter_todo:
        rprint("[green]All chapters already rendered.[/green] Refreshing feed.")
        feed_path = write_feed(conn, book, book_id, output_dir, base_url)
        rprint(f"[dim]Feed:[/dim] {feed_path}")
        return

    engine.load()

    if needs_title:
        rprint("[dim]Rendering title…[/dim]")
        render_chapter(conn, engine, book, book_id, title_ch, cast, title_mp3,
                       run_qc=run_qc)
        dbmod.chapter_mark_done(conn, book_id, 0, title_mp3, engine.name)
        write_feed(conn, book, book_id, output_dir, base_url)

    if chapter_todo:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as bar:
            task = bar.add_task("Rendering chapters", total=len(chapter_todo))
            for ch in chapter_todo:
                slug = slugify(ch.title)
                mp3_path = output_dir / f"{ch.number:0{pad}d}_{slug}.mp3"
                bar.update(task, description=f"Ch {ch.number:>3}: {slug[:40]}")
                render_chapter(conn, engine, book, book_id, ch, cast, mp3_path,
                               run_qc=run_qc)
                dbmod.chapter_mark_done(conn, book_id, ch.number, mp3_path,
                                        engine.name)
                bar.advance(task)
                write_feed(conn, book, book_id, output_dir, base_url)

    feed_path = write_feed(conn, book, book_id, output_dir, base_url)
    rprint(f"\n[green]Done.[/green] MP3s in: {output_dir}")
    rprint(f"[dim]Feed:[/dim] {feed_path}")
