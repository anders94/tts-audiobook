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
from .planning import RenderItem, bucket_by_speaker, plan_chapter


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
                seed: int, allowlist: set[str], run_qc: bool) -> _Rendered:
    best: _Rendered | None = None
    for attempt in range(config.QC_MAX_ATTEMPTS):
        wavs, sr = engine.generate([item.text], prompt, language=language,
                                   seed=seed + attempt)
        wav = audiomod.trim_silence(wavs[0], sr)
        transcript = _transcribe_or_none(wav, sr) if run_qc else None
        result = qcmod.check(item.text, wav, sr, transcript, allowlist)
        cand = _Rendered(wav=wav, qc=result, attempts=attempt + 1)
        if best is None or cand.qc.wer < best.qc.wer:
            best = cand
        if result.passed:
            return best
    return best  # type: ignore[return-value]


def make_batches(bucket: list[RenderItem], engine_max: int) -> list[list[RenderItem]]:
    """Chunk a voice's items by count AND total characters — batch memory
    scales with the longest text times batch size, and a run of long items
    (a letter read aloud) OOMs if batched by count alone."""
    max_count = max(1, min(config.BATCH_SIZE, engine_max))
    batches: list[list[RenderItem]] = []
    cur: list[RenderItem] = []
    cur_chars = 0
    for item in bucket:
        if cur and (len(cur) >= max_count
                    or cur_chars + len(item.text) > config.BATCH_MAX_CHARS):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(item)
        cur_chars += len(item.text)
    if cur:
        batches.append(cur)
    return batches


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
        for batch in make_batches(bucket, engine.max_batch):
            wavs, sr = engine.generate([b.text for b in batch], prompt,
                                       language=book.language, seed=cv.seed)
            sample_rate = sr
            for item, wav in zip(batch, wavs):
                wav = audiomod.trim_silence(wav, sr)
                transcript = _transcribe_or_none(wav, sr) if run_qc else None
                result = qcmod.check(item.text, wav, sr, transcript, allowlist)
                if result.passed:
                    rendered[item.index] = audiomod.normalize_loudness(wav, sr)
                    continue
                # Retry this item alone with varied seeds.
                best = _render_one(engine, prompt, item, language=book.language,
                                   seed=cv.seed + 1000, allowlist=allowlist,
                                   run_qc=run_qc)
                if best.qc.wer < result.wer:
                    keep, keep_qc, attempts = best.wav, best.qc, 1 + best.attempts
                else:
                    keep, keep_qc, attempts = wav, result, 1 + best.attempts
                rendered[item.index] = audiomod.normalize_loudness(keep, sr)
                if not keep_qc.passed:
                    dbmod.qc_flag_add(conn, book_id, chapter.number, item.index,
                                      speaker_key, item.text, keep_qc.wer, attempts)
                    rprint(f"[yellow]QC flag[/yellow] ch{chapter.number} "
                           f"item {item.index} ({speaker_key}): "
                           f"wer={keep_qc.wer:.2f} after {attempts} attempts")

    assert sample_rate is not None
    pieces: list[np.ndarray] = []
    for item in items:
        if item.gap_before_s > 0 and pieces:
            pieces.append(audiomod.silence(sample_rate, item.gap_before_s))
        pieces.append(rendered[item.index])
    full = audiomod.concat(pieces)
    audiomod.encode_mp3(full, sample_rate, output_path)

    device.empty_cache()


def title_chapter(book: Book) -> Chapter:
    text = f"{book.title}, by {book.author}." if book.author else f"{book.title}."
    segments = [Segment(speaker_key=NARRATOR_KEY, text=text, raw_speaker=None)]
    if book.production and book.production.synopsis:
        segments.append(Segment(speaker_key=NARRATOR_KEY,
                                text=book.production.synopsis, raw_speaker=None))
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
