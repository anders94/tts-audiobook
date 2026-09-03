from __future__ import annotations

import hashlib
import shutil
import sqlite3
import subprocess
from pathlib import Path

import soundfile as sf

from . import db as dbmod
from .casting import ClipInfo
from .config import LIBRARY_DIR, ensure_dirs


class LibraryError(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_clip(path: Path) -> float:
    try:
        info = sf.info(str(path))
    except Exception as e:
        raise LibraryError(f"Could not read audio file {path}: {e}") from e
    duration = info.frames / info.samplerate
    if duration < 1.0:
        raise LibraryError(f"Clip is {duration:.1f}s; need at least 1s.")
    if duration < 2.0 or duration > 15.0:
        from rich import print as rprint
        rprint(f"[yellow]Warning:[/yellow] clip is {duration:.1f}s; "
               "best results with 2–15s clips.")
    return duration


def import_clip(conn: sqlite3.Connection, audio_path: Path, *,
                sex: str | None, age_band: str | None, locale: str | None,
                region: str | None, quality: str | None, source: str | None,
                license: str | None, notes: str | None,
                transcript: str | None = None,
                auto_transcribe: bool = True) -> sqlite3.Row:
    if not audio_path.exists():
        raise LibraryError(f"Audio file not found: {audio_path}")
    duration = _validate_clip(audio_path)

    ensure_dirs()
    digest = sha256_file(audio_path)
    dest = LIBRARY_DIR / f"{digest[:16]}{audio_path.suffix.lower() or '.wav'}"
    if not dest.exists():
        shutil.copy2(audio_path, dest)

    if not transcript and auto_transcribe:
        from rich import print as rprint
        rprint(f"[dim]Transcribing {audio_path.name} with Whisper…[/dim]")
        from .asr import transcribe_file
        transcript = transcribe_file(dest)
    if not transcript:
        raise LibraryError("Transcript required (pass --transcript or enable auto-transcribe).")

    clip_id = dbmod.clip_add(
        conn, path=dest, transcript=transcript, duration_s=duration,
        sex=sex, age_band=age_band, locale=locale, region=region,
        quality=quality, source=source, license=license, notes=notes,
        sha256=digest,
    )
    return dbmod.clip_get(conn, clip_id)


def import_clip_array(conn: sqlite3.Connection, wav, sample_rate: int, *,
                      transcript: str | None, sex: str | None,
                      age_band: str | None, locale: str | None,
                      region: str | None, quality: str | None,
                      source: str | None, license: str | None,
                      notes: str | None) -> sqlite3.Row:
    """Import generated/derived audio (morph, synth seed) as a library clip."""
    import tempfile

    import soundfile as sf_

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf_.write(str(tmp_path), wav, sample_rate, subtype="PCM_16")
        return import_clip(conn, tmp_path, sex=sex, age_band=age_band,
                           locale=locale, region=region, quality=quality,
                           source=source, license=license, notes=notes,
                           transcript=transcript)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_clip_audio(row: sqlite3.Row):
    wav, sr = sf.read(row["path"], dtype="float32", always_2d=True)
    return wav.mean(axis=1), sr


def clip_info(row: sqlite3.Row) -> ClipInfo:
    return ClipInfo(
        clip_id=int(row["id"]),
        sex=row["sex"],
        age_band=row["age_band"],
        locale=row["locale"],
        region=row["region"],
        quality=row["quality"],
        notes=row["notes"],
    )


def play_sample(path: Path) -> None:
    if shutil.which("ffplay"):
        subprocess.run(
            ["ffplay", "-autoexit", "-nodisp", "-loglevel", "error", str(path)],
            check=False,
        )
    else:
        from rich import print as rprint
        rprint(f"[yellow]ffplay not found.[/yellow] Clip is at: {path}")
