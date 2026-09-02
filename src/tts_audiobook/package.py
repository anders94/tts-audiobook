from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

from rich import print as rprint

from . import config
from .book import Book, slugify


def _ffprobe_duration_ms(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return int(float(json.loads(out)["format"]["duration"]) * 1000)


def _completed_chapters(conn: sqlite3.Connection, book_id: int) -> list[tuple[int, Path]]:
    rows = conn.execute(
        "SELECT chapter_number, mp3_path FROM chapter_status "
        "WHERE book_id = ? AND completed_at IS NOT NULL ORDER BY chapter_number",
        (book_id,),
    ).fetchall()
    out = []
    for r in rows:
        p = Path(r["mp3_path"])
        if p.exists():
            out.append((int(r["chapter_number"]), p))
    return out


def tag_mp3s(conn: sqlite3.Connection, book: Book, book_id: int) -> int:
    from mutagen.easyid3 import EasyID3
    from mutagen.id3 import ID3NoHeaderError

    titles = {c.number: c.title for c in book.chapters}
    chapters = _completed_chapters(conn, book_id)
    for number, path in chapters:
        try:
            tags = EasyID3(str(path))
        except ID3NoHeaderError:
            from mutagen.mp3 import MP3
            f = MP3(str(path))
            f.add_tags()
            f.save()
            tags = EasyID3(str(path))
        tags["album"] = book.title
        tags["artist"] = book.author or "Unknown"
        tags["albumartist"] = book.author or "Unknown"
        tags["tracknumber"] = str(number)
        tags["title"] = ("Title" if number == 0
                         else f"Chapter {number}: {titles.get(number, '')}".rstrip(": "))
        tags.save()
    return len(chapters)


def build_m4b(conn: sqlite3.Connection, book: Book, book_id: int,
              output_dir: Path, cover: Path | None = None) -> Path:
    chapters = _completed_chapters(conn, book_id)
    if not chapters:
        raise RuntimeError("No completed chapters to package.")

    titles = {c.number: c.title for c in book.chapters}
    meta_lines = [
        ";FFMETADATA1",
        f"title={book.title}",
        f"artist={book.author or 'Unknown'}",
        f"album={book.title}",
        "genre=Audiobook",
    ]
    concat_lines = []
    start_ms = 0
    for number, path in chapters:
        dur = _ffprobe_duration_ms(path)
        title = "Title" if number == 0 else titles.get(number, f"Chapter {number}")
        meta_lines += [
            "[CHAPTER]", "TIMEBASE=1/1000",
            f"START={start_ms}", f"END={start_ms + dur}",
            f"title={title}",
        ]
        escaped = str(path.resolve()).replace("'", r"'\''")
        concat_lines.append(f"file '{escaped}'")
        start_ms += dur

    meta_file = output_dir / "m4b-metadata.txt"
    list_file = output_dir / "m4b-concat.txt"
    meta_file.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    list_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    dest = output_dir / f"{slugify(book.title)}.m4b"
    partial = dest.with_suffix(".m4b.part")
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", str(list_file),
           "-i", str(meta_file)]
    if cover and cover.exists():
        cmd += ["-i", str(cover)]
    cmd += ["-map_metadata", "1", "-map", "0:a"]
    if cover and cover.exists():
        cmd += ["-map", "2:v", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
    cmd += ["-c:a", "aac", "-b:a", config.M4B_BITRATE,
            "-movflags", "+faststart", "-f", "mp4", str(partial)]
    subprocess.run(cmd, check=True)
    partial.replace(dest)
    meta_file.unlink(missing_ok=True)
    list_file.unlink(missing_ok=True)
    rprint(f"[green]m4b:[/green] {dest}")
    return dest
