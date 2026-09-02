from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import DB_PATH, ensure_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS library_clips (
    id          INTEGER PRIMARY KEY,
    path        TEXT UNIQUE NOT NULL,
    transcript  TEXT NOT NULL,
    duration_s  REAL NOT NULL,
    sex         TEXT,             -- male | female
    age_band    TEXT,             -- child|teen|young_adult|adult|middle_aged|elderly
    locale      TEXT,             -- e.g. en-GB
    region      TEXT,             -- e.g. Yorkshire, Southern England
    quality     TEXT,             -- good | ok | poor
    source      TEXT,             -- vctk | librivox | common-voice | custom
    license     TEXT,
    notes       TEXT,
    sha256      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS books (
    id            INTEGER PRIMARY KEY,
    source_path   TEXT UNIQUE NOT NULL,
    title         TEXT,
    author        TEXT,
    gutenberg_id  TEXT,
    output_dir    TEXT NOT NULL,
    engine        TEXT,           -- chosen rendering engine (bakeoff result)
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cast_members (
    book_id         INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    character       TEXT NOT NULL,          -- NARRATOR_KEY or canonical name
    spec_json       TEXT,                   -- VoiceSpec snapshot at cast time
    library_clip_id INTEGER REFERENCES library_clips(id) ON DELETE SET NULL,
    ref_path        TEXT,                   -- frozen reference wav
    ref_transcript  TEXT,
    ref_sha256      TEXT,
    design_seed     INTEGER,                -- voice-design seed, if shaped
    audition_seed   INTEGER,
    engine          TEXT,
    status          TEXT NOT NULL DEFAULT 'proposed',  -- proposed|auditioned|accepted
    PRIMARY KEY (book_id, character)
);

CREATE TABLE IF NOT EXISTS chapter_status (
    book_id        INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    chapter_number INTEGER NOT NULL,
    mp3_path       TEXT,
    engine         TEXT,
    completed_at   TEXT,
    PRIMARY KEY (book_id, chapter_number)
);

CREATE TABLE IF NOT EXISTS qc_flags (
    id             INTEGER PRIMARY KEY,
    book_id        INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    chapter_number INTEGER NOT NULL,
    item_index     INTEGER NOT NULL,
    character      TEXT,
    text           TEXT,
    best_wer       REAL,
    attempts       INTEGER,
    created_at     TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def db(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------- library clips ----------

def clip_add(conn: sqlite3.Connection, *, path: Path, transcript: str,
             duration_s: float, sex: str | None, age_band: str | None,
             locale: str | None, region: str | None, quality: str | None,
             source: str | None, license: str | None, notes: str | None,
             sha256: str) -> int:
    cur = conn.execute(
        "INSERT INTO library_clips (path, transcript, duration_s, sex, age_band, "
        "locale, region, quality, source, license, notes, sha256, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (str(path), transcript, duration_s, sex, age_band, locale, region,
         quality, source, license, notes, sha256, _now()),
    )
    return int(cur.lastrowid)


def clip_get(conn: sqlite3.Connection, clip_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM library_clips WHERE id = ?", (clip_id,)).fetchone()


def clip_list(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute(
        "SELECT * FROM library_clips ORDER BY locale, region, sex, age_band, id"))


def clip_retag(conn: sqlite3.Connection, clip_id: int, field: str, value: str | None) -> None:
    allowed = {"sex", "age_band", "locale", "region", "quality", "source",
               "license", "notes", "transcript"}
    if field not in allowed:
        raise ValueError(f"Cannot retag field {field!r}; one of {sorted(allowed)}")
    conn.execute(f"UPDATE library_clips SET {field} = ? WHERE id = ?", (value, clip_id))


# ---------- books ----------

def book_upsert(conn: sqlite3.Connection, source_path: Path, *, title: str | None,
                author: str | None, gutenberg_id: str | None, output_dir: Path) -> int:
    row = conn.execute(
        "SELECT id FROM books WHERE source_path = ?", (str(source_path),)
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE books SET title=?, author=?, gutenberg_id=?, output_dir=? WHERE id=?",
            (title, author, gutenberg_id, str(output_dir), row["id"]),
        )
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO books (source_path, title, author, gutenberg_id, output_dir, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (str(source_path), title, author, gutenberg_id, str(output_dir), _now()),
    )
    return int(cur.lastrowid)


def book_set_engine(conn: sqlite3.Connection, book_id: int, engine: str) -> None:
    conn.execute("UPDATE books SET engine = ? WHERE id = ?", (engine, book_id))


def book_get(conn: sqlite3.Connection, book_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()


# ---------- cast ----------

def cast_upsert(conn: sqlite3.Connection, book_id: int, character: str, **fields) -> None:
    allowed = {"spec_json", "library_clip_id", "ref_path", "ref_transcript",
               "ref_sha256", "design_seed", "audition_seed", "engine", "status"}
    bad = set(fields) - allowed
    if bad:
        raise ValueError(f"Unknown cast fields: {sorted(bad)}")
    conn.execute(
        "INSERT INTO cast_members (book_id, character) VALUES (?, ?) "
        "ON CONFLICT(book_id, character) DO NOTHING",
        (book_id, character),
    )
    if fields:
        sets = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(
            f"UPDATE cast_members SET {sets} WHERE book_id = ? AND character = ?",
            (*fields.values(), book_id, character),
        )
    conn.commit()


def cast_get(conn: sqlite3.Connection, book_id: int, character: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM cast_members WHERE book_id = ? AND character = ?",
        (book_id, character)).fetchone()


def cast_all(conn: sqlite3.Connection, book_id: int) -> list[sqlite3.Row]:
    return list(conn.execute(
        "SELECT * FROM cast_members WHERE book_id = ? ORDER BY character", (book_id,)))


def cast_delete(conn: sqlite3.Connection, book_id: int, character: str | None = None) -> None:
    if character is None:
        conn.execute("DELETE FROM cast_members WHERE book_id = ?", (book_id,))
    else:
        conn.execute("DELETE FROM cast_members WHERE book_id = ? AND character = ?",
                     (book_id, character))


# ---------- chapter status ----------

def chapter_mark_done(conn: sqlite3.Connection, book_id: int, chapter_number: int,
                      mp3_path: Path, engine: str | None = None) -> None:
    conn.execute(
        "INSERT INTO chapter_status (book_id, chapter_number, mp3_path, engine, completed_at) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(book_id, chapter_number) DO UPDATE SET "
        "  mp3_path = excluded.mp3_path, engine = excluded.engine, "
        "  completed_at = excluded.completed_at",
        (book_id, chapter_number, str(mp3_path), engine, _now()),
    )
    conn.commit()


def chapter_is_done(conn: sqlite3.Connection, book_id: int, chapter_number: int) -> Path | None:
    row = conn.execute(
        "SELECT mp3_path, completed_at FROM chapter_status "
        "WHERE book_id = ? AND chapter_number = ?",
        (book_id, chapter_number),
    ).fetchone()
    if row and row["completed_at"] and row["mp3_path"]:
        p = Path(row["mp3_path"])
        if p.exists():
            return p
    return None


def chapter_status_clear(conn: sqlite3.Connection, book_id: int,
                         chapter_number: int | None = None) -> None:
    if chapter_number is None:
        conn.execute("DELETE FROM chapter_status WHERE book_id = ?", (book_id,))
    else:
        conn.execute(
            "DELETE FROM chapter_status WHERE book_id = ? AND chapter_number = ?",
            (book_id, chapter_number))


# ---------- qc flags ----------

def qc_flag_add(conn: sqlite3.Connection, book_id: int, chapter_number: int,
                item_index: int, character: str | None, text: str,
                best_wer: float, attempts: int) -> None:
    conn.execute(
        "INSERT INTO qc_flags (book_id, chapter_number, item_index, character, "
        "text, best_wer, attempts, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (book_id, chapter_number, item_index, character, text, best_wer,
         attempts, _now()),
    )


def qc_flags_for(conn: sqlite3.Connection, book_id: int) -> list[sqlite3.Row]:
    return list(conn.execute(
        "SELECT * FROM qc_flags WHERE book_id = ? ORDER BY chapter_number, item_index",
        (book_id,)))


def qc_flags_clear(conn: sqlite3.Connection, book_id: int,
                   chapter_number: int | None = None) -> None:
    if chapter_number is None:
        conn.execute("DELETE FROM qc_flags WHERE book_id = ?", (book_id,))
    else:
        conn.execute("DELETE FROM qc_flags WHERE book_id = ? AND chapter_number = ?",
                     (book_id, chapter_number))
