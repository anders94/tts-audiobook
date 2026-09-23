from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import write_enriched

from tts_audiobook import db as dbmod
from tts_audiobook.book import load_book, slugify
from tts_audiobook.feed import ATOM_NS, default_base_url, m4b_path, write_feed

BASE = "https://example.org/pp/"


def _setup(tmp_path):
    book = load_book(write_enriched(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    (out / "01_chapter-i.mp3").write_bytes(b"\xff\xfb" * 100)
    conn = dbmod.connect(tmp_path / "studio.db")
    book_id = dbmod.book_upsert(conn, book.source_path, title=book.title,
                                author=book.author, gutenberg_id=None, output_dir=out)
    dbmod.chapter_mark_done(conn, book_id, 1, out / "01_chapter-i.mp3", "qwen")
    return conn, book, book_id, out


def _channel(path: Path) -> ET.Element:
    return ET.parse(path).getroot().find("channel")


def test_feed_without_m4b_has_no_whole_book_link(tmp_path):
    conn, book, book_id, out = _setup(tmp_path)
    assert m4b_path(book, out) is None
    ch = _channel(write_feed(conn, book, book_id, out, BASE))
    assert "single audiobook file" not in ch.findtext("description")
    assert ch.find(f"{{{ATOM_NS}}}link") is None
    assert len(ch.findall("item")) == 1


def test_feed_links_m4b_when_packaged(tmp_path):
    conn, book, book_id, out = _setup(tmp_path)
    m4b = out / f"{slugify(book.title)}.m4b"
    m4b.write_bytes(b"\0" * 2_500_000)
    assert m4b_path(book, out) == m4b

    ch = _channel(write_feed(conn, book, book_id, out, BASE))
    url = BASE + m4b.name
    assert ch.findtext("description").endswith(f"chapter markers: {url}")
    link = ch.find(f"{{{ATOM_NS}}}link")
    assert link is not None
    assert link.get("href") == url and link.get("type") == "audio/mp4"
    assert "2 MB" in link.get("title")
    # Episodes are untouched: still the per-chapter mp3, not the m4b.
    items = ch.findall("item")
    assert len(items) == 1
    assert items[0].find("enclosure").get("url") == BASE + "01_chapter-i.mp3"


def test_default_base_url_is_publish_root_plus_book_subdir(tmp_path):
    conn, book, book_id, out = _setup(tmp_path)
    assert default_base_url(book) == "https://gutenbergaloud.org/books/1342-pride-and-prejudice/"
    ch = _channel(write_feed(conn, book, book_id, out))
    assert ch.findtext("link") == "https://gutenbergaloud.org/books/1342-pride-and-prejudice/"
    assert ch.find("item/enclosure").get("url") == \
        "https://gutenbergaloud.org/books/1342-pride-and-prejudice/01_chapter-i.mp3"

