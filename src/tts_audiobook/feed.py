from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from .book import Book

FEED_FILENAME = "feed.xml"

ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"


def _attr(s: str) -> str:
    return quoteattr(s)


def _text(s: str) -> str:
    return escape(s)


def _parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _default_base_url(output_dir: Path) -> str:
    return "file://" + str(output_dir.resolve()) + "/"


def _build_item(*, episode: int, title: str, description: str,
                filename: str, file_size: int, pub_date_rfc2822: str,
                base_url: str) -> str:
    url = base_url + filename
    return (
        "    <item>\n"
        f"      <title>{_text(title)}</title>\n"
        f"      <description>{_text(description)}</description>\n"
        f"      <enclosure url={_attr(url)} length=\"{file_size}\" type=\"audio/mpeg\"/>\n"
        f"      <guid isPermaLink=\"false\">{_text(filename)}</guid>\n"
        f"      <pubDate>{pub_date_rfc2822}</pubDate>\n"
        f"      <itunes:episode>{episode}</itunes:episode>\n"
        "      <itunes:episodeType>full</itunes:episodeType>\n"
        "    </item>"
    )


def write_feed(conn: sqlite3.Connection, book: Book, book_id: int,
               output_dir: Path, base_url: str | None = None) -> Path:
    """Emit (or rewrite) feed.xml summarizing every completed mp3."""
    if base_url is None:
        base_url = _default_base_url(output_dir)
    if not base_url.endswith("/"):
        base_url += "/"

    chapter_titles = {c.number: c.title for c in book.chapters}

    rows = conn.execute(
        "SELECT chapter_number, mp3_path, completed_at "
        "FROM chapter_status WHERE book_id = ? AND completed_at IS NOT NULL "
        "ORDER BY chapter_number ASC",
        (book_id,),
    ).fetchall()

    items_xml: list[str] = []
    for row in rows:
        mp3_path = Path(row["mp3_path"])
        if not mp3_path.exists():
            continue
        ep = int(row["chapter_number"])
        if ep == 0:
            title = f"{book.title} — Title"
            description = (f"{book.title}, by {book.author}." if book.author
                           else f"{book.title}.")
        else:
            ch_title = chapter_titles.get(ep, f"Chapter {ep}")
            title = f"Chapter {ep}: {ch_title}" if ch_title and ch_title != f"Chapter {ep}" \
                    else f"Chapter {ep}"
            description = ch_title
        items_xml.append(_build_item(
            episode=ep,
            title=title,
            description=description,
            filename=mp3_path.name,
            file_size=mp3_path.stat().st_size,
            pub_date_rfc2822=format_datetime(_parse_iso(row["completed_at"])),
            base_url=base_url,
        ))

    lang = "en" if book.language.lower().startswith("en") else book.language.lower()[:2]
    author = book.author or "Unknown"
    synopsis = book.production.synopsis if book.production else None
    channel_desc = synopsis or (f"Audiobook performance of {book.title}"
                                + (f" by {author}." if book.author else "."))
    now = format_datetime(datetime.now(timezone.utc))

    feed = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<rss version="2.0" xmlns:itunes="{ITUNES_NS}">\n'
        '  <channel>\n'
        f'    <title>{_text(book.title)}</title>\n'
        f'    <link>{_text(base_url)}</link>\n'
        f'    <description>{_text(channel_desc)}</description>\n'
        f'    <language>{lang}</language>\n'
        f'    <lastBuildDate>{now}</lastBuildDate>\n'
        f'    <itunes:author>{_text(author)}</itunes:author>\n'
        f'    <itunes:summary>{_text(channel_desc)}</itunes:summary>\n'
        '    <itunes:type>serial</itunes:type>\n'
        '    <itunes:explicit>false</itunes:explicit>\n'
        + ("\n".join(items_xml) + "\n" if items_xml else "")
        + '  </channel>\n'
        '</rss>\n'
    )

    dest = output_dir / FEED_FILENAME
    dest.write_text(feed, encoding="utf-8")
    return dest
