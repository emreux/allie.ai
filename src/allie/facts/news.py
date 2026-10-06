"""Headlines from RSS (D39): Google News in the region's edition, a
fallback feed when it does not answer. The titles are other people's
writing and go to the model inside the `<untrusted>` block."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, tzinfo
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl

__all__ = [
    "GOOGLE_NEWS",
    "GOOGLE_NEWS_SEARCH",
    "Headline",
    "describe_headlines",
    "feed_request",
    "parse_rss",
]

GOOGLE_NEWS = "https://news.google.com/rss"
GOOGLE_NEWS_SEARCH = "https://news.google.com/rss/search"


@dataclass(frozen=True, slots=True)
class Headline:
    title: str
    source: str
    published: datetime | None


def feed_request(edition: str, topic: str) -> tuple[str, dict[str, str]]:
    """The address and parameters for the front page, or for a topic."""
    params = dict(parse_qsl(edition))
    words = " ".join(topic.split())
    if not words:
        return GOOGLE_NEWS, params
    return GOOGLE_NEWS_SEARCH, {"q": words, **params}


def parse_rss(xml_text: str, *, limit: int) -> list[Headline]:
    """The first `limit` items of an RSS 2.0 feed; `ValueError` for anything
    that is not XML."""
    try:
        channel = ET.fromstring(xml_text).find("channel")  # noqa: S314  # expat 2.6, as markets.py
    except ET.ParseError as failure:
        raise ValueError(f"not XML: {failure}") from failure
    found: list[Headline] = []
    for item in channel.findall("item") if channel is not None else []:
        title = " ".join((item.findtext("title") or "").split())
        if not title:
            continue
        source = " ".join((item.findtext("source") or "").split())
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        found.append(
            Headline(title=title, source=source, published=_date(item.findtext("pubDate")))
        )
        if len(found) == limit:
            break
    return found


def describe_headlines(headlines: list[Headline], *, tz: tzinfo | None) -> str:
    lines = []
    for number, headline in enumerate(headlines, start=1):
        tail = [part for part in (headline.source, _clock(headline.published, tz)) if part]
        lines.append(f"{number}. {headline.title}" + (f" - {', '.join(tail)}" if tail else ""))
    return "\n".join(lines)


def _date(value: str | None) -> datetime | None:
    try:
        return parsedate_to_datetime(value) if value else None
    except (TypeError, ValueError):
        return None


def _clock(when: datetime | None, tz: tzinfo | None) -> str:
    return f"{when.astimezone(tz):%H:%M}" if when is not None else ""
