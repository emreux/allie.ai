"""What is trending on X (Twitter), by country, from trends24.in (plan.md
D30, 23 September 2026).

**Why trends24.** X's own trends endpoint costs $0.010 a request with no
free credit, and Grok's X search is billed per post read; trends24.in
publishes the list X shows, hour by hour, for 53 countries and the world,
in the page's own HTML (no script needed) and with a robots.txt that
allows every path (checked 2026-09-23). `getdaytrends.com` carries the same
lists and is the fallback if this page ever goes - one file.

**Only the latest hour is read.** The page is a timeline: one
`div.list-container` per hour, newest first, each with its moment in
`h3.title[data-timestamp]` and the topics as `a.trend-link`. The first ten
of the first list are what the user asked about; the rest of the day is
not. A page without that list has changed shape, and says so rather than
reading something else - the stats `article` above the timeline included.

**The country is the site's English name.** The model is told to pass one;
`ALIASES` catches the few everyday others ("UK", "USA", "Türkiye"). No
country is assumed: an empty one is the tool's to ask about (invariant 4,
as `get_weather`'s place).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from loguru import logger
from selectolax.parser import HTMLParser

from allie.store.normalize import normalize_search
from allie.web.page import ACCEPT, FETCH_SECONDS, USER_AGENT

__all__ = [
    "ALIASES",
    "CHANGED",
    "COUNTRIES",
    "NOT_REACHED",
    "TRENDS_READ",
    "TRENDS_URL",
    "WORLDWIDE",
    "WORLDWIDE_URL",
    "Place",
    "Trends",
    "TrendsError",
    "TrendsPage",
    "place_of",
    "read_trends",
]

TRENDS_URL = "https://trends24.in/{slug}/"
WORLDWIDE_URL = "https://trends24.in/"
WORLDWIDE = "worldwide"

# How many topics of the hour are read: what a spoken answer can carry.
TRENDS_READ = 10

# The countries trends24.in lists (its sitemap, 2026-09-23). The address is
# the name in lower case with hyphens: "United Kingdom" -> united-kingdom.
COUNTRIES: tuple[str, ...] = (
    "Algeria", "Argentina", "Australia", "Austria", "Belarus", "Brazil", "Canada", "Chile",
    "Colombia", "Dominican Republic", "Ecuador", "Egypt", "France", "Germany", "Ghana",
    "Greece", "Guatemala", "India", "Indonesia", "Ireland", "Israel", "Italy", "Japan",
    "Jordan", "Kenya", "Korea", "Latvia", "Malaysia", "Mexico", "Netherlands", "New Zealand",
    "Nigeria", "Norway", "Oman", "Pakistan", "Peru", "Philippines", "Poland", "Russia",
    "Saudi Arabia", "Singapore", "South Africa", "Spain", "Sweden", "Switzerland", "Thailand",
    "Turkey", "Ukraine", "United Arab Emirates", "United Kingdom", "United States",
    "Venezuela", "Vietnam",
)  # fmt: skip

# Everyday English names for a listed country, folded as `normalize_search`
# folds them ("Türkiye" -> "turkiye").
ALIASES: dict[str, str] = {
    "usa": "United States",
    "us": "United States",
    "america": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "turkiye": "Turkey",
    "south korea": "Korea",
    "uae": "United Arab Emirates",
    "holland": "Netherlands",
}
_WORLDWIDE_WORDS = frozenset({"worldwide", "world", "global", "everywhere"})

CHANGED = "trends24.in no longer has the list where it used to be: the page has changed."
NOT_REACHED = "trends24.in could not be reached ({reason})."


class TrendsError(Exception):
    """A list that could not be read, worded so that a tool can pass it on."""


@dataclass(frozen=True, slots=True)
class Place:
    """A country trends24 lists, or the world: its name and its page."""

    name: str
    url: str


@dataclass(frozen=True, slots=True)
class Trends:
    """One hour of a place's list: when, and the topics in their order
    with the posts counted where the page counts them."""

    place: str
    at: datetime | None
    names: tuple[str, ...]
    counts: tuple[int | None, ...]


def _key(name: str) -> str:
    return " ".join(normalize_search(name).replace("-", " ").split())


_BY_KEY = {_key(name): name for name in COUNTRIES}


def place_of(country: str) -> Place | None:
    """The listed place `country` names, or `None` for one with no list."""
    key = _key(country)
    if not key:
        return None
    if key in _WORLDWIDE_WORDS:
        return Place(WORLDWIDE, WORLDWIDE_URL)
    name = ALIASES.get(key) or _BY_KEY.get(key)
    if name is None:
        return None
    return Place(name, TRENDS_URL.format(slug=name.lower().replace(" ", "-")))


def read_trends(place: str, html: bytes | str) -> Trends:
    """The first hour of the page: its moment and its first `TRENDS_READ`
    topics. Raises `TrendsError(CHANGED)` when the list is not there."""
    tree = HTMLParser(html)
    newest = tree.css_first(".list-container")
    names: list[str] = []
    counts: list[int | None] = []
    at: datetime | None = None
    if newest is not None:
        heading = newest.css_first("h3.title")
        at = _moment(heading.attributes.get("data-timestamp") if heading is not None else None)
        for entry in newest.css("li"):
            link = entry.css_first("a.trend-link")
            name = " ".join(link.text().split()) if link is not None else ""
            if not name:
                continue
            tally = entry.css_first(".tweet-count")
            names.append(name)
            counts.append(_count(tally.attributes.get("data-count") if tally is not None else None))
            if len(names) == TRENDS_READ:
                break
    if not names:
        logger.warning("trends: {}", CHANGED)
        raise TrendsError(CHANGED)
    return Trends(place=place, at=at, names=tuple(names), counts=tuple(counts))


class TrendsPage:
    """trends24.in over one kept connection, like the weather's."""

    def __init__(
        self, *, client: httpx.AsyncClient | None = None, seconds: float = FETCH_SECONDS
    ) -> None:
        self._client = client
        self._borrowed = client is not None
        self._seconds = seconds

    async def latest(self, place: Place) -> Trends:
        """The latest hour of `place`'s list."""
        return read_trends(place.name, await self._get(place.url))

    async def aclose(self) -> None:
        """Gives back the connection, at shutdown. A borrowed client is left alone."""
        if not self._borrowed and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, url: str) -> bytes:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._seconds, follow_redirects=True)
        try:
            response = await self._client.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": ACCEPT},
                timeout=self._seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as failure:
            reason = f"no answer within {self._seconds:.0f} seconds"
            raise TrendsError(NOT_REACHED.format(reason=reason)) from failure
        except httpx.HTTPStatusError as failure:
            reason = f"HTTP {failure.response.status_code}"
            raise TrendsError(NOT_REACHED.format(reason=reason)) from failure
        except httpx.HTTPError as failure:
            raise TrendsError(NOT_REACHED.format(reason=type(failure).__name__)) from failure
        return response.content


def _moment(value: str | None) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(value or ""), tz=UTC)
    except (ValueError, OverflowError, OSError):
        return None


def _count(value: str | None) -> int | None:
    try:
        return int((value or "").strip())
    except ValueError:
        return None
