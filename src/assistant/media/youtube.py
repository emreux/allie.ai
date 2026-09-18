"""Finding the identifier YouTube needs, so that the model never invents one.

A YouTube address plays exactly one thing: `watch?v=` and eleven characters
that mean nothing, are not derived from the title, and cannot be worked out
from anything the model knows. Asked for one anyway, a language model writes
eleven plausible characters and YouTube answers *"This video isn't available
anymore"*. That sentence is the reason this module exists. Nothing here asks
the model for an identifier; the identifier is looked up first and only then
is an address built, which is also why the model is never left to reach for
`open_url` when the user asks for music (`tools/media.py`).

**Two services, two lookups, for the same reason each.**

A *song* goes to YouTube Music through `ytmusicapi`, which speaks the site's
own InnerTube API without a key. Its `songs` filter answers with recordings
rather than lyric videos and hour-long mixes, which is what somebody who says
"play X" means.

A *video* goes to YouTube's ordinary results page, and the identifier is read
out of `ytInitialData` - the JSON the page's own script reads to draw the
list. That is a scrape, so every step of it is written to fail into a
sentence: a consent wall, a page whose shape changed, a search with no hits
and a refused request all leave here as `SearchError`, never as a `KeyError`
in the middle of a turn. The documented alternative, InnerTube's
`/youtubei/v1/search`, needs an undocumented key lifted out of the same page
and is slower - a second thing to break, for nothing.

**No language and no country are sent.** No `hl`, no `gl`, no
`Accept-Language`: the code carries no locale constant (design.md section
3.12), and both services answer for the user's own region without being told.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from assistant.media.track import SearchError, Track
from assistant.store.normalize import normalize_search

__all__ = ["MUSIC_WATCH", "WATCH", "YouTube", "YouTubeMusic"]

SEARCH_PAGE = "https://www.youtube.com/results"

# The two addresses that *play* rather than search. `music.youtube.com` is not
# a nicer skin over the same page: opened with a `v=`, it starts the song in
# the signed-in listener's player, which is the whole request.
WATCH = "https://www.youtube.com/watch?v={video_id}"
MUSIC_WATCH = "https://music.youtube.com/watch?v={video_id}"

# Long enough for a slow line, short enough that a search which is not coming
# back does not hold a spoken turn open. `[media] search_timeout_seconds`
# overrides it.
SEARCH_SECONDS = 8.0

# How many songs are asked for before one is chosen. More would only give the
# ranking below more covers to sift through.
RESULTS = 10

# How many places a result may climb for being by the artist the user named.
# Three rather than a sort by similarity: the service's own order knows more
# than anything computed here, and this only has to beat a cover listed above
# the original.
RANK_STEP = 3

# What a YouTube identifier looks like. A result whose `videoId` is not one -
# a shelf header, a playlist, a malformed row - is skipped rather than turned
# into an address that would fail in the browser.
VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Where `ytInitialData` begins. Only the assignment is matched; the value is
# read by the JSON decoder itself, so nothing here has to count braces.
_ASSIGNMENT = re.compile(r"ytInitialData\s*=\s*")

# Served without one, YouTube sometimes answers with a page carrying no
# results at all. The version ages; what matters is that it names a browser.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
}


class YouTube:
    """YouTube's results page, read the way the page's own script reads it."""

    def __init__(
        self, *, client: httpx.AsyncClient | None = None, seconds: float = SEARCH_SECONDS
    ) -> None:
        # One client, built on the first search and kept. Measured on this
        # machine, 10 September 2026: a fresh connection per search costs
        # 1.30-1.64 s and a kept one 0.65-1.06 s, so the handshake is most of
        # half a second in the middle of a spoken turn. `aclose` gives it back.
        self._client = client
        self._borrowed = client is not None
        self._seconds = seconds

    async def first_video(self, words: str) -> Track:
        """The first ordinary video YouTube lists for `words`.

        The first, deliberately: the user said a title and meant the video
        that comes up. Anything cleverer here would be this program guessing
        against YouTube's own ranking with far less to go on.
        """
        page = await self._page(words)
        # A results page is over a megabyte of JSON. Decoding and walking it
        # is real CPU work, and section 3.1 rule 4 keeps that off the loop,
        # where audio capture and the announce queue are waiting.
        return await asyncio.to_thread(_first_video, page, words)

    async def aclose(self) -> None:
        """Gives back the connection this opened, at shutdown.

        A client handed in belongs to whoever handed it in and is left alone.
        """
        if not self._borrowed and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _page(self, words: str) -> str:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._seconds, follow_redirects=True)
        return _body(await self._ask(self._client, words))

    async def _ask(self, client: httpx.AsyncClient, words: str) -> httpx.Response:
        try:
            return await client.get(
                SEARCH_PAGE,
                params={"search_query": words},
                headers=HEADERS,
                timeout=self._seconds,
            )
        except httpx.TimeoutException as failure:
            raise SearchError(
                f"YouTube did not answer within {self._seconds:.0f} seconds."
            ) from failure
        except httpx.HTTPError as failure:
            raise SearchError(f"YouTube could not be reached: {failure}") from failure


class YouTubeMusic:
    """YouTube Music's catalogue, through `ytmusicapi` and no key at all."""

    def __init__(self, *, catalogue: Callable[[], Any] | None = None) -> None:
        # The factory rather than the client: building one reads a bundled
        # file and sets up a session, which is cheap but is not free and is
        # not wanted at startup. Tests hand in a stand-in and never reach the
        # network.
        self._new = catalogue if catalogue is not None else _ytmusic
        self._catalogue: Any | None = None

    async def song(self, words: str) -> Track:
        """The song `words` most likely meant, ready to be opened."""
        found = await self._ask(lambda catalogue: catalogue.search(words, "songs", limit=RESULTS))
        chosen = _pick(found, words)
        if chosen is None:
            raise SearchError(f"YouTube Music listed no song for {words!r}.")
        return chosen

    async def anything(self) -> Track:
        """The first song YouTube Music itself offers, for "play some music".

        The front page is what the service would have played anyway, tuned to
        this listener wherever it knows one. Choosing here instead would be
        the assistant having a taste in music, which nobody asked it for.
        """
        for shelf in await self._ask(lambda catalogue: catalogue.get_home()):
            for item in shelf.get("contents") or []:
                track = _song(item) if isinstance(item, dict) else None
                if track is not None:
                    return track
        raise SearchError("YouTube Music's front page offered nothing playable.")

    async def _ask(self, question: Callable[[Any], Any]) -> list[dict[str, Any]]:
        """Runs one question against the catalogue, off the loop and in words.

        `ytmusicapi` is synchronous and talks to the network, so it belongs on
        a thread (section 3.1 rule 4); and it raises whatever its parsers and
        its HTTP client raise, so everything that comes out of it is turned
        into the one exception the tools above know how to report.
        """
        if self._catalogue is None:
            self._catalogue = await asyncio.to_thread(self._new)
        catalogue = self._catalogue

        try:
            answered = await asyncio.to_thread(question, catalogue)
        except asyncio.CancelledError:
            # A turn the user cancelled is not a failed search, and must not
            # be reported to the model as one.
            raise
        except Exception as failure:  # whatever it raises becomes one sentence here
            raise SearchError(f"YouTube Music could not be searched: {failure}") from failure
        return [item for item in answered or [] if isinstance(item, dict)]


def _ytmusic() -> Any:
    """A `ytmusicapi` client.

    The import is here rather than at the top of the module so that the
    package is read by the thread that first needs it: `config.py` imports
    this module for one number, and `live-assistant --help` should not pay for a
    music library to print a usage line.
    """
    from ytmusicapi import YTMusic

    return YTMusic()


# --------------------------------------------------------------------------
# Reading a results page
# --------------------------------------------------------------------------


def _body(response: httpx.Response) -> str:
    if response.status_code != httpx.codes.OK:
        raise SearchError(f"YouTube refused the search with status {response.status_code}.")
    return response.text


def _first_video(page: str, words: str) -> Track:
    for renderer in _renderers(_initial_data(page)):
        video_id = renderer.get("videoId")
        if not isinstance(video_id, str) or not VIDEO_ID.match(video_id):
            continue
        return Track(
            target=WATCH.format(video_id=video_id),
            # The page's own title rather than the words that were searched
            # for: that is how the user hears which video actually opened.
            title=_text(renderer.get("title")) or words,
            artist=_text(renderer.get("ownerText")) or _text(renderer.get("longBylineText")),
        )
    raise SearchError(f"YouTube listed no video for {words!r}.")


def _initial_data(page: str) -> object:
    """The blob the page's own script reads, decoded.

    A page without it is not a page with no results - it is a consent screen,
    a sign-in wall, or a shape that changed, and telling those apart from here
    would be guesswork. What matters is that the tool says nothing was opened.
    """
    assignment = _ASSIGNMENT.search(page)
    if assignment is None:
        raise SearchError(
            "YouTube answered with a page that carries no results at all - a consent "
            "or sign-in screen, most likely. Nothing was opened."
        )
    try:
        # `raw_decode` reads one value and stops where it ends, which is what
        # a JSON object followed by half a megabyte of script needs.
        data, _ = json.JSONDecoder().raw_decode(page, assignment.end())
    except ValueError as failure:
        raise SearchError("YouTube's results page could not be read.") from failure
    return data


def _renderers(node: object) -> Iterator[dict[str, Any]]:
    """Every `videoRenderer` in the blob, in the order the page lists them.

    A walk rather than a path: the results sit under a different chain of keys
    depending on whether YouTube served a plain list, a shelf or a "people
    also watched" row, and the walk survives all three. Rows that are not
    ordinary videos - shorts, playlists, channels, adverts - carry their own
    key and are simply never yielded.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "videoRenderer" and isinstance(value, dict):
                yield value
            else:
                yield from _renderers(value)
    elif isinstance(node, list):
        for item in node:
            yield from _renderers(item)


def _text(node: object) -> str:
    """The words, out of YouTube's two ways of writing a piece of text."""
    if not isinstance(node, dict):
        return ""
    simple = node.get("simpleText")
    if isinstance(simple, str):
        return simple.strip()
    runs = node.get("runs")
    if isinstance(runs, list):
        words = (run.get("text") for run in runs if isinstance(run, dict))
        return "".join(word for word in words if isinstance(word, str)).strip()
    return ""


# --------------------------------------------------------------------------
# Choosing a song
# --------------------------------------------------------------------------


def _pick(results: list[dict[str, Any]], words: str) -> Track | None:
    wanted = f" {normalize_search(words)} "
    best: tuple[int, Track] | None = None
    for place, item in enumerate(results):
        track = _song(item)
        if track is None:
            continue
        score = place - (RANK_STEP if _named(wanted, track.artist) else 0)
        if best is None or score < best[0]:
            best = (score, track)
    return None if best is None else best[1]


def _named(wanted: str, artist: str) -> bool:
    """Whether the user's words actually contain this artist's name.

    Folded on both sides, so "sezen aksu" finds "Sezen Aksu"; padded with
    spaces on both sides, so a name has to be a word in the request rather
    than a run of letters inside a longer one.
    """
    folded = normalize_search(artist).strip()
    return bool(folded) and f" {folded} " in wanted


def _song(item: dict[str, Any]) -> Track | None:
    video_id = item.get("videoId")
    title = item.get("title")
    if not isinstance(video_id, str) or not VIDEO_ID.match(video_id):
        return None
    if not isinstance(title, str) or not title.strip():
        return None
    return Track(
        target=MUSIC_WATCH.format(video_id=video_id),
        title=title.strip(),
        artist=_first_artist(item.get("artists")),
    )


def _first_artist(artists: object) -> str:
    if not isinstance(artists, list):
        return ""
    for artist in artists:
        name = artist.get("name") if isinstance(artist, dict) else None
        if isinstance(name, str) and name.strip():
            return name.strip()
    return ""
