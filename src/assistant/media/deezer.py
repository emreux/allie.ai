"""Naming a recording by its ISRC, through the one catalogue that answers without a key.

This module exists for Spotify's sake. Spotify's own catalogue cannot be
searched without a developer application, and since February 2026 such an
application needs its owner to hold a Premium subscription - which this
build does not ask of anyone (design.md section 3.6, section 12 decision 25).
What Spotify's search box *does* understand without any of that is the
`isrc:` filter: given the International Standard Recording Code, it shows
that one recording and nothing else. So the question "which recording did
the user mean" is put to Deezer instead, whose public API answers a search
and a track's ISRC with no key, no account and no rate to speak of (measured
2026-09-13: eight songs, Turkish and not, 0.2-0.35 s each, all with a code).

Two requests, because Deezer's search rows carry no ISRC: the search names
the track, and `GET /track/<id>` names the code. Whatever comes back is
checked for the shape of an ISRC before it is believed - twelve characters,
two letters first - so that what ends up after `isrc:` in a Spotify search
is a code and never a piece of stray text.

Every way this can fail leaves as a `SearchError` in words, as in
`youtube.py`: a line that is down, a search with no rows, Deezer's own
`error` object (which it sends with HTTP 200), a track without a code. The
caller falls back to a plain word search on Spotify; nothing here is fatal.

No language and no country are sent (design.md section 3.12); Deezer answers
for the user's own region on its own.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from assistant.media.track import Recording, SearchError
from assistant.media.youtube import SEARCH_SECONDS

__all__ = ["ISRC", "SEARCH", "TRACK", "Deezer"]

SEARCH = "https://api.deezer.com/search"
TRACK = "https://api.deezer.com/track/{track_id}"

# What an ISRC looks like once the dashes are gone: a country, a three-place
# registrant, a two-digit year and a five-digit number. Deezer writes them
# without dashes and in upper case already; the check is for the day it does
# not.
ISRC = re.compile(r"^[A-Z]{2}[A-Z0-9]{3}[0-9]{7}$")

# How many rows are asked for. Only the first is used - Deezer's own order
# knows more than anything computed here - but a first row that is not a
# track is skipped, and one more gives that somewhere to go.
RESULTS = 3


class Deezer:
    """Deezer's public catalogue, asked for the ISRC of what the user said."""

    def __init__(
        self, *, client: httpx.AsyncClient | None = None, seconds: float = SEARCH_SECONDS
    ) -> None:
        # One client, built on the first lookup and kept for the connection,
        # as `YouTube` keeps its own; `aclose` gives it back. One handed in
        # belongs to whoever handed it in.
        self._client = client
        self._borrowed = client is not None
        self._seconds = seconds

    async def recording(self, words: str) -> Recording:
        """The recording Deezer lists first for `words`, with its ISRC."""
        rows = _rows(await self._ask(SEARCH, params={"q": words, "limit": RESULTS}))
        track_id = next((row["id"] for row in rows if _is_track(row)), None)
        if track_id is None:
            raise SearchError(f"Deezer listed no recording for {words!r}.")

        track = await self._ask(TRACK.format(track_id=track_id))
        return _recording(track)

    async def aclose(self) -> None:
        """Gives back the connection this opened, at shutdown."""
        if not self._borrowed and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _ask(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._seconds, follow_redirects=True)
        try:
            response = await self._client.get(url, params=params, timeout=self._seconds)
        except httpx.TimeoutException as failure:
            raise SearchError(
                f"Deezer did not answer within {self._seconds:.0f} seconds."
            ) from failure
        except httpx.HTTPError as failure:
            raise SearchError(f"Deezer could not be reached: {failure}") from failure
        return _body(response)


# --------------------------------------------------------------------------
# Reading the answers
# --------------------------------------------------------------------------


def _body(response: httpx.Response) -> dict[str, Any]:
    if response.status_code != httpx.codes.OK:
        raise SearchError(f"Deezer refused the request with status {response.status_code}.")
    try:
        body = response.json()
    except ValueError as failure:
        raise SearchError("Deezer's answer could not be read.") from failure
    if not isinstance(body, dict):
        raise SearchError("Deezer's answer could not be read.")
    if "error" in body:
        # Deezer's way of refusing: HTTP 200 and an `error` object with a
        # message in it. The message is Deezer's own and is passed on.
        problem = body["error"]
        message = problem.get("message", problem) if isinstance(problem, dict) else problem
        raise SearchError(f"Deezer answered with an error: {message}.")
    return body


def _rows(body: dict[str, Any]) -> list[Any]:
    rows = body.get("data")
    return rows if isinstance(rows, list) else []


def _is_track(row: object) -> bool:
    return isinstance(row, dict) and row.get("type") == "track" and isinstance(row.get("id"), int)


def _recording(track: dict[str, Any]) -> Recording:
    title = track.get("title")
    if not isinstance(title, str) or not title.strip():
        raise SearchError("Deezer's track has no title.")
    artist = track.get("artist")
    artist_name = artist.get("name", "") if isinstance(artist, dict) else ""

    code = track.get("isrc")
    isrc = code.replace("-", "").strip().upper() if isinstance(code, str) else ""
    if not isrc:
        raise SearchError(f"Deezer knows {title!r} but lists no ISRC for it.")
    if not ISRC.match(isrc):
        raise SearchError(f"Deezer's ISRC for {title!r} does not read as one: {code!r}.")
    return Recording(isrc=isrc, title=title.strip(), artist=str(artist_name).strip())
