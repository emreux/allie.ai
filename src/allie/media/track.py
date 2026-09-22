"""What a lookup answers with, and what it says when it cannot answer.

Two dataclasses and one exception, together because they are the halves of
the same contract: a search either names something - openable, or at least
identifiable - or explains itself in a sentence a tool can pass straight to
the model.

`target` is deliberately not called `id`. What opens a song on YouTube Music
is an address, what opens one in the Spotify application is a `spotify:` URI,
and what opens a video is a third address; the identifier inside each is the
lookup's business and nobody else's. Whoever holds a `Track` only has to hand
`target` to `shell`.

A `Recording` is the other kind of answer: not something to open but
something to *ask for* - the ISRC that identifies one recording on every
service at once, which is how a song Deezer named is found again on Spotify
(`media/deezer.py`, `media/spotify.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Recording", "SearchError", "Track"]


class SearchError(Exception):
    """A search that could not answer, worded so a tool can pass it on.

    Every failure in this package - a refused page, a search with no results,
    a service that timed out - arrives as one of these rather than as an
    `httpx` error or a `KeyError`, because what the model needs is a sentence
    and not a traceback. The message is English and addressed to the model,
    like every other tool result.
    """


@dataclass(frozen=True, slots=True)
class Track:
    """Something that can be started: what to open, and what it is called."""

    target: str
    title: str
    artist: str = ""

    @property
    def name(self) -> str:
        """ "Artist - Title", or the title alone when nothing named an artist.

        This is what the tool reports and what the model reads out, so it is
        the *service's* spelling of the name rather than the user's: someone
        who asked for "kumralım" is told "Yaşar - Kumralım", which is how they
        find out the right song started.
        """
        return _named(self.title, self.artist)


@dataclass(frozen=True, slots=True)
class Recording:
    """One recording, by the code every service files it under.

    `isrc` is the International Standard Recording Code - twelve characters,
    the same for a recording on Deezer, Spotify and YouTube Music alike. It
    is what lets a song found on one service be asked for on another without
    that other service's search being asked to guess from words.
    """

    isrc: str
    title: str
    artist: str = ""

    @property
    def name(self) -> str:
        """ "Artist - Title", spelled the way the service that named it does."""
        return _named(self.title, self.artist)


def _named(title: str, artist: str) -> str:
    return f"{artist} - {title}" if artist else title
