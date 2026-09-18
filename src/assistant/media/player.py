"""Where a media request goes, and what actually opens.

The lookups in `youtube.py` and `spotify.py` know how to name a thing; this
knows which of them to ask, what to do before starting something, and how to
say what happened. The tools in `tools/media.py` are three docstrings over
the three methods below.

**Three rules decide almost everything here.**

*Play, do not search.* A request for a song ends at an address that starts
playing - `music.youtube.com/watch?v=<id>` - and never at a results page the
user then has to click. Whether that is possible is a property of the
service, not of the request: YouTube Music can be asked for an identifier and
Spotify cannot be asked for one without credentials this build does not have
(`spotify.py`). What Spotify can be asked for is the one recording with a
given ISRC, and Deezer names that code for free (`deezer.py`) - so a Spotify
request opens the exact recording when Deezer knows it and a search when it
does not, and the answer *says* which, and that nothing started. What is
never allowed is opening a page and reporting a song.

*One at a time.* Every song opens in the assistant's own browser window,
and the previous one is closed once the new one is there (`window.py`); what
plays elsewhere - the Spotify application, a tab of the user's own - is
paused through the Windows session rather than by a media key
(`now_playing.py`), and two requests that overlap are served one after the
other rather than racing each other into two windows.

*The failure sentence forbids the sentence the model would otherwise reach
for.* A model told only "the search failed" still tends to tell the user
something is playing. Every failure here therefore ends with what must not be
said, which is cheap and works.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from loguru import logger

from assistant.config import MediaSettings
from assistant.media.deezer import Deezer
from assistant.media.now_playing import pause_current
from assistant.media.spotify import Spotify
from assistant.media.track import Recording, SearchError, Track
from assistant.media.window import MediaWindow, default_browser
from assistant.media.youtube import YouTube, YouTubeMusic
from assistant.store.normalize import normalize_search

__all__ = [
    "MAX_QUERY",
    "SERVICES",
    "MusicSearch",
    "Player",
    "RecordingSearch",
    "Service",
    "VideoSearch",
    "service_keys",
]

YOUTUBE_MUSIC = "youtube_music"
YOUTUBE = "youtube"
SPOTIFY = "spotify"


class MusicSearch(Protocol):
    """Whatever can name a song. `YouTubeMusic` is the one implementation.

    A protocol rather than the class, for the reason section 3.2 gives for the
    provider protocols: what this file needs is two questions answered, and
    binding it to `ytmusicapi` would mean every test of the rules below had a
    music service in it.
    """

    async def song(self, words: str) -> Track: ...

    async def anything(self) -> Track: ...


class VideoSearch(Protocol):
    """Whatever can name a video. `YouTube` is the one implementation.

    `aclose` is part of the contract because the implementation keeps a
    connection open between searches - which is half a second of a spoken
    turn - and something has to be able to give it back.
    """

    async def first_video(self, words: str) -> Track: ...

    async def aclose(self) -> None: ...


class RecordingSearch(Protocol):
    """Whatever can name a recording by its ISRC. `Deezer` is the one
    implementation, and the reason is Spotify: its search takes a code where
    it will not take a key (`spotify.py`)."""

    async def recording(self, words: str) -> Recording: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Service:
    """One place music or video comes from.

    `spoken` is what the user might call it, folded the way search folds
    (`store/normalize.py`), so that "YouTube Music", "youtube music" and the
    "yutub müzik" a recogniser sometimes writes are one service. The keys are
    English and so are the labels: the key is what the model chooses, the
    label is what the answer names, and the language the user hears it in is
    the model's business (section 3.12).
    """

    key: str
    label: str
    spoken: tuple[str, ...]
    # Where "open <service>" goes. Empty for Spotify, whose address depends on
    # whether the application is installed and is asked for at the moment of
    # opening.
    home: str = ""


SERVICES: tuple[Service, ...] = (
    Service(
        key=YOUTUBE_MUSIC,
        label="YouTube Music",
        spoken=("youtube music", "yt music", "ytmusic", "youtubemusic", "yutub muzik"),
        home="https://music.youtube.com",
    ),
    Service(
        key=YOUTUBE,
        label="YouTube",
        spoken=("youtube", "yt", "yutub"),
        home="https://www.youtube.com",
    ),
    Service(key=SPOTIFY, label="Spotify", spoken=("spotify", "spotifay", "spotfy")),
)

# Longer than this, and what arrived is not something a person said out loud.
# A model that pastes a paragraph into a search should be told so rather than
# have the paragraph opened in a browser.
MAX_QUERY = 200

# What every failing answer ends with. Written once because it is the whole
# reason the failures are worded at all.
NOT_PLAYING = "Nothing is playing - do not tell the user a song started."


def service_keys() -> str:
    """The service keys, for a tool description and for a refusal."""
    return ", ".join(service.key for service in SERVICES)


class Player:
    """The one object that turns "play X" into something opening."""

    def __init__(
        self,
        *,
        music: MusicSearch | None = None,
        videos: VideoSearch | None = None,
        recordings: RecordingSearch | None = None,
        spotify: Spotify | None = None,
        settings: MediaSettings | None = None,
        pause: Callable[[], Awaitable[bool]] = pause_current,
        window: MediaWindow | None = None,
    ) -> None:
        self.settings = settings or MediaSettings()
        seconds = self.settings.search_timeout_seconds
        self.music: MusicSearch = music or YouTubeMusic()
        self.videos: VideoSearch = videos or YouTube(seconds=seconds)
        self.recordings: RecordingSearch = recordings or Deezer(seconds=seconds)
        self.spotify = spotify or Spotify()
        # Injected so that a test can watch it happen without a media session
        # on the machine, and so that a Windows that answers strangely is one
        # replaceable function rather than a branch in here.
        self.pause = pause
        # Where a web address opens: one window of the user's browser, the
        # previous one closed once the next is there (`media/window.py`).
        # Injected so that a test sees the address without a browser.
        self.window = window if window is not None else MediaWindow(default_browser())
        # One song at a time: two overlapping requests would otherwise pause
        # each other's window and leave two of them playing.
        self._turn = asyncio.Lock()

    # ----------------------------------------------------------------------
    # What the tools call
    # ----------------------------------------------------------------------

    async def play_music(self, query: str, service: str) -> str:
        """Starts a song, or opens a search where a song cannot be started."""
        chosen = self._service(service)
        if chosen is None:
            return f"No music service called {service!r}; the services are: {service_keys()}."

        words = query.strip()
        if len(words) > MAX_QUERY:
            return f"That is too long to have been spoken ({len(words)} characters). {NOT_PLAYING}"

        if chosen.key == SPOTIFY:
            return await self._search_spotify(words)
        return await self._play_song(chosen, words)

    async def play_video(self, query: str) -> str:
        """Opens the first video YouTube lists for what the user described."""
        words = query.strip()
        if not words:
            return "No video was named. Ask the user which video they want."
        if len(words) > MAX_QUERY:
            return (
                f"That is too long to have been spoken ({len(words)} characters). "
                "Nothing was opened."
            )

        async with self._turn:
            try:
                track = await self.videos.first_video(words)
            except SearchError as failure:
                return f"{failure} Nothing was opened - do not tell the user a video is playing."
            return await self._start(track, "YouTube")

    async def aclose(self) -> None:
        """Lets go of whatever the lookups hold open. Called at shutdown."""
        await self.videos.aclose()
        await self.recordings.aclose()

    async def open_service(self, service: str) -> str:
        """Opens a service the model named by key, playing nothing."""
        chosen = _matching(normalize_search(service).strip())
        if chosen is None:
            return f"No media service called {service!r}; the services are: {service_keys()}."
        return await self._open_home(chosen)

    async def open_named(self, spoken: str) -> str | None:
        """Opens the service the user named, or `None` if they named no service.

        `None` rather than a refusal: this is what `open_app` falls back to
        when the machine has no application by that name (`tools/system.py`),
        and a name that is neither an app nor a service belongs to `open_app`
        to answer.
        """
        service = self._named(spoken)
        return None if service is None else await self._open_home(service)

    # ----------------------------------------------------------------------
    # Doing it
    # ----------------------------------------------------------------------

    async def _play_song(self, service: Service, words: str) -> str:
        wanted = words or self.settings.default_query.strip()
        async with self._turn:
            try:
                # No words at all is a real request - "put some music on" - and
                # the front page is the service's own answer to it. Guessing a
                # song here would be the assistant having a taste in music.
                track = await (self.music.song(wanted) if wanted else self.music.anything())
            except SearchError as failure:
                return f"{failure} {NOT_PLAYING}"
            return await self._start(track, service.label)

    async def _search_spotify(self, words: str) -> str:
        """Opens Spotify as close to the song as a free account allows.

        Nothing is paused first: nothing is about to start, and pausing what
        the user is listening to for a page they still have to click would be
        the assistant interrupting them for nothing.
        """
        if not words:
            return (
                "Spotify cannot be asked for 'something to play' from here - only for a "
                f"search. Ask the user which song or artist they want. {NOT_PLAYING}"
            )

        # The exact recording, when Deezer can name its code (`deezer.py`):
        # one result on Spotify's screen instead of a page of them.
        try:
            found = await self.recordings.recording(words)
        except SearchError as failure:
            logger.debug("no ISRC for the Spotify search: {}", failure)
        else:
            if not await self._open(self.spotify.exact(found.isrc)):
                return f"Spotify would not open. {NOT_PLAYING}"
            return (
                f"Opened Spotify on the exact recording {found.name!r} - it is the only "
                f"result on screen. {NOT_PLAYING} Spotify's catalogue cannot be searched "
                "without credentials this build does not have, so the song is shown but "
                "not started: tell the user to press play, or offer to play it on "
                "YouTube Music."
            )

        # A tidy "Artist - Title" makes Spotify's own search land on the
        # recording rather than on a cover; when the lookup fails, the user's
        # own words were always going to be good enough.
        wanted = words
        try:
            wanted = (await self.music.song(words)).name
        except SearchError as failure:
            logger.debug("no tidy name for the Spotify search: {}", failure)

        if not await self._open(self.spotify.search(wanted)):
            return f"Spotify would not open. {NOT_PLAYING}"
        return (
            f"Opened a Spotify search for {wanted!r}. {NOT_PLAYING} Searching Spotify's "
            "catalogue needs credentials this build does not have, so tell the user the "
            "search is open and they can start it - or offer to play it on YouTube Music."
        )

    async def _open_home(self, service: Service) -> str:
        target = self.spotify.home() if service.key == SPOTIFY else service.home
        if not await self._open(target):
            return f"{service.label} would not open."
        # Opening a service is not playing anything, and a model that has just
        # been asked for music will say otherwise unless it is told.
        return f"Opened {service.label}. {NOT_PLAYING}"

    async def _start(self, track: Track, label: str) -> str:
        if self.settings.pause_before_playing:
            await self._pause()
        if not await self._open(track.target):
            return f"No browser would open {track.target}. {NOT_PLAYING}"
        return f"Playing {track.name} on {label}."

    async def _pause(self) -> None:
        if await self.pause():
            logger.debug("paused what was playing before starting the next thing")

    async def _open(self, target: str) -> bool:
        """Hands `target` to Windows the way its scheme asks to be handled.

        A `spotify:` URI goes to the installed application - started first
        when it is not running, so that the search is not lost on the way
        (`spotify.py`); everything else is a web address and goes to the
        assistant's own browser window (`media/window.py`) - in the profile
        the user is signed in to (`shell.py`).
        """
        try:
            if target.startswith("spotify:"):
                return await self.spotify.open(target)
            return await self.window.show(target)
        except OSError as failure:
            logger.warning("{} would not open: {}", target, failure)
            return False

    # ----------------------------------------------------------------------
    # Naming a service
    # ----------------------------------------------------------------------

    def _service(self, wanted: str) -> Service | None:
        """The service the model asked for, or the one the settings name.

        A service the *model* invented is refused, so that it can call again
        with a real one. A `default_service` the *user* mistyped is not: their
        typo is not a reason for music to stop working, so it is logged and
        the default default is used.
        """
        asked = normalize_search(wanted).strip()
        if asked:
            return _matching(asked)

        configured = _matching(normalize_search(self.settings.default_service).strip())
        if configured is None:
            logger.warning(
                "[media] default_service is {!r}, which is not one of {}; using {}",
                self.settings.default_service,
                service_keys(),
                SERVICES[0].label,
            )
            return SERVICES[0]
        return configured

    def _named(self, spoken: str) -> Service | None:
        """The service the *user* called `spoken`, by name only.

        The keys are not accepted here and neither is an empty string: this
        answers "open YouTube Music", not "open" and not "open youtube_music",
        which is a name nobody says.
        """
        said = normalize_search(spoken).strip()
        if not said:
            return None
        return next((service for service in SERVICES if said in service.spoken), None)


def _matching(asked: str) -> Service | None:
    """The service whose key or spoken name is exactly `asked`, already folded."""
    return next(
        (service for service in SERVICES if asked == service.key or asked in service.spoken), None
    )
