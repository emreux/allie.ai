"""What "on Spotify" can mean without asking the user for credentials.

Spotify *plays* anything its desktop application is handed as a `spotify:`
URI, and that application is the user's own - signed in, on whatever device
they last listened with, Premium or not. So starting something needs nothing
from this program. *Finding* something does, and that is where it stops:
Spotify's catalogue cannot be searched anonymously. The web player's token
endpoint answers `403`, the results page carries no identifiers until its own
script has run (measured again 2026-09-13), and the only documented way in
is a developer application - which, since February 2026, needs its owner to
hold a Premium subscription. This build does not ask that of anyone (the
owner's decision, 10 and 13 September 2026, design.md section 12 decision
25): a user with a free account gets everything a free account can do.

What a free account can do is *one exact result*. Spotify's own search box
understands an `isrc:` filter, and the ISRC of the recording the user meant
can be had without a key from Deezer (`media/deezer.py`). So the words go
to Deezer, the code goes to Spotify, and what opens is the one recording
rather than a page of candidates - in the installed application through
`spotify:search:isrc%3A<code>`, or on the website without it. When Deezer
cannot name the recording, the words themselves are searched, as before.

Either way **nothing starts playing**: a search with one result is still a
search, and the tool result says so in as many words, so that the model
tells the user to press play rather than announcing a song that is not on.
That asymmetry - a real song on YouTube Music, an open page on Spotify - is
visible to the user and is meant to be. The alternative was to have the
assistant claim it had played something it had not.
"""

from __future__ import annotations

import asyncio
import ctypes
import time
from collections.abc import Callable
from ctypes import wintypes
from typing import Any
from urllib.parse import quote

from loguru import logger

from assistant import shell
from assistant.media.window import Win32Desktop

__all__ = [
    "APP_HOME",
    "APP_SEARCH",
    "IMAGE",
    "SCHEME",
    "SETTLE_SECONDS",
    "WAKE_POLL_SECONDS",
    "WAKE_SECONDS",
    "WEB_HOME",
    "WEB_SEARCH",
    "Spotify",
    "app_installed",
    "application_named",
]

# What the installed application answers to, and what the website answers to.
# The scheme is how the caller knows which of the two ways to open it:
# `spotify:` goes to the shell, `https:` to the browser (`media/player.py`).
SCHEME = "spotify"
APP_SEARCH = "spotify:search:{words}"
APP_HOME = "spotify:"
WEB_SEARCH = "https://open.spotify.com/search/{words}"
WEB_HOME = "https://open.spotify.com"

# What the application's process is called, from the Store and from
# Spotify's own installer alike. Where it lives differs (`WindowsApps` for
# the one, the user's profile for the other), which is why windows are
# matched by this name and never by a path (`window.py`).
IMAGE = "Spotify.exe"

# How long a freshly started application is given to show a window before
# the search is sent anyway. A warm machine shows one in a second or two; a
# cold Store package can take several. Measured 2026-09-14 on the owner's
# machine: the running application answers at once, so the wait only ever
# costs on a start.
WAKE_SECONDS = 10.0
WAKE_POLL_SECONDS = 0.2

# A moment between the window appearing and the search being sent. The
# window comes up before the application is listening for URIs, and one
# sent into that gap is the one that gets lost. A guess until measured; the
# owner's ear is the measurement.
SETTLE_SECONDS = 1.0

# `AssocQueryStringW`: the question Windows itself asks before it opens a
# link. `ASSOCF_IS_PROTOCOL` says the string is a scheme and not a file
# type; `ASSOCSTR_FRIENDLYAPPNAME` asks for the name of the application
# behind it, which only exists when there is one.
ASSOCF_IS_PROTOCOL = 0x00001000
ASSOCSTR_FRIENDLYAPPNAME = 4
S_OK = 0


def application_named(scheme: str) -> str | None:
    """The application Windows would open `scheme` links with, by name -
    `None` when nothing is registered for it.

    Measured 2026-09-14 on the owner's machine: the Store build of Spotify
    registers `spotify:` with a `URL Protocol` value and a packaged ProgId,
    and **no** `shell\\open\\command` key - the key the old check looked
    for, which is how a machine with Spotify installed was sent to the
    website. The association query answers for the Store build and for
    Spotify's own installer alike, and refuses, with `ERROR_NO_ASSOCIATION`,
    for a scheme nothing answers.
    """
    shlwapi: Any = ctypes.windll.shlwapi
    query = shlwapi.AssocQueryStringW
    query.argtypes = [
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    query.restype = ctypes.c_long
    size = wintypes.DWORD(0)
    # First for the length, then for the text: the documented two-step.
    if query(ASSOCF_IS_PROTOCOL, ASSOCSTR_FRIENDLYAPPNAME, scheme, "open", None, size) < 0:
        return None
    if size.value == 0:
        return None
    buffer = ctypes.create_unicode_buffer(size.value)
    if query(ASSOCF_IS_PROTOCOL, ASSOCSTR_FRIENDLYAPPNAME, scheme, "open", buffer, size) != S_OK:
        return None
    name = str(buffer.value).strip()
    return name or None


def app_installed(named: Callable[[str], str | None] = application_named) -> bool:
    """Whether a `spotify:` URI would open an application on this machine."""
    name = named(SCHEME)
    if name is None:
        # Nothing registered, or a shell that will not answer. Either way the
        # website is the answer, and neither is worth a warning.
        logger.debug("no application is registered for {}: URIs", SCHEME)
        return False
    return True


# `image name -> the visible windows of the program with that name`.
WindowsNamed = Callable[[str], set[int]]


class Spotify:
    """Where a Spotify request goes: the installed app, or the website."""

    def __init__(
        self,
        *,
        installed: Callable[[], bool] = app_installed,
        windows: WindowsNamed | None = None,
        wake_seconds: float = WAKE_SECONDS,
        poll_seconds: float = WAKE_POLL_SECONDS,
        settle_seconds: float = SETTLE_SECONDS,
    ) -> None:
        # Asked every time rather than once at startup: the user may install
        # Spotify while the assistant is running, and the query is
        # microseconds.
        self._installed = installed
        self._windows: WindowsNamed = (
            windows if windows is not None else Win32Desktop().windows_named
        )
        self._wake_seconds = wake_seconds
        self._poll_seconds = poll_seconds
        self._settle_seconds = settle_seconds

    async def open(self, uri: str) -> bool:
        """Hands `uri` to the application, starting it first when it has to.

        A `spotify:` URI given to Windows starts the application when it is
        not running - and a Store application that is still starting drops
        the URI it was started with: the window comes, the search does not.
        So an application without a window is started with the bare
        `spotify:` first, the window is waited for, and only then is the
        search sent. A running application is handed the search at once.
        Opening the application itself (`APP_HOME`) is never waited on:
        starting it *is* the request.

        `True` once the shell has taken the URI; an `OSError` - nothing
        registered after all - is the caller's to catch, as before.
        """
        if uri != APP_HOME and not await self._running():
            await shell.open_target(APP_HOME)
            if await self._appeared():
                await asyncio.sleep(self._settle_seconds)
            else:
                logger.debug("no {} window appeared within {} s", IMAGE, self._wake_seconds)
        await shell.open_target(uri)
        return True

    async def _running(self) -> bool:
        # Off the loop: enumerating every window and asking each process
        # for its image is milliseconds, and rule 4 of section 3.1 does not
        # make exceptions for milliseconds that add up.
        return bool(await asyncio.to_thread(self._windows, IMAGE))

    async def _appeared(self) -> bool:
        """Whether the application showed a window within `wake_seconds`."""
        deadline = time.monotonic() + self._wake_seconds
        while True:
            if await self._running():
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(self._poll_seconds)

    def search(self, words: str) -> str:
        """Where the search for `words` should be opened.

        The words are percent-encoded, which is what keeps a title with a
        space, an apostrophe or a Turkish letter in it from becoming a
        different search - or, in the URI, from ending it early.
        """
        wanted = quote(words.strip(), safe="")
        return (APP_SEARCH if self._installed() else WEB_SEARCH).format(words=wanted)

    def exact(self, isrc: str) -> str:
        """Where the one recording with `isrc` should be opened.

        A search like any other to Spotify, with the filter its own search
        box understands; the colon is encoded with the rest so that the URI's
        own colons stay the only ones the application has to parse.
        """
        return self.search(f"isrc:{isrc}")

    def home(self) -> str:
        """Where "open Spotify" should go."""
        return APP_HOME if self._installed() else WEB_HOME
