"""WhatsApp, through the official application and nothing else (spec section 5,
design.md section 12 decision 28's neighbour, decision W1 of 2026-09-14).

**What this is.** WhatsApp's own "click to chat" link -
`whatsapp://send?phone=<digits>&text=<encoded>` - opens the chat with the
text already in the composer; the number need not be a saved contact. One
Enter sends it. That is the whole mechanism, and it is the same one a
person uses: the only thing that talks to WhatsApp's servers is WhatsApp,
and a keystroke from `keybd_event` is indistinguishable from the user's.
The routes not taken, and why, are in the spec: an unofficial protocol
client gets the number banned (Meta's 2025-26 waves reached three-year-old
accounts; Hermes Agent's own docs say "do not connect your account"), the
Business API is for businesses, and a browser driven by this program would
be a stranger to the user's session.

**The one rule that matters.** Enter is pressed only when the foreground
window belongs to `WhatsApp.exe`, checked *immediately before* the
keystroke - not a second and a half earlier, when the link was sent, because
focus can move in a second and a half. An Enter that lands in the user's
editor is not an acceptable failure; a message left waiting in the chat is,
and the outcome says so (`placed`) so that the tool can tell the model and
the model can tell the user to press it.

**Waking the application first.** The Windows WhatsApp of December 2025 is
a WebView2 host around `web.whatsapp.com`: a cold start is 5-10 s, and a
Store application that is still starting drops the link it was started
with (measured with Spotify on 2026-09-14). So an application without a
window is started with the bare `whatsapp:` first, the window is waited
for, and the link is sent after a moment for the page inside to load. The
four durations below are guesses until the owner has measured them at home
(spec section 9); the tests set them to zero.

Known limits, on purpose: WhatsApp's "Enter is send" setting off means Enter
adds a line and nothing goes (the tool's answer says the text is in the chat
either way); a WhatsApp not linked to the phone shows a QR screen and Enter
does nothing; the WhatsApp window comes to the front, by design - the user
sees what was sent. Everything that touches Win32 runs off the event loop,
like the media window (design.md section 3.1 rule 4).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import quote

from loguru import logger

from assistant import shell
from assistant.media.spotify import application_named
from assistant.media.window import Win32Desktop

__all__ = [
    "APP_HOME",
    "COMPOSE_SECONDS",
    "FRONT_SECONDS",
    "IMAGE",
    "MAX_TEXT_CHARS",
    "POLL_SECONDS",
    "SCHEME",
    "SETTLE_SECONDS",
    "VK_RETURN",
    "WAKE_SECONDS",
    "Outcome",
    "Screen",
    "WhatsApp",
    "send_link",
]

SCHEME = "whatsapp"
APP_HOME = "whatsapp:"
SEND_LINK = "whatsapp://send?phone={phone}&text={text}"

# What the application's process is called, from the Store.
IMAGE = "WhatsApp.exe"

# A cold WebView2 host showing its window; then web.whatsapp.com loading
# inside it before it will take a link; then the send link bringing the
# window to the front; then the chat opening and the text landing in the
# composer. All four to be measured at the owner's (spec section 9).
WAKE_SECONDS = 10.0
SETTLE_SECONDS = 3.0
FRONT_SECONDS = 5.0
COMPOSE_SECONDS = 1.5
POLL_SECONDS = 0.1

VK_RETURN = 0x0D

# Longer than this and the confirm question, which reads the text out loud
# before anything is sent, would take a minute to ask. A constant rather
# than a setting (spec section 11.3).
MAX_TEXT_CHARS = 1000

# What `send` came to. `pressed`: the link was placed and Enter went to
# WhatsApp - WhatsApp itself gives no confirmation. `placed`: the text is in
# the chat and nothing was pressed, because WhatsApp was not in front when
# it mattered. `no_window`: the application would not show a window in
# time. `not_installed`: nothing on this machine answers `whatsapp:` links.
Outcome = Literal["pressed", "placed", "no_window", "not_installed"]


def send_link(phone: str, text: str) -> str:
    """WhatsApp's click-to-chat link for `phone` (digits only) and `text`.

    `quote` with nothing safe: an `&` in the text would start a second
    parameter and a `#` would end the address, and a space is `%20`, which
    WhatsApp reads and `+` is not guaranteed to be.
    """
    return SEND_LINK.format(phone=phone, text=quote(text.strip(), safe=""))


class Screen(Protocol):
    """What this needs from Windows. `Win32Desktop` is the real one; the
    test's fake records."""

    def windows_named(self, image: str) -> set[int]: ...

    def foreground_image(self) -> str | None: ...

    def press(self, code: int) -> None: ...


class WhatsApp:
    """The installed application, driven by its link and one key."""

    def __init__(
        self,
        *,
        screen: Screen | None = None,
        named: Callable[[str], str | None] = application_named,
        wake_seconds: float = WAKE_SECONDS,
        settle_seconds: float = SETTLE_SECONDS,
        front_seconds: float = FRONT_SECONDS,
        compose_seconds: float = COMPOSE_SECONDS,
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        self._screen: Screen = screen if screen is not None else Win32Desktop()
        # Asked at every send rather than once: the user may install
        # WhatsApp while the assistant runs, and the query is microseconds.
        self._named = named
        self._wake_seconds = wake_seconds
        self._settle_seconds = settle_seconds
        self._front_seconds = front_seconds
        self._compose_seconds = compose_seconds
        self._poll_seconds = poll_seconds
        # One send at a time: two that overlapped would press Enter in
        # each other's chat.
        self._turn = asyncio.Lock()

    def installed(self) -> bool:
        """Whether a `whatsapp:` link would open an application here - the
        question Windows itself asks before opening one (`media/spotify.py`)."""
        return self._named(SCHEME) is not None

    async def send(self, phone: str, text: str) -> Outcome:
        """Places `text` in the chat with `phone` and presses Enter when it
        is safe to. See the module docstring for the one rule."""
        async with self._turn:
            return await self._send(phone, text)

    async def _send(self, phone: str, text: str) -> Outcome:
        if not self.installed():
            return "not_installed"

        if not await self._running():
            await shell.open_target(APP_HOME)
            if not await self._appeared():
                logger.debug("no {} window appeared within {} s", IMAGE, self._wake_seconds)
                return "no_window"
            await asyncio.sleep(self._settle_seconds)

        await shell.open_target(send_link(phone, text))

        if not await self._came_to_front():
            logger.debug("{} did not come to the front within {} s", IMAGE, self._front_seconds)
            return "placed"
        await asyncio.sleep(self._compose_seconds)
        # Again, right before the key: focus can move in a second and a half.
        if not await self._in_front():
            logger.debug("{} lost the foreground before Enter; nothing pressed", IMAGE)
            return "placed"
        await asyncio.to_thread(self._screen.press, VK_RETURN)
        return "pressed"

    async def _running(self) -> bool:
        return bool(await asyncio.to_thread(self._screen.windows_named, IMAGE))

    async def _appeared(self) -> bool:
        return await self._within(self._wake_seconds, self._running)

    async def _came_to_front(self) -> bool:
        return await self._within(self._front_seconds, self._in_front)

    async def _in_front(self) -> bool:
        image = await asyncio.to_thread(self._screen.foreground_image)
        return image is not None and Path(image).name.casefold() == IMAGE.casefold()

    async def _within(self, seconds: float, condition: Callable[[], Awaitable[bool]]) -> bool:
        """Whether `condition` comes true within `seconds`, asked every poll."""
        deadline = time.monotonic() + seconds
        while True:
            if await condition():
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(self._poll_seconds)
