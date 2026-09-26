"""WhatsApp, through the official application and nothing else (spec section 5,
design.md section 12 decision 28's neighbour, decision W1 of 2026-09-14).

**What this is.** WhatsApp's own "click to chat" link -
`whatsapp://send?phone=<digits>&text=<encoded>` - opens the chat with the
text already in the composer; the number need not be a saved contact. The
page's own Send button sends it. That is the whole mechanism, and it is
the same one a person uses: the only thing that talks to WhatsApp's
servers is WhatsApp, and a button pressed through UI Automation is the
button the user would have clicked.
The routes not taken, and why, are in the spec: an unofficial protocol
client gets the number banned (Meta's 2025-26 waves reached three-year-old
accounts; Hermes Agent's own docs say "do not connect your account"), the
Business API is for businesses, and a browser driven by this program would
be a stranger to the user's session.

**The button, not a key (2026-09-26).** Until then one Enter sent it, and
only while the foreground window was WhatsApp's, so that an Enter could
never land in the user's editor. WhatsApp 2.2637.100.0, installed by the
Store that morning, keeps the keyboard in its WinUI shell
(`DesktopChildSiteBridge`) rather than in the page: measured on the owner's
machine, WhatsApp in front and the box holding exactly the message, Enter
reached nothing - and still nothing once UI Automation had moved the focus
into the box. The Send button, pressed through UI Automation
(`Win32Desktop.click_send`), sent at once: from the tray, and with WhatsApp
under every other window. No keystroke goes anywhere now, so the rule about
the foreground went with it. A minimised window is shown first: its page
applied neither the link nor the click until it was.

**The rule that stays (2026-09-21).** Send is pressed only once the chat
box holds *exactly* the message, read through UI Automation
(`Win32Desktop.text_boxes`), and "sent" is said only once the box has
emptied after it. The window being there is not the text being in the
box: measured on the owner's machine, after a restore from the tray the
page inside WhatsApp took 7.5 and 10.5 s to apply the link, while a timer
pressed 1.5 s after the window came forward - on an empty box, with the
text arriving afterwards (the send of 2026-09-20 that "typed but did not
send"). And the link *adds* to a draft rather than replacing it ("deneme 3
(...)deneme 4 (...)"), so a box that holds more than the message is left
alone: what would go is not what the user confirmed. A message left
waiting in the chat is an outcome of its own (`placed`), so that the tool
can tell the model and the model can tell the user to press it.

**Waking the application first.** The Windows WhatsApp of December 2025 is
a WebView2 host around `web.whatsapp.com`: a cold start is 5-10 s, and a
Store application that is still starting drops the link it was started
with (measured with Spotify on 2026-09-14). So an application without a
window is started with the bare `whatsapp:` first, the window is waited
for, and the link is sent after a moment for the page inside to load. The
durations below were guesses until measured on the owner's machine
(2026-09-21, 2026-09-26); the tests set them to zero.

Known limits, on purpose: the button is found by its name, as WhatsApp's
own page gives it (`SEND_NAMES` in `media/window.py`) - a WhatsApp in a
language not listed there leaves the text waiting and says so; a WhatsApp
not linked to the phone shows a QR screen and no box ever holds the text.
Everything that touches Win32 runs off the event loop, like the media
window (design.md section 3.1 rule 4).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Literal, Protocol
from urllib.parse import quote

from loguru import logger

from allie import shell
from allie.media.spotify import application_named
from allie.media.window import Win32Desktop

__all__ = [
    "APP_HOME",
    "IMAGE",
    "IMAGES",
    "MAX_TEXT_CHARS",
    "POLL_SECONDS",
    "SCHEME",
    "SENT_SECONDS",
    "SETTLE_SECONDS",
    "TYPE_SECONDS",
    "WAKE_SECONDS",
    "Holding",
    "Outcome",
    "Screen",
    "WhatsApp",
    "send_link",
]

SCHEME = "whatsapp"
APP_HOME = "whatsapp:"
SEND_LINK = "whatsapp://send?phone={phone}&text={text}"

# What the application's process is called. The Store's WhatsApp of
# December 2025 runs as `WhatsApp.Root.exe` (measured on the owner's
# machine, 2026-09-19: a send that waited for `WhatsApp.exe` saw no window
# in 10 s while the window was open on the screen); older builds, and the
# one the docstrings name, are `WhatsApp.exe`. Either is the application.
IMAGES = ("WhatsApp.exe", "WhatsApp.Root.exe")
IMAGE = IMAGES[0]

# A cold WebView2 host showing its window (0.3 s from the tray, measured
# 2026-09-26); then web.whatsapp.com loading inside it before it will take
# a link; then the page applying the link - the chat open with the text in
# its box (7.5 and 10.5 s after a restore from the tray, measured
# 2026-09-21; 0.5 s on 2026-09-26; a cold start is longer); then the box
# emptying after Send (0.1 s when it goes, measured 2026-09-26).
WAKE_SECONDS = 10.0
SETTLE_SECONDS = 3.0
TYPE_SECONDS = 20.0
SENT_SECONDS = 3.0
POLL_SECONDS = 0.1

# Longer than this and the confirm question, which reads the text out loud
# before anything is sent, would take a minute to ask. A constant rather
# than a setting (spec section 11.3).
MAX_TEXT_CHARS = 1000

# What `send` came to. `sent`: Send was pressed and the box emptied - the
# message left. `pressed`: Send was pressed and the box could not be read
# afterwards, so that is all that is known. `placed`: the text is in the
# chat and nothing went, because the box held other text beside the
# message, the page had no Send button to press, or pressing it did not
# empty the box. `unseen`: the message never showed in the box, so
# nothing was pressed. `no_window`: the application would not show a
# window in time.
# `not_installed`: nothing on this machine answers `whatsapp:` links.
Outcome = Literal["sent", "pressed", "placed", "unseen", "no_window", "not_installed"]

# What the box holds, against the message: exactly it, it among other text
# (a draft the link added to), or not it.
Holding = Literal["exact", "more", "none"]


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

    def text_boxes(self, window: int) -> list[str] | None: ...

    def show(self, window: int) -> None: ...

    def click_send(self, window: int, text: str) -> bool | None: ...


class WhatsApp:
    """The installed application, driven by its link and its Send button."""

    def __init__(
        self,
        *,
        screen: Screen | None = None,
        named: Callable[[str], str | None] = application_named,
        wake_seconds: float = WAKE_SECONDS,
        settle_seconds: float = SETTLE_SECONDS,
        type_seconds: float = TYPE_SECONDS,
        sent_seconds: float = SENT_SECONDS,
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        self._screen: Screen = screen if screen is not None else Win32Desktop()
        # Asked at every send rather than once: the user may install
        # WhatsApp while the assistant runs, and the query is microseconds.
        self._named = named
        self._wake_seconds = wake_seconds
        self._settle_seconds = settle_seconds
        self._type_seconds = type_seconds
        self._sent_seconds = sent_seconds
        self._poll_seconds = poll_seconds
        # One send at a time: two that overlapped would press Send in
        # each other's chat.
        self._turn = asyncio.Lock()

    def installed(self) -> bool:
        """Whether a `whatsapp:` link would open an application here - the
        question Windows itself asks before opening one (`media/spotify.py`)."""
        return self._named(SCHEME) is not None

    async def send(self, phone: str, text: str) -> Outcome:
        """Places `text` in the chat with `phone` and presses Send once the
        box holds exactly it. See the module docstring for why."""
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
        # A minimised window's page takes neither the link nor the click
        # until it is shown (measured 2026-09-26).
        await asyncio.to_thread(self._show)

        holding = await self._shown(text)
        if holding == "none":
            logger.info("whatsapp: the message did not show within {} s", self._type_seconds)
            return "unseen"
        if holding == "more":
            logger.info("whatsapp: the chat box holds other text too; nothing pressed")
            return "placed"
        if not await asyncio.to_thread(self._click_send, text):
            logger.info("whatsapp: no Send button beside the message; nothing pressed")
            return "placed"
        return await self._went(text)

    async def _shown(self, text: str) -> Holding:
        """Waits for the message to show in the chat box: exactly, or among
        other text. `none` after `type_seconds` of neither."""
        deadline = time.monotonic() + self._type_seconds
        while True:
            holding = await asyncio.to_thread(self._holding, text)
            if holding in ("exact", "more"):
                return holding
            if time.monotonic() >= deadline:
                return "none"
            await asyncio.sleep(self._poll_seconds)

    async def _went(self, text: str) -> Outcome:
        """After Send: `sent` once the box no longer holds the message,
        `placed` while it still does after `sent_seconds`, and `pressed`
        when the box cannot be read any more."""
        deadline = time.monotonic() + self._sent_seconds
        while True:
            holding = await asyncio.to_thread(self._holding, text)
            if holding is None:
                logger.info("whatsapp: the chat box could not be read after Send")
                return "pressed"
            if holding == "none":
                return "sent"
            if time.monotonic() >= deadline:
                logger.info(
                    "whatsapp: the chat box still holds the message {} s after Send",
                    self._sent_seconds,
                )
                return "placed"
            await asyncio.sleep(self._poll_seconds)

    def _holding(self, text: str) -> Holding | None:
        """What WhatsApp's text boxes hold against `text`, or `None` when no
        box can be read. Compared without the surrounding whitespace: an
        empty box reads as one newline."""
        wanted = text.strip()
        boxes: list[str] = []
        readable = False
        for image in IMAGES:
            for window in self._screen.windows_named(image):
                found = self._screen.text_boxes(window)
                if found is not None:
                    readable = True
                    boxes.extend(found)
        if not readable:
            return None
        if any(box.strip() == wanted for box in boxes):
            return "exact"
        if any(wanted in box for box in boxes):
            return "more"
        return "none"

    def _show(self) -> None:
        for image in IMAGES:
            for window in self._screen.windows_named(image):
                self._screen.show(window)

    def _click_send(self, text: str) -> bool:
        """Presses Send in the window whose box holds exactly `text`;
        whether one was pressed."""
        for image in IMAGES:
            for window in self._screen.windows_named(image):
                if self._screen.click_send(window, text):
                    return True
        return False

    async def _running(self) -> bool:
        return await asyncio.to_thread(self._has_window)

    def _has_window(self) -> bool:
        return any(self._screen.windows_named(image) for image in IMAGES)

    async def _appeared(self) -> bool:
        return await self._within(self._wake_seconds, self._running)

    async def _within(self, seconds: float, condition: Callable[[], Awaitable[bool]]) -> bool:
        """Whether `condition` comes true within `seconds`, asked every poll."""
        deadline = time.monotonic() + seconds
        while True:
            if await condition():
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(self._poll_seconds)
