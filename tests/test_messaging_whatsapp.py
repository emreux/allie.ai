"""`messaging/whatsapp.py` (15 Sep 2026): the official app, its own link, one Enter.

Nothing here opens WhatsApp, presses a key or reads the registry. The
screen is a fake that answers what the test scripted and records what was
pressed; `shell.launch` is replaced and records what Windows would have
been handed, in order. Every wait is zero seconds.

The one rule that matters is tested from both sides: Enter is pressed only
when the foreground window belongs to `WhatsApp.exe` at *both* checks, and
a message left waiting in the chat is the outcome otherwise - never a
keystroke into whatever the user was typing in.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence

import pytest

from assistant import shell
from assistant.messaging.whatsapp import (
    APP_HOME,
    IMAGE,
    SCHEME,
    VK_RETURN,
    WhatsApp,
    send_link,
)

PHONE = "905320000000"


class FakeScreen:
    """Windows by image, a scripted run of foreground answers, recorded presses."""

    def __init__(
        self, *, windows: set[int] | None = None, foreground: Sequence[str | None] = ()
    ) -> None:
        self.windows: set[int] = set(windows or ())
        # Consumed one per check; the last answer repeats.
        self.foreground: list[str | None] = list(foreground)
        self.pressed: list[int] = []
        self.threads: list[threading.Thread] = []
        # How many `windows_named` calls until a window appears, when the
        # test wants one to come late.
        self.appears_after: int | None = None
        self.polls = 0

    def windows_named(self, image: str) -> set[int]:
        self.threads.append(threading.current_thread())
        self.polls += 1
        if self.appears_after is not None and self.polls > self.appears_after:
            self.windows.add(7)
        return set(self.windows) if image.casefold() == IMAGE.casefold() else set()

    def foreground_image(self) -> str | None:
        self.threads.append(threading.current_thread())
        if len(self.foreground) > 1:
            return self.foreground.pop(0)
        return self.foreground[0] if self.foreground else None

    def press(self, code: int) -> None:
        self.threads.append(threading.current_thread())
        self.pressed.append(code)


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    monkeypatch.setattr(shell, "launch", seen.append)
    return seen


def whatsapp(screen: FakeScreen, *, installed: bool = True) -> WhatsApp:
    return WhatsApp(
        screen=screen,
        named=lambda scheme: "WhatsApp" if installed and scheme == SCHEME else None,
        wake_seconds=0.0,
        settle_seconds=0.0,
        front_seconds=0.0,
        compose_seconds=0.0,
        poll_seconds=0.0,
    )


# --------------------------------------------------------------------------
# The link
# --------------------------------------------------------------------------


def test_the_link_is_whatsapps_own_click_to_chat_with_the_text_fully_encoded() -> None:
    assert send_link(PHONE, "yarın geliyorum") == (
        "whatsapp://send?phone=905320000000&text=yar%C4%B1n%20geliyorum"
    )


def test_nothing_in_the_text_can_reach_the_link_unencoded() -> None:
    """`&` would start a second parameter and `#` would end the address."""
    link = send_link(PHONE, "a&b #1 = c/d?")

    assert link.endswith("&text=a%26b%20%231%20%3D%20c%2Fd%3F")
    assert link.count("&") == 1


# --------------------------------------------------------------------------
# The outcomes, in the order they are decided
# --------------------------------------------------------------------------


async def test_not_installed_opens_nothing(launched: list[str]) -> None:
    screen = FakeScreen()

    assert await whatsapp(screen, installed=False).send(PHONE, "hi") == "not_installed"
    assert launched == []
    assert screen.pressed == []


def test_installed_is_asked_of_windows_by_the_scheme() -> None:
    asked: list[str] = []

    def named(scheme: str) -> str | None:
        asked.append(scheme)
        return None

    assert WhatsApp(screen=FakeScreen(), named=named).installed() is False
    assert asked == ["whatsapp"]


async def test_a_running_app_is_handed_the_link_at_once(launched: list[str]) -> None:
    screen = FakeScreen(windows={1}, foreground=[IMAGE])

    outcome = await whatsapp(screen).send(PHONE, "hi")

    assert outcome == "pressed"
    assert launched == [send_link(PHONE, "hi")]
    assert screen.pressed == [VK_RETURN]


async def test_an_app_without_a_window_is_woken_first_and_the_link_sent_after_it_appears(
    launched: list[str],
) -> None:
    """A Store app that is still starting drops the link it was started with
    (measured with Spotify, 2026-09-14): the bare scheme first, then the link."""
    screen = FakeScreen(foreground=[IMAGE])
    screen.appears_after = 3
    patient = WhatsApp(
        screen=screen,
        named=lambda _: "WhatsApp",
        wake_seconds=1.0,
        settle_seconds=0.0,
        front_seconds=0.0,
        compose_seconds=0.0,
        poll_seconds=0.0,
    )

    outcome = await patient.send(PHONE, "hi")

    assert outcome == "pressed"
    assert launched == [APP_HOME, send_link(PHONE, "hi")]
    assert screen.polls > 3


async def test_no_window_within_the_wait_is_no_window_and_no_link(launched: list[str]) -> None:
    screen = FakeScreen(foreground=[IMAGE])

    outcome = await whatsapp(screen).send(PHONE, "hi")

    assert outcome == "no_window"
    assert launched == [APP_HOME]
    assert screen.pressed == []


async def test_a_foreground_that_never_becomes_whatsapp_is_placed_and_nothing_is_pressed(
    launched: list[str],
) -> None:
    """The user's editor stayed in front: the text is in the chat, waiting."""
    screen = FakeScreen(windows={1}, foreground=["Code.exe"])

    outcome = await whatsapp(screen).send(PHONE, "hi")

    assert outcome == "placed"
    assert launched == [send_link(PHONE, "hi")]
    assert screen.pressed == []


async def test_a_focus_that_moved_between_the_two_checks_is_placed_and_nothing_is_pressed(
    launched: list[str],
) -> None:
    """The one rule that matters: the foreground is checked again right
    before the key. WhatsApp at the first check, the user's editor at the
    second - an Enter into the editor is not an acceptable failure."""
    screen = FakeScreen(windows={1}, foreground=[IMAGE, "Code.exe"])

    outcome = await whatsapp(screen).send(PHONE, "hi")

    assert outcome == "placed"
    assert screen.pressed == []


async def test_whatsapp_at_both_checks_is_one_enter_and_pressed() -> None:
    screen = FakeScreen(windows={1}, foreground=[IMAGE, IMAGE])

    outcome = await whatsapp(screen).send(PHONE, "hi")

    assert outcome == "pressed"
    assert screen.pressed == [VK_RETURN]


async def test_the_image_is_compared_without_regard_to_case_or_path() -> None:
    screen = FakeScreen(windows={1}, foreground=["whatsapp.EXE"])

    assert await whatsapp(screen).send(PHONE, "hi") == "pressed"


async def test_a_foreground_that_comes_to_whatsapp_late_is_waited_for() -> None:
    """The link brings the window to the front; it takes a moment."""
    screen = FakeScreen(windows={1}, foreground=[None, "explorer.exe", IMAGE])
    late = WhatsApp(
        screen=screen,
        named=lambda _: "WhatsApp",
        front_seconds=1.0,
        compose_seconds=0.0,
        poll_seconds=0.0,
    )

    assert await late.send(PHONE, "hi") == "pressed"


# --------------------------------------------------------------------------
# Where and how it runs
# --------------------------------------------------------------------------


async def test_two_sends_do_not_press_enter_in_each_others_chat(launched: list[str]) -> None:
    """One at a time: the second waits for the first to have pressed."""
    screen = FakeScreen(windows={1}, foreground=[IMAGE])
    app = whatsapp(screen)
    order: list[str] = []

    async def one(text: str) -> None:
        outcome = await app.send(PHONE, text)
        order.append(f"{text}:{outcome}")

    await asyncio.gather(one("first"), one("second"))

    assert launched == [send_link(PHONE, "first"), send_link(PHONE, "second")]
    assert order == ["first:pressed", "second:pressed"]


async def test_the_screen_is_never_touched_on_the_event_loop() -> None:
    """Section 3.1 rule 4: enumerating windows and pressing keys are Win32
    calls, and they run on a thread like the media window's."""
    screen = FakeScreen(windows={1}, foreground=[IMAGE])

    await whatsapp(screen).send(PHONE, "hi")

    assert screen.threads and all(t is not threading.main_thread() for t in screen.threads)
