"""`messaging/whatsapp.py` (15 Sep 2026): the official app, its own link, its own Send button.

Nothing here opens WhatsApp, presses a key or reads the registry. The
screen is a fake that answers what the test scripted and records what was
clicked; `shell.launch` is replaced and records what Windows would have
been handed, in order. Every wait is zero seconds.

The fake is a small WhatsApp: the link fills its chat box (on top of
whatever draft was there, as the real one does), its Send button empties
it when it works, the box can be made unreadable, and the window can be
minimised. Send is pressed only once the box holds exactly the message -
measured on the owner's machine, 2026-09-20/21: after a restore from the
tray the page inside WhatsApp takes 7-10 s to apply the link, and a key on
a timer landed on an empty box, with the text arriving afterwards.

Since 2026-09-26 there is no Enter and no foreground rule. WhatsApp
2.2637.100.0 (installed by the Store that morning) keeps the keyboard in
its WinUI shell rather than in the page: with WhatsApp in front and the box
holding exactly the message, Enter reached nothing, and not even after UI
Automation had put the focus in the box. The page's own Send button,
pressed through UI Automation, sent at once - from the tray, and with
WhatsApp under every other window. No keystroke goes anywhere now, so none
can land in the user's editor.
"""

from __future__ import annotations

import asyncio
import threading
from urllib.parse import parse_qs, urlsplit

import pytest

from allie import shell
from allie.messaging.whatsapp import (
    APP_HOME,
    IMAGE,
    IMAGES,
    SCHEME,
    WhatsApp,
    send_link,
)

PHONE = "905320000000"


class FakeScreen:
    """Windows by image, a chat box the link fills, and a Send button that
    empties it - in a window that may be minimised."""

    def __init__(self, *, windows: set[int] | None = None) -> None:
        self.windows: set[int] = set(windows or ())
        # What the process owning `windows` is called: `WhatsApp.exe` unless
        # a test says it is the Store's newer name.
        self.image = IMAGE
        self.threads: list[threading.Thread] = []
        # How many `windows_named` calls until a window appears, when the
        # test wants one to come late.
        self.appears_after: int | None = None
        self.polls = 0
        # The chat box: what it holds, what the link will add to it after
        # how many reads (the page applying the link late), and whether it
        # can be read at all.
        self.box = ""
        self.search = ""
        self.pending: str | None = None
        self.fills_after = 0
        self.reads = 0
        self.readable = True
        # The Send button: whether the page has one, whether it empties the
        # box, and which windows it was pressed in.
        self.has_send = True
        self.send_works = True
        self.clicked: list[int] = []
        # A minimised window's page applies nothing until it is shown
        # (measured 2026-09-26); `show` is what brings it back.
        self.minimised = False
        self.shown: list[int] = []
        self.launched: list[str] = []

    def launch(self, target: str) -> None:
        """`shell.launch`, as WhatsApp would take it: the link's text goes
        into the box, on top of whatever was there."""
        self.launched.append(target)
        query = parse_qs(urlsplit(target).query)
        if "text" in query:
            self.pending = query["text"][0]
            self.reads = 0

    def text_boxes(self, window: int) -> list[str] | None:
        self.threads.append(threading.current_thread())
        if not self.readable:
            return None
        self.reads += 1
        if self.pending is not None and self.reads > self.fills_after and not self.minimised:
            self.box += self.pending
            self.pending = None
        # The real box reads as one newline when empty.
        return [self.search, self.box or "\n"]

    def windows_named(self, image: str) -> set[int]:
        self.threads.append(threading.current_thread())
        self.polls += 1
        if self.appears_after is not None and self.polls > self.appears_after:
            self.windows.add(7)
        # The windows belong to one process, called `self.image`.
        return set(self.windows) if image.casefold() == self.image.casefold() else set()

    def show(self, window: int) -> None:
        self.threads.append(threading.current_thread())
        self.shown.append(window)
        self.minimised = False

    def click_send(self, window: int, text: str) -> bool | None:
        self.threads.append(threading.current_thread())
        if not self.readable:
            return None
        if not self.has_send or self.box.strip() != text.strip():
            return False
        self.clicked.append(window)
        if self.send_works:
            self.box = ""
        return True


@pytest.fixture
def screen(monkeypatch: pytest.MonkeyPatch) -> FakeScreen:
    """A screen whose WhatsApp is running; `shell.launch` is its own, so
    that the link fills its box."""
    fake = FakeScreen(windows={1})
    monkeypatch.setattr(shell, "launch", fake.launch)
    return fake


@pytest.fixture
def launched(screen: FakeScreen) -> list[str]:
    """What Windows was handed, in order."""
    return screen.launched


def whatsapp(screen: FakeScreen, *, installed: bool = True, **seconds: float) -> WhatsApp:
    waits = dict.fromkeys(
        ("wake_seconds", "settle_seconds", "type_seconds", "sent_seconds", "poll_seconds"),
        0.0,
    )
    waits.update(seconds)
    return WhatsApp(
        screen=screen,
        named=lambda scheme: "WhatsApp" if installed and scheme == SCHEME else None,
        **waits,
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


async def test_not_installed_opens_nothing(screen: FakeScreen, launched: list[str]) -> None:
    screen.windows.clear()

    assert await whatsapp(screen, installed=False).send(PHONE, "hi") == "not_installed"
    assert launched == []
    assert screen.clicked == []


def test_installed_is_asked_of_windows_by_the_scheme() -> None:
    asked: list[str] = []

    def named(scheme: str) -> str | None:
        asked.append(scheme)
        return None

    assert WhatsApp(screen=FakeScreen(), named=named).installed() is False
    assert asked == ["whatsapp"]


async def test_a_running_app_is_handed_the_link_at_once(
    screen: FakeScreen, launched: list[str]
) -> None:
    outcome = await whatsapp(screen).send(PHONE, "hi")

    assert outcome == "sent"
    assert launched == [send_link(PHONE, "hi")]
    assert screen.clicked == [1]
    assert screen.box == ""


async def test_an_app_without_a_window_is_woken_first_and_the_link_sent_after_it_appears(
    screen: FakeScreen, launched: list[str]
) -> None:
    """A Store app that is still starting drops the link it was started with
    (measured with Spotify, 2026-09-14): the bare scheme first, then the
    link. Closed into the tray is the same case: the window is hidden, and
    the bare scheme shows it (0.3 s, measured 2026-09-26)."""
    screen.windows.clear()
    screen.appears_after = 3
    patient = whatsapp(screen, wake_seconds=1.0)

    outcome = await patient.send(PHONE, "hi")

    assert outcome == "sent"
    assert launched == [APP_HOME, send_link(PHONE, "hi")]
    assert screen.polls > 3


async def test_no_window_within_the_wait_is_no_window_and_no_link(
    screen: FakeScreen, launched: list[str]
) -> None:
    screen.windows.clear()

    outcome = await whatsapp(screen).send(PHONE, "hi")

    assert outcome == "no_window"
    assert launched == [APP_HOME]
    assert screen.clicked == []


# --------------------------------------------------------------------------
# WhatsApp's own Send button (2026-09-26)
# --------------------------------------------------------------------------


async def test_the_message_goes_by_whatsapps_own_button_not_a_key() -> None:
    """The screen this module is given has no way to press a key at all:
    what sends is the page's Send button, in the window that holds the
    message - wherever the user's focus is."""
    assert not hasattr(FakeScreen, "press")
    assert not hasattr(FakeScreen, "foreground_image")


async def test_a_minimised_window_is_shown_before_the_text_is_waited_for(
    screen: FakeScreen,
) -> None:
    """Measured 2026-09-26: in a minimised window the page applied neither
    the link nor the click until it was shown again."""
    screen.minimised = True

    outcome = await whatsapp(screen).send(PHONE, "hi")

    assert outcome == "sent"
    assert screen.shown == [1]
    assert screen.clicked == [1]


async def test_a_page_without_a_send_button_is_placed_and_nothing_is_pressed(
    screen: FakeScreen,
) -> None:
    """A WhatsApp that names its button otherwise, or changed its page: the
    text is in the chat, waiting, and the user is told so."""
    screen.has_send = False

    outcome = await whatsapp(screen).send(PHONE, "hi")

    assert outcome == "placed"
    assert screen.clicked == []
    assert screen.box == "hi"


# --------------------------------------------------------------------------
# The chat box: Send only once it holds exactly the message (2026-09-21)
# --------------------------------------------------------------------------


async def test_send_waits_for_the_message_to_show_in_the_box(screen: FakeScreen) -> None:
    """The page applies the link late (7-10 s after a restore from the
    tray, measured 2026-09-21): the button waits for the text, not a timer."""
    screen.fills_after = 4
    patient = whatsapp(screen, type_seconds=1.0)

    outcome = await patient.send(PHONE, "hi")

    assert outcome == "sent"
    assert screen.reads > 4
    assert screen.clicked == [1]


async def test_a_message_that_never_shows_is_unseen_and_nothing_is_pressed(
    screen: FakeScreen,
) -> None:
    screen.fills_after = 10_000

    outcome = await whatsapp(screen).send(PHONE, "hi")

    assert outcome == "unseen"
    assert screen.clicked == []


async def test_a_draft_already_in_the_box_keeps_send_and_is_placed(screen: FakeScreen) -> None:
    """The link adds the text to a draft rather than replacing it (measured
    2026-09-21: "deneme 3 (...)deneme 4 (...)"). What would go is not what
    the user confirmed, so nothing is pressed and the user looks."""
    screen.box = "an older draft"

    outcome = await whatsapp(screen).send(PHONE, "hi")

    assert outcome == "placed"
    assert screen.clicked == []
    assert screen.box == "an older drafthi"


async def test_the_box_is_compared_without_the_newline_whatsapp_keeps_in_it(
    screen: FakeScreen, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty box reads as one newline, and a filled one may keep it."""
    screen.box = "hi\n"
    monkeypatch.setattr(shell, "launch", screen.launched.append)

    assert await whatsapp(screen).send(PHONE, " hi ") == "sent"


async def test_a_send_that_did_not_empty_the_box_is_placed(screen: FakeScreen) -> None:
    """The button pressed and the text still there: the user is told so."""
    screen.send_works = False

    outcome = await whatsapp(screen).send(PHONE, "hi")

    assert outcome == "placed"
    assert screen.clicked == [1]


async def test_a_box_that_cannot_be_read_after_send_is_pressed_unverified(
    screen: FakeScreen,
) -> None:
    """The page went away between the button and the look: it was pressed
    and that is all that is known."""
    clicked = screen.click_send

    def click_send(window: int, text: str) -> bool | None:
        answer = clicked(window, text)
        screen.readable = False
        return answer

    screen.click_send = click_send  # type: ignore[method-assign]

    assert await whatsapp(screen).send(PHONE, "hi") == "pressed"


async def test_a_box_that_cannot_be_read_at_all_is_unseen(screen: FakeScreen) -> None:
    """No page to read - WhatsApp changed, or UI Automation is silent: a
    press on a timer is what failed on 2026-09-20, so nothing is pressed."""
    screen.readable = False

    assert await whatsapp(screen).send(PHONE, "hi") == "unseen"
    assert screen.clicked == []


# --------------------------------------------------------------------------
# The application's two names
# --------------------------------------------------------------------------


def test_the_store_application_of_december_2025_is_known_by_its_newer_name() -> None:
    """Measured on the owner's machine, 2026-09-19: the Store's WhatsApp
    runs as `WhatsApp.Root.exe`, and a send that waited for `WhatsApp.exe`
    saw no window in 10 s while the window was open on the screen."""
    assert IMAGES == ("WhatsApp.exe", "WhatsApp.Root.exe")
    assert IMAGES[0] == IMAGE


async def test_a_running_app_under_the_newer_name_is_handed_the_link_at_once(
    screen: FakeScreen, launched: list[str]
) -> None:
    screen.image = "WhatsApp.Root.exe"

    outcome = await whatsapp(screen).send(PHONE, "hi")

    assert outcome == "sent"
    assert launched == [send_link(PHONE, "hi")]
    assert screen.clicked == [1]


# --------------------------------------------------------------------------
# Where and how it runs
# --------------------------------------------------------------------------


async def test_two_sends_do_not_press_send_in_each_others_chat(
    screen: FakeScreen, launched: list[str]
) -> None:
    """One at a time: the second waits for the first to have pressed."""
    app = whatsapp(screen)
    order: list[str] = []

    async def one(text: str) -> None:
        outcome = await app.send(PHONE, text)
        order.append(f"{text}:{outcome}")

    await asyncio.gather(one("first"), one("second"))

    assert launched == [send_link(PHONE, "first"), send_link(PHONE, "second")]
    assert order == ["first:sent", "second:sent"]


async def test_the_screen_is_never_touched_on_the_event_loop(screen: FakeScreen) -> None:
    """Section 3.1 rule 4: enumerating windows, reading the box and pressing
    the button are Win32 calls, and they run on a thread like the media
    window's."""
    await whatsapp(screen).send(PHONE, "hi")

    assert screen.threads and all(t is not threading.main_thread() for t in screen.threads)
