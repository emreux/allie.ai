"""The tray icon (`ui/tray.py`, 4.3): a surface, not a second brain.

What is claimed: the icon and its tooltip follow the state and the mode the
state machine reports; the menu's switch line and "quit" reach the event
loop and nothing else; the folder line opens the settings folder without
going near the loop; and every word is the pack's. The real `pystray`
icon is built once, without being shown, to prove the adapter hands it
labels, actions and a greyed-out state line.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from assistant import locales
from assistant.app import State
from assistant.ui import status, tray
from assistant.ui.tray import ICON_SIZE, TEXT, MenuEntry, Tray, draw_icon, system_icon

CENTRE = (ICON_SIZE // 2, ICON_SIZE // 2)
FOLDER = Path(r"C:\Users\somebody\AppData\Roaming\assistant")


class FakeIcon:
    """What `pystray.Icon` is used for, recorded."""

    def __init__(self, name: str, image: Image.Image, title: str, entries: list[MenuEntry]) -> None:
        self.name = name
        self.icon: Any = image
        self.title = title
        self.entries = entries
        self.running = False
        self.menu_updates = 0

    def run_detached(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def update_menu(self) -> None:
        self.menu_updates += 1


class Built:
    """A tray over a fake icon, with what its clicks did written down."""

    def __init__(self, loop: asyncio.AbstractEventLoop, code: str = "tr") -> None:
        self.toggles = 0
        self.quits = 0
        # Which thread each callback ran on: the claim is that it is the
        # loop's, never the tray's.
        self.threads: list[int] = []
        self.opened: list[Path] = []
        icons: list[FakeIcon] = []

        def icon(name: str, image: Image.Image, title: str, entries: list[MenuEntry]) -> FakeIcon:
            icons.append(FakeIcon(name, image, title, entries))
            return icons[-1]

        def toggled() -> None:
            self.toggles += 1
            self.threads.append(threading.get_ident())

        def quit_() -> None:
            self.quits += 1
            self.threads.append(threading.get_ident())

        self.tray = Tray(
            locales.load(code),
            loop=loop,
            on_toggle=toggled,
            on_quit=quit_,
            settings_folder=FOLDER,
            icon=icon,
            open=self.opened.append,
        )
        [self.icon] = icons

    @property
    def labels(self) -> list[str]:
        return [entry.label() for entry in self.icon.entries]


@pytest.fixture
def built() -> Iterator[Built]:
    """A tray on a loop of its own, so that a click can be posted to it."""
    loop = asyncio.new_event_loop()
    try:
        yield Built(loop)
    finally:
        loop.close()


def turkish(key: str, table: dict[str, str] = TEXT) -> str:
    return locales.load("tr").say(key, table[key])


def label_of(state: State) -> str:
    return turkish(status.label_key(state), status.TEXT)


# --------------------------------------------------------------------------
# What it shows
# --------------------------------------------------------------------------


def test_it_goes_up_idle_with_the_four_lines_and_comes_down_when_told(built: Built) -> None:
    assert built.icon.title == label_of(State.IDLE)
    assert built.icon.icon is not None
    assert len(built.icon.entries) == 4
    assert built.icon.entries[0].action is None
    assert all(entry.action is not None for entry in built.icon.entries[1:])
    assert built.icon.running is False

    built.tray.start()
    assert built.icon.running is True
    built.tray.stop()
    assert built.icon.running is False


def test_the_state_is_the_tooltip_the_first_line_and_the_colour(built: Built) -> None:
    idle = built.icon.icon

    built.tray.state(State.LISTENING)

    assert built.icon.title == label_of(State.LISTENING)
    assert built.labels[0] == label_of(State.LISTENING)
    assert built.icon.icon is not idle
    assert built.icon.icon.getpixel(CENTRE) != idle.getpixel(CENTRE)


@pytest.mark.parametrize("state", list(State))
def test_every_state_has_its_own_label_and_a_picture(built: Built, state: State) -> None:
    built.tray.state(state)

    assert built.icon.title == label_of(state)
    assert built.icon.icon.size == (ICON_SIZE, ICON_SIZE)


def test_an_idle_microphone_that_is_off_says_so_and_offers_to_start(built: Built) -> None:
    """The one thing "ready" would misdescribe. The disc becomes a ring,
    the switch line turns round, and the menu is told to redraw."""
    built.tray.hands_free(False)

    assert built.icon.title == turkish("tray_not_listening")
    assert built.labels[1] == turkish("tray_start_listening")
    assert built.icon.icon.getpixel(CENTRE)[3] == 0
    assert built.icon.menu_updates == 1

    built.tray.hands_free(True)

    assert built.icon.title == label_of(State.IDLE)
    assert built.labels[1] == turkish("tray_stop_listening")
    assert built.icon.icon.getpixel(CENTRE)[3] == 255


def test_a_busy_state_keeps_its_label_even_with_the_microphone_off(built: Built) -> None:
    """Off in the middle of an answer: the answer is what is happening."""
    built.tray.hands_free(False)
    built.tray.state(State.SPEAKING)

    assert built.icon.title == label_of(State.SPEAKING)
    assert built.icon.icon.getpixel(CENTRE)[3] == 0


def test_the_ring_and_the_disc_are_drawn_in_the_states_colour() -> None:
    disc = draw_icon(State.LISTENING, listening=True)
    ring = draw_icon(State.LISTENING, listening=False)

    assert disc.getpixel(CENTRE) == disc.getpixel((4, CENTRE[1]))
    assert ring.getpixel(CENTRE)[3] == 0
    assert ring.getpixel((4, CENTRE[1])) == disc.getpixel((4, CENTRE[1]))


# --------------------------------------------------------------------------
# What its clicks do
# --------------------------------------------------------------------------


async def test_the_switch_and_quit_reach_the_loop_from_the_trays_thread() -> None:
    """A click arrives on pystray's thread; what it asks for runs on the
    loop, and not before the loop gets to it."""
    built = Built(asyncio.get_running_loop())
    switch, quit_ = built.icon.entries[1].action, built.icon.entries[3].action
    assert switch is not None and quit_ is not None
    trays_thread: list[int] = []

    def clicked(action: Callable[[], None]) -> None:
        trays_thread.append(threading.get_ident())
        action()

    await asyncio.to_thread(clicked, switch)
    await asyncio.to_thread(clicked, quit_)
    await asyncio.sleep(0)

    assert (built.toggles, built.quits) == (1, 1)
    assert built.threads == [threading.get_ident()] * 2
    assert threading.get_ident() not in trays_thread


def test_the_folder_line_opens_the_settings_folder_without_the_loop(built: Built) -> None:
    """Explorer is handed the folder from the tray's own thread: nothing to
    post, and nothing the assistant needs to know about."""
    open_folder = built.icon.entries[2].action
    assert open_folder is not None

    open_folder()

    assert built.opened == [FOLDER]
    assert (built.toggles, built.quits) == (0, 0)


def test_the_folder_opens_the_way_a_double_click_would(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[str] = []
    monkeypatch.setattr(tray.shell, "launch", launched.append)

    tray.open_folder(FOLDER)

    assert launched == [str(FOLDER)]


# --------------------------------------------------------------------------
# Whose words
# --------------------------------------------------------------------------


def test_the_menu_speaks_the_pack(built: Built) -> None:
    assert built.labels[1:] == [
        turkish("tray_stop_listening"),
        turkish("tray_open_settings"),
        turkish("tray_quit"),
    ]
    assert built.labels[1:] != [
        TEXT["tray_stop_listening"],
        TEXT["tray_open_settings"],
        TEXT["tray_quit"],
    ]


def test_without_a_pack_the_menu_is_the_english_in_the_code() -> None:
    loop = asyncio.new_event_loop()
    try:
        english = Built(loop, code="en")
    finally:
        loop.close()

    assert english.labels[1:] == [
        TEXT["tray_stop_listening"],
        TEXT["tray_open_settings"],
        TEXT["tray_quit"],
    ]
    assert english.icon.title == status.TEXT["state_idle"]


# --------------------------------------------------------------------------
# The real icon, built and not shown
# --------------------------------------------------------------------------


def test_pystray_is_handed_the_labels_the_actions_and_a_greyed_state_line() -> None:
    clicked: list[str] = []
    entries = [
        MenuEntry(lambda: "ready"),
        MenuEntry(lambda: "Stop listening", lambda: clicked.append("switch")),
    ]

    icon = system_icon("assistant", draw_icon(State.IDLE, listening=True), "ready", entries)

    items = list(icon.menu.items)  # type: ignore[attr-defined]
    assert [str(item.text) for item in items] == ["ready", "Stop listening"]
    assert [item.enabled for item in items] == [False, True]
    items[1](icon)
    assert clicked == ["switch"]
    assert icon.title == "ready"
