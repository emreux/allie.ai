"""The tray icon (design.md section 3.1, phase 4.3; 17 Sep 2026).

A third surface over the same state machine as the status line and the
window, and like them owning no behaviour of its own: what the icon shows is
what `on_state` and `on_mode` say, and what its menu does is what the key and
Ctrl+C already do. `run --tray` puts it up beside whichever of the two is
showing - the window by default, the terminal line under `--terminal` - and
that one keeps showing what was said.

**pystray runs on a thread of its own.** `run_detached` starts it and the
shell's messages arrive there. A click on the menu is one of them, and
everything a click asks of the assistant is posted to the event loop with
`call_soon_threadsafe` - the road the hotkey takes (`audio/capture.py`),
so that nothing about the assistant is touched from two threads. The
other direction, the loop telling the icon what to show, is a property
set on the icon; pystray hands it to the shell from whichever thread sets
it, and the shell's icon calls take a millisecond or two, which is inside
rule 4 of section 3.1.

**Drawn, not shipped.** The icon is a disc in the state's colour - the
reading the status line gives in words - filled while the microphone is
live and a ring while it is not. Each is drawn once with Pillow when the
tray is built, so that a change of state costs a lookup and not a picture.

**No sentence is written here.** The menu's labels come from the pack, and
the state's name is the status line's own label under the same key, so a
pack that translated the terminal has translated the tray. "Show the
window" appears only when `run` gave the tray a window to show (D20).

**The tooltip is the state, the session and the minutes** (plan.md section
4.2): a live model bills by the minute while a session is open, and the
icon is where that is read at a glance. The meter is the status line's
`SessionMinutes`, told the same thing the line is told.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageDraw

from assistant import shell
from assistant.app import State
from assistant.locales import Locale
from assistant.ui.status import TEXT as STATUS_TEXT
from assistant.ui.status import SessionMinutes, label_key

__all__ = ["ICON_SIZE", "TEXT", "MenuEntry", "Tray", "TrayIcon", "draw_icon", "system_icon"]

TEXT: dict[str, str] = {
    "tray_not_listening": "not listening",
    "tray_stop_listening": "Stop listening",
    "tray_start_listening": "Start listening",
    "tray_open_settings": "Open the settings folder",
    "tray_show_window": "Show the window",
    "tray_quit": "Quit",
}

ICON_SIZE = 64
_RING = 8

# The disc's colour by state: the same reading as the status line's colours,
# in pixels. A state with no colour of its own is grey, like idle.
_GREY = (140, 140, 140)
_COLOURS: dict[State, tuple[int, int, int]] = {
    State.OFF: _GREY,
    State.IDLE: _GREY,
    State.SLEEPING: (70, 90, 120),
    State.USER_SPEAKING: (46, 204, 113),
    State.RECONNECTING: (241, 196, 15),
    State.CONFIRMING: (243, 156, 18),
    State.SPEAKING: (155, 89, 182),
    State.ANNOUNCING: (155, 89, 182),
}


@dataclass(frozen=True)
class MenuEntry:
    """One line of the menu: what it says right now, and what a click does.

    `label` is asked every time the menu opens, which is how the toggle
    line reads "Stop listening" one moment and "Start listening" the next.
    A line with no `action` only informs, and is drawn greyed out.
    """

    label: Callable[[], str]
    action: Callable[[], None] | None = None


class TrayIcon(Protocol):
    """What is used of pystray's `Icon`, so that a test can stand in for it."""

    icon: Any
    title: str

    def run_detached(self) -> None: ...

    def stop(self) -> None: ...

    def update_menu(self) -> None: ...


IconFactory = Callable[[str, Image.Image, str, list[MenuEntry]], TrayIcon]


def system_icon(name: str, image: Image.Image, title: str, entries: list[MenuEntry]) -> TrayIcon:
    """pystray's icon, in the real notification area.

    Deferred and untyped: `pystray` ships no stubs and touches the shell
    on import, which `live-assistant --help` has no reason to do.
    """
    import pystray  # type: ignore[import-untyped]

    def item(entry: MenuEntry) -> Any:
        action = entry.action
        if action is None:
            return pystray.MenuItem(lambda _: entry.label(), None, enabled=False)
        return pystray.MenuItem(lambda _: entry.label(), lambda icon, item: action())

    icon: TrayIcon = pystray.Icon(
        name, icon=image, title=title, menu=pystray.Menu(*(item(entry) for entry in entries))
    )
    return icon


def draw_icon(state: State, *, listening: bool) -> Image.Image:
    """A disc in the state's colour, filled while the microphone is live and
    a ring while it is not."""
    image = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    colour = (*_COLOURS.get(state, _GREY), 255)
    box = (2, 2, ICON_SIZE - 3, ICON_SIZE - 3)
    if listening:
        ImageDraw.Draw(image).ellipse(box, fill=colour)
    else:
        ImageDraw.Draw(image).ellipse(box, outline=colour, width=_RING)
    return image


def open_folder(folder: Path) -> None:
    """Shows `folder` in Explorer, the way a double-click would."""
    shell.launch(str(folder))


class Tray:
    """The icon, its tooltip and its menu, kept in step with the state machine."""

    def __init__(
        self,
        locale: Locale,
        *,
        loop: asyncio.AbstractEventLoop,
        on_toggle: Callable[[], None],
        on_quit: Callable[[], None],
        on_show: Callable[[], None] | None = None,
        settings_folder: Path,
        icon: IconFactory = system_icon,
        open: Callable[[Path], None] = open_folder,
        name: str = "assistant",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._said = {key: locale.say(key, default) for key, default in TEXT.items()}
        self._labels = {
            state: locale.say(label_key(state), STATUS_TEXT[label_key(state)]) for state in State
        }
        # The session half of the tooltip: the status line's words under
        # the status line's keys, and the same meter.
        self._meter_said = {
            key: locale.say(key, STATUS_TEXT[key])
            for key in ("session_open", "session_closed", "session_minutes")
        }
        self._session = SessionMinutes(clock=clock)
        self._loop = loop
        self._on_toggle = on_toggle
        self._on_quit = on_quit
        self._on_show = on_show
        self._folder = settings_folder
        self._open = open

        self._state = State.IDLE
        # Live until the capture says otherwise, which it does the moment it
        # starts (`HandsFree.start`).
        self._listening = True
        self._images = {
            (state, live): draw_icon(state, listening=live)
            for state in State
            for live in (True, False)
        }
        self._icon = icon(name, self._images[(State.IDLE, True)], self._title(), self.entries())

    def entries(self) -> list[MenuEntry]:
        """The menu, top to bottom: the state, the switch, the folder, the
        window when there is one to bring back (D20), the end."""
        lines = [
            MenuEntry(self._title),
            MenuEntry(self._toggle_label, self._toggle),
            MenuEntry(lambda: self._said["tray_open_settings"], self._open_settings),
        ]
        if self._on_show is not None:
            lines.append(MenuEntry(lambda: self._said["tray_show_window"], self._show))
        lines.append(MenuEntry(lambda: self._said["tray_quit"], self._quit))
        return lines

    def start(self) -> None:
        """Puts the icon up, on a thread of its own."""
        self._icon.run_detached()

    def stop(self) -> None:
        """Takes the icon down and ends its thread."""
        self._icon.stop()

    # ----------------------------------------------------------------------
    # Called on the event loop, by the state machine.
    # ----------------------------------------------------------------------

    def state(self, state: State) -> None:
        self._state = state
        self._redraw()

    def hands_free(self, listening: bool) -> None:
        """Says whether the microphone is live - in the icon, in the tooltip,
        and in what the switch line offers to do next."""
        self._listening = listening
        self._redraw()
        self._icon.update_menu()

    def session(self, open: bool) -> None:
        """Says whether a session is open - the meter is running - and
        redraws the tooltip with the minutes so far."""
        self._session.told(open)
        self._redraw()

    def _redraw(self) -> None:
        self._icon.icon = self._images[(self._state, self._listening)]
        self._icon.title = self._title()

    def _title(self) -> str:
        """The state, then the session and its minutes. "Not listening" for
        an idle microphone that is off, which is the one thing "ready" would
        misdescribe."""
        if self._state is State.IDLE and not self._listening:
            state = self._said["tray_not_listening"]
        else:
            state = self._labels[self._state]
        which = self._meter_said["session_open" if self._session.open else "session_closed"]
        minutes = self._meter_said["session_minutes"].format(minutes=int(self._session.minutes))
        return f"{state} · {which} · {minutes}"

    def _toggle_label(self) -> str:
        return self._said["tray_stop_listening" if self._listening else "tray_start_listening"]

    # ----------------------------------------------------------------------
    # Called on pystray's thread, by a click. Nothing here touches the
    # assistant: each posts to the loop, or hands a folder to the shell.
    # ----------------------------------------------------------------------

    def _toggle(self) -> None:
        self._loop.call_soon_threadsafe(self._on_toggle)

    def _quit(self) -> None:
        self._loop.call_soon_threadsafe(self._on_quit)

    def _show(self) -> None:
        if self._on_show is not None:
            self._loop.call_soon_threadsafe(self._on_show)

    def _open_settings(self) -> None:
        # Blocking, and on the tray's thread rather than the loop's, which
        # is where a call that waits on Explorer belongs.
        self._open(self._folder)
