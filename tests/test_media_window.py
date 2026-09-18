"""One browser window for what the assistant plays, and the previous one closed.

Nothing here starts a browser or touches a window: the desktop is a fake that
records what it was asked and lets a window "appear" when the test says so,
and `shell.browse` is replaced the way `test_media_player.py` replaces it.
"""

from __future__ import annotations

import asyncio
import threading
import winreg
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from assistant import shell
from assistant.media.window import HTTPS_CHOICE, Browser, MediaWindow, default_browser

MAIN = threading.current_thread()
CHROME = Browser(
    executable=Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    new_window="--new-window",
)
SONG = "https://music.youtube.com/watch?v=UXK9s54VmxQ"
VIDEO = "https://www.youtube.com/watch?v=outny_anbdo"


class FakeDesktop:
    """Windows that appear when told to, and a log of everything asked.

    `appears_after` is how many `windows_of` calls a started window takes to
    show up; `-1` means it never does.
    """

    def __init__(self, *, existing: frozenset[int] = frozenset(), appears_after: int = 1) -> None:
        self.windows: set[int] = set(existing)
        self.events: list[str] = []
        self.threads: list[threading.Thread] = []
        self._next = 100
        self._appears_after = appears_after
        self._polls = 0
        self._pending: int | None = None

    def start(self, command: Sequence[str]) -> None:
        self.threads.append(threading.current_thread())
        self.events.append(f"start {' '.join(command)}")
        if self._appears_after >= 0:
            self._pending = self._next
            self._next += 1
            self._polls = 0

    def windows_of(self, executable: Path) -> set[int]:
        self.threads.append(threading.current_thread())
        if self._pending is not None:
            self._polls += 1
            if self._polls > self._appears_after:
                self.windows.add(self._pending)
                self._pending = None
        return set(self.windows)

    def is_window(self, handle: int) -> bool:
        return handle in self.windows

    def close(self, handle: int) -> None:
        self.events.append(f"close {handle}")
        self.windows.discard(handle)


def window(desktop: FakeDesktop, browser: Browser | None = CHROME) -> MediaWindow:
    return MediaWindow(browser, desktop, appear_seconds=0.2, poll_seconds=0.001)


@pytest.fixture
def browsed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []

    def browse(address: str) -> bool:
        seen.append(address)
        return True

    monkeypatch.setattr(shell, "browse", browse)
    return seen


async def test_the_first_address_starts_the_browser_and_remembers_the_window() -> None:
    desktop = FakeDesktop()
    shown = window(desktop)

    assert await shown.show(SONG) is True

    assert desktop.events == [f"start {CHROME.executable} --new-window {SONG}"]
    assert shown.handle == 100


async def test_the_next_address_closes_the_previous_window_after_its_own_appeared() -> None:
    """Opened first, closed second: a browser whose only window was ours
    would otherwise shut down between the two and lose the address."""
    desktop = FakeDesktop()
    shown = window(desktop)

    await shown.show(SONG)
    await shown.show(VIDEO)

    assert desktop.events == [
        f"start {CHROME.executable} --new-window {SONG}",
        f"start {CHROME.executable} --new-window {VIDEO}",
        "close 100",
    ]
    assert shown.handle == 101


async def test_a_window_the_user_already_closed_is_not_closed_again() -> None:
    desktop = FakeDesktop()
    shown = window(desktop)

    await shown.show(SONG)
    desktop.windows.discard(100)  # the user closed it
    await shown.show(VIDEO)

    assert "close 100" not in desktop.events
    assert shown.handle == 101


async def test_without_a_browser_the_address_goes_to_a_tab(browsed: list[str]) -> None:
    desktop = FakeDesktop()
    shown = window(desktop, browser=None)

    assert await shown.show(SONG) is True

    assert browsed == [SONG]
    assert desktop.events == []


async def test_a_window_that_never_appears_is_not_remembered() -> None:
    desktop = FakeDesktop(appears_after=-1)
    shown = window(desktop)

    assert await shown.show(SONG) is True
    assert shown.handle is None


async def test_a_window_that_was_already_there_is_never_taken() -> None:
    """The user's own browser window, open before the song, is not ours to
    close."""
    desktop = FakeDesktop(existing=frozenset({7}), appears_after=-1)
    shown = window(desktop)

    await shown.show(SONG)
    await shown.show(VIDEO)

    assert shown.handle is None
    assert "close 7" not in desktop.events


async def test_a_window_that_takes_a_moment_is_still_found() -> None:
    desktop = FakeDesktop(appears_after=3)
    shown = window(desktop)

    await shown.show(SONG)

    assert shown.handle == 100


async def test_a_browser_that_will_not_start_is_reported_and_the_old_window_kept() -> None:
    desktop = FakeDesktop()
    shown = window(desktop)
    await shown.show(SONG)

    def refuse(command: Sequence[str]) -> None:
        raise OSError("access denied")

    desktop.start = refuse  # type: ignore[method-assign]

    assert await shown.show(VIDEO) is False
    assert shown.handle == 100
    assert "close 100" not in desktop.events


async def test_the_desktop_is_never_touched_on_the_event_loop() -> None:
    """Starting a process can wait on a cold browser and the window takes a
    moment to appear (design.md section 3.1 rule 4)."""
    desktop = FakeDesktop()

    await window(desktop).show(SONG)

    assert desktop.threads and all(thread is not MAIN for thread in desktop.threads)


async def test_two_requests_do_not_race_each_other_into_two_windows() -> None:
    desktop = FakeDesktop(appears_after=2)
    shown = window(desktop)

    await asyncio.gather(shown.show(SONG), shown.show(VIDEO))

    assert shown.handle == 101
    assert desktop.events.count("close 100") == 1


# --------------------------------------------------------------------------
# Which browser
# --------------------------------------------------------------------------

CHROME_COMMAND = r'"C:\Program Files\Google\Chrome\Application\chrome.exe" --single-argument %1'
EDGE_COMMAND = (
    r'"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --single-argument %1'
)
FIREFOX_COMMAND = r'"C:\Program Files\Mozilla Firefox\firefox.exe" -osint -url "%1"'


def registry(prog_id: str | None, command: str | None) -> Callable[[int, str, str], str | None]:
    values = {
        (winreg.HKEY_CURRENT_USER, HTTPS_CHOICE, "ProgId"): prog_id,
        (winreg.HKEY_CLASSES_ROOT, rf"{prog_id}\shell\open\command", ""): command,
    }

    def read(hive: int, key: str, name: str) -> str | None:
        return values.get((hive, key, name))

    return read


@pytest.mark.parametrize(
    ("prog_id", "command", "executable", "flag"),
    [
        (
            "ChromeHTML",
            CHROME_COMMAND,
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "--new-window",
        ),
        (
            "MSEdgeHTM",
            EDGE_COMMAND,
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "--new-window",
        ),
        (
            "FirefoxURL-308046B0AF4A39CB",
            FIREFOX_COMMAND,
            r"C:\Program Files\Mozilla Firefox\firefox.exe",
            "-new-window",
        ),
    ],
)
def test_the_default_browser_is_read_off_the_registry(
    prog_id: str, command: str, executable: str, flag: str
) -> None:
    browser = default_browser(registry(prog_id, command), exists=lambda path: True)

    assert browser == Browser(executable=Path(executable), new_window=flag)


def test_an_unquoted_command_is_cut_at_the_first_space() -> None:
    browser = default_browser(
        registry("X", r"C:\Browsers\brave.exe --single-argument %1"), exists=lambda path: True
    )

    assert browser is not None
    assert browser.executable == Path(r"C:\Browsers\brave.exe")


@pytest.mark.parametrize(
    ("prog_id", "command"),
    [(None, None), ("ChromeHTML", None), ("ChromeHTML", ""), ("ChromeHTML", '"unterminated')],
)
def test_a_browser_that_cannot_be_named_is_none(prog_id: str | None, command: str | None) -> None:
    assert default_browser(registry(prog_id, command), exists=lambda path: True) is None


def test_a_browser_that_is_not_on_disk_is_none() -> None:
    found = default_browser(registry("ChromeHTML", CHROME_COMMAND), exists=lambda path: False)

    assert found is None
