"""One browser window for what the assistant plays, and the previous one closed.

Nothing here starts a browser or touches a window: the desktop is a fake that
records what it was asked and lets a window "appear" when the test says so,
and `shell.browse` is replaced the way `test_media_player.py` replaces it.
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import threading
import winreg
from collections.abc import Callable, Iterator, Sequence
from ctypes import wintypes
from pathlib import Path

import pytest

from allie import shell
from allie.media.window import (
    HTTPS_CHOICE,
    Browser,
    MediaWindow,
    Win32Desktop,
    default_browser,
)

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
        # The process behind each window; the browser's, 42, unless a test
        # says a window belongs to another.
        self.processes: dict[int, int] = {}
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

    def process_of(self, handle: int) -> int | None:
        return self.processes.get(handle, 42) if handle in self.windows else None

    def close(self, handle: int) -> None:
        self.events.append(f"close {handle}")
        self.windows.discard(handle)


def window(
    desktop: FakeDesktop, browser: Browser | None = CHROME, *, kept: Path | None = None
) -> MediaWindow:
    return MediaWindow(browser, desktop, appear_seconds=0.2, poll_seconds=0.001, kept=kept)


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
# The window of the run before (2026-09-25)
# --------------------------------------------------------------------------


async def test_the_window_is_written_down_with_its_browser_s_process(tmp_path: Path) -> None:
    kept = tmp_path / "media-window"
    shown = window(FakeDesktop(), kept=kept)

    await shown.show(SONG)

    assert kept.read_text(encoding="utf-8").split() == ["100", "42"]


async def test_the_next_run_s_first_song_closes_the_window_of_the_run_before(
    tmp_path: Path,
) -> None:
    """The owner, 2026-09-25: a song, the assistant restarted, another song -
    and both played. The window was remembered only in the memory of the
    run that opened it, and it keeps playing after that run ends."""
    kept = tmp_path / "media-window"
    kept.write_text("7 42", encoding="utf-8")
    desktop = FakeDesktop(existing=frozenset({7}))
    shown = window(desktop, kept=kept)

    await shown.show(SONG)

    assert desktop.events == [f"start {CHROME.executable} --new-window {SONG}", "close 7"]
    assert shown.handle == 100
    assert kept.read_text(encoding="utf-8").split() == ["100", "42"]


async def test_a_window_written_down_whose_process_changed_is_not_the_one(
    tmp_path: Path,
) -> None:
    """A browser that restarted may have given the number to a window of
    the user's own; the process tells them apart."""
    kept = tmp_path / "media-window"
    kept.write_text("7 42", encoding="utf-8")
    desktop = FakeDesktop(existing=frozenset({7}))
    desktop.processes[7] = 99
    shown = window(desktop, kept=kept)

    await shown.show(SONG)

    assert "close 7" not in desktop.events
    assert shown.handle == 100


async def test_a_window_written_down_that_is_gone_is_not_closed(tmp_path: Path) -> None:
    kept = tmp_path / "media-window"
    kept.write_text("7 42", encoding="utf-8")
    desktop = FakeDesktop()
    shown = window(desktop, kept=kept)

    await shown.show(SONG)

    assert "close 7" not in desktop.events
    assert shown.handle == 100


@pytest.mark.parametrize("text", ["", "seven", "7", "7 42 extra", "-1 42"])
async def test_a_note_that_cannot_be_read_is_ignored(tmp_path: Path, text: str) -> None:
    kept = tmp_path / "media-window"
    kept.write_text(text, encoding="utf-8")
    desktop = FakeDesktop(existing=frozenset({7}))
    shown = window(desktop, kept=kept)

    assert await shown.show(SONG) is True

    assert desktop.events == [f"start {CHROME.executable} --new-window {SONG}"]
    assert shown.handle == 100


async def test_no_window_to_remember_leaves_nothing_written_down(tmp_path: Path) -> None:
    kept = tmp_path / "media-window"
    kept.write_text("7 42", encoding="utf-8")
    shown = window(FakeDesktop(appears_after=-1), kept=kept)

    await shown.show(SONG)

    assert not kept.exists()


async def test_the_run_before_is_asked_about_once(tmp_path: Path) -> None:
    """Its window, once closed, is not looked for again at the next song."""
    kept = tmp_path / "media-window"
    kept.write_text("7 42", encoding="utf-8")
    desktop = FakeDesktop(existing=frozenset({7}))
    shown = window(desktop, kept=kept)

    await shown.show(SONG)
    await shown.show(VIDEO)

    assert desktop.events.count("close 7") == 1
    assert desktop.events[-1] == "close 100"


def test_the_player_s_own_window_is_written_down_in_the_data_folder() -> None:
    from allie.config import data_dir
    from allie.media.player import Player
    from allie.media.window import KEPT_FILE_NAME

    assert Player().window.kept == data_dir() / KEPT_FILE_NAME


# --------------------------------------------------------------------------
# The real desktop: which windows are the browser's own
# --------------------------------------------------------------------------

WS_POPUP = 0x80000000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
SW_SHOWNOACTIVATE = 4


@pytest.fixture
def frame_and_bubble() -> Iterator[tuple[int, int]]:
    """Two real windows of this process, one pixel each and off the screen:
    a frame, unowned as a browser window is, and a bubble the frame owns, as
    Chrome's "Translate this page?" is (measured 2026-09-25: owned by the
    song's window, `WS_POPUP`, `WS_EX_TOOLWINDOW`)."""
    user32 = ctypes.windll.user32
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HWND,
        wintypes.HMENU,
        wintypes.HINSTANCE,
        wintypes.LPVOID,
    ]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.DestroyWindow.argtypes = [wintypes.HWND]

    def made(extended: int, owner: int | None) -> int:
        handle = user32.CreateWindowExW(
            extended, "STATIC", "", WS_POPUP, -32000, -32000, 1, 1, owner, None, None, None
        )
        assert handle, ctypes.WinError()
        user32.ShowWindow(handle, SW_SHOWNOACTIVATE)
        return int(handle)

    frame = made(WS_EX_NOACTIVATE, None)
    bubble = made(WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW, frame)
    try:
        yield frame, bubble
    finally:
        user32.DestroyWindow(bubble)
        user32.DestroyWindow(frame)


def test_a_bubble_the_browser_shows_over_its_window_is_not_one_of_its_windows(
    frame_and_bubble: tuple[int, int],
) -> None:
    """Chrome's "Translate this page?" appears two seconds after a song's
    page, as a visible top-level window of chrome.exe. Counted as one, it
    could be taken for the next song's window - which was then never closed,
    and played on under the song after it."""
    frame, bubble = frame_and_bubble
    desktop = Win32Desktop()
    image = desktop._image_of(os.getpid())
    assert image is not None

    found = desktop.windows_of(Path(image))

    assert frame in found
    assert bubble not in found


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
