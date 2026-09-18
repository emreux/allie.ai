"""One browser window for everything the assistant plays, closed before the next.

A song opened as a tab is a tab the user has to close: five songs in an
evening were five tabs, and the only thing this program could do about the
previous one was pause it (`now_playing.py`). A tab cannot be reused or
closed from outside - the browser is the user's, signed in, and Chrome no
longer lets another program drive that profile - but a *window* can: the
default browser is asked for a new one (`--new-window`), the window that
appears is remembered by its handle, and the next song closes it.

Same browser, same profile, so a song still starts by itself (`shell.py`
says why that matters). The registry says which browser: the `https`
association names a ProgId and the ProgId names a command line, the way a
double-click on a link finds it. A browser that cannot be found this way -
a store application, an association nobody set - falls back to the tab.

Finding the window is a diff: the browser's visible top-level windows before
the command, then again until one is new. Windows are matched to the
browser by the executable that owns them, not by class name, so Chrome,
Edge, Brave and Firefox are the same code. The new window is opened
*before* the old one is closed: a browser whose only window was ours would
otherwise shut down between the two and lose the address.

Everything here that touches Win32 runs off the event loop (design.md
section 3.1 rule 4): starting a process can wait on a cold browser, and the
window may take a second or two to appear.
"""

from __future__ import annotations

import asyncio
import ctypes
import time
import winreg
from collections.abc import Callable, Sequence
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from assistant import shell

__all__ = [
    "APPEAR_POLL_SECONDS",
    "APPEAR_SECONDS",
    "HTTPS_CHOICE",
    "Browser",
    "Desktop",
    "MediaWindow",
    "Win32Desktop",
    "default_browser",
]

# How long a new window is waited for. A browser already running shows one in
# well under a second; a cold start takes two or three. Longer than this and
# the answer would be waiting on a window that is not coming.
APPEAR_SECONDS = 3.0
APPEAR_POLL_SECONDS = 0.1

# Where Windows keeps the user's choice of browser for `https` links.
HTTPS_CHOICE = r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice"

# `(hive, key, name) -> value`; a missing key or value is `None`.
ReadValue = Callable[[int, str, str], str | None]


@dataclass(frozen=True, slots=True)
class Browser:
    """The default browser, and how it is asked for a new window."""

    executable: Path
    # `--new-window` for the Chromium family, `-new-window` for Firefox.
    new_window: str


def default_browser(
    read: ReadValue | None = None, *, exists: Callable[[Path], bool] = Path.is_file
) -> Browser | None:
    """The browser a link opens in, or `None` when it cannot be named.

    Two registry reads: the user's choice of ProgId for `https`, and that
    ProgId's open command. Both are injectable so the parsing is tested
    without a registry, and `exists` so a command naming a program that is
    gone is treated as no browser rather than as one that will not start.
    """
    read = _read_value if read is None else read
    prog_id = read(winreg.HKEY_CURRENT_USER, HTTPS_CHOICE, "ProgId")
    if not prog_id:
        return None
    command = read(winreg.HKEY_CLASSES_ROOT, rf"{prog_id}\shell\open\command", "")
    executable = _executable_in(command or "")
    if executable is None or not exists(executable):
        logger.debug("no default browser behind {!r}: command {!r}", prog_id, command)
        return None
    flag = "-new-window" if executable.name.casefold().startswith("firefox") else "--new-window"
    return Browser(executable=executable, new_window=flag)


def _read_value(hive: int, key: str, name: str) -> str | None:
    try:
        with winreg.OpenKey(hive, key) as opened:
            value, _kind = winreg.QueryValueEx(opened, name)
    except OSError:
        return None
    return value if isinstance(value, str) else None


def _executable_in(command: str) -> Path | None:
    """The program at the front of a registry command line: quoted, or up to
    the first space."""
    text = command.strip()
    if not text:
        return None
    if text.startswith('"'):
        end = text.find('"', 1)
        return None if end < 0 else Path(text[1:end])
    return Path(text.split(" ", 1)[0])


class Desktop(Protocol):
    """What the window needs from Windows. `Win32Desktop` is the real one; a
    test hands over one that records."""

    def start(self, command: Sequence[str]) -> None: ...

    def windows_of(self, executable: Path) -> set[int]: ...

    def is_window(self, handle: int) -> bool: ...

    def close(self, handle: int) -> None: ...


WM_CLOSE = 0x0010
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
KEYEVENTF_KEYUP = 0x0002
_EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


class Win32Desktop:
    """The real desktop, through `user32` and `kernel32`.

    Argument types are declared because a window handle is pointer-sized:
    left to `ctypes`' default, a handle above 2^31 would be truncated on the
    way in and name a different window - or none.
    """

    def __init__(self) -> None:
        self._user32: Any = ctypes.windll.user32
        self._kernel32: Any = ctypes.windll.kernel32
        self._user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self._user32.IsWindow.argtypes = [wintypes.HWND]
        self._user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._user32.PostMessageW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        self._user32.EnumWindows.argtypes = [_EnumWindowsProc, wintypes.LPARAM]
        self._user32.GetForegroundWindow.restype = wintypes.HWND
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    def start(self, command: Sequence[str]) -> None:
        shell.start(command)

    def windows_of(self, executable: Path) -> set[int]:
        """The visible top-level windows owned by `executable`."""
        wanted = str(executable).casefold()
        return self._windows_where(lambda image: image.casefold() == wanted)

    def windows_named(self, image: str) -> set[int]:
        """The visible top-level windows of any program whose file is called
        `image` - `Spotify.exe` - wherever it is installed.

        By the name and not the path, because the same program lives under
        `WindowsApps` when it came from the Store and under the user's
        profile when it did not, and the caller has no business knowing
        which (`media/spotify.py`).
        """
        wanted = image.casefold()
        return self._windows_where(lambda path: Path(path).name.casefold() == wanted)

    def _windows_where(self, owned: Callable[[str], bool]) -> set[int]:
        """The visible top-level windows whose owning process's image
        satisfies `owned`."""
        found: set[int] = set()
        # Each process is asked for its image once, not once per window.
        images: dict[int, str | None] = {}

        def visit(handle: int, _: int) -> bool:
            if self._user32.IsWindowVisible(handle):
                pid = wintypes.DWORD()
                self._user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
                if pid.value not in images:
                    images[pid.value] = self._image_of(pid.value)
                image = images[pid.value]
                if image is not None and owned(image):
                    found.add(handle)
            return True

        self._user32.EnumWindows(_EnumWindowsProc(visit), 0)
        return found

    def _image_of(self, pid: int) -> str | None:
        process = self._kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not process:
            # Another user's, or protected: not the browser either way.
            return None
        try:
            size = wintypes.DWORD(32_767)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not self._kernel32.QueryFullProcessImageNameW(
                process, 0, buffer, ctypes.byref(size)
            ):
                return None
            return str(buffer.value)
        finally:
            self._kernel32.CloseHandle(process)

    def foreground_image(self) -> str | None:
        """The file name of the program whose window is in front -
        `WhatsApp.exe` - or `None` when there is none, or it cannot be asked.

        Added for `messaging/whatsapp.py` (2026-09-15), which presses a key
        only when this names the application it means to press it in.
        """
        handle = self._user32.GetForegroundWindow()
        if not handle:
            return None
        pid = wintypes.DWORD()
        self._user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        image = self._image_of(pid.value)
        return None if image is None else Path(image).name

    def press(self, code: int) -> None:
        """One press and release of the key `code`, as the keyboard sends it -
        the three lines `tools/media.py` presses the media keys with, kept
        there as that module's own test seam."""
        self._user32.keybd_event(code, 0, 0, 0)
        self._user32.keybd_event(code, 0, KEYEVENTF_KEYUP, 0)

    def is_window(self, handle: int) -> bool:
        return bool(self._user32.IsWindow(handle))

    def close(self, handle: int) -> None:
        # Posted, not sent: the browser closes the window on its own thread,
        # and this one is not kept waiting for it.
        self._user32.PostMessageW(handle, WM_CLOSE, 0, 0)


class MediaWindow:
    """The one window the assistant plays things in."""

    def __init__(
        self,
        browser: Browser | None,
        desktop: Desktop | None = None,
        *,
        appear_seconds: float = APPEAR_SECONDS,
        poll_seconds: float = APPEAR_POLL_SECONDS,
    ) -> None:
        self.browser = browser
        self._desktop: Desktop = desktop if desktop is not None else Win32Desktop()
        self._appear_seconds = appear_seconds
        self._poll_seconds = poll_seconds
        # The window last opened, while it is believed to exist. Public for
        # whoever wants to know, never closed at shutdown: the user may still
        # be listening.
        self.handle: int | None = None
        # One window at a time: two requests that overlapped would each take
        # the other's window for their own and close the wrong one.
        self._turn = asyncio.Lock()

    async def show(self, address: str) -> bool:
        """Opens `address` in the window, closing the previous one.

        `True` when the browser took it - whether or not the window was
        found afterwards - and `False` when nothing would open.
        """
        if self.browser is None:
            return await shell.open_address(address)
        async with self._turn:
            return await asyncio.to_thread(self._show, self.browser, address)

    def _show(self, browser: Browser, address: str) -> bool:
        before = self._desktop.windows_of(browser.executable)
        try:
            self._desktop.start([str(browser.executable), browser.new_window, address])
        except OSError as failure:
            logger.warning("{} would not start: {}", browser.executable, failure)
            return False

        appeared = self._appeared(browser.executable, before)
        self._close_previous()
        self.handle = appeared
        if appeared is None:
            logger.debug(
                "no new {} window appeared within {} s",
                browser.executable.name,
                self._appear_seconds,
            )
        return True

    def _appeared(self, executable: Path, before: set[int]) -> int | None:
        deadline = time.monotonic() + self._appear_seconds
        while True:
            new = self._desktop.windows_of(executable) - before
            if new:
                return min(new)
            if time.monotonic() >= deadline:
                return None
            time.sleep(self._poll_seconds)

    def _close_previous(self) -> None:
        handle, self.handle = self.handle, None
        if handle is not None and self._desktop.is_window(handle):
            self._desktop.close(handle)
