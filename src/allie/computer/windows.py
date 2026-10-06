"""The windows open on the desktop, found by what the user calls the app
(D41): brought to the front, closed the way their close box closes them,
all minimised.

**Which windows an app has.** A visible, unowned, titled top-level window
that is not a tool window and not cloaked (a suspended Store app keeps a
cloaked frame) is one the user sees in Alt+Tab. Each is known by its title
and the image name of its process. A spoken name is matched against the
image ("chrome"), then against the catalogue's name for the app inside the
title, as whole words ("Google Chrome" in "GitHub - Google Chrome", never
"mail" in "Gmail") - which is how a Store app, whose process is
`ApplicationFrameHost`, is found at all. Allie's own process is never a
match.

**Closing is asking.** `WM_CLOSE` is what the close box sends: an app with
unsaved work asks, and nothing is killed.
"""

from __future__ import annotations

import ctypes
import os
import re
from collections.abc import Iterable
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Protocol

from allie.store.normalize import normalize_search
from allie.tools.system import AppEntry

__all__ = ["AppWindow", "Desktop", "RunningApps", "Win32Desktop"]

GW_OWNER = 4
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
DWMWA_CLOAKED = 14
SW_RESTORE = 9
VK_MENU = 0x12
KEYEVENTF_KEYUP = 0x0002
WM_CLOSE = 0x0010
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# A word shorter than this is inside too many titles to mean one app.
MIN_TITLE_WORD = 3


@dataclass(frozen=True, slots=True)
class AppWindow:
    handle: int
    title: str
    image: str
    pid: int


class Desktop(Protocol):
    def windows(self) -> list[AppWindow]: ...

    def bring_to_front(self, handle: int) -> bool: ...

    def close(self, handle: int) -> None: ...

    def alive(self, handle: int) -> bool: ...

    def minimize_all(self) -> None: ...


class RunningApps:
    """The desktop's windows, by app. Blocking; the tools call it on a thread."""

    def __init__(self, desktop: Desktop, *, own_pid: int | None = None) -> None:
        self._desktop = desktop
        self._own_pid = os.getpid() if own_pid is None else own_pid

    def find(self, spoken: str, entry: AppEntry | None = None) -> list[AppWindow]:
        """The windows of the app `spoken` means, in the desktop's order (the
        front one first); empty when it is open nowhere."""
        windows = [w for w in self._desktop.windows() if w.pid != self._own_pid]
        names = [normalize_search(spoken).strip()]
        if entry is not None:
            names += [normalize_search(entry.name).strip(), normalize_search(entry.spoken).strip()]
        names = [name for name in dict.fromkeys(names) if name]
        return _by_image(windows, names) or _by_title(windows, names)

    def front(self, window: AppWindow) -> bool:
        return self._desktop.bring_to_front(window.handle)

    def close(self, windows: Iterable[AppWindow]) -> None:
        for window in windows:
            self._desktop.close(window.handle)

    def alive(self, windows: Iterable[AppWindow]) -> list[AppWindow]:
        return [window for window in windows if self._desktop.alive(window.handle)]

    def minimize_all(self) -> None:
        self._desktop.minimize_all()


def _by_image(windows: list[AppWindow], names: list[str]) -> list[AppWindow]:
    """The windows whose program is one of the names, or one word of them:
    `chrome.exe` for "chrome" and for "Google Chrome"."""
    known = set(names) | _words(names)
    return [window for window in windows if _stem(window.image) in known]


def _by_title(windows: list[AppWindow], names: list[str]) -> list[AppWindow]:
    """The windows whose title holds one of the names as whole words - how
    a Store app, whose program is `ApplicationFrameHost`, is found. Whole
    words: "mail" is not the Gmail tab of a browser, and closing that
    would close the browser."""
    patterns = [
        re.compile(rf"\b{re.escape(name)}\b") for name in names if len(name) >= MIN_TITLE_WORD
    ]
    return [
        window
        for window in windows
        if any(pattern.search(normalize_search(window.title)) for pattern in patterns)
    ]


def _stem(image: str) -> str:
    return normalize_search(PurePath(image).stem).strip()


def _words(names: list[str]) -> set[str]:
    """Each word of the names, long enough to stand for an app on its own:
    "chrome" of "google chrome"."""
    return {word for name in names for word in name.split() if len(word) >= MIN_TITLE_WORD}


_EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


class Win32Desktop:
    """The real desktop, through user32, kernel32 and dwmapi.

    Argument types are declared, as `media/window.py` declares them: a
    window handle is pointer-sized, and left to `ctypes`' default one above
    2^31 would be cut short on the way in and name a different window - or
    none.
    """

    def __init__(self) -> None:
        self._user32: Any = ctypes.windll.user32
        self._kernel32: Any = ctypes.windll.kernel32
        self._dwmapi: Any = ctypes.windll.dwmapi
        user32 = self._user32
        for name in (
            "IsIconic",
            "IsWindow",
            "IsWindowVisible",
            "SetForegroundWindow",
            "GetWindowTextLengthW",
        ):
            getattr(user32, name).argtypes = [wintypes.HWND]
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindow.restype = wintypes.HWND
        user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
        user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.PostMessageW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        user32.EnumWindows.argtypes = [_EnumWindowsProc, wintypes.LPARAM]
        self._dwmapi.DwmGetWindowAttribute.argtypes = [
            wintypes.HWND,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    def windows(self) -> list[AppWindow]:
        found: list[AppWindow] = []

        def visit(handle: int, _: int) -> bool:
            window = self._window(handle)
            if window is not None:
                found.append(window)
            return True

        self._user32.EnumWindows(_EnumWindowsProc(visit), 0)
        return found

    def bring_to_front(self, handle: int) -> bool:
        if self._user32.IsIconic(handle):
            self._user32.ShowWindow(handle, SW_RESTORE)
        # Windows gives the foreground to a background process only just
        # after a key press; an Alt press and release is the one that
        # changes nothing else.
        self._user32.keybd_event(VK_MENU, 0, 0, 0)
        self._user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)
        self._user32.SetForegroundWindow(handle)
        return bool(self._user32.GetForegroundWindow() == handle)

    def close(self, handle: int) -> None:
        self._user32.PostMessageW(handle, WM_CLOSE, 0, 0)

    def alive(self, handle: int) -> bool:
        return bool(self._user32.IsWindow(handle)) and bool(self._user32.IsWindowVisible(handle))

    def minimize_all(self) -> None:
        # pywin32 ships no type information.
        import pythoncom  # type: ignore[import-untyped]
        import win32com.client  # type: ignore[import-untyped]

        pythoncom.CoInitialize()
        try:
            win32com.client.Dispatch("Shell.Application").MinimizeAll()
        finally:
            pythoncom.CoUninitialize()

    def _window(self, handle: int) -> AppWindow | None:
        user32 = self._user32
        if not user32.IsWindowVisible(handle) or user32.GetWindow(handle, GW_OWNER):
            return None
        if user32.GetWindowLongPtrW(handle, GWL_EXSTYLE) & WS_EX_TOOLWINDOW:
            return None
        cloaked = wintypes.DWORD()
        self._dwmapi.DwmGetWindowAttribute(
            handle, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked)
        )
        if cloaked.value:
            return None
        length = user32.GetWindowTextLengthW(handle)
        if not length:
            return None
        title = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(handle, title, length + 1)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        return AppWindow(
            handle=int(handle), title=title.value, image=self._image(pid.value), pid=int(pid.value)
        )

    def _image(self, pid: int) -> str:
        process = self._kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not process:
            return ""
        try:
            size = wintypes.DWORD(1024)
            path = ctypes.create_unicode_buffer(size.value)
            if not self._kernel32.QueryFullProcessImageNameW(process, 0, path, ctypes.byref(size)):
                return ""
            return PurePath(path.value).name
        finally:
            self._kernel32.CloseHandle(process)
