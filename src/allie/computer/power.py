"""Locking, sleeping, shutting down, restarting (D40). Blocking calls; the
tools run them on a worker thread."""

from __future__ import annotations

import ctypes
import subprocess
from typing import Protocol

__all__ = ["SHUTDOWN_SECONDS", "Power", "WindowsPower"]

# How long a shutdown or restart waits: time for "stop it" to be said and
# heard (the owner's number, D40).
SHUTDOWN_SECONDS = 30

# No console window flashes up for `shutdown.exe`.
_QUIET = subprocess.CREATE_NO_WINDOW


class Power(Protocol):
    def lock(self) -> None: ...

    def sleep(self) -> None: ...

    def shutdown(self, *, restart: bool, seconds: int) -> None: ...

    def cancel_shutdown(self) -> bool: ...


class WindowsPower:
    def lock(self) -> None:
        if not ctypes.windll.user32.LockWorkStation():
            raise OSError("LockWorkStation refused")

    def sleep(self) -> None:
        # Hibernate no, force no, wake events kept: an ordinary sleep.
        if not ctypes.windll.powrprof.SetSuspendState(False, False, False):
            raise OSError("SetSuspendState refused")

    def shutdown(self, *, restart: bool, seconds: int) -> None:
        # Windows' own command with fixed arguments; nothing the model said.
        subprocess.run(  # noqa: S603
            ["shutdown", "/r" if restart else "/s", "/t", str(seconds)],  # noqa: S607
            check=True,
            capture_output=True,
            creationflags=_QUIET,
        )

    def cancel_shutdown(self) -> bool:
        """`True` when a waiting shutdown or restart was stopped; `False`
        when none was waiting (`shutdown /a` exits 1116 then)."""
        done = subprocess.run(
            ["shutdown", "/a"],  # noqa: S607  # Windows' own, fixed arguments
            capture_output=True,
            creationflags=_QUIET,
        )
        return done.returncode == 0
