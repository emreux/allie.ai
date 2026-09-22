"""Starting with Windows (design.md section 8, phase 4.2; 17 Sep 2026).

One value under the user's own `Run` key, and nothing else: no service,
no scheduled task, no administrator. `HKCU\\...\\Run` is what Windows reads
at sign-in for this user alone, and a value there is a decision the user
can see and undo in Task Manager's start-up page, or with `allie
autostart off`. The command registered is the assistant as it is
installed on this machine - the console script beside the interpreter
that is running - with the tray icon, because a program that starts by
itself should have a place to be quit from (`ui/tray.py`).

The value is named after the product, not the package (plan.md D2), so
that this assistant and the one it succeeds can both be registered on
one machine without either taking the other's line.

`winreg` is behind a small protocol so that the tests write to a
dictionary rather than to the developer's sign-in.
"""

from __future__ import annotations

import contextlib
import shutil
import sys
from pathlib import Path
from typing import Protocol

__all__ = [
    "RUN_KEY",
    "VALUE_NAME",
    "Registry",
    "WindowsRegistry",
    "command_line",
    "disable",
    "enable",
    "status",
]

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "allie"
SCRIPT = "allie.exe"


class Registry(Protocol):
    """The three things done to the `Run` key."""

    def get(self, name: str) -> str | None: ...

    def set(self, name: str, command: str) -> None: ...

    def delete(self, name: str) -> None: ...


class WindowsRegistry:
    """The user's own `Run` key, through `winreg`."""

    def get(self, name: str) -> str | None:
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
                value, _ = winreg.QueryValueEx(key, name)
        except FileNotFoundError:
            return None
        return str(value)

    def set(self, name: str, command: str) -> None:
        import winreg

        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, command)

    def delete(self, name: str) -> None:
        import winreg

        with (
            contextlib.suppress(FileNotFoundError),
            winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key,
        ):
            winreg.DeleteValue(key, name)


def command_line(*, tray: bool = True) -> str:
    """The assistant as installed here, quoted for the shell, with `run --tray`.

    The console script sits beside the interpreter that is running this -
    `.venv\\Scripts\\allie.exe` - and is preferred; one on the PATH
    is next; and the interpreter with `-m assistant` is what is left when
    neither exists, which is a checkout run without installing.
    """
    beside = Path(sys.executable).with_name(SCRIPT)
    if beside.is_file():
        program = f'"{beside}"'
    elif found := shutil.which("allie"):
        program = f'"{found}"'
    else:
        program = f'"{sys.executable}" -m assistant'
    return f"{program} run --tray" if tray else f"{program} run"


def enable(registry: Registry, *, command: str | None = None) -> str:
    """Registers the assistant to start at sign-in; returns what was registered."""
    line = command if command is not None else command_line()
    registry.set(VALUE_NAME, line)
    return line


def disable(registry: Registry) -> None:
    """Takes the assistant off the `Run` key. Not being there is not an error."""
    registry.delete(VALUE_NAME)


def status(registry: Registry) -> str | None:
    """The command registered to start at sign-in, or `None`."""
    return registry.get(VALUE_NAME)
