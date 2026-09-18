"""Starting with Windows (`autostart.py`, 4.2): one value under the user's
own `Run` key, written with the assistant as installed here.

The registry is a dictionary in these tests. What is claimed is what gets
written there - the console script beside the running interpreter, quoted,
with `run --tray` - and that "off" twice is not an error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from assistant import autostart
from assistant.autostart import (
    SCRIPT,
    VALUE_NAME,
    command_line,
    disable,
    enable,
    status,
)


class FakeRegistry:
    """The `Run` key as a dictionary."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self.values.get(name)

    def set(self, name: str, command: str) -> None:
        self.values[name] = command

    def delete(self, name: str) -> None:
        self.values.pop(name, None)


@pytest.fixture
def registry() -> FakeRegistry:
    return FakeRegistry()


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An interpreter with the console script beside it, the way a `uv`
    environment lays them out."""
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    (scripts / "python.exe").write_bytes(b"")
    script = scripts / SCRIPT
    script.write_bytes(b"")
    monkeypatch.setattr(sys, "executable", str(scripts / "python.exe"))
    return script


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------


def test_the_command_is_the_script_beside_the_interpreter_with_the_tray(installed: Path) -> None:
    assert command_line() == f'"{installed}" run --tray'


def test_the_tray_can_be_left_out(installed: Path) -> None:
    assert command_line(tray=False) == f'"{installed}" run'


def test_without_the_script_beside_it_the_one_on_the_path_is_taken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    monkeypatch.setattr(autostart.shutil, "which", lambda name: r"C:\Tools\live-assistant.exe")

    assert command_line() == r'"C:\Tools\live-assistant.exe" run --tray'


def test_without_any_script_the_interpreter_runs_the_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkout run without installing: `python -m assistant` still works."""
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    monkeypatch.setattr(autostart.shutil, "which", lambda name: None)

    assert command_line() == f'"{tmp_path / "python.exe"}" -m assistant run --tray'


# --------------------------------------------------------------------------
# On, off, status
# --------------------------------------------------------------------------


def test_on_writes_the_command_under_the_assistants_name(
    registry: FakeRegistry, installed: Path
) -> None:
    written = enable(registry)

    assert registry.values == {VALUE_NAME: f'"{installed}" run --tray'}
    assert written == registry.values[VALUE_NAME]


def test_on_takes_a_command_of_the_callers_choosing(registry: FakeRegistry) -> None:
    enable(registry, command='"D:\\x\\live-assistant.exe" run')

    assert registry.values[VALUE_NAME] == '"D:\\x\\live-assistant.exe" run'


def test_off_takes_it_away_and_is_no_error_when_it_was_never_there(
    registry: FakeRegistry,
) -> None:
    enable(registry, command="x")
    disable(registry)
    disable(registry)

    assert registry.values == {}


def test_status_is_the_command_or_nothing(registry: FakeRegistry) -> None:
    assert status(registry) is None

    enable(registry, command="x run --tray")

    assert status(registry) == "x run --tray"


def test_nothing_else_under_the_key_is_touched(registry: FakeRegistry) -> None:
    registry.values["OneDrive"] = '"C:\\OneDrive.exe" /background'

    enable(registry, command="x")
    disable(registry)

    assert registry.values == {"OneDrive": '"C:\\OneDrive.exe" /background'}
