"""`files` (D42): find, open, show - and what is refused. Over `tmp_path`,
with the opener and the Explorer call recorded instead of run."""

from __future__ import annotations

import os
from pathlib import Path
from typing import get_args

import pytest

from allie.computer.files import KINDS
from allie.tools.files import (
    NOT_INSIDE,
    NOT_OPENABLE,
    NOTHING_FOUND,
    Kind,
    files_for,
)
from allie.tools.registry import Tool

NOW = 1_790_000_000.0


class Shell:
    def __init__(self) -> None:
        self.opened: list[str] = []
        self.shown: list[Path] = []

    async def open(self, target: str) -> None:
        self.opened.append(target)

    def reveal(self, path: Path) -> None:
        self.shown.append(path)


@pytest.fixture
def home(tmp_path: Path) -> dict[str, Path]:
    folders = {"Downloads": tmp_path / "Downloads", "Documents": tmp_path / "Documents"}
    for folder in folders.values():
        folder.mkdir()
    for name in ("Fatura Eylül.pdf", "kurulum.exe"):
        path = folders["Downloads"] / name
        path.write_bytes(b"x")
        os.utime(path, (NOW - 3600, NOW - 3600))
    return folders


@pytest.fixture
def shell() -> Shell:
    return Shell()


def tool(home: dict[str, Path], shell: Shell) -> Tool:
    return files_for(home, opener=shell.open, reveal=shell.reveal, clock=lambda: NOW)


def test_files_is_safe_with_three_actions(home: dict[str, Path], shell: Shell) -> None:
    files = tool(home, shell)

    assert set(get_args(Kind)) == {"any", *KINDS}
    assert files.risk == "safe"
    assert files.spec.parameters["properties"]["action"]["enum"] == ["find", "open", "show"]
    assert files.spec.parameters["required"] == ["action"]


async def test_found_files_are_listed_as_outside_content_with_full_paths(
    home: dict[str, Path], shell: Shell
) -> None:
    said = await tool(home, shell).run(action="find", name="fatura")

    assert said.startswith('<untrusted source="files">\n1. Fatura Eylül.pdf - Downloads, ')
    assert str(home["Downloads"] / "Fatura Eylül.pdf") in said


async def test_nothing_found_is_said(home: dict[str, Path], shell: Shell) -> None:
    assert await tool(home, shell).run(action="find", name="yok böyle") == NOTHING_FOUND


async def test_a_document_is_opened_with_its_own_program(
    home: dict[str, Path], shell: Shell
) -> None:
    path = home["Downloads"] / "Fatura Eylül.pdf"

    said = await tool(home, shell).run(action="open", path=str(path))

    assert shell.opened == [str(path.resolve())]
    assert said == "Opened Fatura Eylül.pdf."


async def test_a_program_is_never_opened_but_may_be_shown(
    home: dict[str, Path], shell: Shell
) -> None:
    path = home["Downloads"] / "kurulum.exe"
    files = tool(home, shell)

    assert await files.run(action="open", path=str(path)) == NOT_OPENABLE.format(name="kurulum.exe")
    assert shell.opened == []
    assert await files.run(action="show", path=str(path)) == ("Showed kurulum.exe in its folder.")
    assert shell.shown == [path.resolve()]


async def test_a_path_outside_the_known_folders_is_refused(
    home: dict[str, Path], shell: Shell, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "secret.pdf"
    elsewhere.write_bytes(b"x")

    for action in ("open", "show"):
        said = await tool(home, shell).run(action=action, path=str(elsewhere))
        assert said == NOT_INSIDE

    assert shell.opened == [] and shell.shown == []


async def test_a_file_that_is_not_there_is_said(home: dict[str, Path], shell: Shell) -> None:
    said = await tool(home, shell).run(action="open", path=str(home["Documents"] / "yok.pdf"))

    assert "no file" in said
