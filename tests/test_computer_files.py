"""The walk over the user's folders (D42): what a name, a kind and a number
of days find, where it stops, and what may be opened. Over `tmp_path`."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from allie.computer.files import KINDS, find_files, folder_of, openable

NOW = 1_790_000_000.0
DAY = 86_400


def make(root: Path, relative: str, *, days_old: float = 0, size: int = 10) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    stamp = NOW - days_old * DAY
    os.utime(path, (stamp, stamp))
    return path


@pytest.fixture
def folders(tmp_path: Path) -> dict[str, Path]:
    made = {name: tmp_path / name for name in ("Documents", "Downloads", "Desktop")}
    for path in made.values():
        path.mkdir()
    return made


def test_every_word_of_the_name_must_be_in_the_file_name(folders: dict[str, Path]) -> None:
    make(folders["Downloads"], "Fatura_Eylül_2026.pdf")
    make(folders["Downloads"], "fatura-ekim.pdf")
    make(folders["Documents"], "Eylül planı.docx")

    walk = find_files(folders, words="eylul fatura", kind="any", days=0, now=NOW)

    assert [found.path.name for found in walk.found] == ["Fatura_Eylül_2026.pdf"]
    assert walk.found[0].folder == "Downloads"
    assert walk.complete


def test_a_kind_is_a_set_of_extensions(folders: dict[str, Path]) -> None:
    make(folders["Desktop"], "tatil.jpg")
    make(folders["Desktop"], "tatil.pdf")

    walk = find_files(folders, words="tatil", kind="image", days=0, now=NOW)

    assert [found.path.name for found in walk.found] == ["tatil.jpg"]
    assert ".pdf" in KINDS["document"] and ".mp4" in KINDS["video"]


def test_days_keeps_the_recent_ones_and_the_newest_come_first(folders: dict[str, Path]) -> None:
    make(folders["Downloads"], "a.pdf", days_old=1)
    make(folders["Downloads"], "b.pdf", days_old=10)
    make(folders["Downloads"], "c.pdf", days_old=0.1)

    walk = find_files(folders, words="", kind="document", days=7, now=NOW)

    assert [found.path.name for found in walk.found] == ["c.pdf", "a.pdf"]


def test_hidden_and_tool_folders_are_not_walked(folders: dict[str, Path]) -> None:
    make(folders["Documents"], ".git/fatura.pdf")
    make(folders["Documents"], "proje/node_modules/fatura.pdf")
    make(folders["Documents"], "proje/fatura.pdf")

    walk = find_files(folders, words="fatura", kind="any", days=0, now=NOW)

    assert [found.path.relative_to(folders["Documents"]).as_posix() for found in walk.found] == [
        "proje/fatura.pdf"
    ]


def test_the_walk_stops_at_its_depth_and_says_it_was_cut(folders: dict[str, Path]) -> None:
    make(folders["Documents"], "1/2/3/derin.pdf")

    shallow = find_files(folders, words="derin", kind="any", days=0, now=NOW, depth=2)
    deep = find_files(folders, words="derin", kind="any", days=0, now=NOW, depth=6)

    assert shallow.found == [] and not shallow.complete
    assert len(deep.found) == 1


def test_the_walk_stops_at_its_entry_count(folders: dict[str, Path]) -> None:
    for number in range(20):
        make(folders["Downloads"], f"dosya{number}.txt")

    walk = find_files(folders, words="dosya", kind="any", days=0, now=NOW, entries=5)

    assert len(walk.found) <= 5 and not walk.complete


def test_the_walk_stops_at_its_time(folders: dict[str, Path]) -> None:
    make(folders["Downloads"], "x.pdf")
    ticks = iter([0.0, 10.0, 20.0, 30.0])

    walk = find_files(
        folders, words="x", kind="any", days=0, now=NOW, seconds=3.0, clock=lambda: next(ticks)
    )

    assert not walk.complete


def test_at_most_ten_are_found(folders: dict[str, Path]) -> None:
    for number in range(15):
        make(folders["Downloads"], f"rapor{number}.pdf", days_old=number)

    walk = find_files(folders, words="rapor", kind="any", days=0, now=NOW)

    assert len(walk.found) == 10
    assert walk.found[0].path.name == "rapor0.pdf"


def test_a_path_belongs_to_a_known_folder_only_when_it_is_inside_one(
    folders: dict[str, Path], tmp_path: Path
) -> None:
    inside = make(folders["Documents"], "a/b.pdf")
    outside = make(tmp_path / "Elsewhere", "b.pdf")

    assert folder_of(inside, folders) == "Documents"
    assert folder_of(outside, folders) is None
    assert folder_of(folders["Documents"] / ".." / "Elsewhere" / "b.pdf", folders) is None


def test_only_known_kinds_may_be_opened(tmp_path: Path) -> None:
    for name in ("a.pdf", "b.JPG", "c.mp3", "d.zip", "e.docx"):
        assert openable(tmp_path / name), name
    for name in ("a.exe", "b.bat", "c.ps1", "d.lnk", "e.vbs", "f.js", "g.msi", "h", "i.py"):
        assert not openable(tmp_path / name), name


def test_nothing_under_a_missing_folder_is_an_error(tmp_path: Path) -> None:
    walk = find_files({"Music": tmp_path / "nope"}, words="x", kind="any", days=0, now=time.time())

    assert walk.found == [] and walk.complete
