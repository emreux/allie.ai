"""`documents/folder.py` (plan.md D35): a folder found by its whole name or
not at all, what it holds, and what one question about it sends.

Every folder and file is made here, under the test's own directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl
import pytest

from allie.documents import folder as folder_module
from allie.documents.folder import (
    EMPTY_FILE,
    NO_FOLDER,
    NO_FOLDERS_YET,
    NO_NAME,
    NO_ROOT,
    NOT_A_DOCUMENT,
    OLD_KINDS,
    TWO_FOLDERS,
    FolderError,
    PdfFile,
    Shelf,
    TextFile,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Two topics, a hidden folder and a file lying loose at the root."""
    shelf_root = tmp_path / "documents"
    for name in ("Bilgi Birikim", "Ali Yılmaz", ".git"):
        (shelf_root / name).mkdir(parents=True)
    (shelf_root / "loose.pdf").write_bytes(b"%PDF-1.7")
    return shelf_root


def failure(shelf: Shelf, spoken: str) -> str:
    with pytest.raises(FolderError) as caught:
        shelf.find(spoken)
    return str(caught.value)


# --------------------------------------------------------------------------
# The names
# --------------------------------------------------------------------------


def test_the_folders_are_named_in_order_and_nothing_else_is(root: Path) -> None:
    assert Shelf(root).names() == ["Ali Yılmaz", "Bilgi Birikim"]


def test_no_root_is_no_folders(tmp_path: Path) -> None:
    assert Shelf(tmp_path / "nope").names() == []


@pytest.mark.parametrize("spoken", ["Ali Yılmaz", "ali yilmaz", "ALİ YILMAZ", "  Ali   Yılmaz "])
def test_the_whole_name_in_any_case_or_spelling_is_the_folder(root: Path, spoken: str) -> None:
    found = Shelf(root).find(spoken)

    assert found.name == "Ali Yılmaz"
    assert found.path == root / "Ali Yılmaz"


@pytest.mark.parametrize("spoken", ["Ali", "Yılmaz", "Ali Yılmazz", "Ali Yılmaz'ın"])
def test_anything_but_the_whole_name_is_no_folder_and_no_other_is_offered(
    root: Path, spoken: str
) -> None:
    said = failure(Shelf(root), spoken)

    assert said == NO_FOLDER.format(name=spoken)
    assert "Bilgi Birikim" not in said


def test_an_empty_name_is_asked_for(root: Path) -> None:
    assert failure(Shelf(root), "  ") == NO_NAME


def test_a_root_that_is_not_there_is_said(tmp_path: Path) -> None:
    missing = tmp_path / "nope"

    assert failure(Shelf(missing), "Ali") == NO_ROOT.format(root=missing)


def test_an_empty_root_says_where_the_folders_go(tmp_path: Path) -> None:
    empty = tmp_path / "documents"
    empty.mkdir()

    assert failure(Shelf(empty), "Ali") == NO_FOLDERS_YET.format(root=empty)


def test_two_folders_that_fold_to_one_name_are_both_refused_by_name(root: Path) -> None:
    (root / "Ali Yilmaz").mkdir()

    assert failure(Shelf(root), "ali yılmaz") == TWO_FOLDERS.format(
        name="ali yılmaz", names="'Ali Yilmaz', 'Ali Yılmaz'"
    )


# --------------------------------------------------------------------------
# What a folder holds
# --------------------------------------------------------------------------


def test_a_folder_lists_what_it_reads_what_it_does_not_and_why(root: Path) -> None:
    folder = root / "Ali Yılmaz"
    for name in ("fatura.pdf", "mizan.xlsx", "Liste.CSV", "yazi.docx", "not.txt"):
        (folder / name).write_bytes(b"x")
    for name in ("eski.doc", "eski.xls", "logo.png"):
        (folder / name).write_bytes(b"x")
    for name in ("~$mizan.xlsx", "desktop.ini", "Thumbs.db", ".hidden"):
        (folder / name).write_bytes(b"x")
    (folder / "arşiv").mkdir()

    listing = Shelf(root).find("Ali Yılmaz").listing()

    assert [(document.name, document.kind) for document in listing.documents] == [
        ("fatura.pdf", "PDF"),
        ("Liste.CSV", "CSV"),
        ("mizan.xlsx", "Excel"),
        ("not.txt", "text"),
        ("yazi.docx", "Word"),
    ]
    assert listing.documents[0].size == 1
    assert dict(listing.not_read) == {
        "eski.doc": OLD_KINDS[".doc"],
        "eski.xls": OLD_KINDS[".xls"],
        "logo.png": NOT_A_DOCUMENT,
    }
    assert listing.subfolders == ("arşiv",)


# --------------------------------------------------------------------------
# What one question sends
# --------------------------------------------------------------------------


def test_a_bundle_carries_pdfs_as_bytes_and_the_rest_as_text(root: Path) -> None:
    folder = root / "Ali Yılmaz"
    (folder / "fatura.pdf").write_bytes(b"%PDF-1.7 fatura")
    book: Any = openpyxl.Workbook()
    book.active.append(["Hesap", "Bakiye"])
    book.active.append(["Kasa", 100])
    book.save(folder / "mizan.xlsx")
    (folder / "eski.doc").write_bytes(b"x")

    bundle = Shelf(root).find("Ali Yılmaz").bundle()

    assert bundle.files == (
        PdfFile("fatura.pdf", b"%PDF-1.7 fatura"),
        TextFile("mizan.xlsx", "## Sheet\n\n| Hesap | Bakiye |\n| --- | --- |\n| Kasa | 100 |"),
    )
    assert bundle.left_out == ()
    assert bundle.not_read == (("eski.doc", OLD_KINDS[".doc"]),)


def test_a_file_that_will_not_open_or_holds_nothing_is_named_not_sent(root: Path) -> None:
    folder = root / "Ali Yılmaz"
    (folder / "bozuk.xlsx").write_bytes(b"not a workbook")
    (folder / "bos.txt").write_text("  \n", encoding="utf-8")

    bundle = Shelf(root).find("Ali Yılmaz").bundle()

    assert bundle.files == ()
    reasons = dict(bundle.not_read)
    assert reasons["bozuk.xlsx"].startswith("could not be read (")
    assert reasons["bos.txt"] == EMPTY_FILE


def test_what_does_not_fit_one_question_is_left_out_by_name(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = root / "Ali Yılmaz"
    (folder / "a.txt").write_text("a" * 60, encoding="utf-8")
    (folder / "b.txt").write_text("b" * 60, encoding="utf-8")
    (folder / "c.pdf").write_bytes(b"%" * 60)
    (folder / "d.pdf").write_bytes(b"%" * 60)
    monkeypatch.setattr(folder_module, "MAX_TOTAL_CHARS", 100)
    monkeypatch.setattr(folder_module, "MAX_PDF_BYTES", 100)

    bundle = Shelf(root).find("Ali Yılmaz").bundle()

    assert [file.name for file in bundle.files] == ["a.txt", "c.pdf"]
    assert bundle.left_out == ("b.txt", "d.pdf")
