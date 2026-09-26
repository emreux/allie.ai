"""The user's document folders (plan.md D35, spec of 2026-09-25 section 3.2).

**The folder is the only source.** One root - `[documents] folder`, or
`%LOCALAPPDATA%\\allie\\documents` - and under it one folder per topic: a
customer, the user's own company, anything; a folder's name is its topic's
name. Nothing is copied, indexed or remembered. The root is listed when a
session opens (for the prompt's list of names) and again whenever a tool is
called, a few milliseconds for a hundred folders; a file dropped in,
renamed or deleted is there or not the next time it is looked for.

**A name is found exactly or not at all.** The name the user said and every
folder's name are folded the same way (`normalize_search`: accents and case
off, so "ali yilmaz" is "Ali Yılmaz"), and only an equal key is a match -
no prefix, no closest match, no suggestion (the owner, 2026-09-25: "sadece
yok desin"). The mistake that matters here is reading out somebody else's
invoice. Two folders that fold to one key are both refused, by name, until
the user renames one.

**What goes to the model is read here.** A PDF as its bytes - Gemini reads
a PDF itself - and everything else as text (`convert.py`), within two caps
for one question: `MAX_PDF_BYTES` of PDFs and `MAX_TOTAL_CHARS` of text. A
file that does not fit, will not open or is of a kind this does not read is
named, never dropped in silence.

Every method here is synchronous and touches the disk: the tools call them
through `asyncio.to_thread`. `names()` alone runs on the loop, once per
session open.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from allie.documents import convert
from allie.store.normalize import normalize_search

__all__ = [
    "EMPTY_FILE",
    "IGNORED_NAMES",
    "KINDS",
    "MAX_PDF_BYTES",
    "MAX_TOTAL_CHARS",
    "NOT_A_DOCUMENT",
    "NO_FOLDER",
    "NO_FOLDERS_YET",
    "NO_NAME",
    "NO_ROOT",
    "OLD_KINDS",
    "PDF_MIME",
    "TWO_FOLDERS",
    "UNREADABLE",
    "Bundle",
    "Document",
    "Folder",
    "FolderError",
    "Listing",
    "PdfFile",
    "Readable",
    "Shelf",
    "TextFile",
]

PDF_MIME = "application/pdf"

# Per question. Ten pages of PDF are well under a megabyte; the cap is for a
# folder somebody filled with scans. Google takes 50 MB a file.
MAX_PDF_BYTES = 20 * 1024 * 1024
# Some 30 000 tokens of text: three times the ten pages a folder is meant
# to hold, and still a small request.
MAX_TOTAL_CHARS = 120_000

# What this reads, by extension, and the kind the model is told.
KINDS: dict[str, str] = {
    ".pdf": "PDF",
    ".xlsx": "Excel",
    ".xlsm": "Excel",
    ".csv": "CSV",
    ".docx": "Word",
    ".txt": "text",
    ".md": "text",
}
# The two older formats an office still has, and what to do about them.
OLD_KINDS: dict[str, str] = {
    ".xls": "the old Excel format; saved as .xlsx it is read",
    ".doc": "the old Word format; saved as .docx it is read",
}
NOT_A_DOCUMENT = "not a kind of document this reads"
UNREADABLE = "could not be read ({reason})"
EMPTY_FILE = "has nothing in it to read"
# Files Windows and Office put into a folder on their own; never the user's.
IGNORED_NAMES = frozenset({"desktop.ini", "thumbs.db"})

NO_NAME = "No folder name was given. Ask the user which folder they mean."
NO_ROOT = (
    "The documents folder {root} does not exist; the folder line under [documents] in "
    "config.toml names it."
)
NO_FOLDERS_YET = (
    "There are no document folders yet. The user makes one folder per topic, with its "
    "documents inside, in {root}."
)
NO_FOLDER = (
    "There is no folder named {name!r}. Tell the user so, and do not offer another folder "
    "in its place."
)
TWO_FOLDERS = (
    "Two folders answer to {name!r}: {names}. Neither is read until the user renames one of them."
)


class FolderError(Exception):
    """A folder that could not be found, worded so that a tool can pass it on."""


@dataclass(frozen=True, slots=True)
class Document:
    """One file of a folder that this reads."""

    name: str
    path: Path
    kind: str
    size: int
    modified: float


@dataclass(frozen=True, slots=True)
class PdfFile:
    """A PDF on its way to the model: its name and its bytes."""

    name: str
    data: bytes


@dataclass(frozen=True, slots=True)
class TextFile:
    """Anything else on its way to the model: its name and its text."""

    name: str
    text: str


# What the model is sent: our own two types. Google's SDK is `model.py`'s
# business alone.
Readable = PdfFile | TextFile


@dataclass(frozen=True, slots=True)
class Listing:
    """What a folder holds: what is read, what is not and why, and the
    folders inside it, which are not read either."""

    documents: tuple[Document, ...]
    not_read: tuple[tuple[str, str], ...]
    subfolders: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Bundle:
    """What one question sends, what did not fit in it, and what was not read."""

    files: tuple[Readable, ...]
    left_out: tuple[str, ...]
    not_read: tuple[tuple[str, str], ...]


# How each kind that is not a PDF becomes text.
_CONVERTERS: dict[str, Callable[[Path], str]] = {
    "Excel": convert.excel_text,
    "CSV": convert.csv_text,
    "Word": convert.word_text,
    "text": convert.plain_text,
}


@dataclass(frozen=True, slots=True)
class Folder:
    """One topic's folder, under the name it is found by."""

    name: str
    path: Path

    def listing(self) -> Listing:
        """The folder's own files, in name order; nothing below it is read."""
        documents: list[Document] = []
        not_read: list[tuple[str, str]] = []
        subfolders: list[str] = []
        for entry in sorted(self.path.iterdir(), key=_order):
            if _ignored(entry.name):
                continue
            if entry.is_dir():
                subfolders.append(entry.name)
                continue
            suffix = entry.suffix.casefold()
            kind = KINDS.get(suffix)
            if kind is None:
                not_read.append((entry.name, OLD_KINDS.get(suffix, NOT_A_DOCUMENT)))
                continue
            status = entry.stat()
            documents.append(Document(entry.name, entry, kind, status.st_size, status.st_mtime))
        return Listing(tuple(documents), tuple(not_read), tuple(subfolders))

    def bundle(self) -> Bundle:
        """What one question about this folder sends to the model."""
        listing = self.listing()
        files: list[Readable] = []
        left_out: list[str] = []
        not_read = list(listing.not_read)
        pdf_bytes = 0
        chars = 0
        for document in listing.documents:
            try:
                readable = _read(document)
            except Exception as failure:
                # A broken or locked file is one line in the answer, not a
                # question that fails.
                not_read.append((document.name, UNREADABLE.format(reason=type(failure).__name__)))
                continue
            if isinstance(readable, PdfFile):
                if pdf_bytes + len(readable.data) > MAX_PDF_BYTES:
                    left_out.append(document.name)
                    continue
                pdf_bytes += len(readable.data)
            elif not readable.text.strip():
                not_read.append((document.name, EMPTY_FILE))
                continue
            elif chars + len(readable.text) > MAX_TOTAL_CHARS:
                left_out.append(document.name)
                continue
            else:
                chars += len(readable.text)
            files.append(readable)
        return Bundle(tuple(files), tuple(left_out), tuple(not_read))


class Shelf:
    """The root and the folders under it, listed afresh each time."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def names(self) -> list[str]:
        """Every folder's name, in the order the prompt lists them."""
        return [folder.name for folder in self._folders()]

    def find(self, spoken: str) -> Folder:
        """The one folder whose name is `spoken`, folded; raises `FolderError`."""
        said = " ".join(spoken.split())
        if not said:
            raise FolderError(NO_NAME)
        if not self.root.is_dir():
            raise FolderError(NO_ROOT.format(root=self.root))
        folders = self._folders()
        if not folders:
            raise FolderError(NO_FOLDERS_YET.format(root=self.root))
        wanted = _key(said)
        matches = [folder for folder in folders if _key(folder.name) == wanted]
        if not matches:
            raise FolderError(NO_FOLDER.format(name=said))
        if len(matches) > 1:
            names = ", ".join(repr(folder.name) for folder in matches)
            raise FolderError(TWO_FOLDERS.format(name=said, names=names))
        return matches[0]

    def _folders(self) -> list[Folder]:
        if not self.root.is_dir():
            return []
        return [
            Folder(entry.name, entry)
            for entry in sorted(self.root.iterdir(), key=_order)
            if entry.is_dir() and not _ignored(entry.name)
        ]


def _read(document: Document) -> Readable:
    if document.kind == "PDF":
        return PdfFile(document.name, document.path.read_bytes())
    return TextFile(document.name, _CONVERTERS[document.kind](document.path))


def _key(name: str) -> str:
    return normalize_search(" ".join(name.split()))


def _order(entry: Path) -> tuple[str, str]:
    """Folded name first, so "ali" sorts beside "Ali"; the name itself
    second, so that two names with one key keep a fixed order."""
    return (_key(entry.name), entry.name)


def _ignored(name: str) -> bool:
    return name.startswith((".", "~$")) or name.casefold() in IGNORED_NAMES
