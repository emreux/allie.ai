"""The user's own folders, walked for a file (D42).

**Which folders.** Desktop, Documents, Downloads, Pictures, Videos and Music,
where Windows says they are (`SHGetKnownFolderPath`) - so a Documents folder
OneDrive has moved is found where it moved to. The document folders of D35
are not here: they are `ask_documents`' and live under AppData.

**A walk has limits.** Depth, a count of entries and a number of seconds:
whichever comes first ends it, and the answer says it was cut. Hidden
folders, `node_modules` and the like are not walked; a file OneDrive keeps
only in the cloud is listed without being downloaded (listing a folder never
fetches a file).

**What may be opened is a list of kinds, not a list of dangers.** A file
whose extension is in no kind - a program, a script, a shortcut, anything
unknown - is never opened; it can be shown in its folder, which runs
nothing.
"""

from __future__ import annotations

import ctypes
import os
import time
import uuid
from collections.abc import Callable, Mapping
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path

from allie.store.normalize import normalize_search

__all__ = [
    "KINDS",
    "KNOWN_FOLDERS",
    "MAX_DEPTH",
    "MAX_ENTRIES",
    "MAX_FOUND",
    "WALK_SECONDS",
    "Found",
    "Walk",
    "find_files",
    "folder_of",
    "known_folders",
    "openable",
]

# Windows' known-folder ids, by the English label the model reads.
KNOWN_FOLDERS: dict[str, str] = {
    "Desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "Documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "Downloads": "374DE290-123F-4565-9164-39C4925E467B",
    "Pictures": "33E28130-4E1E-4676-835A-98395C3BC3BB",
    "Videos": "18989B1D-99B5-455B-841C-AB7C74E4DDFC",
    "Music": "4BD8D571-6D19-48D3-BE97-422220080E43",
}

KINDS: dict[str, frozenset[str]] = {
    "document": frozenset(
        {
            ".pdf",
            ".doc",
            ".docx",
            ".odt",
            ".rtf",
            ".txt",
            ".md",
            ".xls",
            ".xlsx",
            ".ods",
            ".csv",
            ".ppt",
            ".pptx",
            ".odp",
        }
    ),
    "image": frozenset(
        {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".tif", ".tiff"}
    ),
    "video": frozenset({".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm"}),
    "audio": frozenset({".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma"}),
    "archive": frozenset({".zip", ".rar", ".7z"}),
}
OPENABLE = frozenset().union(*KINDS.values())

MAX_DEPTH = 6
MAX_ENTRIES = 50_000
WALK_SECONDS = 3.0
MAX_FOUND = 10
SECONDS_PER_DAY = 86_400

# Folders nobody means when they ask for "the invoice".
SKIPPED = frozenset({"node_modules", "__pycache__", "$recycle.bin", "appdata"})


@dataclass(frozen=True, slots=True)
class Found:
    path: Path
    folder: str
    modified: float
    size: int


@dataclass(slots=True)
class Walk:
    found: list[Found] = field(default_factory=list)
    complete: bool = True


def known_folders() -> dict[str, Path]:
    """The six folders where this user's Windows keeps them; one Windows
    cannot name is left out."""
    folders: dict[str, Path] = {}
    for label, guid in KNOWN_FOLDERS.items():
        path = _known_folder(guid)
        if path is not None:
            folders[label] = path
    return folders


def find_files(
    folders: Mapping[str, Path],
    *,
    words: str,
    kind: str,
    days: int,
    now: float,
    limit: int = MAX_FOUND,
    depth: int = MAX_DEPTH,
    entries: int = MAX_ENTRIES,
    seconds: float = WALK_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> Walk:
    """The newest `limit` files whose name holds every word of `words`, of
    `kind` ("any" for all), changed within `days` (0 for any time). Blocking:
    the tool runs it on a thread."""
    wanted = [word for word in normalize_search(words).split() if word]
    extensions = KINDS.get(kind)
    since = now - days * SECONDS_PER_DAY if days > 0 else None
    started = clock()
    walk = Walk()
    seen = 0
    for label, root in folders.items():
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            folder, level = stack.pop()
            try:
                children = list(os.scandir(folder))
            except OSError:
                continue
            for child in children:
                seen += 1
                if seen > entries or clock() - started > seconds:
                    walk.complete = False
                    return _newest(walk, limit)
                name = child.name
                if child.is_dir(follow_symlinks=False):
                    if name.startswith(".") or name.casefold() in SKIPPED:
                        continue
                    if level + 1 >= depth:
                        walk.complete = False
                        continue
                    stack.append((Path(child.path), level + 1))
                    continue
                if not _matches(name, wanted, extensions):
                    continue
                try:
                    stat = child.stat(follow_symlinks=False)
                except OSError:
                    continue
                if since is not None and stat.st_mtime < since:
                    continue
                walk.found.append(Found(Path(child.path), label, stat.st_mtime, stat.st_size))
    return _newest(walk, limit)


def folder_of(path: Path, folders: Mapping[str, Path]) -> str | None:
    """The label of the known folder `path` is really inside - after `..`
    and links are resolved - or `None`."""
    try:
        real = path.resolve(strict=False)
    except OSError:
        return None
    for label, root in folders.items():
        if real.is_relative_to(root.resolve(strict=False)):
            return label
    return None


def openable(path: Path) -> bool:
    return path.suffix.casefold() in OPENABLE


def _matches(name: str, wanted: list[str], extensions: frozenset[str] | None) -> bool:
    if extensions is not None and Path(name).suffix.casefold() not in extensions:
        return False
    folded = normalize_search(name)
    return all(word in folded for word in wanted)


def _newest(walk: Walk, limit: int) -> Walk:
    walk.found.sort(key=lambda found: found.modified, reverse=True)
    del walk.found[limit:]
    return walk


class _Guid(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _known_folder(text: str) -> Path | None:
    value = uuid.UUID(text)
    guid = _Guid(value.fields[0], value.fields[1], value.fields[2])
    for index, byte in enumerate(value.bytes[8:]):
        guid.Data4[index] = byte
    path = ctypes.c_wchar_p()
    result = ctypes.windll.shell32.SHGetKnownFolderPath(
        ctypes.byref(guid), 0, None, ctypes.byref(path)
    )
    try:
        return Path(path.value) if result == 0 and path.value else None
    finally:
        ctypes.windll.ole32.CoTaskMemFree(path)
