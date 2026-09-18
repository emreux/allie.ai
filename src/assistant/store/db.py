"""Opening the database: one file, one connection, migrated on the way in.

`%LOCALAPPDATA%\\live-assistant\\assistant.db` - beside the log and not beside the
settings, for the same reason: what a machine did stays on that machine and
does not follow the user through a roaming profile (section 3.3). Plain
`sqlite3` and no ORM (section 5.11, decision 4 of section 12).

Three settings are made here because they belong to the connection rather
than to any table. WAL journaling lets a reader look while the assistant
writes - the cost report of 2.4, or `sqlite3` in the owner's hand. With it,
`synchronous=NORMAL` is SQLite's own recommendation: a commit no longer waits
for the disk, and what is written survives a crash of this program, which is
the crash the `started` row of section 3.11 exists for; only the power going
out can lose the last few writes. It is what keeps the 50 ms rule of section
3.1, because a `tool_audit` row is written on the event loop before the tool
runs: measured on this machine on 2026-09-09, a start and a finish together
took 5.8 ms typically and 38 ms at worst with the default, and 0.08 ms
typically and 0.6 ms at worst with this. `Row` makes columns readable by
name, so no repository counts positions.

The path is a parameter and not an environment variable. A test passes
`":memory:"` or a path under `tmp_path`; the composition root passes nothing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from assistant.config import data_dir
from assistant.store.migrations import migrate

__all__ = ["DB_FILE", "database_path", "open_database"]

DB_FILE = "assistant.db"


def database_path() -> Path:
    """`%LOCALAPPDATA%\\live-assistant\\assistant.db`."""
    return data_dir() / DB_FILE


def open_database(path: Path | str | None = None) -> sqlite3.Connection:
    """Opens the database - creating it on the first run - with its schema current."""
    target = database_path() if path is None else path
    if isinstance(target, Path):
        target.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(target)
    connection.row_factory = sqlite3.Row
    # Outside any transaction, which is why they are here and not in a
    # migration; and per connection, which is why they are here and not
    # written into the file once.
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    migrate(connection)
    return connection
