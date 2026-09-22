"""What an audit row still says after a month (`store/retention.py`, 4.6).

The claim is narrow: after `[retention] audit_days` a row keeps everything
but its `result_summary`. The row stays, because the repeat check of
section 3.11 and the recogniser's list of what this user opens read it;
the arguments and the error stay for the same reason; and nothing in the
notes or the reminders is looked at, because those the user wrote to keep.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from allie.config import (
    RetentionSettings,
    Settings,
    config_path,
    load_settings,
    save_settings,
)
from allie.live.base import ToolCall
from allie.store.db import open_database
from allie.store.repos import AuditRepo, NotesRepo
from allie.store.retention import AUDIT_DAYS, SECONDS_PER_DAY, blank_old_audit_summaries

NOW = 1_800_000_000.0
A_PAGE = "Title: Weather - Rain until Thursday, then sun."


@pytest.fixture
def database() -> Iterator[sqlite3.Connection]:
    connection = open_database(":memory:")
    try:
        yield connection
    finally:
        connection.close()


def ran(
    connection: sqlite3.Connection,
    *,
    days_ago: float,
    summary: str | None = A_PAGE,
    error: str | None = None,
) -> int:
    """One finished audit row, `days_ago` days before `NOW`; returns its id."""
    audit = AuditRepo(connection, clock=lambda: NOW - days_ago * SECONDS_PER_DAY)
    call = ToolCall(id="c1", name="fetch_page", arguments={"url": "https://example.test"})
    row_id = audit.start(call, turn_id="turn-1", risk="safe")
    audit.finish(row_id, status="error" if error else "ok", summary=summary, error=error)
    return row_id


def row(connection: sqlite3.Connection, row_id: int) -> sqlite3.Row:
    return connection.execute("SELECT * FROM tool_audit WHERE id = ?", (row_id,)).fetchone()


def blank(connection: sqlite3.Connection, *, days: int = AUDIT_DAYS) -> int:
    return blank_old_audit_summaries(connection, days=days, now=lambda: NOW)


# --------------------------------------------------------------------------
# What goes and what stays
# --------------------------------------------------------------------------


def test_a_summary_older_than_the_window_is_blanked_and_the_row_stays(
    database: sqlite3.Connection,
) -> None:
    old = ran(database, days_ago=31)
    before = dict(row(database, old))

    assert blank(database) == 1

    after = dict(row(database, old))
    assert after["result_summary"] is None
    assert {k: v for k, v in after.items() if k != "result_summary"} == {
        k: v for k, v in before.items() if k != "result_summary"
    }


def test_a_summary_inside_the_window_is_kept(database: sqlite3.Connection) -> None:
    fresh = ran(database, days_ago=29)

    assert blank(database) == 0
    assert row(database, fresh)["result_summary"] == A_PAGE


def test_only_the_summary_goes_the_arguments_and_the_error_stay(
    database: sqlite3.Connection,
) -> None:
    """The arguments are what the user asked for and the repeat check
    reads them; the error is a word about the tool, not about the world."""
    old = ran(database, days_ago=40, summary=None, error="the page could not be read")

    blank(database)

    kept = row(database, old)
    assert kept["error"] == "the page could not be read"
    assert '"https://example.test"' in kept["args_json"]
    assert kept["status"] == "error"


def test_the_window_is_the_users_number(database: sqlite3.Connection) -> None:
    old = ran(database, days_ago=8)

    assert blank(database, days=7) == 1
    assert row(database, old)["result_summary"] is None


def test_zero_days_keeps_every_summary(database: sqlite3.Connection) -> None:
    """Zero is "never", not "everything": a way to keep, not a way to lose."""
    old = ran(database, days_ago=400)

    assert blank(database, days=0) == 0
    assert row(database, old)["result_summary"] == A_PAGE


def test_a_second_pass_finds_nothing_left_to_blank(database: sqlite3.Connection) -> None:
    ran(database, days_ago=31)
    ran(database, days_ago=32)

    assert blank(database) == 2
    assert blank(database) == 0


def test_notes_are_never_touched(database: sqlite3.Connection) -> None:
    notes = NotesRepo(database)
    note = notes.add("the electricity bill is due on the 20th")
    ran(database, days_ago=31)

    blank(database)

    kept = notes.get(note.id)
    assert kept is not None
    assert kept.text == "the electricity bill is due on the 20th"


# --------------------------------------------------------------------------
# The setting
# --------------------------------------------------------------------------


def test_the_default_window_is_thirty_days() -> None:
    assert AUDIT_DAYS == 30
    assert RetentionSettings().audit_days == AUDIT_DAYS


def test_a_negative_window_is_refused() -> None:
    with pytest.raises(ValidationError, match="0 or more days"):
        RetentionSettings(audit_days=-1)


def test_the_window_round_trips_through_the_file(config_home: Path) -> None:
    save_settings(Settings(retention=RetentionSettings(audit_days=7)))

    assert load_settings().retention.audit_days == 7


def test_a_file_written_before_there_was_a_retention_table_keeps_a_month(
    config_home: Path,
) -> None:
    config_path().write_text('[live]\nprimary = "gemini:x"\n', encoding="utf-8")

    assert load_settings().retention.audit_days == AUDIT_DAYS
