"""What an audit row may still say after a month (design.md section 3.7,
phase 4.6; 17 Sep 2026).

`tool_audit` is written before every tool runs and its rows are never
deleted: the row is what makes the repeat check of section 3.11 work, and a
table that forgets its rows forgets that. (It also fed the recogniser the
names this user had opened before, until 2026-09-22 left the recogniser with
the yes and no of a confirmation window.) But a row carries two things of
different weight. Which tool ran, when, and how it went is bookkeeping;
`result_summary` is the first two hundred characters of what the tool
answered - the top of a web page, the text of a note, a forecast - and
there is no reason for that to sit on a disk for years.

So the summary is blanked and the row stays. Once at every start, before
anything reads the table, every row older than `[retention] audit_days`
loses its summary. The arguments stay: the repeat check reads them, and they
are what the user asked for rather than what the world answered back. Notes and
reminders are never touched here - the user wrote those down to keep them, and
`live-assistant purge --all` is the door for them.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable

__all__ = ["AUDIT_DAYS", "SECONDS_PER_DAY", "blank_old_audit_summaries"]

# The default of `[retention] audit_days`: long enough to answer "what did
# it say last week", short enough that a page read in spring is not on the
# disk in autumn.
AUDIT_DAYS = 30
SECONDS_PER_DAY = 86_400


def blank_old_audit_summaries(
    connection: sqlite3.Connection,
    *,
    days: int = AUDIT_DAYS,
    now: Callable[[], float] = time.time,
) -> int:
    """Blanks `result_summary` on every audit row older than `days`, and
    returns how many rows lost theirs.

    Zero days means never: the setting is a way to keep every summary, not
    a way to keep none. A row already blanked is not counted again, so a
    second call in the same minute returns zero.
    """
    if days <= 0:
        return 0
    cutoff = int(now()) - days * SECONDS_PER_DAY
    with connection:
        cursor = connection.execute(
            "UPDATE tool_audit SET result_summary = NULL"
            " WHERE ts < ? AND result_summary IS NOT NULL",
            (cutoff,),
        )
    return int(cursor.rowcount)
