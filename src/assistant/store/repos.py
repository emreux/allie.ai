"""What is written to the database and how, one class per table (section 3.7).

A repository is the only code that knows a table's columns; everything else
calls a method named after what happened. `AuditRepo` is the `tool_audit`
table of 2.1d; `UsageRepo` is the `usage_log` of 2.4, where every turn's
tokens go with their price; `SettingsRepo` is the `settings` of 2.6, one
value per key, where the probe keeps its verdict on a model.

**A call is written down before it runs.** `start` opens the row with
`status = 'started'` and no `finished_at`; `finish` closes it as `ok` or
`error`. A row that stays `started` for ever is what a crash between the two
looks like, and it is kept that way on purpose: it is how the assistant says
"I tried, and I do not know whether it worked" (section 3.11) instead of
guessing. `deny` is the other shape - the gate said no, nothing ran, and one
row says so.

**A call is also written down by what it asked for.** `args_hash` is the
arguments as one line of canonical JSON, hashed: the same request always
reads the same, whatever order the model put the keys in. It is what `recent`
looks a call up by, so the gate can say "you already did this a minute ago"
(section 3.11), and what the repeat check of `agent/limits.py` counts by -
same name, same hash, same call. Never by the call's id, which Gemini leaves
empty. What it does not do is read meaning: `{"to": "a@x.com"}` and
`{"to": "A@x.com"}` are two calls, and the design accepts a missed repeat
over a reported one that never happened.

Every write is one transaction (`with connection:`), and every time is a Unix
epoch second in UTC (section 3.10): local time is for the screen, never for
the file.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from assistant.live.base import ToolCall, Usage
from assistant.store.normalize import normalize_search

__all__ = [
    "HASH_CHARS",
    "MIN_QUERY_CHARS",
    "SEARCH_LIMIT",
    "SUMMARY_CHARS",
    "AuditRepo",
    "EarlierCall",
    "ModelUsage",
    "Note",
    "NotesRepo",
    "Outcome",
    "Reminder",
    "ReminderRepo",
    "ReminderStatus",
    "SettingsRepo",
    "UsageRepo",
    "args_hash",
]

Outcome = Literal["ok", "error"]

# How much of a tool's result is kept. Enough to tell what happened, not the
# whole of a web page; section 3.7 empties even this after thirty days.
SUMMARY_CHARS = 200

# How much of the SHA-256 is kept: sixty-four bits, enough that two different
# requests do not collide by accident and short enough to read in a row.
HASH_CHARS = 16


@dataclass(frozen=True, slots=True)
class EarlierCall:
    """The last time the same call ran, or may have: what became of it, and
    how many seconds ago. What the gate reads; the columns stay here."""

    status: Literal["ok", "started"]
    ago: int


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """What one model was used for over a period, as `live-assistant cost` shows it.

    `cost_usd` is the sum over the turns that have a price, and `None` when
    none has; `unpriced` counts the turns that have none, so that the report
    can say so instead of showing a zero.
    """

    provider: str
    model: str
    turns: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float | None
    unpriced: int


class AuditRepo:
    """The `tool_audit` table: one row per tool call, in two phases.

    `clock` is `time.time` unless a test hands over one it controls.
    """

    def __init__(
        self, connection: sqlite3.Connection, *, clock: Callable[[], float] = time.time
    ) -> None:
        self._connection = connection
        self._clock = clock

    def start(self, call: ToolCall, *, turn_id: str, risk: str) -> int:
        """Opens the row for a call that is about to run, and returns its id."""
        with self._connection:
            row = self._connection.execute(
                "INSERT INTO tool_audit"
                " (ts, turn_id, tool, args_json, args_hash, risk, approved, status)"
                " VALUES (?, ?, ?, ?, ?, ?, 1, 'started') RETURNING id",
                (
                    self._now(),
                    turn_id,
                    call.name,
                    _arguments(call.arguments),
                    args_hash(call.arguments),
                    risk,
                ),
            ).fetchone()
        return int(row[0])

    def finish(
        self,
        row_id: int,
        *,
        status: Outcome,
        summary: str | None = None,
        error: str | None = None,
    ) -> None:
        """Closes the row `start` opened: how it went, and when it ended."""
        with self._connection:
            self._connection.execute(
                "UPDATE tool_audit SET status = ?, result_summary = ?, error = ?, finished_at = ?"
                " WHERE id = ?",
                (
                    status,
                    None if summary is None else summary[:SUMMARY_CHARS],
                    error,
                    self._now(),
                    row_id,
                ),
            )

    def deny(self, call: ToolCall, *, turn_id: str, risk: str) -> int:
        """One row for a call the gate refused. Nothing ran, so the row is
        finished as it is written."""
        now = self._now()
        with self._connection:
            row = self._connection.execute(
                "INSERT INTO tool_audit"
                " (ts, turn_id, tool, args_json, args_hash, risk, approved, status, finished_at)"
                " VALUES (?, ?, ?, ?, ?, ?, 0, 'denied', ?) RETURNING id",
                (
                    now,
                    turn_id,
                    call.name,
                    _arguments(call.arguments),
                    args_hash(call.arguments),
                    risk,
                    now,
                ),
            ).fetchone()
        return int(row[0])

    def recent(self, call: ToolCall, *, within: float) -> EarlierCall | None:
        """The last time this same call ran, or may have, in the last `within` seconds.

        The query of section 3.11: the same tool with the same arguments,
        and only the rows that mean something happened - `ok`, or `started`
        and never finished. A `denied` or `error` row is a call that did
        nothing, and is not an earlier time the user did this.
        """
        now = self._now()
        row = self._connection.execute(
            "SELECT status, ts FROM tool_audit"
            " WHERE tool = ? AND args_hash = ? AND status IN ('ok', 'started') AND ts > ?"
            " ORDER BY ts DESC LIMIT 1",
            (call.name, args_hash(call.arguments), now - within),
        ).fetchone()
        if row is None:
            return None
        return EarlierCall(status=row["status"], ago=max(now - int(row["ts"]), 0))

    def _now(self) -> int:
        return int(self._clock())


class UsageRepo:
    """The `usage_log` table: one row per turn that reached the model (section 6)."""

    def __init__(
        self, connection: sqlite3.Connection, *, clock: Callable[[], float] = time.time
    ) -> None:
        self._connection = connection
        self._clock = clock

    def insert(
        self,
        *,
        turn_id: str,
        provider: str,
        model: str,
        usage: Usage,
        cost_usd: float | None,
    ) -> int:
        """One turn: what it used and what that cost, `None` when the price
        is not known. The row is the receipt; nothing recomputes it later."""
        with self._connection:
            row = self._connection.execute(
                "INSERT INTO usage_log"
                " (ts, provider, model, in_tokens, out_tokens, cached_tokens, cost_usd, turn_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
                (
                    int(self._clock()),
                    provider,
                    model,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cached_tokens,
                    cost_usd,
                    turn_id,
                ),
            ).fetchone()
        return int(row[0])

    def sum_since(self, since: float) -> float:
        """Dollars over the turns from `since` on that have a price.

        A turn with no price adds nothing, which understates: the report
        says how many such turns there were, and the warning of section
        3.11 can only be as right as the prices it was given.
        """
        row = self._connection.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM usage_log WHERE ts >= ?", (since,)
        ).fetchone()
        return float(row[0])

    def by_model_since(self, since: float) -> list[ModelUsage]:
        """The turns from `since` on, added up per model, in a settled order."""
        rows = self._connection.execute(
            "SELECT provider, model, COUNT(*) AS turns,"
            " SUM(in_tokens) AS input_tokens, SUM(out_tokens) AS output_tokens,"
            " SUM(cached_tokens) AS cached_tokens, SUM(cost_usd) AS cost_usd,"
            " SUM(cost_usd IS NULL) AS unpriced"
            " FROM usage_log WHERE ts >= ? GROUP BY provider, model ORDER BY provider, model",
            (since,),
        ).fetchall()
        return [
            ModelUsage(
                provider=row["provider"],
                model=row["model"],
                turns=int(row["turns"]),
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                cached_tokens=int(row["cached_tokens"]),
                cost_usd=None if row["cost_usd"] is None else float(row["cost_usd"]),
                unpriced=int(row["unpriced"]),
            )
            for row in rows
        ]


class SettingsRepo:
    """The `settings` table: one value per key, kept between runs (2.6).

    What the program found out for itself and does not want to find out
    again on every start - not what the user chose, which is `config.toml`.
    The value is text and means whatever the writer meant by it; the probe
    writes JSON, and reads it back itself (`live/probe.py`).
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, key: str) -> str | None:
        """The value under `key`, or `None` if nothing was ever written there."""
        row = self._connection.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row["value"])

    def set(self, key: str, value: str) -> None:
        """Writes `value` under `key`, over whatever was there."""
        with self._connection:
            self._connection.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )


# A trigram index matches nothing shorter than three characters, so a query
# word shorter than that is not a query (section 3.7, phase 4.1).
MIN_QUERY_CHARS = 3

# How many notes a search answers with. Five is what can be read out loud.
SEARCH_LIMIT = 5

_QUERY_WORD = re.compile(r"\w+")


@dataclass(frozen=True, slots=True)
class Note:
    """One note as the user said it, with when it was said (UTC epoch)."""

    id: int
    text: str
    created_at: int


class NotesRepo:
    """The `notes` table and its FTS5 index (section 3.7, phase 4.1).

    Text is kept as said and indexed folded (`store/normalize.py`), so that
    `Işık`, `IŞIK` and `isik` are one word to the search and the note still
    reads back the way it was written. A search is the query folded the
    same way, each word of it a trigram phrase, all of them required.
    """

    def __init__(
        self, connection: sqlite3.Connection, *, clock: Callable[[], float] = time.time
    ) -> None:
        self._connection = connection
        self._clock = clock

    def add(self, text: str) -> Note:
        """Keeps `text` as it is and returns the note it became."""
        now = int(self._clock())
        with self._connection:
            row = self._connection.execute(
                "INSERT INTO notes (text, text_norm, created_at) VALUES (?, ?, ?) RETURNING id",
                (text, normalize_search(text), now),
            ).fetchone()
        return Note(id=int(row[0]), text=text, created_at=now)

    def get(self, note_id: int) -> Note | None:
        row = self._connection.execute(
            "SELECT id, text, created_at FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        return None if row is None else _note(row)

    def delete(self, note_id: int) -> bool:
        """Removes the note; `False` when there was none to remove."""
        with self._connection:
            cursor = self._connection.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        return cursor.rowcount > 0

    def count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM notes").fetchone()
        return int(row[0])

    def latest(self, limit: int = SEARCH_LIMIT) -> list[Note]:
        """The most recent notes, newest first."""
        rows = self._connection.execute(
            "SELECT id, text, created_at FROM notes ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_note(row) for row in rows]

    def search(self, query: str, *, limit: int = SEARCH_LIMIT) -> list[Note]:
        """The notes that contain every usable word of `query`, best match first.

        A word is usable when it is at least `MIN_QUERY_CHARS` long once
        folded; a query with none raises `ValueError`, since the index
        cannot answer it and an empty list would read as "no such note".
        """
        match = query_terms(query)
        if not match:
            raise ValueError(f"a query needs a word of at least {MIN_QUERY_CHARS} characters")
        rows = self._connection.execute(
            "SELECT n.id, n.text, n.created_at FROM notes_fts"
            " JOIN notes AS n ON n.id = notes_fts.rowid"
            " WHERE notes_fts MATCH ? ORDER BY bm25(notes_fts), n.created_at DESC LIMIT ?",
            (match, limit),
        ).fetchall()
        return [_note(row) for row in rows]


ReminderStatus = Literal["pending", "fired", "missed", "cancelled"]


@dataclass(frozen=True, slots=True)
class Reminder:
    """One reminder: what to say, when (UTC epoch), how it repeats, and
    where it stands. `rrule` is `None` for one that fires once."""

    id: int
    text: str
    fire_at: int
    rrule: str | None
    status: ReminderStatus
    created_at: int
    fired_at: int | None = None
    missed_by_sec: int | None = None


class ReminderRepo:
    """The `reminders` table (section 3.10): the source of truth the
    scheduler polls every twenty seconds and the model never sees."""

    def __init__(
        self, connection: sqlite3.Connection, *, clock: Callable[[], float] = time.time
    ) -> None:
        self._connection = connection
        self._clock = clock

    def add(self, text: str, *, fire_at: int, rrule: str | None = None) -> Reminder:
        now = int(self._clock())
        with self._connection:
            row = self._connection.execute(
                "INSERT INTO reminders (text, fire_at, rrule, status, created_at)"
                " VALUES (?, ?, ?, 'pending', ?) RETURNING id",
                (text, fire_at, rrule, now),
            ).fetchone()
        return Reminder(
            id=int(row[0]),
            text=text,
            fire_at=fire_at,
            rrule=rrule,
            status="pending",
            created_at=now,
        )

    def get(self, reminder_id: int) -> Reminder | None:
        row = self._connection.execute(
            "SELECT * FROM reminders WHERE id = ?", (reminder_id,)
        ).fetchone()
        return None if row is None else _reminder(row)

    def pending(self, *, limit: int = 20) -> list[Reminder]:
        """What is still to come, soonest first."""
        rows = self._connection.execute(
            "SELECT * FROM reminders WHERE status = 'pending' ORDER BY fire_at, id LIMIT ?",
            (limit,),
        ).fetchall()
        return [_reminder(row) for row in rows]

    def due(self, now: float) -> list[Reminder]:
        """Every pending reminder whose time has come, earliest first."""
        rows = self._connection.execute(
            "SELECT * FROM reminders WHERE status = 'pending' AND fire_at <= ?"
            " ORDER BY fire_at, id",
            (int(now),),
        ).fetchall()
        return [_reminder(row) for row in rows]

    def close(self, reminder_id: int, *, status: ReminderStatus, at: int, late_by: int) -> None:
        """Ends a one-off reminder as `fired` or `missed`, and says how late."""
        with self._connection:
            self._connection.execute(
                "UPDATE reminders SET status = ?, fired_at = ?, missed_by_sec = ? WHERE id = ?",
                (status, at, late_by, reminder_id),
            )

    def advance(self, reminder_id: int, *, fire_at: int, at: int, late_by: int) -> None:
        """Moves a repeating reminder on to its next time, noting this one."""
        with self._connection:
            self._connection.execute(
                "UPDATE reminders SET fire_at = ?, fired_at = ?, missed_by_sec = ? WHERE id = ?",
                (fire_at, at, late_by, reminder_id),
            )

    def cancel(self, reminder_id: int) -> bool:
        """Cancels a pending reminder; `False` when there was none to cancel."""
        with self._connection:
            cursor = self._connection.execute(
                "UPDATE reminders SET status = 'cancelled' WHERE id = ? AND status = 'pending'",
                (reminder_id,),
            )
        return cursor.rowcount > 0


def _reminder(row: sqlite3.Row) -> Reminder:
    return Reminder(
        id=int(row["id"]),
        text=str(row["text"]),
        fire_at=int(row["fire_at"]),
        rrule=None if row["rrule"] is None else str(row["rrule"]),
        status=row["status"],
        created_at=int(row["created_at"]),
        fired_at=None if row["fired_at"] is None else int(row["fired_at"]),
        missed_by_sec=None if row["missed_by_sec"] is None else int(row["missed_by_sec"]),
    )


def query_terms(query: str) -> str:
    """`query` as an FTS5 match expression: each usable word folded and
    quoted as a phrase, all of them required; empty when no word is usable."""
    words = [
        word
        for word in _QUERY_WORD.findall(normalize_search(query))
        if len(word) >= MIN_QUERY_CHARS
    ]
    return " AND ".join(f'"{word}"' for word in words)


def _note(row: sqlite3.Row) -> Note:
    return Note(id=int(row["id"]), text=str(row["text"]), created_at=int(row["created_at"]))


def args_hash(arguments: Mapping[str, Any]) -> str:
    """The arguments as a short fingerprint that does not depend on their order.

    The first sixteen hex characters of the SHA-256 of the canonical JSON
    below (section 3.11). Two calls with the same name and the same hash are
    the same call, to the gate and to the repeat check alike.
    """
    return hashlib.sha256(_arguments(arguments).encode("utf-8")).hexdigest()[:HASH_CHARS]


def _arguments(arguments: Mapping[str, Any]) -> str:
    """The arguments as one line of JSON, keys sorted, nothing escaped that
    the reader would rather see as it is."""
    return json.dumps(dict(arguments), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
