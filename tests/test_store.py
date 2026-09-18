"""The database opens itself, builds itself, and writes what it is told (2.1d).

Three claims. Opening a database brings its schema to the current version
and does so only once - the second start of the program changes nothing. A
migration is all or nothing, because a file left between two versions is a
file no build can read. And the audit repository writes rows whose shape
section 3.7 fixes: epoch seconds in UTC, arguments as one line of JSON, a
`started` row that has no end until something ends it.

The second migration (2.4) is the first real one: a database at version 1
with rows in it comes up to version 2 with the rows intact and a column
beside them. With it come the fingerprint of a call's arguments, the query
that answers "did this run a moment ago", and the `usage_log` the bill is
kept in. The third (2.6) is the `settings` table: one value per key, where
the probe keeps what it found out about a model.

Every test opens its own database, in memory or under `tmp_path`; the one
under `%LOCALAPPDATA%` is never touched.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from assistant.config import data_dir
from assistant.live.base import ToolCall, Usage
from assistant.store import migrations
from assistant.store.db import DB_FILE, database_path, open_database
from assistant.store.migrations import MIGRATIONS, migrate, schema_version
from assistant.store.repos import (
    SUMMARY_CHARS,
    AuditRepo,
    EarlierCall,
    SettingsRepo,
    UsageRepo,
    args_hash,
)

CALL = ToolCall(id="c1", name="open_app", arguments={"name": "Spotify", "args": "ş"})


@pytest.fixture
def database() -> Iterator[sqlite3.Connection]:
    connection = open_database(":memory:")
    yield connection
    connection.close()


def tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row["name"]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def indexes(connection: sqlite3.Connection) -> set[str]:
    return {
        row["name"]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
    }


def rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute("SELECT * FROM tool_audit ORDER BY id")]


# --------------------------------------------------------------------------
# Opening and migrating
# --------------------------------------------------------------------------


def test_the_first_run_builds_the_schema(database: sqlite3.Connection) -> None:
    assert "tool_audit" in tables(database)
    assert "tool_audit_turn" in indexes(database)
    assert schema_version(database) == len(MIGRATIONS)


def test_the_second_run_finds_nothing_to_do(database: sqlite3.Connection) -> None:
    """Every start of the program calls `migrate`; a migration that ran twice
    would fail on the table it already made."""
    assert migrate(database) == len(MIGRATIONS)
    assert schema_version(database) == len(MIGRATIONS)


def test_the_file_is_made_where_it_is_told_with_its_directory(tmp_path: Path) -> None:
    path = tmp_path / "data" / "assistant.db"

    connection = open_database(path)
    try:
        assert path.is_file()
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        connection.close()


def test_a_commit_does_not_wait_for_the_disk(tmp_path: Path) -> None:
    """`synchronous=NORMAL`, SQLite's own recommendation under WAL: the row
    survives a crash of the program, which is the crash it is written for,
    and a write on the event loop costs a tenth of a millisecond instead of
    six (measured 2026-09-09). The setting is per connection, so it is
    checked on one that was just opened."""
    connection = open_database(tmp_path / "assistant.db")
    try:
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 1
    finally:
        connection.close()


def test_the_default_path_is_beside_the_log_not_beside_the_settings() -> None:
    """`%LOCALAPPDATA%`, not `%APPDATA%`: what a machine did stays on it."""
    assert database_path() == data_dir() / DB_FILE


def test_columns_are_read_by_name(database: sqlite3.Connection) -> None:
    row = database.execute("SELECT 1 AS one").fetchone()

    assert row["one"] == 1


def test_a_migration_that_fails_halfway_leaves_nothing_of_itself(
    database: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The version is bumped in the same transaction as the schema change,
    so a file is always at a whole number of migrations."""
    monkeypatch.setattr(
        migrations,
        "MIGRATIONS",
        (*MIGRATIONS, "CREATE TABLE half (a);\nTHIS IS NOT SQL;"),
    )

    with pytest.raises(sqlite3.Error):
        migrate(database)

    assert schema_version(database) == len(MIGRATIONS)
    assert "half" not in tables(database)
    assert not database.in_transaction


def test_a_database_from_a_newer_build_is_left_as_it_is(database: sqlite3.Connection) -> None:
    """The list is append-only, so what this build needs is already there."""
    database.execute("PRAGMA user_version = 40")

    assert migrate(database) == 40


# --------------------------------------------------------------------------
# The audit repository
# --------------------------------------------------------------------------


@pytest.fixture
def audit(database: sqlite3.Connection) -> AuditRepo:
    return AuditRepo(database, clock=lambda: 1_700_000_000.9)


def test_a_started_call_has_a_beginning_and_no_end(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    row_id = audit.start(CALL, turn_id="t1", risk="confirm")

    [row] = rows(database)
    assert row["id"] == row_id
    assert (row["status"], row["approved"], row["finished_at"]) == ("started", 1, None)
    assert (row["turn_id"], row["tool"], row["risk"]) == ("t1", "open_app", "confirm")
    assert row["ts"] == 1_700_000_000, "epoch seconds, whole"


def test_the_arguments_are_one_line_of_json_in_a_settled_order(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    """Keys sorted and nothing escaped: the same call always reads the same,
    and a Turkish letter reads as itself."""
    audit.start(CALL, turn_id="t1", risk="safe")

    assert rows(database)[0]["args_json"] == '{"args":"ş","name":"Spotify"}'


def test_finishing_ok_keeps_the_result_and_the_time(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    row_id = audit.start(CALL, turn_id="t1", risk="safe")

    audit.finish(row_id, status="ok", summary="Spotify opened")

    [row] = rows(database)
    assert (row["status"], row["result_summary"], row["error"]) == ("ok", "Spotify opened", None)
    assert row["finished_at"] == 1_700_000_000


def test_finishing_in_error_keeps_the_kind_of_error(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    row_id = audit.start(CALL, turn_id="t1", risk="safe")

    audit.finish(row_id, status="error", error="TimeoutError")

    [row] = rows(database)
    assert (row["status"], row["error"], row["result_summary"]) == ("error", "TimeoutError", None)


def test_a_long_result_is_kept_only_up_to_the_summary_length(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    """A web page is not an audit entry; section 3.7 empties even this later."""
    row_id = audit.start(CALL, turn_id="t1", risk="safe")

    audit.finish(row_id, status="ok", summary="x" * (SUMMARY_CHARS * 3))

    assert len(rows(database)[0]["result_summary"]) == SUMMARY_CHARS


def test_a_denied_call_is_finished_as_it_is_written(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    audit.deny(CALL, turn_id="t1", risk="blocked")

    [row] = rows(database)
    assert (row["status"], row["approved"], row["risk"]) == ("denied", 0, "blocked")
    assert row["finished_at"] == row["ts"] == 1_700_000_000


def test_the_names_asked_of_a_tool_are_the_distinct_ones_most_recent_first(
    database: sqlite3.Connection,
) -> None:
    """What this user actually asks to open (2026-09-13): the recogniser is
    told those names before any other. Only calls that ran; each name once,
    by its latest use; other tools' names are not app names."""
    tick = iter(range(1_700_000_000, 1_700_000_100))
    audit = AuditRepo(database, clock=lambda: next(tick))

    def ran(tool: str, name: str, *, status: str = "ok") -> None:
        row = audit.start(
            ToolCall(id="c", name=tool, arguments={"name": name}), turn_id="t", risk="safe"
        )
        audit.finish(row, status=status)  # type: ignore[arg-type]

    ran("open_app", "PyCharm")
    ran("open_app", "Teams")
    ran("open_url", "example.com")
    ran("open_app", "Chrome", status="error")
    ran("open_app", "PyCharm")
    audit.deny(
        ToolCall(id="d", name="open_app", arguments={"name": "Denied"}), turn_id="t", risk="confirm"
    )

    assert audit.names_asked("open_app", limit=10) == ["PyCharm", "Teams"]
    assert audit.names_asked("open_app", limit=1) == ["PyCharm"]
    assert audit.names_asked("play_music", limit=10) == []


def test_each_row_gets_its_own_id(database: sqlite3.Connection, audit: AuditRepo) -> None:
    first = audit.start(CALL, turn_id="t1", risk="safe")
    second = audit.deny(CALL, turn_id="t1", risk="safe")

    assert first != second
    assert [row["id"] for row in rows(database)] == [first, second]


def test_the_clock_is_the_machine_s_unless_a_test_says_otherwise(
    database: sqlite3.Connection,
) -> None:
    """The default clock is `time.time`: epoch seconds, UTC, the same for
    every row and every reader (section 3.10)."""
    AuditRepo(database).start(CALL, turn_id="t1", risk="safe")

    assert rows(database)[0]["ts"] > 1_700_000_000


# --------------------------------------------------------------------------
# The second migration (2.4)
# --------------------------------------------------------------------------


def test_the_second_migration_adds_the_bill_and_the_fingerprint(
    database: sqlite3.Connection,
) -> None:
    columns = {row["name"] for row in database.execute("PRAGMA table_info(tool_audit)")}

    assert "usage_log" in tables(database)
    assert {"tool_audit_repeat", "usage_log_ts"} <= indexes(database)
    assert "args_hash" in columns


def test_a_database_from_before_the_second_migration_comes_up_with_its_rows() -> None:
    """The owner's database was at version 1 with rows in it on 2026-09-09.
    Those rows keep everything they had and get no fingerprint - never "the
    same call" again, which is the honest reading of them."""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(f"BEGIN;\n{MIGRATIONS[0]}\nPRAGMA user_version = 1;\nCOMMIT;")
        connection.execute(
            "INSERT INTO tool_audit (ts, turn_id, tool, args_json, risk, approved, status)"
            " VALUES (1, 't1', 'clock', '{}', 'safe', 1, 'ok')"
        )
        connection.commit()

        assert migrate(connection) == len(MIGRATIONS)

        [old] = rows(connection)
        assert (old["tool"], old["status"], old["args_hash"]) == ("clock", "ok", None)
        AuditRepo(connection).start(CALL, turn_id="t2", risk="safe")
        assert rows(connection)[1]["args_hash"] == args_hash(CALL.arguments)
    finally:
        connection.close()


# --------------------------------------------------------------------------
# The third migration (2.6): settings
# --------------------------------------------------------------------------


def test_the_third_migration_adds_the_settings_table(database: sqlite3.Connection) -> None:
    columns = {row["name"] for row in database.execute("PRAGMA table_info(settings)")}

    assert "settings" in tables(database)
    assert columns == {"key", "value"}


def test_a_database_from_before_the_third_migration_comes_up_with_its_rows() -> None:
    """The owner's database was at version 2 with turns in it on 2026-09-10;
    they are all still there afterwards, and the new table is empty."""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        for number, script in enumerate(MIGRATIONS[:2], start=1):
            connection.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {number};\nCOMMIT;")
        UsageRepo(connection).insert(
            turn_id="t1", provider="gemini", model="x", usage=Usage(1, 2), cost_usd=None
        )

        assert migrate(connection) == len(MIGRATIONS)

        assert len(UsageRepo(connection).by_model_since(0)) == 1
        assert SettingsRepo(connection).get("probe:gemini:x") is None
    finally:
        connection.close()


def test_nothing_written_reads_as_none(database: sqlite3.Connection) -> None:
    assert SettingsRepo(database).get("anything") is None


def test_a_value_written_is_read_back_under_its_key(database: sqlite3.Connection) -> None:
    settings = SettingsRepo(database)

    settings.set("probe:gemini:x", '{"ok": true}')

    assert settings.get("probe:gemini:x") == '{"ok": true}'
    assert settings.get("probe:gemini:y") is None


def test_writing_a_key_again_replaces_the_value(database: sqlite3.Connection) -> None:
    """One value per key: a verdict refreshed a week later is the verdict,
    not a second row beside the first."""
    settings = SettingsRepo(database)

    settings.set("k", "old")
    settings.set("k", "new")

    assert settings.get("k") == "new"
    assert database.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 1


def test_a_value_survives_the_connection(tmp_path: Path) -> None:
    path = tmp_path / "assistant.db"
    first = open_database(path)
    SettingsRepo(first).set("k", "ş")
    first.close()

    second = open_database(path)
    try:
        assert SettingsRepo(second).get("k") == "ş"
    finally:
        second.close()


def test_the_fingerprint_does_not_depend_on_the_order_of_the_arguments() -> None:
    """The model may send the keys in any order; the call is the same call."""
    assert args_hash({"a": 1, "b": "ş"}) == args_hash({"b": "ş", "a": 1})


def test_the_fingerprint_is_sixteen_hex_characters_and_tells_calls_apart() -> None:
    assert len(args_hash({"name": "Spotify"})) == 16
    assert set(args_hash({"name": "Spotify"})) <= set("0123456789abcdef")
    assert args_hash({"name": "Spotify"}) != args_hash({"name": "spotify"})


def test_every_row_carries_the_fingerprint_of_its_arguments(
    database: sqlite3.Connection, audit: AuditRepo
) -> None:
    audit.start(CALL, turn_id="t1", risk="safe")
    audit.deny(CALL, turn_id="t1", risk="confirm")

    assert [row["args_hash"] for row in rows(database)] == [args_hash(CALL.arguments)] * 2


# --------------------------------------------------------------------------
# A moment ago (section 3.11)
# --------------------------------------------------------------------------


class Clock:
    """A clock a test can move."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock(1_700_000_000)


@pytest.fixture
def ticking(database: sqlite3.Connection, clock: Clock) -> AuditRepo:
    return AuditRepo(database, clock=clock)


def test_nothing_earlier_is_nothing(ticking: AuditRepo) -> None:
    assert ticking.recent(CALL, within=600) is None


def test_a_call_that_ran_is_found_with_how_long_ago(ticking: AuditRepo, clock: Clock) -> None:
    row = ticking.start(CALL, turn_id="t1", risk="confirm")
    ticking.finish(row, status="ok", summary="opened")
    clock.now += 40

    assert ticking.recent(CALL, within=600) == EarlierCall(status="ok", ago=40)


def test_a_call_still_started_is_found_as_such(ticking: AuditRepo, clock: Clock) -> None:
    """The crash between the write and the finish: it may have happened."""
    ticking.start(CALL, turn_id="t1", risk="confirm")
    clock.now += 40

    assert ticking.recent(CALL, within=600) == EarlierCall(status="started", ago=40)


def test_a_denied_or_failed_call_is_not_something_that_happened(
    ticking: AuditRepo, clock: Clock
) -> None:
    ticking.deny(CALL, turn_id="t1", risk="confirm")
    row = ticking.start(CALL, turn_id="t2", risk="confirm")
    ticking.finish(row, status="error", error="RuntimeError")
    clock.now += 40

    assert ticking.recent(CALL, within=600) is None


def test_a_call_from_before_the_window_is_not_found(ticking: AuditRepo, clock: Clock) -> None:
    row = ticking.start(CALL, turn_id="t1", risk="confirm")
    ticking.finish(row, status="ok")
    clock.now += 601

    assert ticking.recent(CALL, within=600) is None


def test_the_latest_of_several_is_the_one_found(ticking: AuditRepo, clock: Clock) -> None:
    row = ticking.start(CALL, turn_id="t1", risk="confirm")
    ticking.finish(row, status="ok")
    clock.now += 100
    ticking.start(CALL, turn_id="t2", risk="confirm")
    clock.now += 20

    assert ticking.recent(CALL, within=600) == EarlierCall(status="started", ago=20)


def test_another_tool_or_other_arguments_are_another_call(ticking: AuditRepo) -> None:
    row = ticking.start(CALL, turn_id="t1", risk="confirm")
    ticking.finish(row, status="ok")
    other_tool = ToolCall(id="c2", name="open_url", arguments=CALL.arguments)
    other_arguments = ToolCall(id="c3", name="open_app", arguments={"name": "Chrome"})

    assert ticking.recent(other_tool, within=600) is None
    assert ticking.recent(other_arguments, within=600) is None


# --------------------------------------------------------------------------
# The bill (2.4)
# --------------------------------------------------------------------------


@pytest.fixture
def usage(database: sqlite3.Connection, clock: Clock) -> UsageRepo:
    return UsageRepo(database, clock=clock)


def spend(usage: UsageRepo, turn_id: str, cost: float | None, model: str = "m") -> None:
    usage.insert(turn_id=turn_id, provider="gemini", model=model, usage=Usage(), cost_usd=cost)


def test_a_turn_is_one_row_with_its_tokens_and_its_price(
    database: sqlite3.Connection, usage: UsageRepo
) -> None:
    usage.insert(
        turn_id="t1", provider="gemini", model="m", usage=Usage(300, 10, 5), cost_usd=0.0004
    )

    [row] = database.execute("SELECT * FROM usage_log").fetchall()
    assert (row["ts"], row["provider"], row["model"]) == (1_700_000_000, "gemini", "m")
    assert (row["in_tokens"], row["out_tokens"], row["cached_tokens"]) == (300, 10, 5)
    assert (row["cost_usd"], row["turn_id"]) == (pytest.approx(0.0004), "t1")


def test_a_turn_with_no_known_price_is_written_with_none_and_not_zero(
    database: sqlite3.Connection, usage: UsageRepo
) -> None:
    spend(usage, "t1", None)

    assert database.execute("SELECT cost_usd FROM usage_log").fetchone()[0] is None


def test_the_sum_counts_the_rows_from_the_moment_on_and_skips_the_unpriced(
    usage: UsageRepo, clock: Clock
) -> None:
    spend(usage, "t1", 0.5)
    clock.now += 100
    spend(usage, "t2", 0.25)
    spend(usage, "t3", None)

    assert usage.sum_since(1_700_000_100) == pytest.approx(0.25)
    assert usage.sum_since(0) == pytest.approx(0.75)


def test_an_empty_table_sums_to_nothing(usage: UsageRepo) -> None:
    assert usage.sum_since(0) == 0.0


def test_the_report_adds_the_turns_up_per_model_in_a_settled_order(usage: UsageRepo) -> None:
    usage.insert(turn_id="t1", provider="gemini", model="b", usage=Usage(100, 10, 5), cost_usd=0.1)
    usage.insert(turn_id="t2", provider="gemini", model="b", usage=Usage(200, 20, 0), cost_usd=None)
    usage.insert(turn_id="t3", provider="gemini", model="a", usage=Usage(1, 1, 0), cost_usd=0.01)

    [first, second] = usage.by_model_since(0)

    assert (first.provider, first.model, first.turns, first.cost_usd) == ("gemini", "a", 1, 0.01)
    assert (second.model, second.turns, second.unpriced) == ("b", 2, 1)
    assert (second.input_tokens, second.output_tokens, second.cached_tokens) == (300, 30, 5)
    assert second.cost_usd == pytest.approx(0.1)


def test_a_model_none_of_whose_turns_has_a_price_reports_no_cost(usage: UsageRepo) -> None:
    spend(usage, "t1", None)

    [only] = usage.by_model_since(0)
    assert (only.cost_usd, only.unpriced) == (None, 1)


def test_the_report_of_an_empty_table_is_empty(usage: UsageRepo) -> None:
    assert usage.by_model_since(0) == []
