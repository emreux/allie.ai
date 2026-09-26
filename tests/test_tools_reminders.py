"""`create_reminder`, `list_reminders`, `cancel_reminder` (design.md phase
4.2; 17 Sep 2026), and the line that tells the model what time it is.

The first two over the real table in memory with a clock the test
holds; the third through the real gate, for the reason `test_tools_notes.py`
gives: cancelling is not done without a yes, and the yes is to the
reminder's own words.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone

import pytest
from dateutil.tz import tzlocal

from allie.agent.core import Confirm
from allie.agent.policy import DECLINED, dispatch
from allie.live.base import ToolCall
from allie.store.db import open_database
from allie.store.repos import ReminderRepo
from allie.tools.registry import Tool, ToolRegistry
from allie.tools.reminders import (
    EMPTY,
    MAX_REMINDER_CHARS,
    NONE_PENDING,
    TEXT,
    cancel_reminder_for,
    create_reminder_for,
    current_time_line,
    list_reminders_for,
)

# Thursday 17 September 2026, 09:00 in the machine's own zone: the tool
# reads and writes local times, so the test speaks local time too.
NOW = datetime(2026, 9, 17, 9, 0, tzinfo=tzlocal())


class FakeConfirm:
    def __init__(self, *, answer: bool) -> None:
        self.answer = answer
        self.asked: list[str] = []

    async def __call__(self, question: str) -> bool:
        self.asked.append(question)
        return self.answer


@pytest.fixture
def database() -> Iterator[sqlite3.Connection]:
    connection = open_database(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def reminders(database: sqlite3.Connection) -> ReminderRepo:
    return ReminderRepo(database, clock=lambda: NOW.timestamp())


@pytest.fixture
def create_reminder(reminders: ReminderRepo) -> Tool:
    return create_reminder_for(reminders, clock=lambda: NOW.timestamp())


@pytest.fixture
def list_reminders(reminders: ReminderRepo) -> Tool:
    return list_reminders_for(reminders)


@pytest.fixture
def cancel_reminder(reminders: ReminderRepo) -> Tool:
    return cancel_reminder_for(reminders, confirm_prompt="'{text}' hatırlatıcısı iptal edilecek.")


async def through_the_gate(tool: Tool, confirm: Confirm, **arguments: object) -> str:
    call = ToolCall(id="c1", name=tool.spec.name, arguments=dict(arguments))
    return await dispatch(call, turn_id="t1", registry=ToolRegistry([tool]), confirm=confirm)


# --------------------------------------------------------------------------
# create_reminder
# --------------------------------------------------------------------------


def test_create_and_list_are_safe_and_cancel_asks(
    create_reminder: Tool, list_reminders: Tool, cancel_reminder: Tool
) -> None:
    assert (create_reminder.risk, list_reminders.risk, cancel_reminder.risk) == (
        "safe",
        "safe",
        "confirm",
    )
    assert create_reminder.spec.parameters["required"] == ["text", "at"]
    assert create_reminder.spec.parameters["properties"]["repeat"]["enum"] == [
        "none",
        "daily",
        "weekly",
        "weekdays",
        "monthly",
    ]
    assert cancel_reminder.spec.parameters["required"] == ["reminder_id", "text"]
    # "remove", not "cancel": "cancel" is a no word, and "yes, cancel it"
    # would have been heard as a no (2026-09-26).
    assert TEXT["reminder_cancel_confirm"] == "Shall I remove the reminder '{text}'?"


async def test_a_reminder_is_set_for_a_local_time_and_said_back(
    create_reminder: Tool, reminders: ReminderRepo
) -> None:
    said = await create_reminder.run(text="Ahmet'i ara", at="2026-09-18T09:00")

    assert said == (
        'Reminder #1 set for 2026-09-18T09:00 (Friday): "Ahmet\'i ara". '
        "It will be said aloud when the time comes."
    )
    row = reminders.get(1)
    assert row is not None
    assert row.fire_at == int(datetime(2026, 9, 18, 9, 0, tzinfo=tzlocal()).timestamp())
    assert row.rrule is None and row.status == "pending"


async def test_a_time_with_its_own_zone_is_kept_as_that_moment(
    create_reminder: Tool, reminders: ReminderRepo
) -> None:
    await create_reminder.run(text="x", at="2026-09-18T06:00+00:00")

    row = reminders.get(1)
    assert row is not None
    assert row.fire_at == int(datetime(2026, 9, 18, 6, 0, tzinfo=UTC).timestamp())


async def test_a_repeat_is_kept_as_a_rule_and_said_back(
    create_reminder: Tool, reminders: ReminderRepo
) -> None:
    said = await create_reminder.run(text="İlaç", at="2026-09-18T08:00", repeat="weekdays")

    assert "set for 2026-09-18T08:00 (Friday), weekdays:" in said
    row = reminders.get(1)
    assert row is not None and row.rrule == "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"


async def test_a_time_already_past_is_refused_with_the_time_it_is(
    create_reminder: Tool, reminders: ReminderRepo
) -> None:
    said = await create_reminder.run(text="Geç", at="2026-09-17T08:59")

    assert said == (
        "The time 2026-09-17T08:59 (Thursday) is already past (it is 2026-09-17T09:00 "
        "(Thursday)). Ask the user for a time still to come."
    )
    assert reminders.pending() == []


async def test_a_time_that_is_not_iso_is_explained(create_reminder: Tool) -> None:
    said = await create_reminder.run(text="x", at="yarın sabah")

    assert said.startswith("The time 'yarın sabah' is not ISO 8601.")


async def test_an_empty_or_overlong_reminder_is_refused(
    create_reminder: Tool, reminders: ReminderRepo
) -> None:
    assert await create_reminder.run(text="  ", at="2026-09-18T09:00") == EMPTY
    said = await create_reminder.run(text="x" * (MAX_REMINDER_CHARS + 1), at="2026-09-18T09:00")
    assert said.startswith("Too long")
    assert reminders.pending() == []


# --------------------------------------------------------------------------
# list_reminders
# --------------------------------------------------------------------------


async def test_pending_reminders_are_listed_soonest_first(
    create_reminder: Tool, list_reminders: Tool
) -> None:
    assert await list_reminders.run() == NONE_PENDING

    await create_reminder.run(text="Sonra", at="2026-09-19T10:00", repeat="daily")
    await create_reminder.run(text="Önce", at="2026-09-18T09:00")

    said = await list_reminders.run()

    assert said == (
        "Pending reminders, soonest first:\n"
        "#2 2026-09-18T09:00 (Friday): Önce\n"
        "#1 2026-09-19T10:00 (Saturday), daily: Sonra"
    )


# --------------------------------------------------------------------------
# cancel_reminder, through the gate
# --------------------------------------------------------------------------


async def test_a_reminder_is_not_cancelled_without_a_yes(
    create_reminder: Tool, cancel_reminder: Tool, reminders: ReminderRepo
) -> None:
    await create_reminder.run(text="Dişçi", at="2026-09-18T09:00")
    user = FakeConfirm(answer=False)

    said = await through_the_gate(cancel_reminder, user, reminder_id=1, text="Dişçi")

    assert said == DECLINED
    assert user.asked == ["'Dişçi' hatırlatıcısı iptal edilecek."]
    assert len(reminders.pending()) == 1


async def test_a_reminder_is_cancelled_after_a_yes_to_its_own_words(
    create_reminder: Tool, cancel_reminder: Tool, list_reminders: Tool, reminders: ReminderRepo
) -> None:
    await create_reminder.run(text="Dişçi randevusu", at="2026-09-18T09:00")
    user = FakeConfirm(answer=True)

    said = await through_the_gate(cancel_reminder, user, reminder_id=1, text="dişçi randevusu")

    assert said == "Cancelled reminder #1: 'Dişçi randevusu'."
    assert await list_reminders.run() == NONE_PENDING
    row = reminders.get(1)
    assert row is not None and row.status == "cancelled"


async def test_a_paraphrased_text_is_refused_even_after_a_yes(
    create_reminder: Tool, cancel_reminder: Tool, reminders: ReminderRepo
) -> None:
    await create_reminder.run(text="Dişçi randevusu saat üçte", at="2026-09-18T09:00")
    user = FakeConfirm(answer=True)

    said = await through_the_gate(cancel_reminder, user, reminder_id=1, text="Dişçi")

    assert said.startswith("Reminder #1 reads 'Dişçi randevusu saat üçte', not 'Dişçi'.")
    assert len(reminders.pending()) == 1


async def test_a_number_that_is_no_pending_reminder_is_said(cancel_reminder: Tool) -> None:
    said = await through_the_gate(
        cancel_reminder, FakeConfirm(answer=True), reminder_id=9, text="x"
    )

    assert said.startswith("There is no pending reminder #9.")


# --------------------------------------------------------------------------
# The time, at the end of the prompt
# --------------------------------------------------------------------------


def test_the_time_line_is_local_iso_to_the_minute_with_weekday_and_offset() -> None:
    moment = datetime(2026, 9, 17, 14, 5, 42, tzinfo=timezone(timedelta(hours=3)))

    assert current_time_line(moment) == (
        "The current local date and time is 2026-09-17T14:05 (Thursday, UTC+03:00)."
    )


def test_the_time_line_without_an_argument_is_now() -> None:
    line = current_time_line()

    assert line.startswith("The current local date and time is 20")
    assert line.endswith(").")
