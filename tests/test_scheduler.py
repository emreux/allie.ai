"""The scheduler fires reminders on the clock it is given, and catches up
on the ones it missed (design.md section 3.10, invariant 7; CLAUDE.md
testing notes: the clock is fast-forwarded, never waited on).

Every test is a table in memory, a clock that is a number, and one call
to `tick`. What comes out is what went on the announce queue and what the
rows say afterwards. The model is nowhere in this file, which is the
point of invariant 7.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import pytest
from loguru import logger

from assistant.announce.queue import Announcement, AnnounceQueue
from assistant.scheduler.runner import (
    GRACE_SECONDS,
    LATE_LIMIT_SECONDS,
    Scheduler,
    next_occurrence,
)
from assistant.store.db import open_database
from assistant.store.repos import ReminderRepo

# A fixed zone, so that the weekday arithmetic below does not depend on
# the machine the tests run on. Turkey has no clock change.
ZONE = timezone(timedelta(hours=3))

# Thursday 17 September 2026, 09:00 in that zone.
NINE = int(datetime(2026, 9, 17, 9, 0, tzinfo=ZONE).timestamp())
HOUR = 60 * 60
DAY = 24 * HOUR


class Clock:
    def __init__(self, now: int) -> None:
        self.now = now

    def __call__(self) -> float:
        return float(self.now)


@pytest.fixture
def database() -> Iterator[sqlite3.Connection]:
    connection = open_database(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def clock() -> Clock:
    return Clock(NINE - HOUR)


@pytest.fixture
def reminders(database: sqlite3.Connection, clock: Clock) -> ReminderRepo:
    return ReminderRepo(database, clock=clock)


@pytest.fixture
def queue() -> AnnounceQueue:
    return AnnounceQueue()


@pytest.fixture
def scheduler(reminders: ReminderRepo, queue: AnnounceQueue, clock: Clock) -> Scheduler:
    return Scheduler(reminders, queue, clock=clock, zone=ZONE)


def drained(queue: AnnounceQueue) -> list[Announcement]:
    out: list[Announcement] = []
    while not queue.empty():
        out.append(queue._queue.get_nowait())
    return out


# --------------------------------------------------------------------------
# One-off reminders
# --------------------------------------------------------------------------


def test_nothing_is_said_before_its_time(
    scheduler: Scheduler, reminders: ReminderRepo, queue: AnnounceQueue, clock: Clock
) -> None:
    reminders.add("Dişçi", fire_at=NINE)
    clock.now = NINE - 1

    assert scheduler.tick() == 0
    assert queue.empty()
    assert reminders.get(1) is not None and reminders.get(1).status == "pending"


def test_a_reminder_on_time_is_said_as_it_is(
    scheduler: Scheduler, reminders: ReminderRepo, queue: AnnounceQueue, clock: Clock
) -> None:
    made = reminders.add("Dişçi randevusu", fire_at=NINE)
    clock.now = NINE + 15

    assert scheduler.tick() == 1
    assert drained(queue) == [Announcement(text="Dişçi randevusu", reminder_id=made.id)]
    row = reminders.get(made.id)
    assert row is not None
    assert (row.status, row.fired_at, row.missed_by_sec) == ("fired", NINE + 15, 15)


def test_a_reminder_within_the_grace_is_not_called_late(
    scheduler: Scheduler, reminders: ReminderRepo, queue: AnnounceQueue, clock: Clock
) -> None:
    reminders.add("Su iç", fire_at=NINE)
    clock.now = NINE + GRACE_SECONDS

    scheduler.tick()

    assert drained(queue)[0].text == "Su iç"


def test_a_reminder_half_an_hour_late_says_how_late(
    scheduler: Scheduler, reminders: ReminderRepo, queue: AnnounceQueue, clock: Clock
) -> None:
    reminders.add("Dişçi", fire_at=NINE)
    clock.now = NINE + 30 * 60 + 40

    scheduler.tick()

    assert drained(queue)[0].text == "A reminder from 30 minutes ago: Dişçi"
    row = reminders.get(1)
    assert row is not None and row.status == "fired" and row.missed_by_sec == 30 * 60 + 40


def test_a_reminder_five_hours_late_is_missed_and_said_once_as_such(
    scheduler: Scheduler, reminders: ReminderRepo, queue: AnnounceQueue, clock: Clock
) -> None:
    reminders.add("Dişçi", fire_at=NINE)
    clock.now = NINE + 5 * HOUR

    assert scheduler.tick() == 1
    assert drained(queue) == [
        Announcement(text="While I was off, one reminder went by: Dişçi", reminder_id=1)
    ]
    row = reminders.get(1)
    assert row is not None and row.status == "missed" and row.missed_by_sec == 5 * HOUR


def test_several_missed_reminders_are_one_sentence_naming_the_last(
    scheduler: Scheduler, reminders: ReminderRepo, queue: AnnounceQueue, clock: Clock
) -> None:
    reminders.add("Birinci", fire_at=NINE - 2 * DAY)
    reminders.add("İkinci", fire_at=NINE - DAY)
    last = reminders.add("Üçüncü", fire_at=NINE)
    clock.now = NINE + LATE_LIMIT_SECONDS + 1

    assert scheduler.tick() == 1
    assert drained(queue) == [
        Announcement(
            text="While I was off, 3 reminders went by. The last was: Üçüncü", reminder_id=last.id
        )
    ]
    assert [reminders.get(n).status for n in (1, 2, 3)] == ["missed"] * 3  # type: ignore[union-attr]


def test_late_and_missed_together_are_said_in_order(
    scheduler: Scheduler, reminders: ReminderRepo, queue: AnnounceQueue, clock: Clock
) -> None:
    reminders.add("Eski", fire_at=NINE - DAY)
    reminders.add("Yeni", fire_at=NINE)
    clock.now = NINE + 10 * 60

    assert scheduler.tick() == 2
    texts = [item.text for item in drained(queue)]
    assert texts == [
        "A reminder from 10 minutes ago: Yeni",
        "While I was off, one reminder went by: Eski",
    ]


def test_a_reminder_is_said_once_and_not_on_the_next_tick(
    scheduler: Scheduler, reminders: ReminderRepo, queue: AnnounceQueue, clock: Clock
) -> None:
    reminders.add("Bir kere", fire_at=NINE)
    clock.now = NINE

    scheduler.tick()
    clock.now = NINE + 20
    scheduler.tick()

    assert len(drained(queue)) == 1


def test_a_cancelled_reminder_is_never_said(
    scheduler: Scheduler, reminders: ReminderRepo, queue: AnnounceQueue, clock: Clock
) -> None:
    made = reminders.add("İptal", fire_at=NINE)
    assert reminders.cancel(made.id)
    clock.now = NINE + 5

    assert scheduler.tick() == 0
    assert not reminders.cancel(made.id)


# --------------------------------------------------------------------------
# Repeating reminders
# --------------------------------------------------------------------------


def test_a_daily_reminder_moves_on_a_day_when_it_fires(
    scheduler: Scheduler, reminders: ReminderRepo, queue: AnnounceQueue, clock: Clock
) -> None:
    reminders.add("İlaç", fire_at=NINE, rrule="FREQ=DAILY")
    clock.now = NINE + 5

    scheduler.tick()

    assert drained(queue)[0].text == "İlaç"
    row = reminders.get(1)
    assert row is not None
    assert (row.status, row.fire_at, row.fired_at) == ("pending", NINE + DAY, NINE + 5)


def test_five_days_of_a_daily_reminder_are_said_once_and_the_rest_skipped(
    scheduler: Scheduler, reminders: ReminderRepo, queue: AnnounceQueue, clock: Clock
) -> None:
    """The laptop was closed for five days. The latest occurrence is
    half an hour old, so it is said as late; the four before it are
    skipped, and the row points at tomorrow."""
    reminders.add("İlaç", fire_at=NINE, rrule="FREQ=DAILY")
    clock.now = NINE + 5 * DAY + 30 * 60
    skipped: list[str] = []
    sink = logger.add(lambda message: skipped.append(str(message)), level="INFO")
    try:
        assert scheduler.tick() == 1
    finally:
        logger.remove(sink)

    assert drained(queue)[0].text == "A reminder from 30 minutes ago: İlaç"
    row = reminders.get(1)
    assert row is not None and row.fire_at == NINE + 6 * DAY and row.status == "pending"
    assert any("skipped 5 occurrences" in line for line in skipped)


def test_a_daily_reminder_three_hours_late_is_missed_but_still_repeats(
    scheduler: Scheduler, reminders: ReminderRepo, queue: AnnounceQueue, clock: Clock
) -> None:
    reminders.add("İlaç", fire_at=NINE, rrule="FREQ=DAILY")
    clock.now = NINE + 3 * HOUR

    scheduler.tick()

    assert drained(queue)[0].text == "While I was off, one reminder went by: İlaç"
    row = reminders.get(1)
    assert row is not None and row.fire_at == NINE + DAY and row.status == "pending"


def test_a_weekday_reminder_skips_the_weekend() -> None:
    friday = int(datetime(2026, 9, 18, 8, 0, tzinfo=ZONE).timestamp())
    monday = int(datetime(2026, 9, 21, 8, 0, tzinfo=ZONE).timestamp())

    assert (
        next_occurrence("FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR", start=friday, after=friday, zone=ZONE)
        == monday
    )


def test_a_monthly_reminder_keeps_its_day_and_hour() -> None:
    first = int(datetime(2026, 9, 30, 18, 30, tzinfo=ZONE).timestamp())
    following = next_occurrence("FREQ=MONTHLY", start=first, after=first, zone=ZONE)

    assert datetime.fromtimestamp(following, ZONE) == datetime(2026, 10, 30, 18, 30, tzinfo=ZONE)


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


async def test_the_loop_ticks_on_its_interval_and_survives_a_bad_tick(
    reminders: ReminderRepo, queue: AnnounceQueue, clock: Clock
) -> None:
    reminders.add("Tık", fire_at=NINE)
    clock.now = NINE
    scheduler = Scheduler(reminders, queue, clock=clock, zone=ZONE, interval_seconds=0.001)
    ticks = 0
    original = scheduler.tick

    def tick() -> int:
        nonlocal ticks
        ticks += 1
        if ticks == 1:
            raise sqlite3.OperationalError("database is locked")
        return original()

    scheduler.tick = tick  # type: ignore[method-assign]
    task = asyncio.create_task(scheduler.run())
    try:
        for _ in range(200):
            if not queue.empty():
                break
            await asyncio.sleep(0.005)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert ticks >= 2
    assert drained(queue)[0].text == "Tık"


def test_the_pack_words_the_sentences(
    reminders: ReminderRepo, queue: AnnounceQueue, clock: Clock
) -> None:
    scheduler = Scheduler(
        reminders,
        queue,
        clock=clock,
        zone=ZONE,
        wording={"reminder_late": "{minutes} dakika gecikmeli hatırlatma: {text}"},
    )
    reminders.add("Dişçi", fire_at=NINE)
    clock.now = NINE + 5 * 60

    scheduler.tick()

    assert drained(queue)[0].text == "5 dakika gecikmeli hatırlatma: Dişçi"
