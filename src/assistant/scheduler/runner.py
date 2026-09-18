"""The scheduler: reminders said when their time comes, model or no model
(design.md section 3.10, invariant 7; 17 Sep 2026).

One asyncio task, one query every twenty seconds: "which pending
reminders have a `fire_at` at or before now?" Whatever comes back is put
on the announce queue (invariant 5) and marked; nothing here calls the
model, the network or the speaker. That is why a reminder is the most
reliable thing the assistant does: an API outage, a key that stopped
working and a provider that is down all leave this loop exactly as it is.

**A reminder that is late is said late, up to a point.** The program was
not running, or the machine was asleep, and the time went by. Within
`LATE_LIMIT_SECONDS` the reminder is still said, with how late it is in
front of it - "yarım saat gecikmeli hatırlatma" - because a dentist at
three is worth hearing about at half past. Past the limit it is `missed`:
said once, in one sentence for all of them, so that a laptop opened after
a week does not read out the week. The one sentence names the last of
them, which is the most likely to still matter.

**A repeating reminder never falls behind.** Its row holds the *next*
time; when that time has passed the rule (`python-dateutil`, RFC 5545)
gives the first occurrence after now and the row moves there. Occurrences
in between - five days of a daily reminder on a closed laptop - are not
said five times: the latest is said or counted as missed by the rules
above, the rest are skipped and counted in the log.

**The clock is a parameter.** `test_scheduler.py` moves it by hours in a
line; a test that waited for a real reminder would be the slowest test in
the project and the least trusted.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from datetime import datetime, tzinfo

from dateutil import rrule
from dateutil.tz import tzlocal
from loguru import logger

from assistant.announce.queue import Announcement, AnnounceQueue
from assistant.store.repos import Reminder, ReminderRepo

__all__ = [
    "GRACE_SECONDS",
    "INTERVAL_SECONDS",
    "LATE_LIMIT_SECONDS",
    "TEXT",
    "Scheduler",
    "next_occurrence",
]

# How often the table is asked (section 3.10). Twenty seconds: a reminder
# is never more than that late by the scheduler's own doing, and the query
# is one indexed read.
INTERVAL_SECONDS = 20.0

# Up to this late is on time: the poll interval, a slow start, a second of
# transcription. Said without a word about lateness.
GRACE_SECONDS = 60

# Past this a reminder is not late but missed: not said as a reminder, only
# counted in the one sentence that says what went by while the assistant
# was off.
LATE_LIMIT_SECONDS = 2 * 60 * 60

# The last link of the chain of section 3.12 for the three sentences this
# loop composes; the pack answers first. `{text}` is the reminder as the
# user gave it, `{minutes}` how late, `{count}` how many were missed.
TEXT: dict[str, str] = {
    "reminder_late": "A reminder from {minutes} minutes ago: {text}",
    "reminder_missed_one": "While I was off, one reminder went by: {text}",
    "reminder_missed_many": "While I was off, {count} reminders went by. The last was: {text}",
}


class Scheduler:
    """Polls `reminders` and announces what is due. One instance, one task."""

    def __init__(
        self,
        reminders: ReminderRepo,
        queue: AnnounceQueue,
        *,
        clock: Callable[[], float] = time.time,
        wording: Mapping[str, str] = TEXT,
        interval_seconds: float = INTERVAL_SECONDS,
        zone: tzinfo | None = None,
    ) -> None:
        self._reminders = reminders
        self._queue = queue
        self._clock = clock
        self._said = {key: wording.get(key) or default for key, default in TEXT.items()}
        self._interval = interval_seconds
        self._zone = zone if zone is not None else tzlocal()

    async def run(self) -> None:
        """Ticks until cancelled. A tick that fails is logged and the next
        one comes on time: one bad row must not stop every reminder after it."""
        while True:
            try:
                self.tick()
            except Exception:
                logger.exception("the scheduler's tick failed")
            await asyncio.sleep(self._interval)

    def tick(self) -> int:
        """Says what is due, once, and returns how many announcements were made.

        Synchronous on purpose: two indexed reads and a write per due
        reminder, well inside the 50 ms of section 3.1 rule 4.
        """
        now = int(self._clock())
        missed: list[Reminder] = []
        announced = 0
        for reminder in self._reminders.due(now):
            if self._fire(reminder, now, missed):
                announced += 1

        if missed:
            last = missed[-1]
            key = "reminder_missed_one" if len(missed) == 1 else "reminder_missed_many"
            text = self._said[key].format(count=len(missed), text=last.text)
            self._queue.put(Announcement(text=text, reminder_id=last.id))
            announced += 1
        return announced

    def _fire(self, reminder: Reminder, now: int, missed: list[Reminder]) -> bool:
        """One due reminder: said, or counted as missed; the row closed or
        moved on. `True` when something was put on the queue."""
        if reminder.rrule is None:
            late = now - reminder.fire_at
            if late > LATE_LIMIT_SECONDS:
                self._reminders.close(reminder.id, status="missed", at=now, late_by=late)
                missed.append(reminder)
                return False
            self._reminders.close(reminder.id, status="fired", at=now, late_by=late)
            self._say(reminder, late)
            return True

        latest, upcoming, skipped = _catch_up(reminder, now, self._zone)
        late = now - latest
        self._reminders.advance(reminder.id, fire_at=upcoming, at=now, late_by=late)
        if skipped:
            logger.info(
                "reminder {id} skipped {count} occurrences while off", id=reminder.id, count=skipped
            )
        if late > LATE_LIMIT_SECONDS:
            missed.append(reminder)
            return False
        self._say(reminder, late)
        return True

    def _say(self, reminder: Reminder, late: int) -> None:
        if late > GRACE_SECONDS:
            text = self._said["reminder_late"].format(minutes=late // 60, text=reminder.text)
        else:
            text = reminder.text
        self._queue.put(Announcement(text=text, reminder_id=reminder.id))


def next_occurrence(rule: str, *, start: int, after: int, zone: tzinfo | None = None) -> int:
    """The first time `rule` fires strictly after `after`, given that it
    fires at `start` and every time the rule says from there on.

    Times in and out are UTC epoch seconds; the rule is read in `zone`
    (the machine's by default), so a daily reminder at nine stays at nine
    across a change of clocks.
    """
    tz = zone if zone is not None else tzlocal()
    rules = rrule.rrulestr(rule, dtstart=datetime.fromtimestamp(start, tz))
    following = rules.after(datetime.fromtimestamp(after, tz))
    if following is None:
        # A rule with a COUNT or an UNTIL that has run out. Not something
        # the tool writes; a day later is a wrong answer that does no harm.
        return after + 24 * 60 * 60
    return int(following.timestamp())


def _catch_up(reminder: Reminder, now: int, zone: tzinfo) -> tuple[int, int, int]:
    """For a repeating reminder whose time has passed: the latest occurrence
    at or before now, the next one after it, and how many in between were
    skipped."""
    rules = rrule.rrulestr(
        reminder.rrule or "", dtstart=datetime.fromtimestamp(reminder.fire_at, zone)
    )
    moment = datetime.fromtimestamp(now, zone)
    passed = rules.between(datetime.fromtimestamp(reminder.fire_at, zone), moment, inc=True)
    latest = int(passed[-1].timestamp()) if passed else reminder.fire_at
    upcoming = next_occurrence(reminder.rrule or "", start=reminder.fire_at, after=now, zone=zone)
    return latest, upcoming, max(len(passed) - 1, 0)
