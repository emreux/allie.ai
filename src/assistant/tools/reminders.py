"""Reminders: set, listed, cancelled (design.md section 3.10, phase 4.2;
17 Sep 2026).

"Yarın dokuzda Ahmet'i aramayı hatırlat" is a time and a sentence. The
model turns the words into the time - it is told what time it is now at
the end of every request (`current_time_line`), so that "yarın" has a
date - and `create_reminder` writes the row the scheduler will fire
(`scheduler/runner.py`). Nothing here says anything out loud: when the
time comes, the scheduler puts the sentence on the announce queue, and
the state machine says it between turns (invariant 5).

`create_reminder` and `list_reminders` are `safe`; `cancel_reminder` is
`confirm` and takes the reminder's own text beside its number, for the
reason `delete_note` does (`tools/notes.py`): the user hears the real
words before saying yes, and a paraphrase is refused.

Times cross this file in two shapes and never a third: ISO 8601 in the
machine's own zone on the model's side, because every model reads that
without ambiguity; UTC epoch seconds on the table's side, like every time
in the database (section 3.10). Repeats are the four the tool offers,
written as RFC 5545 rules for `python-dateutil` to read.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Literal

from dateutil.tz import tzlocal

from assistant.store.normalize import normalize_search
from assistant.store.repos import Reminder, ReminderRepo
from assistant.tools.registry import Tool, tool

__all__ = [
    "MAX_REMINDER_CHARS",
    "REPEATS",
    "TEXT",
    "cancel_reminder_for",
    "create_reminder_for",
    "current_time_line",
    "list_reminders_for",
]

# The last link of the chain of section 3.12 for the one sentence a user
# hears from these tools. `{text}` is the reminder as the user gave it.
TEXT: dict[str, str] = {
    "reminder_cancel_confirm": "The reminder '{text}' will be cancelled.",
}

# A reminder is said out loud, so it is a sentence.
MAX_REMINDER_CHARS = 200

Repeat = Literal["none", "daily", "weekly", "weekdays", "monthly"]

# The four repeats the model may choose, as the rules the scheduler reads.
REPEATS: dict[str, str | None] = {
    "none": None,
    "daily": "FREQ=DAILY",
    "weekly": "FREQ=WEEKLY",
    "weekdays": "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
    "monthly": "FREQ=MONTHLY",
}

# How many pending reminders are listed. Enough for a week of a busy user.
LIST_LIMIT = 20

# The answers, addressed to the model.
SET = "Reminder #{id} set for {when}{repeat}: {text!r}. It will be said aloud when the time comes."
EMPTY = "Nothing to remind about: the text was empty."
TOO_LONG = "Too long: a reminder is at most {limit} characters. Keep its essence and call again."
BAD_TIME = (
    "The time {at!r} is not ISO 8601. Write it as YYYY-MM-DDTHH:MM in the user's local "
    "time, for example 2026-09-18T09:00."
)
PAST = "The time {when} is already past (it is {now}). Ask the user for a time still to come."
NONE_PENDING = "No reminders are pending."
PENDING = "Pending reminders, soonest first:\n{reminders}"
CANCELLED = "Cancelled reminder #{id}: {text!r}."
NO_SUCH = "There is no pending reminder #{id}. Call list_reminders to find the one the user means."
MISMATCH = (
    "Reminder #{id} reads {stored!r}, not {given!r}. Nothing was cancelled; call again with "
    "the reminder's own text so that the user hears what will be cancelled."
)

# What the model is told at the end of every request, so that "tomorrow at
# nine" is a date. Addressed to the model: English, like the tool answers.
NOW_LINE = "The current local date and time is {now} ({weekday}, UTC{offset})."

_WORD = re.compile(r"\w+")


def current_time_line(now: datetime | None = None) -> str:
    """One sentence with the time, for the end of the system prompt.

    At the end and not the beginning: everything before it stays byte for
    byte the same between requests, which is what a provider's prompt
    cache needs (architecture guide section 2). To the minute, so that
    the line changes sixty times an hour and not sixty times a minute.
    """
    moment = now if now is not None else datetime.now(tzlocal())
    offset = moment.strftime("%z")
    return NOW_LINE.format(
        now=moment.strftime("%Y-%m-%dT%H:%M"),
        weekday=moment.strftime("%A"),
        offset=f"{offset[:3]}:{offset[3:]}" if offset else "",
    )


def create_reminder_for(reminders: ReminderRepo, *, clock: Callable[[], float] = time.time) -> Tool:
    """`create_reminder`, bound to the table and to the clock that says
    whether a time is still to come."""

    @tool(risk="safe")
    async def create_reminder(
        text: Annotated[str, "What to say when the time comes, in the user's own words."],
        at: Annotated[
            str,
            "When, as ISO 8601 in the user's local time: 2026-09-18T09:00. Work it out from "
            "the current date and time you were given.",
        ],
        repeat: Annotated[
            Repeat, "'none' for once; 'daily', 'weekly', 'weekdays' or 'monthly' from that time on."
        ] = "none",
    ) -> str:
        """Sets a reminder to be said aloud at a time: "remind me tomorrow
        at nine to call Ahmet", "every weekday at eight". Use it whenever
        the user wants to be told something at a time; for a note with no
        time, add_note keeps it instead. Say back the time you set."""
        cleaned = " ".join(text.split())
        if not cleaned:
            return EMPTY
        if len(cleaned) > MAX_REMINDER_CHARS:
            return TOO_LONG.format(limit=MAX_REMINDER_CHARS)
        try:
            moment = datetime.fromisoformat(at.strip())
        except ValueError:
            return BAD_TIME.format(at=at)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=tzlocal())

        now = clock()
        if moment.timestamp() <= now:
            return PAST.format(when=_local(int(moment.timestamp())), now=_local(int(now)))

        made = reminders.add(cleaned, fire_at=int(moment.timestamp()), rrule=REPEATS[repeat])
        return SET.format(
            id=made.id,
            when=_local(made.fire_at),
            repeat="" if repeat == "none" else f", {repeat}",
            text=made.text,
        )

    return create_reminder


def list_reminders_for(reminders: ReminderRepo) -> Tool:
    """`list_reminders`, bound to the table it reads."""

    @tool(risk="safe")
    async def list_reminders() -> str:
        """Lists the reminders still to come, soonest first, with their
        numbers. Use it when the user asks what reminders they have, or
        before cancelling one."""
        pending = reminders.pending(limit=LIST_LIMIT)
        if not pending:
            return NONE_PENDING
        return PENDING.format(reminders="\n".join(_line(reminder) for reminder in pending))

    return list_reminders


def cancel_reminder_for(
    reminders: ReminderRepo, *, confirm_prompt: str = TEXT["reminder_cancel_confirm"]
) -> Tool:
    """`cancel_reminder`, bound to the table and to the question it asks."""

    @tool(risk="confirm", confirm_prompt=confirm_prompt)
    async def cancel_reminder(
        reminder_id: Annotated[int, "The reminder's number, as list_reminders showed it."],
        text: Annotated[str, "The reminder's text exactly as list_reminders showed it."],
    ) -> str:
        """Cancels a pending reminder, once the user has confirmed out
        loud. Call list_reminders first and pass the number and the text
        as listed: the user hears the text before deciding, and a text
        that does not match the reminder is not cancelled. A repeating
        reminder is cancelled for good."""
        stored = reminders.get(reminder_id)
        if stored is None or stored.status != "pending":
            return NO_SUCH.format(id=reminder_id)
        if _phrase(stored.text) != _phrase(text):
            return MISMATCH.format(id=reminder_id, stored=stored.text, given=text)
        reminders.cancel(reminder_id)
        return CANCELLED.format(id=reminder_id, text=stored.text)

    return cancel_reminder


def _line(reminder: Reminder) -> str:
    """`#4 2026-09-18T09:00 (Friday), daily: the text`."""
    repeat = next((name for name, rule in REPEATS.items() if rule == reminder.rrule), "none")
    tail = "" if repeat == "none" else f", {repeat}"
    return f"#{reminder.id} {_local(reminder.fire_at)}{tail}: {reminder.text}"


def _local(epoch: int) -> str:
    """An epoch second as the model reads times: local ISO to the minute
    and the weekday, since "Thursday" is what "perşembe" is a question about."""
    moment = datetime.fromtimestamp(epoch, tzlocal())
    return f"{moment.strftime('%Y-%m-%dT%H:%M')} ({moment.strftime('%A')})"


def _phrase(text: str) -> str:
    return " ".join(_WORD.findall(normalize_search(text)))
