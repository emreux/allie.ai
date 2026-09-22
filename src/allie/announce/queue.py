"""What the assistant says without having been asked (design.md section
3.1, invariant 5; 17 Sep 2026).

A reminder that falls due, and one day a watcher that noticed something,
have a sentence to say and no turn to say it in. They do not say it: they
put it here, and the state machine reads the queue when - and only when -
it is `IDLE`. An announcement made from anywhere else would be spoken over
the user, or over an answer, or into a confirmation window that then hears
it as an answer. One queue, one reader, one place where "is anyone
talking?" is known.

The queue carries `Announcement`s rather than strings so that the reader
can log what it said by number and the writer can say why it is saying it.
Nothing here formats a sentence: the scheduler composes it from the pack's
words, and this file does not know what a reminder is.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

__all__ = ["AnnounceQueue", "Announcement"]


@dataclass(frozen=True, slots=True)
class Announcement:
    """One sentence to be said when the assistant is free, and where it
    came from. `reminder_id` is `None` for anything that is not a
    reminder."""

    text: str
    reminder_id: int | None = None


class AnnounceQueue:
    """An `asyncio.Queue` of announcements, written from the event loop
    and read by the state machine between turns."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Announcement] = asyncio.Queue()

    def put(self, announcement: Announcement) -> None:
        """Adds one to be said. Never blocks: the queue has no ceiling, and
        a writer must not wait on the speaker."""
        self._queue.put_nowait(announcement)

    async def get(self) -> Announcement:
        """The next one, waited for."""
        return await self._queue.get()

    def empty(self) -> bool:
        return self._queue.empty()

    def __len__(self) -> int:
        return self._queue.qsize()
