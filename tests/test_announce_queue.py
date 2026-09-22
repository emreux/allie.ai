"""The announce queue is read between turns and nowhere else (design.md
section 3.1, invariant 5; plan.md 4.4 rule 5, D4).

`test_scheduler.py` proves what goes on the queue. This file proves what
the state machine does with it: an announcement waiting when nothing is
under way is said, in the local voice with the session's input paused; one
that arrives while a turn is under way - the user talking, an answer owed,
a tool running, an answer playing - is said after the turn; listening
switched off does not silence it; a voice cuts it off like an answer; and
the session, when one is open, is told what was said, so that the model
does not say it again.

The queue itself is three lines over `asyncio.Queue` and is tested in
passing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from allie.agent.prompts import ANNOUNCED_PREFIX
from allie.announce.queue import Announcement, AnnounceQueue
from allie.app import LiveAssistant, State, Turn
from allie.live.base import TurnComplete
from tests.test_app import (
    FakeCapture,
    FakeSpeaker,
    FakeTTS,
    Held,
    Provider,
    Room,
    assistant_with,
    calls,
    runner_with,
    until,
    voice,
)


async def running(coroutine: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
    task = asyncio.create_task(coroutine)
    await asyncio.sleep(0)
    return task


async def stopped(task: asyncio.Task[None]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def started(assistant: LiveAssistant, capture: FakeCapture) -> asyncio.Task[None]:
    """`run`, up to the point where the door is open."""
    task = await running(assistant.run())
    await until(lambda: capture.started)
    return task


# --------------------------------------------------------------------------
# The queue
# --------------------------------------------------------------------------


async def test_the_queue_hands_announcements_out_in_order() -> None:
    queue = AnnounceQueue()
    assert queue.empty() and len(queue) == 0

    queue.put(Announcement(text="bir", reminder_id=1))
    queue.put(Announcement(text="iki"))

    assert len(queue) == 2
    assert await queue.get() == Announcement(text="bir", reminder_id=1)
    assert await queue.get() == Announcement(text="iki", reminder_id=None)
    assert queue.empty()


# --------------------------------------------------------------------------
# The state machine
# --------------------------------------------------------------------------


async def test_an_announcement_is_said_when_nothing_is_under_way() -> None:
    queue = AnnounceQueue()
    capture = FakeCapture()
    speaker = FakeSpeaker()
    tts = FakeTTS()
    states: list[State] = []
    assistant = assistant_with(
        capture=capture, speaker=speaker, tts=tts, on_state=states.append, announcements=queue
    )
    queue.put(Announcement(text="Dişçi randevusu", reminder_id=3))

    task = await started(assistant, capture)
    try:
        await until(lambda: speaker.heard == "Dişçi randevusu")
        await until(lambda: assistant.state is State.IDLE)
    finally:
        await stopped(task)

    assert states == [State.IDLE, State.ANNOUNCING, State.IDLE]
    assert tts.said == ["Dişçi randevusu"]
    # Deaf while it was said, listening again after: the microphone would
    # otherwise hear the reminder and answer it. And the session's input
    # paused for it (D3), though there was no session to pause.
    assert capture.switches == [True, False]
    assert not capture.deaf
    assert capture.pauses == [True, False]


async def test_an_announcement_waits_for_the_turn_under_way() -> None:
    """The user has spoken and the answer is owed: the reminder follows the
    answer rather than talking over it."""
    queue = AnnounceQueue()
    capture = FakeCapture()
    speaker = FakeSpeaker()
    tts = FakeTTS()
    states: list[State] = []
    assistant = assistant_with(
        capture=capture, speaker=speaker, tts=tts, on_state=states.append, announcements=queue
    )

    task = await started(assistant, capture)
    try:
        capture.speak()
        capture.quiet()
        queue.put(Announcement(text="Su iç"))
        await until(lambda: speaker.heard.endswith("Su iç"))
        await until(lambda: assistant.state is State.IDLE)
    finally:
        await stopped(task)

    assert speaker.heard == "Üçü dört geçiyor.Su iç"
    assert tts.said == ["Su iç"]
    assert states == [
        State.IDLE,
        State.USER_SPEAKING,
        State.IDLE,
        State.SPEAKING,
        State.IDLE,
        State.ANNOUNCING,
        State.IDLE,
    ]


async def test_an_announcement_waits_for_the_tool_under_way() -> None:
    """A tool round is the middle of a turn, model silent or not."""
    queue = AnnounceQueue()
    capture = FakeCapture()
    speaker = FakeSpeaker()
    gate = Held()
    room = Room(calls(), TurnComplete(), after_result=[voice("Üç."), TurnComplete()])
    assistant = assistant_with(
        capture=capture,
        speaker=speaker,
        provider=Provider(room),
        runner=runner_with(gate),
        announcements=queue,
    )

    task = await started(assistant, capture)
    try:
        capture.speak()
        capture.quiet()
        await until(lambda: len(gate.calls) == 1)
        queue.put(Announcement(text="Su iç"))
        await asyncio.sleep(0.02)
        assert speaker.heard == ""
        gate.release.set()
        await until(lambda: speaker.heard.endswith("Su iç"))
    finally:
        await stopped(task)

    assert speaker.heard == "Üç.Su iç"


async def test_an_announcement_that_arrives_later_is_said_then() -> None:
    queue = AnnounceQueue()
    capture = FakeCapture()
    speaker = FakeSpeaker()
    assistant = assistant_with(capture=capture, speaker=speaker, announcements=queue)

    task = await started(assistant, capture)
    try:
        await asyncio.sleep(0.02)
        assert speaker.heard == ""
        queue.put(Announcement(text="Şimdi"))
        await until(lambda: speaker.heard == "Şimdi")
    finally:
        await stopped(task)


async def test_listening_switched_off_does_not_silence_a_reminder() -> None:
    queue = AnnounceQueue()
    capture = FakeCapture()
    speaker = FakeSpeaker()
    assistant = assistant_with(capture=capture, speaker=speaker, announcements=queue)

    task = await started(assistant, capture)
    try:
        capture.switch_off()
        queue.put(Announcement(text="Dişçi"))
        await until(lambda: speaker.heard == "Dişçi")
        await until(lambda: assistant.state is State.OFF)
    finally:
        await stopped(task)

    assert assistant.state is State.OFF


async def test_a_voice_cuts_an_announcement_off_like_an_answer() -> None:
    queue = AnnounceQueue()
    capture = FakeCapture()
    speaker = FakeSpeaker(on_play=lambda: capture.speak())
    tts = FakeTTS()
    assistant = assistant_with(capture=capture, speaker=speaker, tts=tts, announcements=queue)
    queue.put(Announcement(text="Uzun bir hatırlatma"))

    task = await started(assistant, capture)
    try:
        await until(lambda: speaker.stopped >= 1)
    finally:
        await stopped(task)

    assert tts.said[0] == "Uzun bir hatırlatma"


async def test_the_session_is_told_what_was_said_and_not_asked_to_answer() -> None:
    """D4: one line to the session that is open, with the prefix the
    prompt's rule names, and no answer asked for."""
    queue = AnnounceQueue()
    capture = FakeCapture()
    room = Room()
    turns: list[Turn] = []
    assistant = assistant_with(
        capture=capture, provider=Provider(room), announcements=queue, on_turn=turns.append
    )

    task = await started(assistant, capture)
    try:
        capture.speak()
        capture.quiet()
        await until(lambda: assistant.session_open)
        room.arrives(voice("Üç."), TurnComplete())
        await until(lambda: turns and assistant.state is State.IDLE)
        queue.put(Announcement(text="Su iç"))
        await until(lambda: room.texts != [])
    finally:
        await stopped(task)

    assert room.texts == [(f"{ANNOUNCED_PREFIX}Su iç", "user", False)]
    assert capture.pauses == [True, False]


async def test_without_a_queue_nothing_changes() -> None:
    """Most tests and a run with no scheduler: `run` behaves as it did."""
    capture = FakeCapture()
    speaker = FakeSpeaker()
    assistant = assistant_with(capture=capture, speaker=speaker)

    task = await started(assistant, capture)
    try:
        capture.speak()
        capture.quiet()
        await until(lambda: speaker.heard == "Üçü dört geçiyor.")
    finally:
        await stopped(task)

    assert not capture.started


async def test_a_reminder_is_said_while_asleep_and_it_sleeps_on() -> None:
    """The dentist does not wait for the wake word (D21): said in the local
    voice between turns as ever, and the machine goes back to sleep."""
    queue = AnnounceQueue()
    capture = FakeCapture(asleep=True)
    speaker = FakeSpeaker()
    assistant = assistant_with(capture=capture, speaker=speaker, announcements=queue)

    task = await started(assistant, capture)
    try:
        assert assistant.state is State.SLEEPING
        queue.put(Announcement(text="Diş hekimi yarın onda.", reminder_id=1))
        await until(lambda: speaker.heard == "Diş hekimi yarın onda.")
        await until(lambda: assistant.state is State.SLEEPING)
    finally:
        await stopped(task)

    assert capture.sleeps == 0
    assert capture.asleep
