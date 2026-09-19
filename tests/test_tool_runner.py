"""The tool round of a live session (plan.md section 4.4 rule 3, task L1.3).

The old agent loop was a `while`: send the conversation, and if the model
asked for a tool, run the call through the gate, put the result back in and
ask again. A live session has no request to loop over - the model asks in
the middle of speaking and is answered in the middle of it - so what the
loop did *per round* is all that is left, and `ToolRunner` does it: the
guard of section 3.11 is asked, the gate is handed the call, and the words
go back through `LiveSession.send_tool_result`. The tests of that round are
lifted from the old loop's, with the request loop cut out of them.

The gate here is a fake that lets everything through and remembers what
came; the real one has its own suite in `test_policy.py`, untouched.

Three claims are the file's own.

**The caller is not held.** The one who hands a call in is the loop reading
the session's events, and a tool takes anywhere from milliseconds to the
fifteen seconds of a confirmation window. Held for that long, the loop
would miss the server withdrawing the call, interrupting, or hanging up -
so `run` returns at once and the round goes on beside it.

**One at a time, in the order they came.** Two calls in one message are
two questions the user may be asked, and there is one microphone to ask
them with. The second waits for the first, and the results go back in the
order the calls came - as the old loop sent them.

**A call the model withdrew is not answered** (`ToolCallCancelled`): the
user spoke over the model before the tool was done, and a result for a
call the server has forgotten is a result it will refuse or misfile.
Hanging up withdraws everything.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from assistant.agent.core import Confirm, ToolRunner, decline
from assistant.agent.limits import DUPLICATE_CALL, TOOL_LIMIT_REACHED, TOOL_TIMED_OUT, Limits
from assistant.live.base import ProviderError, ToolCall
from assistant.tools.registry import ToolRegistry, tool
from tests.live_contract import FakeLiveSession


@tool(risk="safe")
async def clock() -> str:
    """Tells the time."""
    return "15:04"


@tool(risk="safe")
async def calendar() -> str:
    """Tells the date."""
    return "Monday"


TOOLS = ToolRegistry([clock, calendar])
LIMIT = Limits().tool_calls_per_turn


class FakeGate:
    """A gate that lets everything through and remembers what came, and when.

    The tools' own bodies are never run here: what the runner does with a
    call is hand it over, and what it does with the answer is send it back.
    Both are visible from outside without running anything.
    """

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []
        self.turn_ids: list[str] = []
        self.confirms: list[Confirm] = []

    async def __call__(self, call: ToolCall, *, turn_id: str, confirm: Confirm) -> str:
        self.calls.append(call)
        self.turn_ids.append(turn_id)
        self.confirms.append(confirm)
        return f"{call.name}: done"


class Held(FakeGate):
    """A gate whose tools take as long as the test says: every call waits
    for `release` before it is answered, and a call stopped while waiting
    is written down."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.stopped: list[str] = []

    async def __call__(self, call: ToolCall, *, turn_id: str, confirm: Confirm) -> str:
        self.calls.append(call)
        self.turn_ids.append(turn_id)
        self.confirms.append(confirm)
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.stopped.append(call.id)
            raise
        return f"{call.name}: done"


class Broken(FakeGate):
    """A gate with a bug in it: the kind of failure the gate itself never
    lets a tool cause."""

    async def __call__(self, call: ToolCall, *, turn_id: str, confirm: Confirm) -> str:
        raise RuntimeError("the key is sk-secret")


class Gone(FakeLiveSession):
    """A session whose socket died while the tool was working."""

    async def send_tool_result(self, call: ToolCall, content: str) -> None:
        raise ProviderError("socket closed", kind="unreachable")


def asks(name: str, call_id: str = "c1", **arguments: object) -> ToolCall:
    """The model asking for `name`, as a session delivers it."""
    return ToolCall(id=call_id, name=name, arguments=arguments)


def calls(count: int) -> list[ToolCall]:
    """`count` calls of one tool, every one with an argument of its own, so
    that the repeat check stays out of a test about the ceiling."""
    return [asks("clock", f"c{n}", n=n) for n in range(count)]


def runner(gate: FakeGate | None = None, limits: Limits | None = None) -> ToolRunner:
    return ToolRunner(TOOLS, gate if gate is not None else FakeGate(), limits)


async def tick() -> None:
    """Lets what was scheduled run up to its next wait."""
    for _ in range(3):
        await asyncio.sleep(0)


def logged() -> tuple[list[str], int]:
    lines: list[str] = []
    return lines, logger.add(lines.append, format="{message}")


# --------------------------------------------------------------------------
# The round: the model asks, the gate decides, the words go back
# --------------------------------------------------------------------------


async def test_a_call_goes_through_the_gate_and_its_result_goes_back_to_the_session() -> None:
    """The old loop's fifteen lines with the request cut out: the call is
    handed to the gate, and what the gate said is what the session sends."""
    gate = FakeGate()
    session = FakeLiveSession()
    call = asks("clock")

    tools = runner(gate)
    tools.run(call, session)
    await tools.settled()

    assert gate.calls == [call]
    assert session.results == [(call, "clock: done")]


async def test_the_tools_in_the_registry_are_what_a_session_is_opened_with() -> None:
    """A live session declares its tools when it opens (`SessionConfig.tools`),
    and the runner is the one who knows them."""
    assert runner().specs() == TOOLS.specs()


async def test_the_caller_is_not_held_while_the_tool_works() -> None:
    """The loop reading the session's events hands the call in and goes back
    to reading: the round is pending, nothing has been sent yet."""
    gate = Held()
    session = FakeLiveSession()
    tools = runner(gate)

    tools.run(asks("clock"), session)

    assert tools.pending == 1
    assert session.results == []
    gate.release.set()
    await tools.settled()
    assert tools.pending == 0
    assert [content for _, content in session.results] == ["clock: done"]


async def test_two_calls_are_answered_one_after_the_other_in_the_order_they_came() -> None:
    """Two questions cannot share the microphone: the second call is not
    handed to the gate until the first has been answered."""
    gate = Held()
    session = FakeLiveSession()
    tools = runner(gate)
    first, second = asks("clock", "c1"), asks("calendar", "c2")

    tools.run(first, session)
    tools.run(second, session)
    await tick()

    assert gate.calls == [first]
    gate.release.set()
    await tools.settled()
    assert session.results == [(first, "clock: done"), (second, "calendar: done")]


async def test_the_turn_id_reaches_the_gate_with_every_call() -> None:
    """It is what `tool_audit` files the calls under (section 3.9); the
    runner carries it from the turn to the gate and never reads it."""
    gate = FakeGate()
    tools = runner(gate)
    tools.new_turn("turn-7")

    tools.run(asks("clock", "c1"), FakeLiveSession())
    tools.run(asks("calendar", "c2"), FakeLiveSession())
    await tools.settled()

    assert gate.turn_ids == ["turn-7", "turn-7"]


async def test_before_any_turn_was_named_a_call_is_filed_under_no_name() -> None:
    gate = FakeGate()
    tools = runner(gate)

    tools.run(asks("clock"), FakeLiveSession())
    await tools.settled()

    assert gate.turn_ids == [""]


async def test_a_session_that_died_meanwhile_loses_the_answer_and_not_the_program() -> None:
    """The socket dropped while the tool worked. `Closed` reaches the loop
    on its own; here the result has nowhere to go, and that goes to the log
    by its kind - never by the provider's words."""
    tools = runner()
    lines, sink = logged()
    try:
        tools.run(asks("clock"), Gone())
        await tools.settled()
    finally:
        logger.remove(sink)

    assert tools.pending == 0
    assert any("clock" in line and "unreachable" in line for line in lines)
    assert not any("socket closed" in line for line in lines)


async def test_a_round_that_crashed_is_written_down_by_its_kind_and_the_next_one_runs() -> None:
    """A bug on our side of the gate - not a tool failing, which the gate
    already turns into words. Nothing is sent for it, the class name goes to
    the log and the message does not, and the call behind it still runs."""
    session = FakeLiveSession()
    tools = ToolRunner(TOOLS, Broken())
    lines, sink = logged()
    try:
        tools.run(asks("clock", "c1"), session)
        await tools.settled()
    finally:
        logger.remove(sink)

    assert session.results == []
    assert any("clock" in line and "RuntimeError" in line for line in lines)
    assert not any("sk-secret" in line for line in lines)


# --------------------------------------------------------------------------
# Who answers a tool's question
# --------------------------------------------------------------------------


async def test_the_one_who_answers_a_tool_s_question_is_handed_to_the_gate() -> None:
    """`app.py` owns the microphone and the gate owns the question; the
    runner carries the one to the other and reads neither."""
    gate = FakeGate()

    async def says_yes(question: str) -> bool:
        return True

    tools = runner(gate)
    tools.run(asks("clock"), FakeLiveSession(), confirm=says_yes)
    await tools.settled()

    assert gate.confirms == [says_yes]


async def test_with_nobody_to_ask_the_answer_is_no() -> None:
    """Never a quiet yes: a tool that needs asking about does not run until
    somebody can be asked."""
    gate = FakeGate()

    tools = runner(gate)
    tools.run(asks("clock"), FakeLiveSession())
    await tools.settled()

    [confirm] = gate.confirms
    assert confirm is decline
    assert await confirm("Spotify will be opened.") is False


# --------------------------------------------------------------------------
# The limits of section 3.11: the runner asks the guard before every call
# --------------------------------------------------------------------------


async def test_a_call_over_the_limit_is_answered_with_the_limit_and_not_run() -> None:
    """A session's tools cannot be withdrawn the way the old loop withdrew
    them from a request, so the ninth call is answered instead: with the
    limit, in the one channel the model has to read."""
    gate = FakeGate()
    session = FakeLiveSession()
    tools = runner(gate)

    for call in calls(LIMIT + 1):
        tools.run(call, session)
    await tools.settled()

    assert len(gate.calls) == LIMIT
    assert session.results[-1] == (calls(LIMIT + 1)[-1], TOOL_LIMIT_REACHED)
    assert len(session.results) == LIMIT + 1


async def test_the_limit_is_the_eight_calls_section_3_11_asks_for() -> None:
    assert LIMIT == 8


async def test_the_same_call_once_too_often_is_refused_in_the_tool_s_channel_and_not_run() -> None:
    """Architecture guide section 12: the model reads the refusal as the
    tool's answer and changes course."""
    gate = FakeGate()
    session = FakeLiveSession()
    tools = runner(gate)

    for call_id in ("c1", "c2", "c3"):
        tools.run(asks("clock", call_id), session)
    await tools.settled()

    assert len(gate.calls) == 2
    assert session.results[-1] == (asks("clock", "c3"), DUPLICATE_CALL.format(times=2))


async def test_a_different_call_in_between_starts_the_count_again() -> None:
    gate = FakeGate()
    session = FakeLiveSession()
    tools = runner(gate)

    for name, call_id in (
        ("clock", "c1"),
        ("clock", "c2"),
        ("calendar", "c3"),
        ("clock", "c4"),
        ("clock", "c5"),
    ):
        tools.run(asks(name, call_id), session)
    await tools.settled()

    assert len(gate.calls) == 5


async def test_a_refused_repeat_counts_towards_the_turn_s_calls() -> None:
    """Every call the model makes counts, refused or not: with three calls
    allowed and one repeat, the fourth is over the limit."""
    gate = FakeGate()
    session = FakeLiveSession()
    tools = runner(gate, Limits(tool_calls_per_turn=3, duplicate_calls=1))

    for n in range(6):
        tools.run(asks("clock", f"c{n}"), session)
    await tools.settled()

    assert len(gate.calls) == 1
    assert [content for _, content in session.results] == [
        "clock: done",
        DUPLICATE_CALL.format(times=1),
        DUPLICATE_CALL.format(times=1),
        TOOL_LIMIT_REACHED,
        TOOL_LIMIT_REACHED,
        TOOL_LIMIT_REACHED,
    ]


async def test_the_limits_are_the_runner_s_to_be_given() -> None:
    """From `config.toml`, through the composition root. A test that says
    nothing gets the defaults of section 3.11."""
    gate = FakeGate()
    tools = runner(gate, Limits(tool_calls_per_turn=2))

    for call in calls(5):
        tools.run(call, FakeLiveSession())
    await tools.settled()

    assert len(gate.calls) == 2


async def test_the_calls_the_gate_ran_are_counted_for_the_log() -> None:
    """Ran, not asked for: a refused repeat is not a call that ran."""
    gate = FakeGate()
    tools = runner(gate)

    for call_id in ("c1", "c2", "c3"):
        tools.run(asks("clock", call_id), FakeLiveSession())
    await tools.settled()

    assert tools.ran == 2


async def test_a_new_turn_forgets_the_last_call_and_the_count() -> None:
    """D9: the guard is the turn's, and a turn is one utterance and
    everything the model did until it was done - the state machine says
    when that is. The count starts again with it."""
    gate = FakeGate()
    session = FakeLiveSession()
    tools = runner(gate, Limits(duplicate_calls=1))
    tools.run(asks("clock", "c1"), session)
    tools.run(asks("clock", "c2"), session)
    await tools.settled()

    tools.new_turn("turn-2")
    tools.run(asks("clock", "c3"), session)
    await tools.settled()

    assert [call.id for call in gate.calls] == ["c1", "c3"]
    assert session.results[-1] == (asks("clock", "c3"), "clock: done")
    assert tools.ran == 1


# --------------------------------------------------------------------------
# A call withdrawn, a tool overdue, a hang-up
# --------------------------------------------------------------------------


async def test_a_call_the_model_withdrew_is_not_answered() -> None:
    """`ToolCallCancelled`: the user spoke over the model before the tool
    was done. The tool is stopped and nothing is sent for it."""
    gate = Held()
    session = FakeLiveSession()
    tools = runner(gate)
    tools.run(asks("clock", "c1"), session)
    await tick()

    tools.cancel(("c1",))
    await tools.settled()

    assert session.results == []
    assert tools.pending == 0
    assert gate.stopped == ["c1"]


async def test_withdrawing_one_call_leaves_the_others_to_run() -> None:
    gate = Held()
    session = FakeLiveSession()
    tools = runner(gate)
    first, second = asks("clock", "c1"), asks("calendar", "c2")
    tools.run(first, session)
    tools.run(second, session)
    await tick()

    tools.cancel(("c1",))
    gate.release.set()
    await tools.settled()

    assert session.results == [(second, "calendar: done")]


async def test_a_call_still_waiting_behind_another_can_be_withdrawn_too() -> None:
    gate = Held()
    session = FakeLiveSession()
    tools = runner(gate)
    first, second = asks("clock", "c1"), asks("calendar", "c2")
    tools.run(first, session)
    tools.run(second, session)
    await tick()

    tools.cancel(("c2",))
    gate.release.set()
    await tools.settled()

    assert gate.calls == [first]
    assert session.results == [(first, "clock: done")]
    # The guard counted it - every call the model makes counts - but the
    # gate never saw it, and `ran` counts what the gate saw.
    assert tools.ran == 1


async def test_withdrawing_an_id_nobody_has_changes_nothing() -> None:
    session = FakeLiveSession()
    tools = runner()

    tools.run(asks("clock", "c1"), session)
    tools.cancel(("nobody",))
    await tools.settled()

    assert [content for _, content in session.results] == ["clock: done"]


async def test_a_tool_that_outlasts_the_turn_s_clock_is_stopped_and_the_model_told() -> None:
    """D9: `turn_seconds` watches one call. The tool is stopped, and the
    model hears so in the tool's channel rather than waiting for ever on a
    result that is not coming."""
    gate = Held()
    session = FakeLiveSession()
    tools = runner(gate, Limits(turn_seconds=0.02))
    lines, sink = logged()
    try:
        tools.run(asks("clock", "c1"), session)
        await tools.settled()
    finally:
        logger.remove(sink)

    assert gate.stopped == ["c1"]
    assert session.results == [(asks("clock", "c1"), TOOL_TIMED_OUT.format(seconds=0.02))]
    assert any("clock" in line and "0.02" in line for line in lines)


def test_the_model_is_told_how_long_the_tool_had() -> None:
    assert TOOL_TIMED_OUT.format(seconds=60.0) == (
        "The tool took longer than 60 seconds and was stopped."
    )


async def test_hanging_up_drops_what_is_pending() -> None:
    """Rule 4: switching off is a hang-up. The tool under way is stopped,
    the one behind it never starts, and nothing is sent to a session that
    is being closed."""
    gate = Held()
    session = FakeLiveSession()
    tools = runner(gate)
    tools.run(asks("clock", "c1"), session)
    tools.run(asks("calendar", "c2"), session)
    await tick()

    await tools.close()

    assert tools.pending == 0
    assert session.results == []
    assert gate.stopped == ["c1"]
    assert gate.calls == [asks("clock", "c1")]


async def test_hanging_up_with_nothing_pending_is_harmless() -> None:
    tools = runner()

    await tools.close()
    await tools.settled()

    assert tools.pending == 0


async def test_after_a_hang_up_the_next_call_still_runs() -> None:
    gate = Held()
    tools = runner(gate)
    tools.run(asks("clock", "c1"), FakeLiveSession())
    await tick()
    await tools.close()

    session = FakeLiveSession()
    gate.release.set()
    tools.run(asks("calendar", "c2"), session)
    await tools.settled()

    assert session.results == [(asks("calendar", "c2"), "calendar: done")]
