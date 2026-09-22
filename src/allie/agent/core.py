"""The tool round of a live session, and what the gate and the composition
root share (design.md section 3.9, plan.md section 4.4 rule 3).

The agent loop of the old repository - the `while` that sent a request, ran
the model's calls through the gate and asked again - is not here, and will
not be: a live session has no request to loop over. The model asks for a
tool in the middle of speaking and is answered in the middle of it, so what
the loop used to do per round is all that is left to do, and `ToolRunner`
does it: the guard of section 3.11 is asked, the gate is handed the call,
and the words go back through `LiveSession.send_tool_result`.

Three things about the round are decided here rather than in `app.py`.

**The caller is not held.** The one who hands a call in is the loop reading
the session's events, and a tool takes anywhere from milliseconds to the
fifteen seconds of a confirmation window. Held for that long the loop would
miss the server withdrawing the call, interrupting, or hanging up - so `run`
returns at once and the round goes on as a task beside it.

**One at a time, in the order they came.** Two calls in one message are two
questions the user may be asked, and there is one microphone to ask them
with. The second waits for the first, and the results go back in the order
the calls came - as the old loop sent them.

**A call the model withdrew is not answered.** `ToolCallCancelled` means the
user spoke over the model before the tool was done; a result for a call the
server has forgotten is one it refuses or misfiles. `cancel` stops the tool
and sends nothing; `close` - the hang-up of rule 4 - does that to every call
pending.

The guard is the turn's, and the turn is the state machine's to end (D9):
`new_turn` is called when the user's utterance has been answered whole, not
at every `TurnComplete` the server sends - Gemini ends a server turn at the
tool call itself (ADR-001 section 8).

The rest is the vocabulary the gate and the composition root still speak:
who answers a tool's question, what the answer is when nobody can, where
the system prompt comes from, and the shape of the gate as its caller sees
it.

**The gate is handed in, not imported.** `policy.py` is never imported from
here; the composition root builds the gate and passes it, a test passes a
fake, and a tool cannot be run any other way (invariant 1). The one who
answers a tool's question comes with each call from `app.py`, because the
microphone that hears the yes lives there, and the gate that needs it is
built first - neither can be built holding the other.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Protocol

from loguru import logger

from allie.agent.limits import TOOL_TIMED_OUT, Limits, TurnGuard
from allie.live.base import LiveSession, ProviderError, ToolCall, ToolSpec
from allie.tools.registry import ToolRegistry

__all__ = [
    "Confirm",
    "Dispatch",
    "PromptSource",
    "ToolRunner",
    "decline",
]


# Asks the user a question out loud and answers yes or no. Who actually asks
# is decided by whoever runs the tool: the state machine hands over the
# microphone, a test hands over a fake, and the gate never learns which.
Confirm = Callable[[str], Awaitable[bool]]


async def decline(question: str) -> bool:
    """Nobody to ask means no - never a quiet yes."""
    return False


# The system prompt, when part of it lives outside the code: called when a
# session opens, and expected to answer the same bytes until something
# actually changed (`store/memory.py`).
PromptSource = Callable[[], str]


class Dispatch(Protocol):
    """The gate, as its caller sees it: one call in, the words for the model out.

    `turn_id` names the turn in `tool_audit` (section 3.9) and `confirm` is
    whoever can ask the user a question; the caller carries both from
    `app.py` to the gate and reads neither.
    """

    async def __call__(self, call: ToolCall, *, turn_id: str, confirm: Confirm) -> str: ...


class ToolRunner:
    """The old loop's tool round, without the loop: the guard, the gate, the
    words back.

    One per conversation, built by the composition root with the registry
    the session is opened with (`specs`), the gate and the limits of section
    3.11. `run` takes a call the session delivered and answers it on that
    session, later; `pending` says how many are still on their way and
    `ran` how many the gate was handed this turn, for the log. Not thread
    safe: called from the event loop, by the loop that reads the events.

    The gate is handed in, not imported - `policy.py` is never seen from
    here (invariant 1) - and so is the one who answers a tool's question:
    `confirm` comes with each call from `app.py`, because the microphone
    that hears the yes lives there. A call stopped by `cancel` or `close`
    while its question is being asked is stopped inside that `confirm`,
    which has to let the microphone go on the way out.
    """

    def __init__(
        self, tools: ToolRegistry, dispatch: Dispatch, limits: Limits | None = None
    ) -> None:
        self._tools = tools
        self._dispatch = dispatch
        self._limits = limits if limits is not None else Limits()
        self._guard = TurnGuard(self._limits)
        self._turn_id = ""
        self._ran = 0
        # The rounds still on their way, each with the call it answers; the
        # latest of them is what the next one waits for.
        self._pending: dict[asyncio.Task[None], ToolCall] = {}
        self._last: asyncio.Task[None] | None = None

    def specs(self) -> list[ToolSpec]:
        """The tools a session is opened with (`SessionConfig.tools`)."""
        return self._tools.specs()

    @property
    def pending(self) -> int:
        """How many calls have not been answered yet - or dropped."""
        return len(self._pending)

    @property
    def ran(self) -> int:
        """How many calls the gate was handed this turn: ran, not asked
        for - a refused repeat is not a call that ran."""
        return self._ran

    def new_turn(self, turn_id: str = "") -> None:
        """The user's utterance has been answered whole (D9): the counts
        start again, and what comes next is filed under `turn_id` in
        `tool_audit`."""
        self._guard = TurnGuard(self._limits)
        self._turn_id = turn_id
        self._ran = 0

    def run(self, call: ToolCall, session: LiveSession, *, confirm: Confirm = decline) -> None:
        """Answers `call` on `session`, once the calls before it are answered.

        The guard answers first, here and in the order the calls came: a
        call over the limit, or the same call once too often in a row, gets
        its sentence instead of a run (section 3.11). Either way the words
        go back through `session.send_tool_result`, later; this returns at
        once.
        """
        refused = self._guard.allow(call)
        task = asyncio.create_task(
            self._answer(call, session, confirm, refused=refused, after=self._last),
            name=f"tool {call.name}",
        )
        self._pending[task] = call
        self._last = task
        task.add_done_callback(self._forget)

    def cancel(self, ids: Iterable[str]) -> None:
        """The model withdrew these calls (`ToolCallCancelled`): whichever of
        them is under way is stopped, whichever is waiting never starts, and
        nothing is sent for any of them. An id nobody has is ignored."""
        withdrawn = set(ids)
        for task, call in list(self._pending.items()):
            if call.id in withdrawn:
                task.cancel()

    async def close(self) -> None:
        """The hang-up: every call pending is withdrawn, and this returns
        once they are gone. Harmless with nothing pending."""
        for task in list(self._pending):
            task.cancel()
        await self.settled()

    async def settled(self) -> None:
        """Returns once every call handed in so far has been answered - or
        dropped."""
        while self._pending:
            await asyncio.wait(list(self._pending))

    async def _answer(
        self,
        call: ToolCall,
        session: LiveSession,
        confirm: Confirm,
        *,
        refused: str | None,
        after: asyncio.Task[None] | None,
    ) -> None:
        if after is not None:
            # One at a time: the round before this one ends first, however
            # it ends. `wait` rather than `await`, so that its cancellation
            # is not mistaken for ours.
            await asyncio.wait([after])
        if refused is None:
            self._ran += 1
            try:
                result = await asyncio.wait_for(
                    self._dispatch(call, turn_id=self._turn_id, confirm=confirm),
                    self._limits.turn_seconds,
                )
            except TimeoutError:
                seconds = self._limits.turn_seconds
                logger.warning(
                    "tool {name} took longer than {seconds:g} s and was stopped",
                    name=call.name,
                    seconds=seconds,
                )
                result = TOOL_TIMED_OUT.format(seconds=seconds)
        else:
            result = refused
        try:
            await session.send_tool_result(call, result)
        except ProviderError as failure:
            # The socket died while the tool worked. `Closed` reaches the
            # loop on its own; the answer is lost, and the log says so by
            # kind - the provider's words are where a key could travel.
            logger.warning(
                "tool {name}: the session is gone, its answer was dropped: {kind}",
                name=call.name,
                kind=failure.kind,
            )

    def _forget(self, task: asyncio.Task[None]) -> None:
        call = self._pending.pop(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            # A bug on our side of the gate - the gate turns a tool's own
            # failure into words. Nothing was sent for the call; the class
            # name is enough, the message could carry anything.
            logger.error(
                "tool {name}: the round failed: {kind}", name=call.name, kind=type(error).__name__
            )
