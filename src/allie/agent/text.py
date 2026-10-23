"""The agent loop (design.md item 1.9 and 2.1c, architecture guide section 1).

The whole product is arranged around a handful of lines: put what the user
said at the end of the recent conversation, send it to the model, and if the
model asks for a tool, run it through the gate of section 3.9, put the result
back in and ask again. It is a loop because one sentence can take several
requests - "yarın Ahmet'i aramayı hatırlat" is the time, then the reminder,
then the answer - and the answer is whatever the model says once it stops
asking. Phase 1 ran this once per turn, with nothing to go round for; phase
2.1c gave it its `while`.

Five decisions here are worth more than the code that implements them.

**A turn is not a message.** The window of section 3.7 keeps the last twelve
*turns*, where a turn is what the user said plus everything that followed it.
Counting messages instead would be simpler and would eventually cut between a
tool call and the result that answers it - a shape every provider rejects. The
cut is at what the user said, so a tool call and its result stay together.

**Nothing is remembered until the answer is whole.** The conversation is
replaced in one assignment at the end of the turn. A dropped connection, the
sixty second `THINKING` timeout of section 3.1, a cancelled turn: all of them
leave the history exactly as it was, so the user repeats themselves and gets a
clean turn rather than one built on half of a previous one.

**The gate is handed in, not imported.** This file never sees `policy.py`. The
composition root builds the gate and passes it; a test passes a fake; and the
loop cannot be talked into running a tool any other way, because it has no
other way (invariant 1). Offering tools without a gate is refused outright.
So is the one who answers a tool's question: `confirm` comes with each turn
from `app.py`, because the microphone that hears the yes lives there and the
gate that needs it is built first - neither can be built holding the other.

**The system prompt is not part of the conversation.** It is prepended to each
request instead of living in the history, which keeps it out of reach of the
window and byte-identical from turn to turn - the one thing prompt caching
needs (architecture guide section 2). It may be handed in as a source rather
than a sentence (2.10): the user's remembered facts live outside this file
and are read into the prompt at every request, so a fact kept a moment ago
is in the next request, and the bytes still change only when the facts do.

**The words leave as they arrive** (2.8, architecture guide section 11).
`stream_reply` hands each piece of text out the moment the provider produced
it, so that `app.py` has the first sentence spoken while the model is still
writing the second. What the turn came to as a whole - the text, what it
cost over every request, how it ended, how many tools ran - is only known
once the stream is over, and is left in `last_answer` for whoever read it;
`reply` is that reader, for a caller with no use for the pieces. A tool
round is announced through `on_tool_round` the moment the loop turns to run
one: that is when the silence a tool costs begins, and the clock of the
filler in `app.py` starts on it.

The loop has no exit of its own, so it does not count for itself: the limits
of section 3.11 live beside this file in `limits.py`, and a `TurnGuard` is
asked before every call. Beside this file rather than inside an adapter,
because written in an adapter they would be written three times, and one of
the three would be forgotten (invariant 3).

Copied from the old repository's `agent/core.py` on 2026-10-23 for the
phone's text path (plan.md D58), where a written message is the turn and the
one who answers a tool's question is the phone's buttons
(`remote/questions.py`) rather than the microphone. `Confirm`, `Dispatch`,
`PromptSource` and `decline` are `agent/core.py`'s, so that one gate serves
the voice and the phone alike, and an unanswered question is `None` here too.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field

from allie.agent.core import Confirm, Dispatch, PromptSource, decline
from allie.agent.limits import Limits, TurnGuard
from allie.agent.prompts import SYSTEM_PROMPT
from allie.text.base import (
    Delta,
    LLMProvider,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
    was_cut_off,
)
from allie.tools.registry import ToolRegistry

__all__ = [
    "WINDOW_TURNS",
    "Agent",
    "Answer",
    "Confirm",
    "Dispatch",
    "OnToolRound",
    "PromptSource",
    "decline",
    "window",
]

# How many turns the model is allowed to remember (section 3.7). Cost grows
# roughly with the square of a conversation's length, because every turn resends
# every turn before it; this is the ceiling that stops it.
WINDOW_TURNS = 12


# Told, once per round, that the loop is about to run the model's calls: the
# moment a turn goes quiet for as long as the tool takes. `app.py` starts the
# filler's clock on it (2.8); the loop itself reads nothing back.
OnToolRound = Callable[[], None]


@dataclass(frozen=True, slots=True)
class Answer:
    """What one turn produced: the words, what they cost, and how it ended.

    `usage` is the total over every request the turn made - a turn that ran
    a tool made at least two - and is what `usage_log` bills from (section
    6). `finish_reason` is that of the last request, in the provider's own
    word; `cut_off` is what that word means when it means "the token limit
    of section 3.11 ended the answer", so that `app.py` can say so out loud
    - a sentence that stops halfway with nothing said about it is a bug
    nobody can find. `tool_calls` is how many calls the gate ran, for the
    log.
    """

    text: str
    usage: Usage
    finish_reason: str | None
    cut_off: bool = False
    tool_calls: int = 0


@dataclass(slots=True)
class _Reply:
    """What one request came back with, once its stream has ended.

    Filled in by `_ask` as the stream goes by, because the words are handed
    out on the way and an async generator cannot also hand back a value.
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None


class Agent:
    """One conversation with one model. The turns live in memory and die with it.

    Not thread safe and not re-entrant: the state machine of section 3.1 is in
    exactly one state at a time, so a second turn cannot start while the first
    is still running.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str,
        system_prompt: str | PromptSource = SYSTEM_PROMPT,
        tools: ToolRegistry | None = None,
        dispatch: Dispatch | None = None,
        limits: Limits | None = None,
    ) -> None:
        if tools is not None and dispatch is None:
            # A tool offered without a gate is a tool that would run without
            # one, which is the one thing section 3.9 forbids.
            raise ValueError("tools were offered without a gate to run them through")
        self._provider = provider
        self._model = model
        self._system_prompt = system_prompt
        self._tools = tools
        self._dispatch = dispatch
        self._limits = limits if limits is not None else Limits()
        self._history: list[Message] = []
        # What the last turn came to, filled in when its stream ends and
        # `None` while one is running (`stream_reply`).
        self.last_answer: Answer | None = None

    async def reply(self, said: str, *, turn_id: str = "", confirm: Confirm = decline) -> Answer:
        """Answers one thing the user said, running whatever tools it takes,
        and remembers having done so.

        `confirm` is who a tool that wants a yes asks. Left out, the answer
        is no: a tool that needs asking about is a tool that does not run
        until somebody can be asked.

        The pieces of the answer are of no interest here: this is
        `stream_reply` read to its end, for a caller that wants the whole.
        """
        async for _ in self.stream_reply(said, turn_id=turn_id, confirm=confirm):
            pass
        if self.last_answer is None:
            # The stream sets it before it ends, so this cannot happen -
            # and is said rather than typed away, so that a change to the
            # stream cannot make this hand back the previous turn's answer.
            raise RuntimeError("the turn ended without an answer")
        return self.last_answer

    async def stream_reply(
        self,
        said: str,
        *,
        turn_id: str = "",
        confirm: Confirm = decline,
        on_tool_round: OnToolRound | None = None,
    ) -> AsyncGenerator[str]:
        """Answers one thing the user said, handing the words out as they come.

        Every piece of text the model produces is yielded the moment it
        arrives - the words of a tool round too, when it has any: a model
        that says "let me look" before it looks means them to be heard.
        Once the stream ends, `last_answer` holds what the turn came to:
        the whole text, what it cost over every request, how it ended, and
        how many tools ran. Until then it is `None`, so that a reader who
        stopped early cannot mistake the previous turn's answer for this
        one's.

        `on_tool_round` is called each time the loop turns to run the
        model's calls, before the first of them is dispatched: the moment
        the silence a tool costs begins (2.8).

        A reader that walks away - the key going down, the turn timing
        out - closes the provider's stream with it, and nothing is
        remembered, exactly as for a turn that failed.
        """
        self.last_answer = None
        conversation = window([*self._history, Message.user(said)])
        spent = Usage()
        ran = 0
        guard = TurnGuard(self._limits)

        while True:
            # Past the limit the model is offered nothing, so the only thing
            # left for it to do is answer.
            offered = [] if guard.exhausted else self._offered()
            got = _Reply()
            # `aclosing`: a reader that leaves this generator leaves `_ask`
            # too, and with it the provider's stream, rather than letting
            # the garbage collector find them.
            async with aclosing(self._ask(conversation, offered, got)) as pieces:
                async for text in pieces:
                    yield text
            spent = spent + got.usage

            # Nothing asked for, no gate to ask it of, or a call made after
            # being offered nothing: the turn ends with the words there are.
            if not got.tool_calls or self._dispatch is None or not offered:
                break

            if on_tool_round is not None:
                on_tool_round()
            conversation.append(Message.assistant(got.text, got.tool_calls))
            for call in got.tool_calls:
                # The guard answers first: a call over the limit, or the same
                # call once too often in a row, gets its sentence instead of
                # a run (section 3.11).
                refused = guard.allow(call)
                if refused is None:
                    result = await self._dispatch(call, turn_id=turn_id, confirm=confirm)
                    ran += 1
                else:
                    result = refused
                conversation.append(Message.tool_result(call, result))

        # A model that produced no text said nothing, and a message with no
        # content is refused outright by several providers - so a silent turn
        # is not written down at all rather than poisoning every turn after
        # it. When it did speak, everything in between - the calls and their
        # results - is remembered with it, so the model knows what it did.
        if got.text:
            self._history = [*conversation, Message.assistant(got.text)]
        self.last_answer = Answer(
            text=got.text,
            usage=spent,
            finish_reason=got.finish_reason,
            cut_off=was_cut_off(got.finish_reason),
            tool_calls=ran,
        )

    def _offered(self) -> list[ToolSpec]:
        return [] if self._tools is None else self._tools.specs()

    def _system(self) -> Message:
        """The system prompt as of now: the sentence given, or what the
        source answers today."""
        source = self._system_prompt
        return Message.system(source() if callable(source) else source)

    async def _ask(
        self, conversation: list[Message], tools: list[ToolSpec], got: _Reply
    ) -> AsyncGenerator[str]:
        """Runs one request, yielding its text as it streams, and leaves the
        rest of what came back in `got` once the stream has ended.

        The output token limit of section 3.11 goes to the provider here,
        which is where an answer can actually be stopped. The provider's
        stream is closed however this one ends: an adapter's stream is an
        async generator holding a connection, and a reader that walked away
        would otherwise leave the request running until the connection was
        collected.
        """
        spoken: list[str] = []
        calls: list[ToolCall] = []
        stream = self._provider.stream(
            [self._system(), *conversation],
            tools,
            model=self._model,
            max_tokens=self._limits.output_tokens,
        )
        try:
            async for delta in stream:
                if delta.text:
                    spoken.append(delta.text)
                    yield delta.text
                if delta.tool_call is not None:
                    calls.append(delta.tool_call)
                # Both arrive at most once per stream and each adapter promises
                # it - a running total added up would bill the request several
                # times over.
                if delta.usage is not None:
                    got.usage = delta.usage
                if delta.finish_reason is not None:
                    got.finish_reason = delta.finish_reason
        finally:
            await _close(stream)

        got.text = "".join(spoken)
        got.tool_calls = tuple(calls)


async def _close(stream: AsyncIterator[Delta]) -> None:
    """Closes a provider's stream, when it is the kind that can be closed.

    The protocol promises an iterator and every adapter hands back an async
    generator; closing it is what ends the request underneath. An iterator
    with nothing to close is left alone.
    """
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        await aclose()


def window(conversation: Sequence[Message], *, turns: int = WINDOW_TURNS) -> list[Message]:
    """The last `turns` turns of a conversation, each of them whole.

    A turn starts where the user spoke and ends where the next one starts, so
    everything the assistant did in between - a tool call and the result that
    answers it - is kept or dropped together.
    """
    starts = [index for index, message in enumerate(conversation) if message.role == "user"]
    if len(starts) <= turns:
        return list(conversation)
    return list(conversation[starts[-turns] :])
