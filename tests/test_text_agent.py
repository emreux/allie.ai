"""The agent loop (design.md section 8, items 1.9 and 2.1c).

Phase 1 proved the loop end to end: what the user said goes to the model with
the last twelve turns for company, and the text that comes back is the answer.
Phase 2.1c gave it the other branch - the `while` of the architecture guide:
the model asks for a tool, the call goes through a gate the loop was handed,
the result goes back in, and the loop goes round again until the model
answers in words. The gate here is a fake that lets everything through and
remembers what came; the real one has its own suite in `test_policy.py`.

Three claims here are worth the file on their own.

**A turn is not a message.** The window keeps twelve *turns*, and a turn is the
user's message plus everything the assistant did about it. Counting messages
would be simpler and wrong: one day the cut lands between a tool call and the
result that answers it, and every provider rejects that.

**The prefix never moves.** The system prompt is frozen - no clock, no date -
because a provider that caches a prompt only does so while the bytes match
(architecture guide section 2). A cache that stops hitting reports nothing; it
just costs more.

**A turn that failed did not happen.** History is written once the answer is
whole, so a dropped connection, the sixty second `THINKING` timeout of section
3.1 or a cancelled turn leaves the conversation exactly as it was.

**The loop does not count for itself.** The limits of section 3.11 come in
through `Limits` and are kept by a guard the loop asks before every call;
what the guard decides is `test_limits.py`'s, what the loop does with the
decision is here.

**The words leave as they arrive** (2.8). `stream_reply` yields each piece
of text the moment the provider produced it, announces a tool round the
moment the loop turns to run one, and leaves the whole answer behind only
once the stream has ended; a reader that walks away closes the provider's
stream with it. `reply` is that stream read to its end.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from allie.agent import prompts
from allie.agent.limits import DUPLICATE_CALL, TOOL_LIMIT_REACHED, Limits
from allie.agent.prompts import LANGUAGE_FALLBACK, SYSTEM_PROMPT
from allie.agent.text import WINDOW_TURNS, Agent, Answer, Confirm, decline, window
from allie.text.base import Delta, Message, ModelInfo, ToolCall, ToolSpec, Usage
from allie.tools.registry import ToolRegistry, tool

MODEL = "scripted-1"


@dataclass(frozen=True, slots=True)
class Request:
    """One thing the agent asked of a provider."""

    messages: list[Message]
    tools: list[ToolSpec]
    model: str
    max_tokens: int

    @property
    def turns(self) -> list[Message]:
        """Everything after the system prompt - the conversation itself."""
        return self.messages[1:]


Step = Delta | BaseException | Callable[[], None] | Callable[[], Awaitable[object]] | float


class ScriptedProvider:
    """A provider that answers from a script and remembers what it was asked.

    One entry per turn, each a sequence of steps the stream takes in order: a
    `Delta` to yield, an exception to raise where it stands - which is how a
    connection that dies halfway through an answer is written down - a number
    of seconds to spend not answering, or something to do while the model is
    supposedly writing - which may be something to wait for, when what the
    test is about is what happened before the model went on (2.8).
    """

    id = "scripted"

    def __init__(self, *turns: Sequence[Step]) -> None:
        self._turns = list(turns)
        self.calls: list[Request] = []
        # How many of its streams are still open: entered and neither
        # exhausted nor closed. A reader that walked away and left one is
        # a request still running.
        self.open_streams = 0

    async def validate_credentials(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id=MODEL, display_name="Scripted 1")]

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[Delta]:
        self.calls.append(
            Request(messages=list(messages), tools=list(tools), model=model, max_tokens=max_tokens)
        )
        turn = self._turns.pop(0) if self._turns else [Delta(text="Tamam.")]

        self.open_streams += 1
        try:
            for item in turn:
                if isinstance(item, BaseException):
                    raise item
                if isinstance(item, int | float):
                    await asyncio.sleep(item)
                elif callable(item):
                    outcome = item()
                    if inspect.isawaitable(outcome):
                        await outcome
                else:
                    yield item
        finally:
            self.open_streams -= 1


def answers(count: int) -> list[list[Delta]]:
    """A script of `count` turns, each a one word answer."""
    return [[Delta(text=f"cevap {number}")] for number in range(count)]


# --------------------------------------------------------------------------
# One turn
# --------------------------------------------------------------------------


async def test_what_the_user_said_is_what_the_model_is_asked() -> None:
    provider = ScriptedProvider([Delta(text="Saat üç.")])

    answer = await Agent(provider, model=MODEL).reply("Saat kaç?")

    assert answer.text == "Saat üç."
    assert provider.calls[-1].turns == [Message.user("Saat kaç?")]
    assert provider.calls[-1].model == MODEL


async def test_the_answer_is_every_fragment_the_model_streamed() -> None:
    """An empty delta is legal - providers send chunks that only advance their
    own state - and must not end the answer or land in it."""
    provider = ScriptedProvider([Delta(text="Mer"), Delta(), Delta(text="haba.")])

    answer = await Agent(provider, model=MODEL).reply("selam")

    assert answer.text == "Merhaba."


async def test_the_token_counts_of_the_turn_are_carried_out_of_it() -> None:
    """Item 1.11 logs them per turn, and the cost report of section 6 is built
    from that log."""
    spent = Usage(input_tokens=812, output_tokens=97)
    provider = ScriptedProvider([Delta(text="Tamam."), Delta(usage=spent)])

    answer = await Agent(provider, model=MODEL).reply("selam")

    assert answer.usage == spent


async def test_a_provider_that_counts_no_tokens_still_gives_an_answer() -> None:
    provider = ScriptedProvider([Delta(text="Tamam.")])

    answer = await Agent(provider, model=MODEL).reply("selam")

    assert (answer.text, answer.usage) == ("Tamam.", Usage())


async def test_the_reason_the_model_stopped_survives_the_turn() -> None:
    """A reply cut off at the token limit is not a reply that ended; the user
    is told so in phase 2, and cannot be if the loop drops this."""
    provider = ScriptedProvider(
        [Delta(text="Uzun cevabın başı"), Delta(finish_reason="MAX_TOKENS")]
    )

    answer = await Agent(provider, model=MODEL).reply("anlat")

    assert answer.finish_reason == "MAX_TOKENS"


async def test_an_agent_given_no_tools_offers_none() -> None:
    """The phase 1 shape, still legal: a loop with nothing to go round for."""
    provider = ScriptedProvider([Delta(text="Tamam.")])

    await Agent(provider, model=MODEL).reply("not al")

    assert provider.calls[-1].tools == []


# --------------------------------------------------------------------------
# What the model is allowed to remember
# --------------------------------------------------------------------------


async def test_the_model_is_told_what_was_already_said() -> None:
    provider = ScriptedProvider([Delta(text="Merhaba.")], [Delta(text="Ahmet.")])
    conversation = Agent(provider, model=MODEL)

    await conversation.reply("selam")
    await conversation.reply("adım neydi?")

    assert provider.calls[-1].turns == [
        Message.user("selam"),
        Message.assistant("Merhaba."),
        Message.user("adım neydi?"),
    ]


async def test_only_the_last_twelve_turns_are_still_in_the_room() -> None:
    """Section 3.7: the window is what keeps a long conversation from growing
    roughly quadratically in cost."""
    provider = ScriptedProvider(*answers(WINDOW_TURNS + 1))
    conversation = Agent(provider, model=MODEL)

    for number in range(WINDOW_TURNS + 1):
        await conversation.reply(f"soru {number}")

    asked = [message.content for message in provider.calls[-1].turns if message.role == "user"]
    assert len(asked) == WINDOW_TURNS
    assert asked[0] == "soru 1"


def test_the_window_is_the_twelve_turns_section_3_7_asks_for() -> None:
    """A number the design arrived at rather than one the code picked: twelve
    turns is what section 3.7 budgets for against the cost of resending them."""
    assert WINDOW_TURNS == 12


def test_a_turn_is_not_a_message() -> None:
    """Phase 2 puts a tool call and its result inside a single turn. The cut
    has to be at what the user said, or it eventually orphans a tool result
    from the call it answers - which every provider rejects.
    """
    call = ToolCall(id="c1", name="get_current_time", arguments={})
    conversation = [
        Message.user("saat kaç?"),
        Message.assistant(tool_calls=(call,)),
        Message.tool_result(call, "15:04"),
        Message.assistant("Üçü dört geçiyor."),
        Message.user("teşekkürler"),
        Message.assistant("Rica ederim."),
    ]

    assert window(conversation, turns=1) == conversation[4:]


def test_a_conversation_shorter_than_the_window_is_left_whole() -> None:
    conversation = [Message.user("selam"), Message.assistant("Merhaba.")]

    assert window(conversation, turns=WINDOW_TURNS) == conversation


async def test_the_system_prompt_is_no_turn_and_cannot_fall_out_of_the_window() -> None:
    """It is prepended to every request rather than kept in the conversation,
    so there is no length of chat that can push it out."""
    provider = ScriptedProvider(*answers(WINDOW_TURNS * 2))
    conversation = Agent(provider, model=MODEL, system_prompt="You are an assistant.")

    for number in range(WINDOW_TURNS * 2):
        await conversation.reply(f"soru {number}")

    assert provider.calls[-1].messages[0] == Message.system("You are an assistant.")


async def test_every_request_starts_with_the_very_same_bytes() -> None:
    """A cached prefix is a prefix that matches byte for byte. Nothing reports
    the moment it stops - the bill just grows (architecture guide section 2)."""
    provider = ScriptedProvider(*answers(3))
    conversation = Agent(provider, model=MODEL)

    for number in range(3):
        await conversation.reply(f"soru {number}")

    assert [call.messages[0] for call in provider.calls] == [Message.system(SYSTEM_PROMPT)] * 3


async def test_the_prompt_may_be_a_source_read_at_every_request() -> None:
    """The user's facts live outside the loop (2.10). A fact kept a moment
    ago is in the very next request, not the next start."""
    facts = ["Bana Emre de."]
    provider = ScriptedProvider(*answers(2))
    conversation = Agent(provider, model=MODEL, system_prompt=lambda: "\n".join(["BASE", *facts]))

    await conversation.reply("selam")
    facts.append("Kahveyi sade içerim.")
    await conversation.reply("bir daha")

    assert provider.calls[0].messages[0] == Message.system("BASE\nBana Emre de.")
    assert provider.calls[1].messages[0] == Message.system(
        "BASE\nBana Emre de.\nKahveyi sade içerim."
    )


async def test_a_source_that_answers_the_same_keeps_the_bytes_the_same() -> None:
    """The cache of architecture guide section 2 goes on hitting between
    the changes."""
    provider = ScriptedProvider(*answers(3))
    conversation = Agent(provider, model=MODEL, system_prompt=lambda: "BASE")

    for number in range(3):
        await conversation.reply(f"soru {number}")

    assert [call.messages[0] for call in provider.calls] == [Message.system("BASE")] * 3


# --------------------------------------------------------------------------
# When the turn goes wrong
# --------------------------------------------------------------------------


async def test_a_turn_that_failed_did_not_happen() -> None:
    """The user hears "I could not connect" (item 1.10) and says it again. What
    they must not be answered from is half a turn nobody heard."""
    provider = ScriptedProvider(
        [Delta(text="Yarım "), ConnectionError("connection reset")],
        [Delta(text="Merhaba.")],
    )
    conversation = Agent(provider, model=MODEL)

    with pytest.raises(ConnectionError):
        await conversation.reply("selam")
    await conversation.reply("selam?")

    assert provider.calls[-1].turns == [Message.user("selam?")]


async def test_a_cancelled_turn_leaves_nothing_behind() -> None:
    """`THINKING` gives up after sixty seconds (section 3.1 rule 5), which
    cancels the turn from outside. Cancellation is not an error to swallow."""
    provider = ScriptedProvider(
        [Delta(text="Düşün"), asyncio.CancelledError()],
        [Delta(text="Buyur.")],
    )
    conversation = Agent(provider, model=MODEL)

    with pytest.raises(asyncio.CancelledError):
        await conversation.reply("uzun bir soru")
    await conversation.reply("selam")

    assert provider.calls[-1].turns == [Message.user("selam")]


async def test_a_model_that_said_nothing_is_not_remembered_as_having_spoken() -> None:
    """An empty assistant message is not merely noise in the window: several
    providers refuse a message with no content at all, so one silent turn would
    break every turn after it."""
    provider = ScriptedProvider([Delta(finish_reason="SAFETY")], [Delta(text="Merhaba.")])
    conversation = Agent(provider, model=MODEL)

    answer = await conversation.reply("...")
    await conversation.reply("selam")

    assert answer.text == ""
    assert provider.calls[-1].turns == [Message.user("selam")]


# --------------------------------------------------------------------------
# The system prompt
# --------------------------------------------------------------------------


# The old repository's `test_the_prompt_is_made_of_the_rules_it_names` pinned
# that product's five-rule prompt; the voice's prompt here has six and is
# pinned by `test_prompts.py` (dropped with the copy, 2026-10-23, plan.md D58).


def test_the_prompt_pins_no_language() -> None:
    """Section 3.12: the reply mirrors whatever language the user just used.
    Naming one here would quietly beat the rule that says so."""
    said = SYSTEM_PROMPT.casefold()

    assert not [name for name in ("turkish", "türkçe", "english", "german") if name in said]


def test_a_message_with_no_language_in_it_keeps_the_last_one() -> None:
    """Measured 2026-08-31: a transcript of digits alone was answered in
    English. There was no language to mirror, so the rule has to say what to
    do when there is none - without naming one."""
    said = LANGUAGE_FALLBACK.casefold()

    assert "digits" in said
    assert "last" in said
    assert not [character for character in LANGUAGE_FALLBACK if character.isdigit()]


def test_the_prompt_carries_no_clock_and_no_calendar() -> None:
    """A date, a time or a count is a digit, and a digit in the prefix is a
    cache that stops hitting. The date comes from a tool in phase 2."""
    assert not [character for character in SYSTEM_PROMPT if character.isdigit()]


def test_nothing_in_the_prompt_module_can_change_between_requests() -> None:
    """Frozen means nothing computes it. Imports are where that stops being
    true - `datetime`, `locale`, a settings read - so there are none."""
    source = Path(inspect.getfile(prompts)).read_text(encoding="utf-8")
    imported: set[str] = set()

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= {"__future__"}


# --------------------------------------------------------------------------
# The tool branch (2.1c): the model asks, the gate decides, the result goes back
# --------------------------------------------------------------------------


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

    The tools' own bodies are never run here: what the loop does with a call
    is hand it over, and what it does with the answer is put it back. Both
    are visible from outside without running anything.
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


def asks(name: str, call_id: str = "c1", **arguments: object) -> Delta:
    """The model asking for `name`, as a provider streams it."""
    return Delta(tool_call=ToolCall(id=call_id, name=name, arguments=arguments))


def rounds(count: int) -> list[list[Delta]]:
    """`count` rounds of one call each, every one with an argument of its own,
    so that the repeat check of 2.4 stays out of a test about the ceiling."""
    return [[asks("clock", f"c{n}", n=n)] for n in range(count)]


def with_tools(provider: ScriptedProvider, gate: FakeGate | None = None) -> Agent:
    return Agent(provider, model=MODEL, tools=TOOLS, dispatch=gate if gate else FakeGate())


async def test_the_tools_in_the_registry_are_what_the_model_is_offered() -> None:
    provider = ScriptedProvider([Delta(text="Tamam.")])

    await with_tools(provider).reply("not al")

    assert provider.calls[-1].tools == TOOLS.specs()


def test_tools_cannot_be_offered_without_a_gate_to_run_them_through() -> None:
    """Invariant 1 at the composition root: a tool offered without a gate is
    a tool that would run without one."""
    with pytest.raises(ValueError, match="gate"):
        Agent(ScriptedProvider(), model=MODEL, tools=TOOLS)


async def test_a_call_goes_through_the_gate_and_its_result_goes_back_to_the_model() -> None:
    """The fifteen lines of architecture guide section 1, end to end: the
    request is kept in the conversation, the result answers it, and the
    words come from the request after."""
    gate = FakeGate()
    provider = ScriptedProvider([asks("clock")], [Delta(text="Üçü dört geçiyor.")])

    answer = await with_tools(provider, gate).reply("saat kaç?")

    [call] = gate.calls
    assert call.name == "clock"
    assert answer.text == "Üçü dört geçiyor."
    assert provider.calls[-1].turns == [
        Message.user("saat kaç?"),
        Message.assistant("", (call,)),
        Message.tool_result(call, "clock: done"),
    ]


async def test_the_words_said_alongside_a_call_stay_with_it() -> None:
    gate = FakeGate()
    provider = ScriptedProvider([Delta(text="Bakıyorum."), asks("clock")], [Delta(text="Üç.")])

    await with_tools(provider, gate).reply("saat kaç?")

    assert provider.calls[-1].turns[1] == Message.assistant("Bakıyorum.", (gate.calls[0],))


async def test_two_calls_in_one_reply_are_answered_in_the_order_they_came() -> None:
    gate = FakeGate()
    provider = ScriptedProvider(
        [asks("clock", "c1"), asks("calendar", "c2")], [Delta(text="Pazartesi, üç.")]
    )

    await with_tools(provider, gate).reply("bugün ne, saat kaç?")

    assert [call.name for call in gate.calls] == ["clock", "calendar"]
    assert provider.calls[-1].turns[2:] == [
        Message.tool_result(gate.calls[0], "clock: done"),
        Message.tool_result(gate.calls[1], "calendar: done"),
    ]


async def test_the_turn_id_reaches_the_gate_with_every_call() -> None:
    """It is what `tool_audit` files the calls under (section 3.9); the loop
    carries it and never reads it."""
    gate = FakeGate()
    provider = ScriptedProvider([asks("clock", "c1"), asks("calendar", "c2")], [Delta(text="Üç.")])

    await with_tools(provider, gate).reply("saat kaç?", turn_id="turn-7")

    assert gate.turn_ids == ["turn-7", "turn-7"]


async def test_at_the_limit_the_model_is_offered_nothing_and_left_to_answer() -> None:
    """Section 3.11: a model that keeps asking is a model in a loop, and every
    round is a request paid for. After the eighth call the ninth request
    offers no tools, so the only thing left to do is answer."""
    gate = FakeGate()
    provider = ScriptedProvider(*rounds(LIMIT), [Delta(text="Yeter.")])

    answer = await with_tools(provider, gate).reply("dön dur")

    assert len(gate.calls) == LIMIT
    assert len(provider.calls) == LIMIT + 1
    assert all(request.tools == TOOLS.specs() for request in provider.calls[:-1])
    assert provider.calls[-1].tools == []
    assert answer.text == "Yeter."


async def test_a_call_made_after_being_offered_nothing_ends_the_turn() -> None:
    """A model that ignores an empty tool list is not argued with: the turn
    ends with whatever words there are, rather than going round for ever."""
    gate = FakeGate()
    provider = ScriptedProvider(*rounds(LIMIT + 5))

    answer = await with_tools(provider, gate).reply("dön dur")

    assert len(gate.calls) == LIMIT
    assert len(provider.calls) == LIMIT + 1
    assert answer.text == ""


async def test_a_call_over_the_limit_is_answered_with_the_limit_and_not_run() -> None:
    """Two calls in one reply when only one is left: the second is not run,
    and the model is told so in the one channel it has to read."""
    gate = FakeGate()
    provider = ScriptedProvider(
        *rounds(LIMIT - 1),
        [asks("clock", "c8", n=8), asks("calendar", "c9")],
        [Delta(text="Tamam.")],
    )

    answer = await with_tools(provider, gate).reply("dön dur")

    assert len(gate.calls) == LIMIT
    assert [call.name for call in gate.calls][-1] == "clock"
    refused = provider.calls[-1].turns[-1]
    assert (refused.tool_name, refused.content) == ("calendar", TOOL_LIMIT_REACHED)
    assert provider.calls[-1].tools == []
    assert answer.text == "Tamam."


async def test_the_limit_is_the_eight_calls_section_3_11_asks_for() -> None:
    assert LIMIT == 8


async def test_a_turn_that_ran_a_tool_is_remembered_whole() -> None:
    """Architecture guide section 1, consequence (a): the model knows what it
    did only because the call and the result are still in the conversation
    next time. Drop them and "bir saat sonra kaç olur" has nothing to
    count from."""
    provider = ScriptedProvider([asks("clock")], [Delta(text="Üç.")], [Delta(text="Dört.")])
    conversation = with_tools(provider)

    await conversation.reply("saat kaç?")
    await conversation.reply("bir saat sonra?")

    assert [message.role for message in provider.calls[-1].turns] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]


async def test_the_cost_of_a_turn_is_the_sum_of_every_request_it_made() -> None:
    """Two requests, two counts. The log line and the bill are per turn."""
    provider = ScriptedProvider(
        [asks("clock"), Delta(usage=Usage(100, 5))],
        [Delta(text="Üç."), Delta(usage=Usage(150, 20, 64))],
    )

    answer = await with_tools(provider).reply("saat kaç?")

    assert answer.usage == Usage(250, 25, 64)


async def test_without_a_gate_a_call_the_model_makes_anyway_is_not_acted_on() -> None:
    """The phase 1 shape. No tools were offered, so a call is a model's
    slip; the words are kept and nothing is run, because there is nothing
    to run it through."""
    provider = ScriptedProvider([asks("clock"), Delta(text="Saat üç.")])

    answer = await Agent(provider, model=MODEL).reply("saat kaç?")

    assert answer.text == "Saat üç."
    assert len(provider.calls) == 1


# --------------------------------------------------------------------------
# Who answers a tool's question (2.3)
# --------------------------------------------------------------------------


async def test_the_one_who_answers_a_tool_s_question_is_handed_to_the_gate() -> None:
    """`app.py` owns the microphone and the gate owns the question; the loop
    carries the one to the other and reads neither."""
    gate = FakeGate()
    provider = ScriptedProvider([asks("clock")], [Delta(text="Üç.")])

    async def says_yes(question: str) -> bool:
        return True

    await with_tools(provider, gate).reply("saat kaç?", confirm=says_yes)

    assert gate.confirms == [says_yes]


async def test_with_nobody_to_ask_the_answer_is_no() -> None:
    """Never a quiet yes: a tool that needs asking about does not run until
    somebody can be asked."""
    gate = FakeGate()
    provider = ScriptedProvider([asks("clock")], [Delta(text="Üç.")])

    await with_tools(provider, gate).reply("saat kaç?")

    [confirm] = gate.confirms
    assert confirm is decline
    assert await confirm("Spotify will be opened.") is False


# --------------------------------------------------------------------------
# The limits of section 3.11 (2.4): the loop asks the guard and tells the provider
# --------------------------------------------------------------------------


async def test_the_same_call_once_too_often_is_refused_in_the_tool_s_channel_and_not_run() -> None:
    """Architecture guide section 12: the model reads the refusal as the
    tool's answer and changes course."""
    gate = FakeGate()
    provider = ScriptedProvider(
        [asks("clock", "c1")], [asks("clock", "c2")], [asks("clock", "c3")], [Delta(text="Peki.")]
    )

    answer = await with_tools(provider, gate).reply("saat kaç, emin misin?")

    assert len(gate.calls) == 2
    refused = provider.calls[-1].turns[-1]
    assert (refused.tool_name, refused.content) == ("clock", DUPLICATE_CALL.format(times=2))
    assert answer.text == "Peki."


async def test_a_different_call_in_between_starts_the_count_again() -> None:
    gate = FakeGate()
    provider = ScriptedProvider(
        [asks("clock", "c1")],
        [asks("clock", "c2")],
        [asks("calendar", "c3")],
        [asks("clock", "c4")],
        [asks("clock", "c5")],
        [Delta(text="Peki.")],
    )

    await with_tools(provider, gate).reply("dön dur")

    assert len(gate.calls) == 5


async def test_a_model_repeating_itself_is_not_given_unlimited_rounds() -> None:
    """Every refused repeat counts towards the turn's calls: with three calls
    allowed and one repeat, the fourth request offers no tools."""
    gate = FakeGate()
    provider = ScriptedProvider(*([[asks("clock")]] * 6))
    limits = Limits(tool_calls_per_turn=3, duplicate_calls=1)
    agent = Agent(provider, model=MODEL, tools=TOOLS, dispatch=gate, limits=limits)

    await agent.reply("dön dur")

    assert len(gate.calls) == 1
    assert len(provider.calls) == 4
    assert provider.calls[-1].tools == []


async def test_the_limits_are_the_agent_s_to_be_given() -> None:
    """From `config.toml`, through the composition root. A test that says
    nothing gets the defaults of section 3.11."""
    gate = FakeGate()
    provider = ScriptedProvider(*rounds(5))
    agent = Agent(
        provider, model=MODEL, tools=TOOLS, dispatch=gate, limits=Limits(tool_calls_per_turn=2)
    )

    await agent.reply("dön dur")

    assert len(gate.calls) == 2


async def test_the_output_token_limit_goes_to_the_provider_with_every_request() -> None:
    """The provider is where an answer can actually be stopped: the number
    is ours, the stopping is theirs (invariant 3)."""
    provider = ScriptedProvider([asks("clock")], [Delta(text="Üç.")])
    limits = Limits(output_tokens=1234)
    agent = Agent(provider, model=MODEL, tools=TOOLS, dispatch=FakeGate(), limits=limits)

    await agent.reply("saat kaç?")

    assert [request.max_tokens for request in provider.calls] == [1234, 1234]


async def test_the_default_output_limit_is_the_four_thousand_tokens_of_section_3_11() -> None:
    provider = ScriptedProvider([Delta(text="Tamam.")])

    await Agent(provider, model=MODEL).reply("selam")

    assert provider.calls[-1].max_tokens == 4000


@pytest.mark.parametrize(
    ("reason", "cut_off"),
    [("MAX_TOKENS", True), ("length", True), ("max_tokens", True), ("STOP", False), (None, False)],
)
async def test_an_answer_the_token_limit_ended_is_marked_as_cut_off(
    reason: str | None, cut_off: bool
) -> None:
    """In each vendor's own word - Gemini's, OpenAI's, Anthropic's - read as
    one, so that `app.py` can say so without knowing which answered."""
    provider = ScriptedProvider([Delta(text="Uzun cevabın başı"), Delta(finish_reason=reason)])

    answer = await Agent(provider, model=MODEL).reply("anlat")

    assert answer.cut_off is cut_off


async def test_the_calls_the_gate_ran_are_counted_for_the_log() -> None:
    """Ran, not asked for: a refused repeat is not a call that ran."""
    gate = FakeGate()
    provider = ScriptedProvider(
        [asks("clock", "c1")], [asks("clock", "c2")], [asks("clock", "c3")], [Delta(text="Peki.")]
    )

    answer = await with_tools(provider, gate).reply("dön dur")

    assert answer.tool_calls == 2


# --------------------------------------------------------------------------
# The words as they come (2.8): the stream, and what is known when it ends
# --------------------------------------------------------------------------


async def streamed(agent: Agent, said: str, **rest: Any) -> list[str]:
    """Every piece the stream handed out, in order."""
    return [piece async for piece in agent.stream_reply(said, **rest)]


async def test_the_words_come_out_piece_by_piece_in_the_order_the_model_wrote_them() -> None:
    """The point of 2.8: the first sentence is spoken while the model is
    still writing the second. An empty delta is not a piece."""
    provider = ScriptedProvider([Delta(text="Saat "), Delta(), Delta(text="üç.")])

    pieces = await streamed(Agent(provider, model=MODEL), "saat kaç?")

    assert pieces == ["Saat ", "üç."]


async def test_the_answer_as_a_whole_is_known_once_the_stream_has_ended() -> None:
    spent = Usage(input_tokens=302, output_tokens=8)
    provider = ScriptedProvider([Delta(text="Üç."), Delta(usage=spent, finish_reason="STOP")])
    agent = Agent(provider, model=MODEL)

    await streamed(agent, "saat kaç?")

    assert agent.last_answer == Answer(text="Üç.", usage=spent, finish_reason="STOP")


async def test_the_answer_is_not_known_before_the_stream_has_ended() -> None:
    """A reader that stopped early must not be handed the previous turn's
    answer and take it for this one's."""
    provider = ScriptedProvider([Delta(text="Bir.")], [Delta(text="İki."), Delta(text="Üç.")])
    agent = Agent(provider, model=MODEL)
    await agent.reply("bir")

    stream = agent.stream_reply("iki")
    first = await anext(stream)
    known = agent.last_answer
    await stream.aclose()

    assert (first, known) == ("İki.", None)


async def test_reply_is_the_stream_read_to_its_end() -> None:
    provider = ScriptedProvider([Delta(text="Mer"), Delta(text="haba.")])
    agent = Agent(provider, model=MODEL)

    answer = await agent.reply("selam")

    assert answer.text == "Merhaba."
    assert agent.last_answer == answer


async def test_the_words_of_a_tool_round_are_handed_out_too() -> None:
    """A model that says "let me look" before it looks means it to be
    heard - and the whole answer still counts the tool it ran."""
    gate = FakeGate()
    provider = ScriptedProvider([Delta(text="Bakıyorum."), asks("clock")], [Delta(text="Üç.")])
    agent = with_tools(provider, gate)

    pieces = await streamed(agent, "saat kaç?")

    assert pieces == ["Bakıyorum.", "Üç."]
    assert agent.last_answer is not None
    assert (agent.last_answer.text, agent.last_answer.tool_calls) == ("Üç.", 1)


async def test_a_tool_round_is_announced_before_its_first_call_is_run() -> None:
    """Once per round, and before the gate is asked: the silence the tool
    costs begins here, and the filler of `app.py` is timed from it."""
    gate = FakeGate()
    provider = ScriptedProvider(
        [asks("clock", "c1"), asks("calendar", "c2")],
        [asks("clock", "c3", n=3)],
        [Delta(text="Üç.")],
    )
    announced: list[int] = []

    await streamed(
        with_tools(provider, gate),
        "saat kaç?",
        on_tool_round=lambda: announced.append(len(gate.calls)),
    )

    assert announced == [0, 2]


async def test_a_turn_without_a_tool_round_announces_none() -> None:
    provider = ScriptedProvider([Delta(text="Üç.")])
    announced: list[int] = []

    await streamed(with_tools(provider), "saat kaç?", on_tool_round=lambda: announced.append(1))

    assert announced == []


async def test_a_reader_that_walks_away_leaves_nothing_behind() -> None:
    """The key went down while the model was writing (`app.py`). The turn
    did not happen, exactly as a turn that failed did not."""
    provider = ScriptedProvider([Delta(text="Uzun "), Delta(text="cevap")], [Delta(text="Selam.")])
    agent = Agent(provider, model=MODEL)

    stream = agent.stream_reply("anlat")
    await anext(stream)
    await stream.aclose()
    await agent.reply("selam")

    assert provider.calls[-1].turns == [Message.user("selam")]


async def test_the_provider_s_stream_is_closed_with_the_reader() -> None:
    """The adapter's stream holds the connection. Left to the garbage
    collector, the request would run on after the reader stopped listening
    - and the reader stops listening exactly when the user is talking."""
    provider = ScriptedProvider([Delta(text="bir"), Delta(text="iki"), Delta(text="üç")])
    stream = Agent(provider, model=MODEL).stream_reply("say")

    await anext(stream)
    while_reading = provider.open_streams
    await stream.aclose()

    assert (while_reading, provider.open_streams) == (1, 0)


async def test_a_stream_read_to_its_end_leaves_no_stream_open_either() -> None:
    provider = ScriptedProvider([asks("clock")], [Delta(text="Üç.")])

    await streamed(with_tools(provider), "saat kaç?")

    assert provider.open_streams == 0
