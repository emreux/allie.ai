"""The one path from a model-produced `ToolCall` to a running tool (design.md section 3.9).

The model cannot tell an instruction of the user's from text it read somewhere
- both are tokens in the same window - so the only defence that holds is
outside the model: code between the model's request and the action. This
module is that code, and there is deliberately no second way to run a tool.

`dispatch` is a referee. It looks the tool up, reads the risk the tool
declared for itself, and either refuses, asks, or runs. Whatever happens it
answers in words, because the answer goes back to the model as the tool's
result: a refusal the model can read is a turn that continues, an exception
that escapes is a turn that dies.

**It writes before it runs.** With an `AuditRepo` in hand the gate opens a
`tool_audit` row *before* the tool's body starts and closes it afterwards;
a refusal is one row that is closed as it is written. A crash between the
two leaves a row that says `started` and nothing more - which is the truth,
and is what section 3.11 builds "I tried, and I do not know whether it
worked" on. The order is the whole point; do not move the write after the
call to save a millisecond.

**The question knows what happened a moment ago** (section 3.11). Before a
risky tool's question is asked, the same audit table is read: the same tool
with the same arguments, run or left `started` inside the last ten minutes,
adds one sentence - "you already did this forty seconds ago". The harm this
guards against is not the mail going out twice; it is the user saying yes
to a question they were already expecting, because they gave the command a
moment ago and the connection dropped. The gate asks the right question
either way; this is the piece of information the user did not have. Nothing
is skipped on their behalf: the decision stays theirs, and a repeat they
meant is a repeat they get.

The answers to the model are English constants and stay that way. The one
sentence addressed to the user goes through the locale chain of section
3.12 like every other: `TEXT` below is the end of it, and the composition
root hands `dispatch` the pack's wording.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from loguru import logger

from assistant.agent.core import Confirm
from assistant.agent.limits import Limits
from assistant.live.base import ToolCall
from assistant.store.repos import AuditRepo, EarlierCall
from assistant.tools.registry import ToolRegistry

__all__ = [
    "DECLINED",
    "DISABLED",
    "DUPLICATE_WINDOW_SECONDS",
    "FAILED",
    "MISSING_ARGUMENT",
    "NO_SUCH_TOOL",
    "TEXT",
    "Confirm",
    "dispatch",
]

NO_SUCH_TOOL = "There is no tool named {name!r}."
DISABLED = "This tool is disabled in the current configuration."
DECLINED = "The user declined this action."
MISSING_ARGUMENT = "The call is missing the argument {name!r}."
FAILED = "The tool failed: {kind}"

# What the gate adds to a question when the audit table says the same call
# ran a moment ago (section 3.11). The last link of the chain of section
# 3.12, as `TEXT` is in every module that says something: the pack answers
# first, through the `wording` the composition root hands `dispatch`.
# `{ago}` is a count of seconds or of minutes, in the coarse shape below
# until the locale's own formatter arrives in phase 3.
TEXT: dict[str, str] = {
    "duplicate_done": "You already did this {ago} ago.",
    "duplicate_uncertain": "You tried this {ago} ago, and the outcome is not known.",
    "seconds_ago": "{count} seconds",
    "minutes_ago": "{count} minutes",
}

# How far back the gate looks for the same call. Ten minutes: an outage and
# the command said again fit inside it, and a repeat the user means an hour
# later is not bothered (section 3.11). The number is the section 3.11
# table's, written once in `limits.py`.
DUPLICATE_WINDOW_SECONDS = Limits().duplicate_window_sec


async def dispatch(
    call: ToolCall,
    *,
    turn_id: str,
    registry: ToolRegistry,
    confirm: Confirm,
    unblocked: Iterable[str] = (),
    audit: AuditRepo | None = None,
    wording: Mapping[str, str] = TEXT,
    duplicate_window: float = DUPLICATE_WINDOW_SECONDS,
) -> str:
    """Runs one tool call the way its risk allows, and reports back in words.

    `unblocked` names the `blocked` tools the user switched on in `config.toml`.
    A tool named there is treated as `confirm`, never as `safe`: opening it
    was a decision made in a file, running it is still a decision made aloud.

    `turn_id` groups the rows of one turn in `tool_audit`, and `audit` is
    where they go. Without one - most tests - nothing is written, and
    nothing is known about a moment ago either.

    `confirm` is whoever can put a question to the user and hear the
    answer: the state machine's own microphone in life (`app.py`), a
    fake in a test. It is a parameter rather than something built here so
    that the gate never holds the microphone and the loop never holds the
    gate's insides.

    `wording` is the pack's version of `TEXT`, and `duplicate_window` how
    many seconds back the audit table is read for the same call.
    """
    tool = registry.get(call.name)
    if tool is None:
        # The model asked for a tool it was never offered. Nothing to run,
        # and nothing to write down: no tool was judged.
        return NO_SUCH_TOOL.format(name=call.name)

    if tool.risk == "blocked" and call.name not in unblocked:
        return _refused(DISABLED, call, tool.risk, turn_id=turn_id, audit=audit)

    if tool.risk != "safe":
        if tool.confirm_prompt is None:
            # `tool()` refuses to build one of these, but a `Tool` made by hand
            # could exist. A question that cannot be asked is a tool that
            # cannot run - failing closed is the whole point of the gate.
            return _refused(DISABLED, call, tool.risk, turn_id=turn_id, audit=audit)
        try:
            # The real argument values go into the sentence: the user hears
            # exactly what the model asked for, which is what catches an
            # injected request (architecture guide section 6).
            question = tool.confirm_prompt.format(**call.arguments)
        except KeyError as missing:
            # The model left out an argument the question needs. It is told
            # which; nobody is asked and nothing is written, since no
            # decision was reached.
            return MISSING_ARGUMENT.format(name=missing.args[0])
        if audit is not None:
            # What the user does not know and the table does: that the same
            # call ran, or may have, a moment ago (section 3.11).
            earlier = audit.recent(call, within=duplicate_window)
            if earlier is not None:
                question = f"{question} {_a_moment_ago(earlier, wording)}"
        if not await confirm(question):
            return _refused(DECLINED, call, tool.risk, turn_id=turn_id, audit=audit)

    # Written down before it runs (section 3.9). From here on a crash leaves a
    # row that says `started`, and that is the honest record of it.
    row = audit.start(call, turn_id=turn_id, risk=tool.risk) if audit is not None else None
    try:
        result = await tool.run(**call.arguments)
    except Exception as error:
        # Deliberately broad: whatever broke inside the tool, the turn goes on
        # and the model gets to tell the user. The class name is enough - the
        # message could carry anything, and nothing here is allowed to log it.
        kind = type(error).__name__
        logger.warning("tool {name} failed: {kind}", name=call.name, kind=kind)
        if audit is not None and row is not None:
            audit.finish(row, status="error", error=kind)
        return FAILED.format(kind=kind)

    if audit is not None and row is not None:
        audit.finish(row, status="ok", summary=result)
    return result


def _a_moment_ago(earlier: EarlierCall, wording: Mapping[str, str]) -> str:
    """ "You already did this forty seconds ago", in the pack's words.

    An `ok` row is something that was done; a `started` row is something
    that may have been, and the sentence says so rather than guess either
    way (section 3.11). Under a minute the span is said in seconds, from a
    minute on in whole minutes - coarse, and coarse is enough for "a moment
    ago" until the locale formatter of phase 3.
    """
    if earlier.ago < 60:
        ago = wording["seconds_ago"].format(count=earlier.ago)
    else:
        ago = wording["minutes_ago"].format(count=earlier.ago // 60)
    key = "duplicate_done" if earlier.status == "ok" else "duplicate_uncertain"
    return wording[key].format(ago=ago)


def _refused(
    answer: str, call: ToolCall, risk: str, *, turn_id: str, audit: AuditRepo | None
) -> str:
    """The gate said no: one closed row, if there is somewhere to write it."""
    if audit is not None:
        audit.deny(call, turn_id=turn_id, risk=risk)
    return answer
