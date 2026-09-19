"""The tool-use probe: does this model actually call a tool? (design.md section 3.2, 2.6)

Every model can be asked a question; not every model can be asked to *do*
something. A tool is offered to it as a schema, and a model that was never
trained on that shape answers in words instead of calling the tool - and
nothing complains. `ModelInfo.supports_tools` has said `None`, "nobody has
tested this", since phase 1. The whole of the secretary depends on the
answer being yes, so the setup wizard does not take the user's choice of
model until this file has asked it once, and `live-assistant run` asks
again when the answer is a week old (the provider may have changed the
model since).

**One session, one tool, one question.** The tool is the canonical
`get_current_time(city)` of section 3.2; the question is one that ought to
make the model reach for it - "what time is it in Istanbul" cannot be
answered from memory - and it is sent as a text turn, so the probe needs no
microphone. A model that emits a call to that tool passes, whatever it
says beside it; a model that only talks fails, with the reason written
down. The time to the model's first sign of life is measured on the way,
because the session is open anyway and section 3.3 wants the number. The
session is closed the moment the verdict is in: the call is never
answered, and a spoken answer is not listened to.

**The question comes from the locale pack.** Section 3.2 wanted it Turkish
so that one request checks both tool calling and understanding the user's
language. Both still hold: the pack's `[probe] question` asks in the
product's language, and `QUESTION` below is the English at the end of the
chain (section 3.12), as for every other sentence. This file carries no
Turkish.

**The verdict is remembered in the `settings` table**, under
`probe:<provider>:<model>`, as one line of JSON with the time it was
reached. `remembered` returns it while it is younger than
`PROBE_TTL_SECONDS`, and `None` - "ask again" - once it is not, or when it
cannot be read: a verdict of unknown age is no verdict.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from assistant.live.base import (
    AudioChunk,
    Closed,
    LiveProvider,
    OutputText,
    ProviderError,
    SessionConfig,
    ToolCallEvent,
    ToolSpec,
    TurnComplete,
)
from assistant.store.repos import SettingsRepo

__all__ = [
    "CANONICAL_TOOL_TEST",
    "NO_TOOL_CALL",
    "PROBE_SECONDS",
    "PROBE_TTL_SECONDS",
    "QUESTION",
    "ProbeResult",
    "probe_key",
    "probe_tool_support",
    "remember",
    "remembered",
]

# The one tool every model is tested with (section 3.2). Its name is the
# name of the real tool of `tools/system.py` on purpose: a model that calls
# this one will call that one.
CANONICAL_TOOL_TEST = ToolSpec(
    name="get_current_time",
    description="Returns the current local time for the given city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
)

# The end of the chain of section 3.12: what is asked when the pack offers
# no `[probe] question`. A question the model cannot answer from memory, so
# that the only good answer is to call the tool.
QUESTION = "What time is it in Istanbul?"

# How long the verdict is waited for, from the open. A live model that is
# going to call the tool does so within a second or two (measured 2026-09-18:
# session open 0.6-0.8 s, text turn to the call about a second); past this
# the provider is reported as not answering, which is its own sentence.
PROBE_SECONDS = 10.0

# How long a verdict is trusted (section 3.2). A provider may swap what is
# behind a model name; a week is short enough to notice and long enough not
# to spend a request on every start.
PROBE_TTL_SECONDS = 7 * 86400

# Why a probe failed, in the one word `settings` keeps.
NO_TOOL_CALL = "no_tool_call_emitted"


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """What one probe found: whether the model called the tool, why not if
    it did not, and how long its first sign of life - a call, a word, a
    sound - took in milliseconds from the session being asked for; `None`
    when nothing at all arrived. The field keeps the name the `settings` rows were written with."""

    ok: bool
    reason: str | None = None
    first_token_ms: float | None = None


async def probe_tool_support(provider: LiveProvider, model: str, *, question: str) -> ProbeResult:
    """Opens a session on `model` with the canonical tool on offer, asks the
    question as a text turn, and reports whether the model called the tool.

    The session is listened to until the verdict is in: the call, or the end
    of the turn without one, or the server hanging up. A call to some other
    tool is not a pass and not the end either - what the model says beside
    the call does not count, and the turn ends on its own. A refusal by the
    provider comes out as the `ProviderError` the adapter raised - the caller
    has a sentence for that, and it is not "cannot call tools"; an answer
    that does not come within `PROBE_SECONDS` is one too, in the same
    currency, so the caller needs no second sentence.
    """
    config = SessionConfig(model=model, tools=[CANONICAL_TOOL_TEST], transcripts=False)
    started = time.perf_counter()
    first_token_ms: float | None = None
    called = False

    try:
        async with asyncio.timeout(PROBE_SECONDS), provider.connect(config) as session:
            await session.send_text(question)
            async for event in session.events():
                if first_token_ms is None and isinstance(
                    event, ToolCallEvent | OutputText | AudioChunk
                ):
                    first_token_ms = (time.perf_counter() - started) * 1000
                if isinstance(event, ToolCallEvent) and event.call.name == CANONICAL_TOOL_TEST.name:
                    called = True
                    break
                if isinstance(event, TurnComplete):
                    break
                if isinstance(event, Closed):
                    if event.error is not None:
                        raise event.error
                    break
    except TimeoutError as waited:
        raise ProviderError(
            f"no answer from {model!r} within {PROBE_SECONDS:.0f} s", kind="timeout"
        ) from waited

    if called:
        return ProbeResult(ok=True, first_token_ms=first_token_ms)
    return ProbeResult(ok=False, reason=NO_TOOL_CALL, first_token_ms=first_token_ms)


def probe_key(provider: str, model: str) -> str:
    """`probe:gemini:gemini-2.5-flash` - the row of `settings` a verdict is kept in."""
    return f"probe:{provider}:{model}"


def remember(
    verdicts: SettingsRepo,
    provider: str,
    model: str,
    result: ProbeResult,
    *,
    now: float | None = None,
) -> None:
    """Writes the verdict down with the time it was reached."""
    record = {
        "ok": result.ok,
        "ts": time.time() if now is None else now,
        "first_token_ms": result.first_token_ms,
        "reason": result.reason,
    }
    verdicts.set(probe_key(provider, model), json.dumps(record))


def remembered(
    verdicts: SettingsRepo,
    provider: str,
    model: str,
    *,
    now: float | None = None,
    ttl: float = PROBE_TTL_SECONDS,
) -> ProbeResult | None:
    """The verdict written down for `model`, or `None` when there is none
    worth reading: nothing stored, a row that does not parse, or one older
    than `ttl` seconds."""
    stored = verdicts.get(probe_key(provider, model))
    if stored is None:
        return None

    try:
        record: Any = json.loads(stored)
    except ValueError:
        return None
    if not isinstance(record, dict):
        return None

    reached = record.get("ts")
    ok = record.get("ok")
    if not isinstance(reached, int | float) or not isinstance(ok, bool):
        return None
    if (time.time() if now is None else now) - reached > ttl:
        return None

    first_token_ms = record.get("first_token_ms")
    reason = record.get("reason")
    return ProbeResult(
        ok=ok,
        reason=reason if isinstance(reason, str) else None,
        first_token_ms=float(first_token_ms) if isinstance(first_token_ms, int | float) else None,
    )
