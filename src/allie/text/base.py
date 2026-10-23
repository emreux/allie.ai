"""The contract every LLM provider is reduced to (design.md section 3.2).

Providers disagree about almost everything: Anthropic sends content blocks,
OpenAI sends a string plus `tool_calls`, Gemini sends `parts`. Tool definitions,
tool results and streaming events differ again. Let those differences reach the
rest of the application and the project is nailed to one vendor.

So the application sees only this module: five value types and one protocol.
An adapter translates its vendor's shapes into these on the way in and out, and
that is the only place vendor knowledge is allowed to live.

Phase 1 shipped a single adapter and called `stream` with `tools=[]`. The tool
types were defined here from the start anyway - the permission gate and the
agent loop of phase 2 are written against them, and retrofitting a type this
central breaks every call site (design.md section 8). Phase 2.1 turned them
on, and two things the first real round trip taught are recorded on the types
themselves: a tool result carries the name of the tool as well as the id of
the call, and a call carries back whatever opaque token the provider attached
to it.

Copied from the old repository's `llm/base.py` on 2026-10-23 for the phone's
text path (plan.md D58): the text agent and the Gemini text adapter speak it.
The six types it shares with the live protocol - `ProviderError`,
`AuthenticationError`, `ToolSpec`, `ToolCall`, `Usage`, `ModelInfo` - are
imported from `allie.live.base` rather than defined twice, so that the one
gate judges one kind of call whichever model asked for it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from allie.live.base import (
    AuthenticationError,
    ModelInfo,
    ProviderError,
    ToolCall,
    ToolSpec,
    Usage,
)

__all__ = [
    "AuthenticationError",
    "Delta",
    "LLMProvider",
    "Message",
    "ModelInfo",
    "ProviderError",
    "Role",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "was_cut_off",
]

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of the conversation, independent of any provider's wire format.

    Frozen because the agent loop keeps a window of past turns and hands the
    same objects to the adapter on every request; a message that could be
    edited in place would rewrite history nobody meant to change.
    """

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_call_id: str | None = None
    tool_name: str | None = None

    def __post_init__(self) -> None:
        if self.role == "tool" and (self.tool_call_id is None or self.tool_name is None):
            raise ValueError(
                "a tool result needs the tool_call_id and the tool_name of the call it answers"
            )
        if self.tool_calls and self.role != "assistant":
            raise ValueError(f"only an assistant message carries tool calls, not {self.role!r}")

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: tuple[ToolCall, ...] = ()) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(cls, answering: ToolCall, content: str) -> Message:
        """What a tool said back, tied to the call that asked for it.

        Both the id and the name travel: OpenAI and Anthropic match a result
        to its call by id, Gemini by name - and Gemini refuses a result that
        has none (measured 2026-09-09, "Name cannot be empty"). Built from the
        call itself, a result cannot carry a mismatched pair.
        """
        return cls(
            role="tool", content=content, tool_call_id=answering.id, tool_name=answering.name
        )


@dataclass(frozen=True, slots=True)
class Delta:
    """One piece of a streaming response.

    A chunk carries whichever of these the provider just produced: a piece of
    text, one finished tool call, the reason generation stopped, or the token
    counts that arrive last. An empty `Delta` is legal - providers do send
    chunks that only advance their own state.

    `usage` appears on **at most one** `Delta` per stream and carries the
    totals for the whole request. Providers disagree about this - Gemini
    repeats a running total on every chunk - so each adapter holds the counts
    back and reports them once. Without that rule the obvious way to read them,
    adding up every `Delta.usage`, would overstate a request several times over
    and the cost report of section 6 would be quietly wrong.
    """

    text: str | None = None
    tool_call: ToolCall | None = None
    finish_reason: str | None = None
    usage: Usage | None = None


# The word each vendor uses when the output token limit ended the answer
# rather than the model: Anthropic `max_tokens`, OpenAI `length`, Gemini
# `MAX_TOKENS`. An adapter passes its vendor's word through untouched in
# `Delta.finish_reason`; this is the one place the three are read as one.
_CUT_OFF = frozenset({"max_tokens", "length"})


def was_cut_off(finish_reason: str | None) -> bool:
    """Whether `finish_reason` says the token limit ended the answer (section 3.11).

    The case is the vendor's own and is not held against it.
    """
    return finish_reason is not None and finish_reason.casefold() in _CUT_OFF


@runtime_checkable
class LLMProvider(Protocol):
    """What an adapter has to offer. Nothing here mentions a vendor.

    Vendor-only features - Anthropic prompt caching, OpenAI `reasoning_effort` -
    stay inside their adapter and are announced through a `capabilities`
    attribute the adapter defines for itself. Adding them here would drag every
    other adapter down to the common denominator (design.md section 3.2).
    """

    id: str

    async def validate_credentials(self) -> bool:
        """Reports whether the stored key actually works, before it is saved."""
        ...

    async def list_models(self) -> list[ModelInfo]:
        """Lists the models this key can reach, for the setup command to offer."""
        ...

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[Delta]:
        """Sends a request and yields the response as it arrives.

        Declared `def`, not `async def`, and this is deliberate. Adapters write
        it as `async def ... yield`, an async generator: calling it returns an
        `AsyncIterator` immediately, with no `await`. Declaring `async def` here
        would type it as a coroutine that returns an iterator, callers would
        have to await it first, and no adapter would satisfy the protocol.
        `test_llm_adapters.py` guards this.
        """
        ...
