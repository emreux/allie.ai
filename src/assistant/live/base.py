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
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

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


class ProviderError(Exception):
    """Something the provider refused to do, in words the application knows.

    An adapter never lets its vendor's own exception out. `app.py` has to
    decide what the assistant says out loud, and it cannot import three SDKs to
    find out which of them just failed - that would put vendor knowledge in the
    one place section 3.2 keeps it out of.
    """


class AuthenticationError(ProviderError):
    """The key was refused: mistyped, revoked, or out of credit.

    Kept apart from every other refusal because the answer is different
    (section 3.2). A connection that dropped will probably work next turn, so
    it is retried; a key that was refused will not, so it is never retried and
    never quietly failed over to another model the user is then billed for.
    The three ways a key can be refused all end in the same sentence: renew it.
    """


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool offered to the model, described in the one language all three speak.

    `parameters` is a JSON Schema object. Anthropic calls it `input_schema`,
    OpenAI wraps it in a function object, Gemini calls it a function
    declaration - all three are that same schema in a different envelope.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A complete request from the model to run one tool.

    Every provider streams the arguments as partial JSON. An adapter buffers
    those fragments and emits this object only once the arguments parse, so a
    `ToolCall` is never half-built. The permission gate of section 3.9 depends
    on that: it cannot judge an action it can only see the beginning of.

    `signature` is whatever the provider attached to the call and wants back
    with it, byte for byte, when the call is resent as history. Nothing above
    the adapter reads it; the loop only carries it. Gemini 3 attaches one to
    every function call it makes - a "thought signature", the encrypted trace
    of the reasoning behind the call - and refuses the conversation without
    it (measured 2026-09-09: a 400 on the second round of "saat kaç?").
    """

    id: str
    name: str
    arguments: Mapping[str, Any]
    signature: bytes | None = None


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts for one request, in the shape the `usage_log` table stores.

    `cached_tokens` is a subset of `input_tokens`, not an addition to it. Only
    some providers report it; the rest leave it at zero, which reads correctly
    as "no cache hit" (design.md section 6).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        """What two requests cost together.

        A turn that went round the tool loop made more than one request, and
        the turn's cost is their sum. This is the one place counts are added:
        within a request `Delta.usage` is already the total, and adding
        *those* up is the mistake its docstring warns about.
        """
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """One entry of a provider's model list, as the setup command shows it.

    The optional fields are genuinely unknown for some providers rather than
    merely absent. `supports_tools=None` means "nobody has tested this model
    yet" - the phase 2 probe replaces it with a measured answer instead of
    letting the assistant fail silently at two in the morning (section 3.2).
    """

    id: str
    display_name: str
    context_window: int | None = None
    supports_tools: bool | None = None
    input_price_per_mtok: float | None = None
    output_price_per_mtok: float | None = None


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
