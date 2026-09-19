"""The contract every live provider is reduced to (plan.md section 4.3).

Providers disagree about almost everything: Gemini Live speaks protobuf
messages over the SDK's socket, OpenAI's live API speaks JSON events over a
raw one; one ends the model's turn with `turn_complete`, the other with
`response.done`; one hands out a resumption handle, the other has none. Let
those differences reach the rest of the application and the project is
nailed to one vendor.

So the application sees only this module: the value types, one provider
protocol, one session protocol and the eleven events a session can produce.
An adapter translates its vendor's shapes into these on the way in and out,
and that is the only place vendor knowledge is allowed to live.

The shape is the old repository's text contract with the request loop taken
out. `ToolSpec`, `ToolCall`, `Usage`, `ModelInfo` and the two errors are the
same types every module of the tree already shares; `Message`, `Delta` and
the stream went with the pipeline. What replaced them is a *session*: opened
once per conversation, fed microphone audio and text, handing back audio,
transcripts, tool calls and the moments that matter - the model was
interrupted, the turn is over, the server is about to hang up.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AudioChunk",
    "AuthenticationError",
    "Closed",
    "GoingAway",
    "InputText",
    "Interrupted",
    "LiveEvent",
    "LiveProvider",
    "LiveSession",
    "ModelInfo",
    "OutputText",
    "ProviderError",
    "Resumable",
    "SessionConfig",
    "ToolCall",
    "ToolCallCancelled",
    "ToolCallEvent",
    "ToolSpec",
    "TurnComplete",
    "Usage",
    "UsageReport",
]


class ProviderError(Exception):
    """Something the provider refused to do, in words the application knows.

    An adapter never lets its vendor's own exception out. `app.py` has to
    decide what the assistant says out loud, and it cannot import two SDKs to
    find out which of them just failed - that would put vendor knowledge in
    the one place section 4.3 keeps it out of.

    `kind` is the one word the state machine reads to choose its sentence
    and its next move: `"refused"` (the provider said no, for reasons of its
    own), `"rate_limit"` (it said "slow down" - the free tier's word),
    `"unreachable"` (the network, not the provider: a socket refused, a name
    that did not resolve, a stream that stopped), `"timeout"` (an answer that
    did not come in time). The message carries the provider's own words,
    which are the only thing that makes a report of this diagnosable.
    """

    def __init__(self, message: str, *, kind: str = "refused") -> None:
        super().__init__(message)
        self.kind = kind


class AuthenticationError(ProviderError):
    """The key was refused: mistyped, revoked, or out of credit.

    Kept apart from every other refusal because the answer is different. A
    connection that dropped will probably work next turn, so it is retried;
    a key that was refused will not, so it is never retried and never quietly
    failed over to another model the user is then billed for. The three ways
    a key can be refused all end in the same sentence: renew it.
    """

    def __init__(self, message: str, *, kind: str = "key") -> None:
        super().__init__(message, kind=kind)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool offered to the model, described in the one language both speak.

    `parameters` is a JSON Schema object. Gemini calls it a function
    declaration, OpenAI wraps it in a function object - both are that same
    schema in a different envelope.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A complete request from the model to run one tool.

    A live session delivers a call whole - the arguments are not streamed as
    partial JSON the way a text stream's were - and the adapter still
    promises it: the permission gate cannot judge an action it can only see
    the beginning of. The id is what the result is matched to on the way
    back; the name travels with it because Gemini matches by name and
    refuses a result without one.
    """

    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts for one report, in the shape the `usage_log` table stores.

    `cached_tokens` is a subset of `input_tokens`, not an addition to it. Only
    some providers report it; the rest leave it at zero, which reads correctly
    as "no cache hit". The minutes of audio (plan.md D8) are not here: they
    are counted by the session itself (`LiveSession.audio_in_ms`), exactly,
    from the bytes that went each way.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        """What two reports cost together.

        A turn with a tool call in it is two server turns (the model stops at
        the call and speaks the answer as a new one, measured 2026-09-18),
        each with its own report; the turn's cost is their sum.
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
    yet" - the probe replaces it with a measured answer instead of letting
    the assistant fail silently at two in the morning.
    """

    id: str
    display_name: str
    context_window: int | None = None
    supports_tools: bool | None = None
    input_price_per_mtok: float | None = None
    output_price_per_mtok: float | None = None


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Everything a session is opened with, independent of any vendor's config.

    `voice` empty is the model's own default (plan.md D19: the owner's ear
    preferred it). `language_code` is the BCP-47 hint of the locale pack -
    `tr-TR` - for the recogniser behind the model and for its voice; without
    it a short or quiet Turkish sentence is heard as Hindi (ADR-001).
    `end_sensitivity` (`""`, `"HIGH"`, `"LOW"`) and `silence_ms` (0 = the
    server's default) are the two server-side turn-detection knobs the same
    ADR kept for the owner to tune. `resume_handle` is what a previous
    session handed out in `Resumable`; given, the conversation continues.
    """

    model: str
    voice: str = ""
    system_prompt: str = ""
    tools: Sequence[ToolSpec] = ()
    input_rate: int = 16_000
    transcripts: bool = True
    resume_handle: str | None = None
    language_code: str = ""
    end_sensitivity: str = ""
    silence_ms: int = 0


# --------------------------------------------------------------------------
# What a session sends up: eleven events, and nothing else
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A piece of the model's voice: 16-bit little-endian mono PCM at
    `sample_rate` (Gemini and OpenAI both speak at 24 kHz). Played the
    moment it arrives; nothing is buffered for the sake of a whole answer."""

    pcm16: bytes
    sample_rate: int


@dataclass(frozen=True, slots=True)
class Interrupted:
    """The server heard the user speak over the model and stopped it. What
    is still queued for the speaker is thrown away, and nothing said after
    this point counts as said."""


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    """The model asks for one tool, whole. The answer goes back through
    `LiveSession.send_tool_result` - after the gate, and nowhere else."""

    call: ToolCall


@dataclass(frozen=True, slots=True)
class ToolCallCancelled:
    """The model withdrew calls it had made - the user interrupted before
    they were answered. A result for one of these ids is not sent."""

    ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InputText:
    """A piece of the transcript of what the user said, as the server heard
    it. For the status line and the log; the model has already understood
    the audio itself."""

    text: str


@dataclass(frozen=True, slots=True)
class OutputText:
    """A piece of the transcript of what the model is saying."""

    text: str


@dataclass(frozen=True, slots=True)
class TurnComplete:
    """The model has finished its turn. Gemini sends this *at* a tool call
    too, before the result is even sent, and speaks the answer as a new turn
    (measured 2026-09-18); the state machine counts the two as one."""


@dataclass(frozen=True, slots=True)
class UsageReport:
    """What the provider says a turn cost in tokens. Reported when the
    provider reports it and never invented: a session that says nothing
    about its cost sends no report."""

    usage: Usage


@dataclass(frozen=True, slots=True)
class Resumable:
    """A handle with which a later session continues this conversation
    (`SessionConfig.resume_handle`). Gemini hands one out at the open and
    again as the conversation goes on; the latest is the one to keep."""

    handle: str


@dataclass(frozen=True, slots=True)
class GoingAway:
    """The server will hang up soon - the session's hard time cap is near.
    `time_left` is the provider's own text for how soon. The state machine
    reconnects with the latest handle before it does."""

    time_left: str


@dataclass(frozen=True, slots=True)
class Closed:
    """The session is over and no more events are coming. `error` is `None`
    when the socket closed the way sockets close - we hung up, or the server
    did after `GoingAway` - and the `ProviderError` that says why otherwise,
    in the same currency as a refusal at the open."""

    error: ProviderError | None = None


LiveEvent = (
    AudioChunk
    | Interrupted
    | ToolCallEvent
    | ToolCallCancelled
    | InputText
    | OutputText
    | TurnComplete
    | UsageReport
    | Resumable
    | GoingAway
    | Closed
)


# --------------------------------------------------------------------------
# The two protocols
# --------------------------------------------------------------------------


@runtime_checkable
class LiveSession(Protocol):
    """One open conversation with the model. Nothing here mentions a vendor.

    Audio goes in at `SessionConfig.input_rate` as 16-bit mono PCM and comes
    out as `AudioChunk` events. The session counts both ways in milliseconds
    - exactly, from the bytes, which is what live models bill by (plan.md
    D8) - and the counts only grow: the state machine reads them at the end
    of a turn and at the close and takes the difference.
    """

    @property
    def audio_in_ms(self) -> int:
        """Milliseconds of audio sent so far."""
        ...

    @property
    def audio_out_ms(self) -> int:
        """Milliseconds of the model's voice received so far."""
        ...

    async def send_audio(self, pcm16: bytes) -> None:
        """Microphone audio, at the rate the session was opened with."""
        ...

    async def send_text(self, text: str, *, role: str = "user", turn_complete: bool = True) -> None:
        """A text turn. `turn_complete` asks the model to answer it; `False`
        only adds it to the conversation - what an announcement the assistant
        already said aloud needs (plan.md D4), and a probe question does not."""
        ...

    async def send_tool_result(self, call: ToolCall, content: str) -> None:
        """What the tool said back, tied to the call that asked for it."""
        ...

    async def interrupt(self) -> None:
        """Local playback has been stopped by hand (the hotkey); the provider
        is told if it needs telling. Gemini does not - its own detector
        stopped it already."""
        ...

    def events(self) -> AsyncIterator[LiveEvent]:
        """Everything the server sends, translated, until `Closed`.

        Declared `def`, not `async def`, and this is deliberate. Adapters
        write it as `async def ... yield`, an async generator: calling it
        returns an `AsyncIterator` immediately, with no `await`. Declaring
        `async def` here would type it as a coroutine that returns an
        iterator, callers would have to await it first, and no adapter would
        satisfy the protocol. `test_live_adapters.py` guards this.
        """
        ...


@runtime_checkable
class LiveProvider(Protocol):
    """What an adapter has to offer. Nothing here mentions a vendor.

    Vendor-only features - a resumption handle, transcripts, the server's own
    turn detection - are announced through `capabilities` and never added
    here, which would drag the other adapter down to the common denominator.
    """

    id: str
    capabilities: frozenset[str]

    async def validate_credentials(self) -> bool:
        """Reports whether the stored key actually works, before it is saved."""
        ...

    async def list_models(self) -> list[ModelInfo]:
        """Lists the live-capable models this key can reach, for the setup command."""
        ...

    def connect(self, config: SessionConfig) -> AbstractAsyncContextManager[LiveSession]:
        """Opens a session: `async with provider.connect(config) as session`.

        Leaving the block closes the socket. A key the provider refuses is
        `AuthenticationError`, anything else it refuses `ProviderError`, and a
        network that is not there `ProviderError` with `kind="unreachable"` -
        all raised on entering the block, never the SDK's own.
        """
        ...
