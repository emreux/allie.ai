"""The vocabulary the live adapter contract is written in (plan.md section 4.3).

`test_live_adapters.py` says what every adapter must do; it must say it
without naming a vendor, or the suite stops being a contract and becomes a
second copy of one adapter's tests. So the tests script a *session* in the
words below - the server speaks, says what it said, hears what the user
said, asks for a tool, interrupts, reports what the turn cost, hangs up -
and each adapter's own test file translates that script into the messages
its SDK actually produces.

That translation is the mirror of the production rule: vendor knowledge
lives in the adapter, and nowhere else. Here it lives in the adapter's test
file, and nowhere else. The OpenAI adapter of L2 arrives as a `Build`
function beside its own tests and one more entry in the suite's list - the
contract itself does not change, which is exactly the claim section 4.3
makes about the production code.

At the bottom is the reference implementation: a provider and a session
that satisfy the protocols without a network, replaying events they were
given and recording what was sent. The probe, the registry and the wizard
tests stand it in a real adapter's place, which is only honest as long as
it answers the protocol the way the real ones do - so the contract suite
runs it too.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from assistant.live.base import (
    AudioChunk,
    AuthenticationError,
    Closed,
    LiveEvent,
    LiveProvider,
    ModelInfo,
    ProviderError,
    SessionConfig,
    ToolCall,
)

__all__ = [
    "COMPLAINT",
    "Adapter",
    "Build",
    "Calls",
    "Cancels",
    "FakeLiveProvider",
    "FakeLiveSession",
    "Hears",
    "Interrupts",
    "Leaves",
    "Nothing",
    "Refuses",
    "Resumes",
    "Says",
    "Speaks",
    "Spends",
    "Step",
    "Stops",
]


@dataclass(frozen=True, slots=True)
class Speaks:
    """The model's voice: this many milliseconds of audio, at whatever rate
    the provider speaks at. The contract checks the milliseconds, not the
    rate - the rate is the adapter's to report."""

    ms: int


@dataclass(frozen=True, slots=True)
class Says:
    """A piece of the transcript of what the model said."""

    text: str


@dataclass(frozen=True, slots=True)
class Hears:
    """A piece of the transcript of what the server heard the user say."""

    text: str


@dataclass(frozen=True, slots=True)
class Nothing:
    """A message that only advances the provider's own state.

    Every provider sends them - a setup acknowledgement, a keep-alive, an
    empty content message. None of them is an event, and turning one into
    an event would have the state machine act on something that never
    happened.
    """


@dataclass(frozen=True, slots=True)
class Calls:
    """A request to run a tool, whole."""

    name: str
    arguments: Mapping[str, Any]
    id: str = "c1"


@dataclass(frozen=True, slots=True)
class Cancels:
    """The model withdraws calls it made: the user spoke over it."""

    ids: tuple[str, ...] = ("c1",)


@dataclass(frozen=True, slots=True)
class Interrupts:
    """The server stopped the model because the user spoke."""


@dataclass(frozen=True, slots=True)
class Spends:
    """What the provider says the turn cost.

    Providers disagree about when this arrives. Scripting it as a step
    rather than as an attribute of the turn is what lets one test ask the
    same question of a provider that reports at the end of a turn and one
    that reports whenever it likes.
    """

    input: int
    output: int
    cached: int = 0


@dataclass(frozen=True, slots=True)
class Stops:
    """The model's turn is over."""


@dataclass(frozen=True, slots=True)
class Resumes:
    """The server hands out a handle a later session can continue with."""

    handle: str = "handle-1"


@dataclass(frozen=True, slots=True)
class Leaves:
    """The server says it will hang up soon."""

    time_left: str = "30s"


Step = (
    Speaks
    | Says
    | Hears
    | Nothing
    | Calls
    | Cancels
    | Interrupts
    | Spends
    | Stops
    | Resumes
    | Leaves
)


class Refuses(StrEnum):
    """The refusals the application tells apart (section 4.3).

    Each adapter answers these in whatever shape its own provider uses:
    Gemini refuses a key with a 400 whose message says so, others with a
    401; the free tier says "slow down" with a 429.

    `THE_NETWORK` is not a refusal by the provider at all - a socket that
    was refused, a name that did not resolve, a stream that stopped - and
    that is exactly why it is here: the SDK raises its transport library's
    own exception for it, which derives from neither `ProviderError` nor
    `OSError`.
    """

    THE_KEY = "the key"
    THE_REQUEST = "the request"
    THE_QUOTA = "the quota"
    THE_NETWORK = "the network"


# What a provider is made to say when it refuses the request, so that a test
# can look for it in the exception without knowing whose provider it was.
COMPLAINT = "the model is overloaded"


class Build(Protocol):
    """Builds an adapter whose provider will do exactly what the script says."""

    def __call__(
        self,
        *script: Step,
        refuses: Refuses | None = None,
        after: int = 0,
        models: Sequence[tuple[str, str]] = (),
    ) -> LiveProvider:
        """The script is what the server sends once a session is open, in
        order; when it runs out the server hangs up the way servers do.
        `after` is how much of it arrives before the refusal does: zero
        refuses the open outright (and `list_models`), anything else is a
        session that dies part way through. `models` is what `list_models`
        should find, as (id, display name) pairs."""
        ...


@dataclass(frozen=True, slots=True)
class Adapter:
    """One adapter, and the way to make its provider say things."""

    # The key it is registered under in `live/registry.py::ADAPTERS`, which
    # is what lets the suite notice an adapter that was added without a
    # harness.
    name: str
    build: Build


# --------------------------------------------------------------------------
# The reference implementation
# --------------------------------------------------------------------------


class FakeLiveSession:
    """A session that replays the events it was given and records what was
    sent. `events()` yields the script and then `Closed()`, unless the
    script ends in a `Closed` of its own."""

    def __init__(self, events: Sequence[LiveEvent] = (), *, input_rate: int = 16_000) -> None:
        self.script = list(events)
        self.audio: list[bytes] = []
        self.texts: list[tuple[str, str, bool]] = []
        self.results: list[tuple[ToolCall, str]] = []
        self.interrupts = 0
        self.audio_in_ms = 0
        self.audio_out_ms = 0
        self.read = 0
        self._input_rate = input_rate

    async def send_audio(self, pcm16: bytes) -> None:
        self.audio.append(pcm16)
        self.audio_in_ms += len(pcm16) * 1000 // (self._input_rate * 2)

    async def send_text(self, text: str, *, role: str = "user", turn_complete: bool = True) -> None:
        self.texts.append((text, role, turn_complete))

    async def send_tool_result(self, call: ToolCall, content: str) -> None:
        self.results.append((call, content))

    async def interrupt(self) -> None:
        self.interrupts += 1

    async def events(self) -> AsyncIterator[LiveEvent]:
        for event in self.script:
            self.read += 1
            if isinstance(event, AudioChunk):
                self.audio_out_ms += len(event.pcm16) * 1000 // (event.sample_rate * 2)
            yield event
            if isinstance(event, Closed):
                return
        yield Closed()


class FakeLiveProvider:
    """A provider that opens `FakeLiveSession`s scripted with `events`, or
    refuses to with `refusal`, and remembers every session it opened."""

    id = "fake"
    capabilities: frozenset[str] = frozenset()

    def __init__(
        self,
        events: Sequence[LiveEvent] = (),
        *,
        models: Sequence[ModelInfo] = (),
        refusal: ProviderError | None = None,
    ) -> None:
        self.events = list(events)
        self.models = list(models)
        self.refusal = refusal
        self.opened: list[SessionConfig] = []
        self.sessions: list[FakeLiveSession] = []

    async def validate_credentials(self) -> bool:
        # The same rule as the real adapters: only a refused key is `False`;
        # a provider that could not be reached raises, so that the wizard
        # does not send the user to renew a key that works.
        try:
            await self.list_models()
        except AuthenticationError:
            return False
        return True

    async def list_models(self) -> list[ModelInfo]:
        if self.refusal is not None:
            raise self.refusal
        return list(self.models)

    @asynccontextmanager
    async def connect(self, config: SessionConfig) -> AsyncIterator[FakeLiveSession]:
        self.opened.append(config)
        if self.refusal is not None:
            raise self.refusal
        session = FakeLiveSession(self.events, input_rate=config.input_rate)
        self.sessions.append(session)
        yield session
