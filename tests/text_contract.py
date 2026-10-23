"""The vocabulary the adapter contract is written in (design.md section 3.2).

`test_llm_adapters.py` says what every adapter must do; it must say it without
naming a vendor, or the suite stops being a contract and becomes a second copy
of one adapter's tests. So the tests script a provider in the words below - it
says something, it asks for a tool, it reports what the request cost, it
refuses - and each adapter's own test file translates that script into the
shapes its SDK actually produces.

That translation is the mirror of the production rule: vendor knowledge lives
in the adapter, and nowhere else. Here it lives in the adapter's test file, and
nowhere else. Adding a provider in phase 2.7 is a `Build` function beside the
new adapter's tests and one more entry in the list - the contract suite itself
does not change, which is exactly the claim section 3.2 makes about the
production code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from allie.text.base import LLMProvider

__all__ = [
    "COMPLAINT",
    "Adapter",
    "Build",
    "Calls",
    "Nothing",
    "Refuses",
    "Says",
    "Spends",
    "Starts",
    "Step",
    "Stops",
]


@dataclass(frozen=True, slots=True)
class Says:
    """A piece of the answer, as the provider streams it."""

    text: str


@dataclass(frozen=True, slots=True)
class Nothing:
    """A chunk that only advances the provider's own state.

    Every provider sends them - a role announcement, a keep-alive, an empty
    delta before the first word. None of them is an answer, and turning one
    into a `Delta` would have the agent loop count a turn as having said
    something it never said.
    """


@dataclass(frozen=True, slots=True)
class Calls:
    """A finished request to run a tool: every argument has arrived."""

    name: str
    arguments: Mapping[str, Any]
    id: str = "c1"


@dataclass(frozen=True, slots=True)
class Starts:
    """A tool call whose arguments are still arriving.

    All three providers stream tool arguments as partial JSON, each in its own
    way. This is the half the adapter has to keep to itself until it is whole -
    the permission gate of section 3.9 cannot judge an action it can only see
    the beginning of.
    """

    name: str
    arguments: Mapping[str, Any]
    id: str = "c1"


@dataclass(frozen=True, slots=True)
class Spends:
    """What the provider says the request has cost so far.

    Providers disagree about when this arrives and how often. Scripting it as
    a step rather than as an attribute of the answer is what lets one test ask
    the same question of a provider that reports once and one that repeats a
    running total on every chunk.
    """

    input: int
    output: int
    cached: int = 0


@dataclass(frozen=True, slots=True)
class Stops:
    """Generation ended. What the provider calls the reason is its own affair -
    `STOP`, `stop`, `end_turn` - so no test here reads the word."""


Step = Says | Nothing | Calls | Starts | Spends | Stops


class Refuses(StrEnum):
    """The refusals the application tells apart (section 3.2).

    Each adapter answers these in whatever shape its own provider uses: Gemini
    refuses a key with a 400 whose message says so, others with a 401.

    `THE_NETWORK` is not a refusal by the provider at all - a socket that was
    refused, a name that did not resolve, a stream that timed out - and that
    is exactly why it is here: the SDK raises its transport library's own
    exception for it, which derives from neither `ProviderError` nor `OSError`.
    Measured 2026-09-05, one of those ended the program with a traceback.
    """

    THE_KEY = "the key"
    THE_REQUEST = "the request"
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
    ) -> LLMProvider:
        """`after` is how much of the script arrives before the refusal does:
        zero refuses the request outright, anything else is a stream that dies
        part way through. `models` is what `list_models` should find, as
        (id, display name) pairs."""
        ...


@dataclass(frozen=True, slots=True)
class Adapter:
    """One adapter, and the way to make its provider say things."""

    # The key it is registered under in `llm/registry.py::ADAPTERS`, which is
    # what lets the suite notice an adapter that was added without a harness.
    name: str
    build: Build
