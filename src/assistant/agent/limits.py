"""The limits of design.md section 3.11: what one turn may do, what a day may cost.

The loop of `core.py` has no exit of its own. A model that did not get what
it wanted from a tool asks again, with the same arguments, and would go on
asking - every round a request paid for (architecture guide section 12). So
the loop is given a referee for the turn, `TurnGuard`, asked before every
tool call and answering either "go" or the sentence the model reads instead
of a result. Three limits are its to keep: how many calls a turn may make,
how many times in a row the same call may be made, and - through
`Limits.output_tokens`, which the loop hands to the provider - how long an
answer may be. The two spending limits, the gate's look-back window and the
turn's own clock ride in the same `Limits`, because section 3.11 is one table;
who reads which row is written on the class.

**Written here and not in an adapter** (invariant 3). Inside an adapter each
limit would be written three times, once per vendor, and one of the three
would be forgotten. This file sees a `ToolCall` and nothing of any vendor.

**The refusal is a tool result, not a lecture.** It goes back in the tool's
own channel, where the model reads it as what the tool said and moves on -
a user message it could argue with, and would.

**A repeat is the same name with the same arguments,** judged by the hash
`store/repos.py` writes into `tool_audit`, so that the guard, the gate and the
audit row agree about what "the same call" means. Not by the call's id:
Gemini sends none (measured 2026-09-09, `id=""` on every call), and two ids
for the same request are exactly the repeat that has to be caught.

**Every call the model makes counts, refused or not.** A refusal that did
not count would leave a model repeating itself with no ceiling on the rounds
it costs; counted, the eighth refusal is the last thing the model is offered
tools for.

The numbers are the defaults of the section 3.11 table, written once, here.
`config.py` takes its `[limits]` defaults from this class, and what
`config.toml` names replaces them through `from_settings`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from assistant.live.base import ToolCall
from assistant.store.repos import args_hash

if TYPE_CHECKING:
    from assistant.config import LimitSettings

__all__ = ["DUPLICATE_CALL", "TOOL_LIMIT_REACHED", "Limits", "TurnGuard"]

# What a call over the limit is answered with - as a tool result, in the
# model's own channel, rather than as a user message it might argue with.
TOOL_LIMIT_REACHED = "Tool limit reached; answer with what you have."

# The identical call one time too many. The model reads this and changes
# course (architecture guide section 12).
DUPLICATE_CALL = (
    "You already called this tool {times} times in a row with the same arguments; "
    "try a different approach."
)


@dataclass(frozen=True, slots=True)
class Limits:
    """The table of section 3.11, as numbers.

    `tool_calls_per_turn`, `duplicate_calls` and `output_tokens` are the
    loop's (`TurnGuard`, `core.py`). `turn_seconds` is the `THINKING`
    timeout of `app.py`. `daily_usd`, `monthly_usd` and `hard_stop` are read
    by `usage/tracker.py`. `duplicate_window_sec` is the gate's, for the "you
    already did this" sentence.
    """

    tool_calls_per_turn: int = 8
    output_tokens: int = 4000
    duplicate_calls: int = 2
    turn_seconds: float = 60.0
    daily_usd: float = 2.0
    monthly_usd: float = 30.0
    hard_stop: bool = False
    duplicate_window_sec: int = 600

    @classmethod
    def from_settings(cls, settings: LimitSettings) -> Limits:
        """What `config.toml` `[limits]` says, field for field."""
        return cls(
            tool_calls_per_turn=settings.tool_calls_per_turn,
            output_tokens=settings.output_tokens,
            duplicate_calls=settings.duplicate_calls,
            turn_seconds=settings.turn_seconds,
            daily_usd=settings.daily_usd,
            monthly_usd=settings.monthly_usd,
            hard_stop=settings.hard_stop,
            duplicate_window_sec=settings.duplicate_window_sec,
        )


class TurnGuard:
    """One turn's referee: asked before every tool call, answers go or no.

    A new one per turn - the counts start at zero with the turn and die
    with it. Nothing here runs a tool or talks to a provider; it counts.
    """

    def __init__(self, limits: Limits) -> None:
        self._limits = limits
        self._made = 0
        self._last: tuple[str, str] | None = None
        self._run = 0

    @property
    def exhausted(self) -> bool:
        """Whether the turn has made every call it may.

        Past this point the loop offers the model no tools, so the only
        move left to it is to answer (section 3.11).
        """
        return self._made >= self._limits.tool_calls_per_turn

    def allow(self, call: ToolCall) -> str | None:
        """`None` to let `call` through; otherwise what the model is told instead."""
        if self.exhausted:
            return TOOL_LIMIT_REACHED
        self._made += 1

        key = (call.name, args_hash(call.arguments))
        self._run = self._run + 1 if key == self._last else 1
        self._last = key
        if self._run > self._limits.duplicate_calls:
            return DUPLICATE_CALL.format(times=self._limits.duplicate_calls)
        return None
