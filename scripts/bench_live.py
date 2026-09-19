"""The product's own numbers, from a real run (plan.md section 7, L1.9).

`live-assistant run`, with a stopwatch on the state machine's hooks. Nothing
in `src/` changes and nothing is stood in for: the real microphone, the real
doorman, the real session, the real sound card. What is measured is what the
user feels, and it is printed once the run ends (Ctrl+C), in the shape the
worklog wants:

- how long a session takes to open, from the doorman's onset to the socket
  being ready (and, after a `GoingAway`, from the old socket closing to the
  new one being ready);
- how long after you stop talking the model's first sound arrives, per turn
  (`Turn.first_sound_ms`, the number the log line carries too);
- how long after you start talking over the model it falls silent - from the
  doorman's onset while the model speaks to the sound card being stopped;
- how many minutes went over the wire each way, per turn and in all, and how
  long sessions were open (what a live model bills for).

    uv run python scripts/bench_live.py
    uv run python scripts/bench_live.py --device "Intel Smart Sound"

The same flags as `run` (`--device`, `--tray`). Beware one number: a voice the
doorman takes for yours while the model speaks - the room carrying its voice
back, a cough - is counted as talking over it, and if the server does not
agree, the "silence" that follows is the answer's own end, seconds later. A
long interruption number is that, not a slow stop.
"""

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from assistant import app
from assistant.__main__ import main as run
from assistant.app import LiveAssistant, State, Turn

# How much of a transcript one summary line carries.
SHOWN_CHARS = 40


@dataclass
class Meter:
    """The stopwatch: what the hooks said, and when."""

    state: State = State.IDLE
    session_open: bool = False
    onset_at: float | None = None
    opened_at: float | None = None
    closed_at: float | None = None
    barge_at: float | None = None
    started_at: float = field(default_factory=time.monotonic)
    opens_ms: list[float] = field(default_factory=list)
    reopens_ms: list[float] = field(default_factory=list)
    interrupts_ms: list[float] = field(default_factory=list)
    open_seconds: float = 0.0
    turns: list[Turn] = field(default_factory=list)

    def speech(self, speaking: bool) -> None:
        if not speaking:
            return
        now = time.monotonic()
        if not self.session_open and self.onset_at is None:
            self.onset_at = now
        if self.state is State.SPEAKING:
            self.barge_at = now

    def session(self, opened: bool) -> None:
        now = time.monotonic()
        self.session_open = opened
        if opened:
            if self.onset_at is not None:
                self.opens_ms.append((now - self.onset_at) * 1000)
            elif self.state is State.RECONNECTING and self.closed_at is not None:
                self.reopens_ms.append((now - self.closed_at) * 1000)
            self.onset_at = None
            self.opened_at = now
        else:
            if self.opened_at is not None:
                self.open_seconds += now - self.opened_at
            self.opened_at = None
            self.closed_at = now

    def told(self, state: State) -> None:
        if self.barge_at is not None and state is not State.SPEAKING:
            self.interrupts_ms.append((time.monotonic() - self.barge_at) * 1000)
            self.barge_at = None
        self.state = state

    def turn(self, finished: Turn) -> None:
        self.turns.append(finished)


METER = Meter()


class Watched:
    """The product's capture, with the doorman's word passed through the
    stopwatch on its way to the state machine. Everything else is the
    capture's own."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def on_speech(self) -> Callable[[bool], None] | None:
        return self._inner.on_speech

    @on_speech.setter
    def on_speech(self, told: Callable[[bool], None] | None) -> None:
        if told is None:
            self._inner.on_speech = None
            return

        def timed(speaking: bool) -> None:
            METER.speech(speaking)
            told(speaking)

        self._inner.on_speech = timed

    @property
    def on_mode(self) -> Callable[[bool], None] | None:
        return self._inner.on_mode

    @on_mode.setter
    def on_mode(self, told: Callable[[bool], None] | None) -> None:
        self._inner.on_mode = told

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class Timed(LiveAssistant):
    """The state machine as it is, its hooks read by the stopwatch first."""

    def __init__(
        self,
        *,
        capture: Any,
        on_state: Callable[[State], None] | None = None,
        on_turn: Callable[[Turn], None] | None = None,
        on_session: Callable[[bool], None] | None = None,
        **rest: Any,
    ) -> None:
        super().__init__(
            capture=Watched(capture),
            on_state=_both(METER.told, on_state),
            on_turn=_both(METER.turn, on_turn),
            on_session=_both(METER.session, on_session),
            **rest,
        )


def _both[T](first: Callable[[T], None], then: Callable[[T], None] | None) -> Callable[[T], None]:
    def told(value: T) -> None:
        first(value)
        if then is not None:
            then(value)

    return told


def _spread(values: Sequence[float]) -> str:
    """`min / median / max ms (n)`, or the admission that nothing was measured."""
    if not values:
        return "not measured"
    low, mid, high = min(values), statistics.median(values), max(values)
    return f"{low:.0f} / {mid:.0f} / {high:.0f} ms  (n={len(values)})"


def _shown(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SHOWN_CHARS else text[: SHOWN_CHARS - 1] + "…"


def summary(meter: Meter) -> str:
    wall = (time.monotonic() - meter.started_at) / 60
    lines = [
        "",
        f"bench_live: {len(meter.opens_ms) + len(meter.reopens_ms)} sessions, "
        f"{len(meter.turns)} turns, {meter.open_seconds / 60:.1f} min open, {wall:.1f} min wall",
        f"  session open (onset → ready)      {_spread(meter.opens_ms)}",
        f"  reopen after GoingAway            {_spread(meter.reopens_ms)}",
        f"  first sound after you stop        "
        f"{_spread([t.first_sound_ms for t in meter.turns if t.first_sound_ms is not None])}",
        f"  interruption → silence            {_spread(meter.interrupts_ms)}",
    ]
    if meter.turns:
        lines.append("  per turn:")
        for number, turn in enumerate(meter.turns, start=1):
            first = "-" if turn.first_sound_ms is None else f"{turn.first_sound_ms:.0f} ms"
            cost = "price unknown" if turn.cost_usd is None else f"${turn.cost_usd:.4f}"
            failed = f", failed: {turn.failure}" if turn.failure else ""
            lines.append(
                f"    #{number} {_shown(turn.heard)!r:<44} first {first:>8}, "
                f"in {turn.audio_in_ms / 1000:5.1f} s, out {turn.audio_out_ms / 1000:5.1f} s, "
                f"{turn.tool_calls} tools, {cost}{failed}"
            )
        total_in = sum(t.audio_in_ms for t in meter.turns) / 60_000
        total_out = sum(t.audio_out_ms for t in meter.turns) / 60_000
        lines.append(
            f"  audio over the wire               in {total_in:.2f} min, out {total_out:.2f} min"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """`run` with the stopwatch on, then the summary."""
    app.LiveAssistant = Timed  # type: ignore[misc]
    code = run(["run", *(sys.argv[1:] if argv is None else argv)])
    print(summary(METER))
    return code


if __name__ == "__main__":
    sys.exit(main())
