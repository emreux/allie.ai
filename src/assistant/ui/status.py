"""The terminal status line (item 1.11), and the `Screen` both surfaces answer to.

It was the whole interface once. The window of D20 is what `run` opens now, the
tray icon (4.3) can sit beside either, and this line is what `run --terminal`
keeps - still the shortest answer to the only question a user has while nothing
is being said: is it listening, is it thinking, or has it stopped?

**One line, overwritten.** The state changes four times in a turn. Printed as
four lines each, a minute of use buries what was actually said under a hundred
lines of `thinking`. `rich`'s `Live` keeps the state at the bottom and lets
what was heard and answered scroll past above it.

**It does not animate.** The line is redrawn when the state changes and at no
other time - no spinner, no refresh thread. Rule 4 of section 3.1 gives the
event loop fifty milliseconds, and a status line is not what should spend them.

**No sentence is written here.** The pack answers first and the English
constants below are the end of the chain (section 3.12), as everywhere else.
A state is looked up by its own name, so a state added in a later phase needs
a line in `TEXT` and nothing else.

**The session and its minutes are on the line** (plan.md section 4.2). A live
model bills by the minute while a session is open, silent or not (D5), so
the line says whether one is open and how many whole minutes this run has
had one open - cost awareness is a feature now, not a report. Counted here
from the clock, since the usage rows do not carry minutes yet (L1.5).
Whether the server hears the microphone loudly enough is said too (D18):
a level under `QUIET_DBFS` is the first thing to check when the model
answers in another language, and it is said once, on a line that stays.
The line and the window (D20) are both a `Screen`, which is all `run`
knows of either.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from types import TracebackType
from typing import Protocol, runtime_checkable

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from assistant.app import State, Turn
from assistant.audio.capture import DEFAULT_TOGGLE_HOTKEY, QUIET_DBFS
from assistant.live.base import Usage
from assistant.locales import Locale

__all__ = ["TEXT", "QuietNotice", "Screen", "SessionMinutes", "StatusLine", "label_key", "spell"]

# The mark at the start of the line. A shape rather than a word, so it needs
# no translation and no room.
BULLET = "●"


def label_key(state: State) -> str:
    """The `TEXT` key holding the label for `state`.

    Derived from the state's own name rather than kept in a second table: two
    tables to add a state to is one table somebody forgets.
    """
    return f"state_{state}"


TEXT: dict[str, str] = {
    "paused": "Not listening. Press {toggle} to listen again. Ctrl+C stops.",
    "hands_free": (
        "Listening - just talk. {toggle} stops listening (and cuts an answer short). "
        "Ctrl+C stops everything."
    ),
    "loading_speech": "Loading the speech model...",
    "checking_model": "Checking whether the model calls tools...",
    "state_off": "off",
    "state_idle": "ready",
    "state_sleeping": "asleep",
    "state_user_speaking": "hearing you",
    "state_confirming": "waiting for a yes or no",
    "state_speaking": "speaking",
    "state_announcing": "reminding",
    "state_reconnecting": "reconnecting",
    "you_said": "you",
    "it_said": "assistant",
    "turn_cost": "{input} in, {output} out",
    # The session and the meter (plan.md 4.2): shown beside the state.
    "session_open": "session open",
    "session_closed": "session closed",
    "session_minutes": "{minutes} min this run",
    # Said once when the server heard the microphone too quietly (D18); the
    # numbers are dBFS, written by the code.
    "microphone_quiet": (
        "The microphone is quiet: {level} dBFS reached the server, under {quiet} dBFS. "
        "Raise its level or boost in Windows' sound settings."
    ),
}

# Colour is the fastest way to read a line somebody is not looking at, and it
# carries nothing that is not also written in words - a terminal without colour
# loses no information. A state with no colour of its own is simply plain.
_STYLES: dict[State, str] = {
    State.OFF: "dim",
    State.IDLE: "dim",
    State.SLEEPING: "dim",
    State.USER_SPEAKING: "bold green",
    State.CONFIRMING: "bold yellow",
    State.SPEAKING: "magenta",
    State.RECONNECTING: "yellow",
}


def spell(hotkey: str) -> str:
    """`<ctrl>+<alt>+h` as `Ctrl+Alt+H`.

    `pynput` spells a combination for its own parser; the user reads it off
    their keyboard, where none of the angle brackets appear.
    """
    return "+".join(part.strip("<>").capitalize() for part in hotkey.split("+"))


class QuietNotice:
    """Says once, per stretch of quiet, that the microphone is quiet (D18).

    The line and the window show the same sentence by the same rule, and the
    rule was written out in both until 2026-09-22 - eight identical lines, the
    kind that drift apart the day one of them is fixed. `sentence` is the
    pack's, already resolved; `say` is whoever puts a notice on the screen.
    """

    def __init__(self, sentence: str, say: Callable[[str], None]) -> None:
        self._sentence = sentence
        self._say = say
        # Whether it has been said for the stretch of quiet under way; a level
        # that is loud enough arms it again, because a headset changed or a
        # Windows update deserves a new line.
        self._told = False

    def level(self, dbfs: float | None) -> None:
        """What the microphone sent when a stream closed. `None` says nothing:
        either nothing was sent, or the threshold does not apply to the path
        this microphone was opened on (`LiveCapture.level_judged`)."""
        if dbfs is None:
            return
        if dbfs >= QUIET_DBFS:
            self._told = False
            return
        if self._told:
            return
        self._told = True
        self._say(self._sentence.format(level=round(dbfs), quiet=int(QUIET_DBFS)))


class SessionMinutes:
    """How long sessions have been open since the program started.

    What a live model bills by (plan.md D5, D8): the time between open and
    close, silent or not, whole minutes when shown. Told the same thing the
    state machine tells everyone (`on_session`), and safe to be told it
    twice - a reconnect may say "open" while open.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._opened_at: float | None = None
        self._before = 0.0

    @property
    def open(self) -> bool:
        return self._opened_at is not None

    @property
    def minutes(self) -> float:
        """Minutes so far, the session under way included."""
        seconds = self._before
        if self._opened_at is not None:
            seconds += self._clock() - self._opened_at
        return seconds / 60

    def told(self, open: bool) -> None:
        if open and self._opened_at is None:
            self._opened_at = self._clock()
        elif not open and self._opened_at is not None:
            self._before += self._clock() - self._opened_at
            self._opened_at = None


@runtime_checkable
class Screen(Protocol):
    """What `run` shows the assistant on: the terminal's line, or the window
    (`ui/window.py`, plan.md D20). Everything `__main__._talk` asks of one.
    A method here is a queue put on the window, so a call may cost the
    loop nothing more than that."""

    def starting(self) -> None: ...

    def checking_model(self) -> None: ...

    def notice(self, message: str) -> None: ...

    def state(self, state: State) -> None: ...

    def session(self, open: bool) -> None: ...

    def microphone_level(self, dbfs: float | None) -> None: ...

    def hands_free(self, listening: bool) -> None: ...

    def turn(self, finished: Turn) -> None: ...

    def level(self, dbfs: float) -> None: ...


class StatusLine:
    """One line of terminal, kept up to date with what the assistant is doing."""

    def __init__(
        self,
        locale: Locale,
        *,
        toggle: str = DEFAULT_TOGGLE_HOTKEY,
        console: Console | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._said = {key: locale.say(key, default) for key, default in TEXT.items()}
        self._console = console if console is not None else Console()
        # The meter (plan.md 4.2): closed until the state machine opens one.
        self._session = SessionMinutes(clock=clock)
        self._quiet = QuietNotice(self._said["microphone_quiet"], self.notice)

        keys = {"toggle": spell(toggle)}
        # Both are built up front; only which one is shown changes when the
        # mode does. `hands_free` runs on the event loop between two other
        # things, and should cost a lookup rather than a format. Paused until
        # the capture says otherwise - which it does the moment it starts.
        self._hints = {
            False: self._said["paused"].format(**keys),
            True: self._said["hands_free"].format(**keys),
        }
        self._hint = self._hints[False]

        # What is on the line now, so that switching the mode can redraw it
        # without knowing which state the assistant is in.
        self._line = ("", "", False)

        # Drawn only when something changes: `auto_refresh` would start a
        # thread to redraw a line that has not moved.
        self._live = Live(console=self._console, auto_refresh=False)

    def __enter__(self) -> StatusLine:
        self._live.start()
        self.state(State.IDLE)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._live.stop()

    def starting(self) -> None:
        """Whisper is loading. It takes seconds, and a blank terminal during
        them reads as a program that failed to start."""
        self._show(self._said["loading_speech"], "yellow", hint=False)

    def checking_model(self) -> None:
        """The probe of 2.6 is asking the model one question: a second or two
        on the network, before anything else is loaded."""
        self._show(self._said["checking_model"], "yellow", hint=False)

    def notice(self, message: str) -> None:
        """Writes `message` above the line, where it stays.

        For the one thing the user should read once and not have overwritten
        by the next state: that the model failed the probe (2.6). Worded by
        the caller, because the sentence is the caller's.
        """
        self._console.print(Text(message, style="yellow"))

    @property
    def minutes(self) -> float:
        """Minutes of open session this run, the one under way included."""
        return self._session.minutes

    def state(self, state: State) -> None:
        self._show(self._said[label_key(state)], _STYLES.get(state, ""))

    def session(self, open: bool) -> None:
        """Says whether a session is open - the meter is running - and
        redraws the line with the minutes so far."""
        self._session.told(open)
        message, style, hint = self._line
        self._show(message, style, hint=hint)

    def microphone_level(self, dbfs: float | None) -> None:
        """What the server heard, in dBFS, when a stream closed (D18): a
        quiet microphone is said once, on a line that stays, and said again
        only after it was heard loudly enough in between."""
        self._quiet.level(dbfs)

    def level(self, dbfs: float) -> None:
        """The sound, block by block (plan.md D20). The line has no meter
        for it and would not redraw fifty times a second if it had."""

    def hands_free(self, listening: bool) -> None:
        """Says whether the microphone is live.

        The only thing on screen that answers it. A mode the user cannot see
        the state of is a mode they leave on by accident in a room with other
        people in it, which is the one way this feature can cost them money.
        """
        self._hint = self._hints[listening]
        message, style, hint = self._line
        self._show(message, style, hint=hint)

    def turn(self, finished: Turn) -> None:
        """Writes a finished turn above the line, where it stays.

        This is the whole record the user gets in phase 1: nothing is kept
        after the process ends, and the log deliberately holds the numbers
        rather than the words (`logs.py`).
        """
        if not finished.heard:
            # A key tapped by accident, a recording of silence, or a question
            # withdrawn mid-turn. None of them is a turn the user had. (The
            # old pipeline also had turns it could not read, shown here as a
            # confidence; a live model hears the user itself.)
            return

        heard = Text(finished.heard)

        answer = Text(finished.said)
        spent = self._spent(finished.usage)
        if spent:
            answer.append(f"   {spent}", style="dim")

        exchange = Table.grid(padding=(0, 2))
        exchange.add_column(style="dim", justify="right")
        exchange.add_column()
        exchange.add_row(self._said["you_said"], heard)
        exchange.add_row(self._said["it_said"], answer)
        self._console.print(exchange)

    # ----------------------------------------------------------------------

    def _show(self, message: str, style: str, *, hint: bool = True) -> None:
        self._line = (message, style, hint)

        line = Text()
        line.append(f"{BULLET} {message}", style=style or None)
        line.append(f"  {self._meter()}", style="dim")
        if hint:
            line.append(f"    {self._hint}", style="dim")
        self._live.update(line, refresh=True)

    def _meter(self) -> str:
        """`session open · 3 min this run` - whole minutes, never rounded up."""
        which = self._said["session_open" if self._session.open else "session_closed"]
        minutes = self._said["session_minutes"].format(minutes=int(self._session.minutes))
        return f"{which} · {minutes}"

    def _spent(self, usage: Usage) -> str:
        """What the turn cost, or nothing at all for a turn that failed.

        A failed turn reports no tokens (`app.py`), and `0 in, 0 out` next to
        an error message reads as a claim that the request was free rather than
        as the absence of a number.
        """
        if not usage.input_tokens and not usage.output_tokens:
            return ""
        return self._said["turn_cost"].format(input=usage.input_tokens, output=usage.output_tokens)
