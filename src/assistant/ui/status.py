"""The terminal status line - the whole interface of phase 1 (item 1.11).

There is no window and no tray icon yet, and a voice assistant gives the user
nothing to look at by design. So one line of the terminal has to answer the
only question they have while nothing is being said: is it listening, is it
thinking, or has it stopped?

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
"""

from __future__ import annotations

from types import TracebackType

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from assistant.app import State, Turn
from assistant.audio.capture import DEFAULT_TOGGLE_HOTKEY
from assistant.live.base import Usage
from assistant.locales import Locale

__all__ = ["TEXT", "StatusLine", "label_key", "spell"]

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
    "state_idle": "ready",
    "state_listening": "listening",
    "state_transcribing": "writing it down",
    "state_thinking": "thinking",
    "state_confirming": "waiting for a yes or no",
    "state_speaking": "speaking",
    "state_announcing": "reminding",
    "you_said": "you",
    "it_said": "assistant",
    "turn_cost": "{input} in, {output} out",
    "not_caught": "(not caught - confidence {confidence})",
}

# Colour is the fastest way to read a line somebody is not looking at, and it
# carries nothing that is not also written in words - a terminal without colour
# loses no information. A state with no colour of its own is simply plain.
_STYLES: dict[State, str] = {
    State.IDLE: "dim",
    State.LISTENING: "bold green",
    State.TRANSCRIBING: "yellow",
    State.THINKING: "cyan",
    State.CONFIRMING: "bold yellow",
    State.SPEAKING: "magenta",
}


def spell(hotkey: str) -> str:
    """`<ctrl>+<alt>+h` as `Ctrl+Alt+H`.

    `pynput` spells a combination for its own parser; the user reads it off
    their keyboard, where none of the angle brackets appear.
    """
    return "+".join(part.strip("<>").capitalize() for part in hotkey.split("+"))


class StatusLine:
    """One line of terminal, kept up to date with what the assistant is doing."""

    def __init__(
        self,
        locale: Locale,
        *,
        toggle: str = DEFAULT_TOGGLE_HOTKEY,
        console: Console | None = None,
    ) -> None:
        self._said = {key: locale.say(key, default) for key, default in TEXT.items()}
        self._console = console if console is not None else Console()

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

    def state(self, state: State) -> None:
        self._show(self._said[label_key(state)], _STYLES.get(state, ""))

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
        if not finished.heard and not finished.missed:
            # A key tapped by accident, a recording of silence, or a question
            # withdrawn mid-turn. None of them is a turn the user had.
            return

        # A missed turn shows the number instead of the words. There is no
        # transcript worth printing - that is what missed means - and the
        # number is what tells the user whether speaking up would have helped.
        if finished.missed:
            confidence = "-" if finished.confidence is None else f"{finished.confidence:.2f}"
            heard = Text(self._said["not_caught"].format(confidence=confidence), style="dim")
        else:
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
        if hint:
            line.append(f"    {self._hint}", style="dim")
        self._live.update(line, refresh=True)

    def _spent(self, usage: Usage) -> str:
        """What the turn cost, or nothing at all for a turn that failed.

        A failed turn reports no tokens (`app.py`), and `0 in, 0 out` next to
        an error message reads as a claim that the request was free rather than
        as the absence of a number.
        """
        if not usage.input_tokens and not usage.output_tokens:
            return ""
        return self._said["turn_cost"].format(input=usage.input_tokens, output=usage.output_tokens)
