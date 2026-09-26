"""The state machine of the live product (plan.md sections 4.2-4.4).

`IDLE` at the door, `USER_SPEAKING` while the doorman hears a voice,
`SPEAKING` while the model's voice plays, and back; `CONFIRMING` is the
window of the permission gate, opened from inside a tool round; `ANNOUNCING`
the announce queue being read between turns; `RECONNECTING` a session being
reopened with the handle it left behind; `OFF` the switch. The session itself
is opened by speech and closed by silence (D5), and everything in between is
an event the session sends and this file answers.

Everything is injected - the microphone, the provider, the runner, the local
recogniser, the program's own voice, the sound card - so a whole conversation can be driven in a
test without any of them. That is also the reason this file is short: each
piece already knows how to do its own job, and what is left here is the
order they do it in, and what happens when one of them fails.

**A session opens on speech and closes on silence** (rule 1). The doorman -
the local detector inside the capture - reports the onset, the capture has
already queued the pre-roll, and a session is opened for it; sixty seconds
of nothing said and nothing played closes it again, because a live model
bills by the minute while open. An opening that fails is said out loud and
the door is kept; three failures in a row switch the assistant off, so that
a dead network does not cost a sentence a minute.

**The model's voice plays as it arrives and stops the instant the server
says so** (rule 2). Nothing is buffered for the sake of a whole answer, and
`Interrupted` drops what is queued and aborts what the sound card holds.
Nothing said after that point is written down as said.

**A tool call runs through the gate and nowhere else** (rule 3). The runner
of `agent/core.py` asks the guard, hands the call to the gate and sends the
words back; what this file adds is the one who answers a tool's question -
`confirm` - and the filler said when a tool takes its time.

**A tool that wants a yes gets one out loud, or does not run.** The gate
hands `confirm` the sentence with the real argument values in it; this file
waits for the model to finish what it was saying, pauses the session's input
(D3: nothing said in the window reaches the model), reads the question in
the assistant's voice from a session of its own (D32), tells the user how
to answer, and opens the microphone for six seconds; Google's recogniser
reads what was said (D36). A "no" anywhere in the answer wins
over a "yes"; silence is a no, and so is a voice or the switch while the
question is still being read; an answer with neither word in it is asked
about once more, and a second such answer is a no as well. The exchange
belongs to the gate, not to the model: only the tool's result reaches it.
The words that count as yes and no come from the locale pack; the English
ones below are the end of the chain.

**Switching off is an interruption and a hang-up** (rule 4). `Ctrl+Alt+H`
is the only key, and off means silent as well as deaf: the speaker stops,
the session closes, and the turn ends without a word. The capture reports
the switch through `on_mode`; this file acts on it and only then passes it
to the screen.

**A reminder is said between turns, and only then** (rule 5, D4). The
scheduler writes to the announce queue and never to the speaker; the queue
is read when nothing is under way - no voice, no answer owed, no tool, no
sound - and said in the assistant's voice with the session's input paused; then
one line goes to the session so the model knows what was said. Listening
switched off does not silence it: the assistant was asked not to listen,
not to forget the dentist. A voice that starts cuts it off like an answer.

**Asleep, only the wake word is heard** (D21, 2026-09-21). With a wake word
configured the machine starts `SLEEPING`: the capture feeds the detector
and nothing else, no session can open. The phrase wakes it - a chime or a
sentence, then a session at once - and the idle close puts it back to
sleep. The key still switches it off and on; on from off is awake. A
reminder is said asleep as it is said off.

**What a turn cost is written down** (rule 6) when the model is done with
it: what the user said and the model answered, the audio minutes both ways,
the tokens when the provider reports them, priced by the tracker. A turn is
one utterance and everything the model did until it said it was done - and
Gemini says so *at* a tool call too, so a `TurnComplete` with a call in it
or a tool still running is not the end of anything (ADR-001 section 8).
Past the day's or the month's limit every new session opens with a warning,
and with `hard_stop` on none is opened at all.

**Three failures are said out loud, and no others** (rule 7). A refused key
means the user has to go and renew it, a provider that cannot be reached
means try again, and one that took too long means the same. Anything else
is a bug in this project, and swallowing it into "I could not connect" is
how it would never get fixed: it comes out of `run` as the exception it is.

**A recording is judged by whether it held speech, never by how sure the
decoder was of its words.** The local engine this was measured on
(2026-09-05; gone since D36) answered silence with confident looking words
and a correct single word with unsure ones - a subtitle credit over silence
scored 0.54 and "Merhaba." alone 0.48 - while its own estimate of whether
anything was said separated them (0.86 against 0.06). Google's recogniser
says the same thing its own way: no final transcript is "nothing to
decode". `hear` below rests on that and on nothing else, and serves one
place: the yes-and-no window.

**No user-facing sentence is written here** (rule 8). The pack answers first
and the English constants below are the end of the chain, exactly as in the
wizard (section 3.12).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Literal, Protocol

from loguru import logger

from allie.agent.core import ToolRunner
from allie.agent.prompts import ANNOUNCED_PREFIX
from allie.announce.queue import Announcement, AnnounceQueue
from allie.audio.player import LivePlayback, PlaybackError, Speaker
from allie.audio.wake import CHIME_RATE, chime
from allie.config import LiveSettings
from allie.live.base import (
    AudioChunk,
    AuthenticationError,
    Closed,
    GoingAway,
    InputText,
    Interrupted,
    LiveEvent,
    LiveProvider,
    LiveSession,
    OutputText,
    ProviderError,
    Resumable,
    SessionConfig,
    ToolCallCancelled,
    ToolCallEvent,
    TurnComplete,
    Usage,
    UsageReport,
)
from allie.locales import Locale
from allie.store.normalize import normalize_search
from allie.stt.base import NO_SPEECH_CEILING, SAMPLE_RATE, Audio, STTProvider, Transcript
from allie.tts.base import TTSProvider
from allie.usage.tracker import UsageTracker
from allie.web.search import Searched

__all__ = [
    "CONFIRM_WINDOW_SECONDS",
    "FILLERS",
    "FILLER_DELAY_SECONDS",
    "IDLE_CLOSE_SECONDS",
    "MAX_OPEN_FAILURES",
    "MIN_UTTERANCE_SECONDS",
    "NO_WORDS",
    "RESUME_MINUTES",
    "TEXT",
    "YES_WORDS",
    "Answer",
    "Capture",
    "Heard",
    "LiveAssistant",
    "State",
    "Turn",
    "confirm_words",
    "hear",
    "read_answer",
]


class State(StrEnum):
    """Where the assistant is. A string so the status line and the logs can
    print it without a table of names.

    The set of plan.md section 4.2. `USER_SPEAKING` is the local detector's
    opinion and nothing else - the server decides where a turn ends - and is
    there for the screen. `OFF` is a state rather than a mode, because a
    session is something the switch closes. `SLEEPING` is behind the wake
    word (D21): the microphone open, only the phrase listened for.
    """

    OFF = "off"
    SLEEPING = "sleeping"
    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    SPEAKING = "speaking"
    CONFIRMING = "confirming"
    ANNOUNCING = "announcing"
    RECONNECTING = "reconnecting"


# Section 3.1 rule 6. How long the microphone stays open for a yes or a no
# after the question has been read; what comes after it is a no.
CONFIRM_WINDOW_SECONDS = 6.0

# How one confirmation window ended (`LiveAssistant._ask`): a word of the
# pack's, neither word (worth one more question), nothing heard, or the
# exchange stopped from outside.
Answer = Literal["yes", "no", "neither", "unheard", "stopped"]

# Shorter than this and it was a noise rather than a word. Below a syllable,
# so nothing anybody meant to say is thrown away.
MIN_UTTERANCE_SECONDS = 0.35

# The session policy of D5, written once in `config.py`: `config.toml`
# `[live]` replaces both through the constructor.
IDLE_CLOSE_SECONDS = LiveSettings().idle_close_seconds
RESUME_MINUTES = LiveSettings().resume_minutes

# Rule 1: how many sessions in a row may fail to open before the assistant
# switches itself off rather than say the same sentence at every onset.
MAX_OPEN_FAILURES = 3

# The last link of the chain of section 3.12 for the two words the window
# listens for, as `TEXT` is for the sentences: the pack's `[speech]` table
# answers first, and a pack that has none gets these.
YES_WORDS = ("yes", "ok", "okay", "confirm")
NO_WORDS = ("no", "cancel", "stop")

# The end of the same chain for what is said while a tool takes its time:
# the pack's `[speech] filler` answers first. Said in turn, so a pack that
# lists two is not heard saying the same one every time.
FILLERS = ("One moment, let me check...",)

# How long a tool round may stay silent before the filler is said. A native
# tool answers in milliseconds and the model's first sound follows within a
# second of the result (ADR-001 section 3: the model is silent for the whole
# of a tool call, `NON_BLOCKING` or not); above a second the round is slow
# for some other reason - a page, the Store, the weather - and that is the
# silence worth saying something about. Measured 2026-09-10 in the old
# product: a filler said at 300 ms held a ready answer back for two seconds.
FILLER_DELAY_SECONDS = 1.0

# What is done at the wake word (D21): the chime, a sentence in the local
# voice, or nothing. The config validates the word; this is its type.
Greeting = Literal["chime", "sentence", "none"]

# The last link of the chain of section 3.12: what is said when no locale pack
# offers a translation. Keys are unique across the whole project - the pack has
# one table of sentences, and `test_locales.py` checks that no two modules
# claim the same key.
TEXT: dict[str, str] = {
    # The three failures said out loud when there is no session to say them
    # (plan.md 4.4 rule 7), in the assistant's voice.
    "unreachable": "I could not reach the provider. Will you try again?",
    "key_invalid": "Your API key is not being accepted any more. You need to renew it.",
    "took_too_long": "That took too long. Will you try again?",
    # Read after the gate's question, so the user knows what kind of answer is
    # being listened for - and again, alone, when the answer had neither.
    "confirm_hint": "Say yes or no.",
    "confirm_again": "I did not catch that. Yes, or no?",
    # Said before a session opens once a spending limit is passed - every
    # time, until the day or the month turns; and instead of a session with
    # `hard_stop` on.
    "daily_over": "You have gone over today's spending limit.",
    "monthly_over": "You have gone over this month's spending limit.",
    "spend_stopped": "The spending limit has been passed, so I am not asking the model.",
    # Said in the assistant's voice at the wake word when `[wake] greeting =
    # "sentence"` (D21); the chime is the default.
    "wake_greeting": "I am listening.",
}


@dataclass(frozen=True, slots=True)
class Heard:
    """What the recogniser made of a recording, and whether it is worth a reply.

    Three outcomes rather than two, because "nothing happened" and "you said
    something and I could not read it" are different things to the person in
    the chair. Silence over a quiet room deserves silence; speech that came
    back as no words deserves to be asked about again, or the assistant
    looks broken at exactly the moment it is working as designed.
    """

    text: str = ""
    missed: bool = False
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Turn:
    """What one turn came to, for whoever is showing or logging it.

    Nothing else keeps any of this: the transcript is gone once the model has
    it and the answer once it has been spoken. The minutes and the tokens go
    to the log every turn, which is what the cost report of section 6 is
    built from - so they have to leave the turn, and a turn that only
    reported success would hide the ones that cost money and still failed.

    A live turn is one utterance and everything the model did until it said
    it was done (D9). `heard` and `said` are the server's transcripts of the
    two, when it sends them; `audio_in_ms` and `audio_out_ms` are what went
    over the wire both ways, exact from the bytes (D8); `usage` is the tokens
    when the provider reported them, and a real zero when it did not.

    A turn that *failed* carries the key of the sentence that was said instead
    of an answer - `unreachable`, `key_invalid`, `took_too_long`,
    `spend_stopped` - and nothing else about the failure. The log needs the
    kind; the provider's own words are where a key could travel and stay in
    the exception.

    `turn_id` is the name the turn goes by in `tool_audit` (section 3.9),
    so that a row there and a line in the log can be read together. A turn
    that never reached a session has none.

    `cost_usd` is what the turn cost at its model's price - `None` when the
    price is not known, or the turn never reached the model - and
    `tool_calls` how many calls the gate ran. `first_sound_ms` is how long
    after the user stopped talking the model's first sound arrived - the
    number a live product is about - and `None` when there was none, or
    nobody was heard to stop.

    `searched` is what Google was asked while the turn ran (D29), for the
    row the screen shows between the two; empty when nothing was looked up.

    It carried three more fields until 2026-09-22 - `missed`, `confidence`,
    `intent` - which the old pipeline set when its recogniser could not read a
    recording or when a short command was answered without the model (D7).
    A live turn set none of the three, so every reader of them was a branch
    that could not run: the log line, the transcript row, a pack key in two
    languages.
    """

    heard: str = ""
    said: str = ""
    usage: Usage = field(default_factory=Usage)
    failure: str | None = None
    turn_id: str = ""
    cost_usd: float | None = None
    tool_calls: int = 0
    first_sound_ms: float | None = None
    audio_in_ms: int = 0
    audio_out_ms: int = 0
    searched: tuple[str, ...] = ()


@dataclass(slots=True)
class _Turn:
    """The turn under way, gathered as the session's events go by; `Turn`
    is made of it once the model is done."""

    turn_id: str = ""
    heard: list[str] = field(default_factory=list)
    said: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    # The session's counters when the turn began, so that what the turn sent
    # and received is a difference.
    in_at: int = 0
    out_at: int = 0
    # Something is under way: the user spoke, the model owes an answer or is
    # giving it. Cleared once the answer has been heard.
    open: bool = False
    # A tool was asked for in the server turn under way: its `TurnComplete`
    # is not the end of the user's turn (ADR-001 section 8).
    called: bool = False
    # The server stopped the model: nothing after that is written down as said.
    cut: bool = False
    first_sound_ms: float | None = None
    # Whether the model has made a sound since the last tool call, and the
    # filler waiting to be said if it does not (once a turn, however many
    # rounds): dropped the moment the model sounds, or the turn ends.
    sounded: bool = True
    filler: asyncio.Task[None] | None = None
    audio_in_ms: int = 0
    audio_out_ms: int = 0
    tool_calls: int = 0
    # What was looked up during it (D29), taken the moment it ends so that
    # the next turn's searches are never the last one's.
    searched: tuple[str, ...] = ()

    @property
    def counts(self) -> bool:
        """Whether there is anything to write down about it: words either
        way, a tool, tokens, or minutes - a turn with transcripts off is
        sound alone, and still cost."""
        return bool(
            self.heard
            or self.said
            or self.tool_calls
            or self.usage.input_tokens
            or self.audio_in_ms
            or self.audio_out_ms
        )


class Capture(Protocol):
    """The microphone, as the state machine needs it (`audio/capture.py`
    `LiveCapture`, plan.md section 4.1)."""

    # Called on the event loop when the doorman hears a voice begin, and
    # again when it stops. The onset is what opens a session; the capture
    # has already queued the pre-roll by then.
    on_speech: Callable[[bool], None] | None

    # Called on the event loop when listening is switched on or off - and
    # once at `start`, with the mode it began in. Off is a hang-up.
    on_mode: Callable[[bool], None] | None

    # Called on the event loop when the wake word was heard (D21), after
    # the stream has been opened. `None` where there is no wake word.
    on_wake: Callable[[], None] | None

    @property
    def listening(self) -> bool: ...

    @property
    def asleep(self) -> bool:
        """Whether only the wake word is listened for."""
        ...

    @property
    def wake_word(self) -> bool:
        """Whether there is a wake word to sleep behind."""
        ...

    def sleep(self) -> None:
        """Back behind the wake word: the stream ends; nothing without one."""
        ...

    def wake(self) -> None: ...

    @property
    def taking(self) -> bool: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def toggle(self) -> None: ...

    def mute(self) -> None:
        """Stops listening while the assistant speaks - on a microphone that
        needs it (half duplex, D18); nothing on one that does not."""
        ...

    def unmute(self) -> None: ...

    def pause(self) -> None:
        """Nothing is forwarded to the session until `resume`; the doorman
        keeps watching."""
        ...

    def resume(self) -> None: ...

    def close(self) -> None:
        """Ends the stream `chunks` reads; the next onset opens a new one."""
        ...

    def chunks(self) -> AsyncIterator[bytes]:
        """The stream the doorman opened, 16-bit PCM at `SAMPLE_RATE`, until
        `close`; empty at once when none is open."""
        ...

    async def listen_for(self, seconds: float) -> Audio | None:
        """One sentence within `seconds`, listening on or off; `None` if none came.

        The window of the gate. What is said in it is an answer to the
        assistant, not a question for it: not reported through `on_speech`
        and, with the input paused, never sent to the session.
        """
        ...


class LiveAssistant:
    """One conversation after another, for as long as the program runs."""

    def __init__(
        self,
        *,
        capture: Capture,
        provider: LiveProvider,
        session_config: Callable[[], SessionConfig],
        tool_runner: ToolRunner,
        tts: TTSProvider,
        stt: STTProvider,
        speaker: Speaker,
        locale: Locale,
        tracker: UsageTracker | None = None,
        announcements: AnnounceQueue | None = None,
        idle_close_seconds: float = IDLE_CLOSE_SECONDS,
        resume_minutes: float = RESUME_MINUTES,
        filler_delay: float = FILLER_DELAY_SECONDS,
        greeting: Greeting = "chime",
        searched: Searched | None = None,
        on_state: Callable[[State], None] | None = None,
        on_turn: Callable[[Turn], None] | None = None,
        on_mode: Callable[[bool], None] | None = None,
        on_session: Callable[[bool], None] | None = None,
    ) -> None:
        self._capture = capture
        self._provider = provider
        # What is done at the wake word (D21).
        self._greeting = greeting
        # What `look_up` and `x_trends` searched (D29), taken at every
        # turn's end for the screen. Without one, no turn has searches.
        self._searched = searched
        # What a session is opened with, read at every open: the prompt
        # carries the user's facts and the time, and both move.
        self._session_config = session_config
        self._runner = tool_runner
        self._tts = tts
        self._stt = stt
        self._speaker = speaker
        self._locale = locale
        self._tracker = tracker
        # What the scheduler wants said (invariant 5), read between turns.
        # Without one - most tests, and a run with no scheduler - nothing
        # is ever announced.
        self._announcements = announcements
        self._idle_close = idle_close_seconds
        self._resume_seconds = resume_minutes * 60
        self._filler_delay = filler_delay
        self._on_state = on_state
        self._on_turn = on_turn
        # The screen's listener for the mode. Told after this file has acted
        # on a switch-off, never before (`_mode_changed`).
        self._on_mode = on_mode
        self._on_session = on_session

        self._said = {key: locale.say(key, default) for key, default in TEXT.items()}
        self._yes, self._no = confirm_words(locale)
        self._fillers = locale.fillers or FILLERS
        self._fillers_said = 0
        self._state = State.IDLE
        # The model's voice, queued as it arrives (rule 2).
        self._playback = LivePlayback(speaker)
        # How many of the assistant's own sentences are being played at once
        # - a question, a reminder, a failure. The microphone is deafened by
        # the outermost and listens again when that one is done.
        self._playing = 0
        # Whether the microphone is deaf for the model's voice (half duplex).
        self._answering = False
        # Set the moment a voice starts or listening is switched off while
        # one of the assistant's own sentences is being read, so that what
        # follows it is not said (`_withdrawn`).
        self._interrupted = asyncio.Event()
        # Set at every change of state, for whoever waits for one.
        self._changed = asyncio.Event()

        self._session: LiveSession | None = None
        self._session_open = False
        self._conversation: asyncio.Task[None] | None = None
        # Everything running beside the event loop: the conversation, the
        # settling of an answer, the filler. Waited for by `settled`.
        self._tasks: set[asyncio.Task[None]] = set()
        self._crash: BaseException | None = None
        self._crashed = asyncio.Event()
        # The provider's resumption handle and when the session that gave
        # it closed (D5): reused within `resume_minutes`.
        self._handle: str | None = None
        self._handle_at = 0.0
        self._failures = 0
        self._last_activity = 0.0
        self._user_speaking = False
        # When the doorman last heard the user stop, for `first_sound_ms`;
        # kept apart from the turn, which may not exist yet - the user is
        # usually done talking before the session has opened.
        self._quiet_at: float | None = None
        self._turn = _Turn()

    @property
    def state(self) -> State:
        return self._state

    @property
    def session_open(self) -> bool:
        """Whether a session is open right now - what the tray shows."""
        return self._session_open

    def fixed_sentences(self) -> list[str]:
        """What the program always says in the same words (D32): the
        fillers, the hint after a question and the one after an answer with
        neither word in it, the three failures, the limits, the greeting.
        The voice keeps these, so that they cost no request and need no
        network - the failures are said exactly when there is none."""
        return [*self._fillers, *self._said.values()]

    async def begin(self) -> None:
        """Opens the microphone, before anything is said."""
        self._capture.on_speech = self._speech
        self._capture.on_mode = self._mode_changed
        self._capture.on_wake = self._woken

        # Where the machine stands before the microphone opens, so that the
        # mode the capture reports as it starts finds it there (off stays off).
        self._state = State.IDLE if self._capture.listening else State.OFF
        self._capture.start()
        # Said out loud to whoever is watching, rather than merely being true:
        # the status line went on showing the last thing it was told until
        # something happened, and a program that looks like it never finished
        # starting is one nobody speaks to. Said only once the microphone is
        # open (2026-09-23): a start that failed at the device left "ready" on
        # the screen above the sentence saying why. Behind the wake word
        # (D21) the first thing said is asleep - the door is not watched until
        # the phrase is heard.
        self._enter(State.SLEEPING if self._capture.asleep else self._state)

    async def run(self) -> None:
        """Opens the door and keeps it, until something stops the program.

        Everything happens beside this: the doorman opens a session, the
        session's events drive the machine, a reminder is said when nothing
        is under way. What this waits for is the one thing that ends a run
        from the inside - a bug in a task, which comes out here as the
        exception it is rather than dying quietly in the background.
        """
        await self.begin()
        announcing = self._spawn(self._announcing())
        # The sentences the program always says, read once in the
        # assistant's voice and kept (D32) - beside everything else, since
        # the first start after a new voice reads a dozen of them.
        preparing = self._spawn(self._tts.prepare(self.fixed_sentences()))
        try:
            await self._crashed.wait()
            if self._crash is not None:
                raise self._crash
        finally:
            announcing.cancel()
            preparing.cancel()
            self._hang_up()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await self.settled()
            self._capture.stop()

    async def settled(self) -> None:
        """Returns once nothing is under way beside the loop: no session, no
        answer still draining, no filler. A task that crashed meanwhile
        comes out of here as its exception."""
        while self._tasks:
            await asyncio.wait(set(self._tasks))
        if self._crash is not None:
            raise self._crash

    # ----------------------------------------------------------------------
    # The door: what the capture reports
    # ----------------------------------------------------------------------

    def _speech(self, speaking: bool) -> None:
        """The doorman heard a voice begin, or stop (plan.md 4.2)."""
        if self._state in (State.OFF, State.SLEEPING):
            # Asleep, the capture hands its blocks to the wake word and to
            # nothing else, so this is not called (D21). Guarded all the
            # same: the one thing a voice must never do while the assistant
            # is asleep is open a session that costs money.
            return
        self._user_speaking = speaking
        self._active()
        if not speaking:
            self._quiet_at = _now()
            if self._state is State.USER_SPEAKING:
                self._enter(State.IDLE)
            return

        self._turn.open = True
        if self._state is State.IDLE:
            self._enter(State.USER_SPEAKING)
        if self._playing:
            # Talking over a question being read, a reminder, a failure: the
            # sentence is cut off and what depended on it is off. A voice
            # inside the window itself never arrives here - the capture
            # keeps it as the answer. The model's own voice is not stopped
            # from here: the server hears the user too, and says so.
            self._interrupted.set()
            self._speaker.stop()
        if self._conversation is None and self._capture.listening:
            # Speech while the session is closed: the pre-roll is queued,
            # and the session is opened for it (D5).
            self._conversation = self._spawn(self._converse())

    def _mode_changed(self, listening: bool) -> None:
        """The toggle went one way or the other. Acted on first, shown second:
        the screen must not say "off" over an assistant still talking."""
        if listening:
            if self._state is State.OFF:
                self._enter(State.IDLE)
        else:
            self._switched_off()
        if self._on_mode is not None:
            self._on_mode(listening)

    def _switched_off(self) -> None:
        """Listening went off, and with it whatever was under way: the
        model's voice, a question, a reminder, the session (rule 4).
        Nothing is said about it - the user asked for silence."""
        if self._state is State.OFF:
            return
        self._interrupted.set()
        self._user_speaking = False
        self._playback.stop()
        if self._playing:
            self._speaker.stop()
        self._hang_up()
        self._enter(State.OFF)

    def _hang_up(self) -> None:
        """Closes the session, if one is open or opening. Its task finishes
        on its own, and closes the stream the doorman opened with it."""
        if self._conversation is not None:
            self._conversation.cancel()

    def _woken(self) -> None:
        """The wake word was heard (D21): awake, greeted, and listening for
        real - the session opens now, not at the first sentence, so that
        "hey Friday, saat kaç" is not answered a socket late. A detection
        that arrives while awake changes nothing."""
        if self._state is not State.SLEEPING:
            return
        self._active()
        self._enter(State.IDLE)
        self._spawn(self._greet_and_listen())

    async def _greet_and_listen(self) -> None:
        if self._greeting == "sentence":
            await self._speak(self._said["wake_greeting"])
            self._rest()
        elif self._greeting == "chime":
            await self._sound(chime(), CHIME_RATE)
        if self._conversation is None and self._capture.listening and not self._capture.asleep:
            self._conversation = self._spawn(self._converse())

    def _doze(self) -> None:
        """Back to sleep, when there is a wake word to sleep behind (D21):
        the idle close of D5 is where the day's conversations end."""
        if not self._capture.wake_word or self._state is State.OFF:
            return
        self._capture.sleep()
        self._enter(State.SLEEPING)

    # ----------------------------------------------------------------------
    # One session, from the onset that opened it to its close
    # ----------------------------------------------------------------------

    async def _converse(self) -> None:
        """A session and, when it drops, the one that continues it."""
        self._interrupted.clear()
        try:
            if self._tracker is not None and self._tracker.stopped():
                # `hard_stop` and a limit passed (rule 6): no session, and
                # the user hears why instead of an answer.
                self._capture.close()
                await self._speak(self._said["spend_stopped"])
                self._finish(Turn(failure="spend_stopped"))
                self._rest()
                return
            warning = None if self._tracker is None else self._tracker.warning()
            if warning is not None:
                # The one sentence the user has to act on, before the
                # session that costs more.
                await self._speak(self._said[warning])
                self._rest()

            resume = False
            while await self._attend_one(resume=resume):
                resume = True
        finally:
            self._conversation = None

    async def _attend_one(self, *, resume: bool) -> bool:
        """Opens one session and attends to it until it ends. `True` when
        the next one should follow at once, with the handle."""
        config = self._session_config()
        if resume or self._handle_fresh():
            config = replace(config, resume_handle=self._handle)
        opened = False
        try:
            async with self._provider.connect(config) as session:
                opened = True
                return await self._attend(session)
        except AuthenticationError as refusal:
            logger.warning("session refused: {words}", words=refusal)
            failure = "key_invalid"
        except ProviderError as refusal:
            logger.warning("session {kind}: {words}", kind=refusal.kind, words=refusal)
            if not opened and config.resume_handle is not None and refusal.kind == "refused":
                # The handle, not the network: the provider would not
                # continue the conversation it handed out (Gemini keeps one
                # for a few minutes, not the two hours it documents -
                # measured 2026-09-21: taken at 4 minutes, refused at 5 with
                # close code 1011). Offered again it is refused again, so it
                # is forgotten here and the session opened afresh, with
                # nothing said: the context is lost, the turn is not.
                logger.info("the resumption handle was refused; opening without it")
                self._handle = None
                return await self._attend_one(resume=False)
            failure = _failure_of(refusal)
        except OSError as failure_below:
            # A socket refused below the adapter's transport never reaches
            # it to be translated.
            logger.warning("session unreachable: {words}", words=failure_below)
            failure = "unreachable"
        self._failures += 1
        await self._failed(failure)
        return False

    async def _failed(self, failure: str) -> None:
        """A session that could not be opened, or died: said out loud (rule
        7), written down, and after `MAX_OPEN_FAILURES` in a row the switch
        goes off rather than the sentence being said at every onset."""
        self._capture.close()
        logger.warning("session failed: {kind}", kind=failure)
        await self._speak(self._said[failure])
        self._finish(Turn(failure=failure))
        if self._failures >= MAX_OPEN_FAILURES and self._capture.listening:
            logger.warning("{count} sessions failed in a row; switching off", count=self._failures)
            self._failures = 0
            self._capture.toggle()
        self._rest()

    async def _attend(self, session: LiveSession) -> bool:
        """Everything the session sends, answered, until it closes.

        The audio goes in beside this, from the stream the doorman opened;
        the idle clock ticks beside it too. `True` to reopen at once: the
        server is about to hang up, or already did with an error.
        """
        self._session = session
        self._new_turn(session)
        self._tell_session(True)
        if self._state is State.RECONNECTING:
            self._rest()
        pump = asyncio.create_task(self._pump(session))
        watch = asyncio.create_task(self._idle_watch())
        ending: str | None = None
        try:
            async for event in session.events():
                ending = self._on_event(event, session)
                if ending is not None:
                    break
        finally:
            for task in (pump, watch):
                task.cancel()
            await asyncio.gather(pump, watch, return_exceptions=True)
            self._session = None
            # A tool still running has no session to answer.
            await self._runner.close()
            if ending in (None, "closed"):
                # Hung up, by us or politely by the server: what is queued
                # of the answer is still heard to its end.
                self._playback.flush()
            else:
                self._playback.stop()
            self._handle_at = _now()
            self._close_turn(session)
            self._tell_session(False)
            if ending == "reopen":
                # The stream the doorman opened stays: what the user says
                # while the socket is down is sent to the next session.
                self._enter(State.RECONNECTING)
            else:
                self._capture.close()
            self._spawn(self._quieten())

        if ending == "reopen":
            return self._capture.listening
        if ending is not None and ending != "closed":
            await self._failed(ending)
        return False

    def _on_event(self, event: LiveEvent, session: LiveSession) -> str | None:
        """One event, answered. Returns why the session is over, when it is:
        `"closed"`, `"reopen"`, or the key of the failure to say."""
        self._active()
        turn = self._turn
        match event:
            case AudioChunk(pcm16=pcm16, sample_rate=rate):
                turn.open = True
                turn.sounded = True
                _drop(turn.filler)
                if turn.first_sound_ms is None and self._quiet_at is not None:
                    turn.first_sound_ms = (_now() - self._quiet_at) * 1000
                self._start_answer()
                self._playback.push(pcm16, sample_rate=rate)
            case Interrupted():
                # The server heard the user over the model: what is queued is
                # dropped and what the sound card holds is aborted (rule 2).
                # Nothing said from here belongs to the answer, but the turn
                # stays open until the server closes it: measured in the
                # spikes of 2026-09-18 (`docs/spikes/A-default.jsonl`) and
                # again on 2026-09-21, `interrupted` is followed by the
                # cut answer's `usage_metadata` and then by `turn_complete`.
                # Ending the turn here booked those tokens to the next turn,
                # which had no words in it and went to the log as a turn of
                # its own.
                self._playback.stop()
                turn.cut = True
                # A tool call in the server turn just cut no longer owes us
                # a `TurnComplete` of its own: the next one ends the turn.
                turn.called = False
                # The machine rests now rather than when the turn is closed:
                # the sound is already gone, and a server that went quiet
                # after `interrupted` must not leave the microphone deaf and
                # the idle close - which only fires in `IDLE` - disarmed.
                self._spawn(self._quieten())
            case ToolCallEvent(call=call):
                turn.open = True
                turn.called = True
                turn.sounded = False
                self._runner.run(call, session, confirm=self.confirm)
                if turn.filler is None:
                    # The clock of the filler starts here (D19): the model is
                    # silent for as long as the tool takes.
                    turn.filler = self._spawn(self._filler_after(turn))
            case ToolCallCancelled(ids=ids):
                self._runner.cancel(ids)
            case InputText(text=text):
                turn.open = True
                turn.heard.append(text)
            case OutputText(text=text):
                if not turn.cut:
                    turn.said.append(text)
            case TurnComplete():
                self._playback.flush()
                if turn.called:
                    # Gemini ends its turn at the tool call, before the
                    # result is even sent; the answer comes as a new turn
                    # (ADR-001 section 8). The user's turn goes on.
                    turn.called = False
                else:
                    self._end_turn(session)
            case UsageReport(usage=usage):
                turn.usage = turn.usage + usage
            case Resumable(handle=handle):
                self._handle = handle
            case GoingAway(time_left=time_left):
                logger.info("the server is about to hang up ({left}); reopening", left=time_left)
                return "reopen"
            case Closed(error=error):
                if error is None:
                    return "closed"
                logger.warning("session dropped ({kind}): {words}", kind=error.kind, words=error)
                self._failures += 1
                if self._failures < MAX_OPEN_FAILURES:
                    return "reopen"
                return _failure_of(error)
        return None

    async def _pump(self, session: LiveSession) -> None:
        """The stream the doorman opened, into the session, block by block
        (plan.md 4.1). Ends with the stream; a socket that died under it is
        left to `Closed` to report."""
        try:
            async for chunk in self._capture.chunks():
                await session.send_audio(chunk)
        except ProviderError as failure:
            logger.warning("sending audio failed: {kind}", kind=failure.kind)

    async def _idle_watch(self) -> None:
        """Closes the session after `idle_close_seconds` of nothing said and
        nothing played (D5): a live session bills by the minute, silent or
        not."""
        while True:
            remaining = self._last_activity + self._idle_close - _now()
            if remaining > 0:
                await asyncio.sleep(remaining)
                continue
            if self._quiet():
                logger.info(
                    "session closed after {seconds:g} s of silence", seconds=self._idle_close
                )
                self._hang_up()
                self._doze()
                return
            self._active()

    def _quiet(self) -> bool:
        return (
            self._state is State.IDLE
            and not self._user_speaking
            and self._runner.pending == 0
            and not self._playback.playing
        )

    # ----------------------------------------------------------------------
    # The turn
    # ----------------------------------------------------------------------

    def _new_turn(self, session: LiveSession) -> None:
        """Minted here, where the turn becomes something that may act: every
        tool call it makes is written down under this name (section 3.9),
        and the guard counts from zero with it (D9)."""
        turn_id = uuid.uuid4().hex
        self._runner.new_turn(turn_id)
        self._turn = _Turn(
            turn_id=turn_id,
            in_at=session.audio_in_ms,
            out_at=session.audio_out_ms,
            open=self._user_speaking,
        )

    def _end_turn(self, session: LiveSession) -> None:
        """The model is done with the user's turn (rule 6): what it came to
        is written down once the last of its voice has been heard, and the
        next turn starts now - what the server sends from here on is the
        next utterance's."""
        finished = self._turn
        _drop(finished.filler)
        finished.audio_in_ms = session.audio_in_ms - finished.in_at
        finished.audio_out_ms = session.audio_out_ms - finished.out_at
        finished.tool_calls = self._runner.ran
        finished.searched = self._take_searched()
        # A turn came through: whatever failed before, the network is fine.
        self._failures = 0
        self._quiet_at = None
        self._new_turn(session)
        self._spawn(self._settle(finished))

    def _take_searched(self) -> tuple[str, ...]:
        """What was looked up since the last turn ended (D29)."""
        return self._searched.take() if self._searched is not None else ()

    async def _quieten(self) -> None:
        """Once the last of the model's voice has been heard: the microphone
        listens again and the machine rests."""
        await self._playback.drained()
        self._answered()

    async def _settle(self, finished: _Turn) -> None:
        """Waits for the answer to be heard to its end, then writes the turn
        down and rests. Beside the event loop, so that an interruption
        arriving meanwhile is still seen - and still stops the sound."""
        await self._playback.drained()
        self._answered()
        if finished.counts:
            self._finish(self._record(finished))

    def _record(self, finished: _Turn) -> Turn:
        cost = (
            self._tracker.record(finished.turn_id, finished.usage)
            if self._tracker is not None
            else None
        )
        return Turn(
            heard="".join(finished.heard).strip(),
            said="".join(finished.said).strip(),
            usage=finished.usage,
            turn_id=finished.turn_id,
            cost_usd=cost,
            tool_calls=finished.tool_calls,
            first_sound_ms=finished.first_sound_ms,
            audio_in_ms=finished.audio_in_ms,
            audio_out_ms=finished.audio_out_ms,
            searched=finished.searched,
        )

    def _close_turn(self, session: LiveSession) -> None:
        """The session ended under the turn: what there is of it is written
        down as it stands, and the rest is forgotten."""
        turn = self._turn
        _drop(turn.filler)
        turn.tool_calls = self._runner.ran
        turn.searched = self._take_searched()
        turn.audio_in_ms = session.audio_in_ms - turn.in_at
        turn.audio_out_ms = session.audio_out_ms - turn.out_at
        if turn.counts:
            self._finish(self._record(turn))
        self._turn = _Turn(open=self._user_speaking)

    def _finish(self, turn: Turn) -> None:
        if self._on_turn is not None:
            self._on_turn(turn)

    # ----------------------------------------------------------------------
    # The filler, and the question a tool asks
    # ----------------------------------------------------------------------

    async def _filler_after(self, turn: _Turn) -> None:
        """Says the filler once `filler_delay` has passed with no sound from
        the model since the tool was called - through the same queue as the
        model's voice, so that the answer follows it rather than talking
        over it. Never while a question is being asked: the user is
        answering it."""
        await asyncio.sleep(self._filler_delay)
        if (
            turn is not self._turn
            or turn.sounded
            or self._session is None
            or self._state in (State.CONFIRMING, State.ANNOUNCING, State.OFF)
        ):
            return
        filler = self._filler()
        self._start_answer()
        try:
            async with contextlib.aclosing(self._tts.stream([filler])) as buffers:
                async for buffer in buffers:
                    self._playback.push(buffer, sample_rate=self._tts.sample_rate)
        finally:
            # Also when the model's voice cut it short: what was pushed of
            # it is an answer of its own, and the model's follows.
            self._playback.flush()

    def _filler(self) -> str:
        """The next of the pack's fillers, in turn."""
        chosen = self._fillers[self._fillers_said % len(self._fillers)]
        self._fillers_said += 1
        return chosen

    async def confirm(self, question: str) -> bool | None:
        """Asks `question` out loud and listens for a yes (section 3.1 rule 2).

        The gate's `Confirm`, run inside the tool's own round. `question`
        already holds the real argument values; what is added is how to
        answer, since the user cannot know that only two words are being
        listened for. The model's voice is heard to its end first - it may
        have said "let me check" before it asked - and the session's input
        is paused for the whole exchange (D3), so that the yes never reaches
        the model: only the tool's result does. Nothing runs without a clear
        yes. `False` is a no the user gave: a no word, a no beside a yes, a
        voice or the switch while the question is still being read. `None`
        is an answer nobody heard - silence, a noise, two answers with
        neither word in them - which the gate tells the model apart from a
        refusal (2026-09-26). Stopped from outside - the model withdrew the
        call, or the switch went off - it lets the microphone go on the way
        out.
        """
        await self._playback.drained()
        self._answered()
        if self._state is State.OFF:
            return False
        self._interrupted.clear()

        self._enter(State.CONFIRMING)
        self._capture.pause()
        try:
            # Two utterances: the question live, with the real values in it,
            # and the hint kept on disk (D32).
            answer = await self._ask(question, self._said["confirm_hint"])
            if answer == "neither":
                answer = await self._ask(self._said["confirm_again"])
        finally:
            self._capture.resume()
            self._active()
            if self._state is State.CONFIRMING:
                self._rest()
        if answer == "yes":
            return True
        if answer in ("unheard", "neither"):
            return None
        return False

    async def _ask(self, *prompt: str) -> Answer:
        """Reads `prompt`, opens the window, and reads the answer.

        `neither`: something was said with neither word in it, or the
        recogniser could not read it - worth one more try. `unheard`:
        silence, or a noise the recogniser calls nothing - not worth one,
        nobody was there to hear it. `stopped`: the user spoke over the
        question, switched the assistant off, or the call was withdrawn.
        Every answer the window gets is one line in the log: three declines
        on 2026-09-25 left no trace of what had been heard.
        """
        await self._play(prompt)
        if self._withdrawn():
            return "stopped"

        # The window hears: `_play` has already unmuted the microphone on its
        # way out, and the echo tail that left behind counts down before the
        # window listens, so the room repeating the question is not a yes
        # (`audio/capture.py`). The old product opened and closed the
        # microphone around this line as well; on this path it was always
        # open already, and `test_app.py` pins that the window is never deaf.
        pcm = await self._capture.listen_for(CONFIRM_WINDOW_SECONDS)
        if self._withdrawn():
            return "stopped"
        if pcm is None:
            logger.info("confirm answer: nothing within {:.0f} s", CONFIRM_WINDOW_SECONDS)
            return "unheard"

        heard = await self._heard(pcm)
        if not heard.text:
            empty: Answer = "neither" if heard.missed else "unheard"
            logger.info("confirm answer: no words -> {answer}", answer=empty)
            return empty
        verdict = read_answer(heard.text, yes=self._yes, no=self._no)
        answer: Answer = "yes" if verdict is True else "no" if verdict is False else "neither"
        logger.info("confirm answer: {text!r} -> {answer}", text=heard.text, answer=answer)
        return answer

    async def _heard(self, pcm: Audio) -> Heard:
        """What the user answered, and what to make of it when they said nothing."""
        if len(pcm) < MIN_UTTERANCE_SECONDS * SAMPLE_RATE:
            # A noise rather than a word. Sending it to Google costs a
            # second and a half, for nothing - and there is nothing here
            # to have misheard.
            return Heard()

        return hear(await self._stt.transcribe(pcm, hint=self._locale.stt_language))

    # ----------------------------------------------------------------------
    # Reminders (rule 5)
    # ----------------------------------------------------------------------

    async def _announcing(self) -> None:
        """Reads the queue, and says what it finds when nothing is under way."""
        if self._announcements is None:
            await asyncio.Event().wait()
            return
        while True:
            announcement = await self._announcements.get()
            while not self._free():
                self._changed.clear()
                await self._changed.wait()
            await self._announce(announcement)

    def _free(self) -> bool:
        """Between turns: no voice, no answer owed or being given, no tool,
        no sound. Off counts - the assistant was asked not to listen, not
        to forget the dentist. Asleep counts too: the wake word does not
        gate the dentist."""
        return (
            self._state in (State.IDLE, State.OFF, State.SLEEPING)
            and not self._turn.open
            and self._runner.pending == 0
            and not self._playback.playing
        )

    async def _announce(self, announcement: Announcement) -> None:
        """Says one announcement in the assistant's voice, with the
        session's input paused, and tells the session what was said (D4)."""
        self._interrupted.clear()
        self._enter(State.ANNOUNCING)
        logger.info("announcing reminder {id}", id=announcement.reminder_id)
        self._capture.pause()
        try:
            await self._play([announcement.text])
        finally:
            self._capture.resume()
        session = self._session
        if session is not None and not self._withdrawn():
            # One line, without asking for an answer: the model learns what
            # the assistant already said aloud, and says nothing about it.
            with contextlib.suppress(ProviderError):
                await session.send_text(
                    f"{ANNOUNCED_PREFIX}{announcement.text}", role="user", turn_complete=False
                )
        self._active()
        if self._state is State.ANNOUNCING:
            # Back to where it was: off, when the switch is off - the
            # reminder was said all the same.
            if self._capture.listening:
                self._rest()
            else:
                self._enter(State.OFF)

    # ----------------------------------------------------------------------
    # Saying things in the assistant's voice (D32)
    # ----------------------------------------------------------------------

    async def _speak(self, said: str) -> None:
        """Says one sentence of the assistant's own: a failure, a limit."""
        if not said or self._state is State.OFF:
            return
        self._interrupted.clear()
        self._enter(State.SPEAKING)
        await self._play([said])

    async def _play(self, texts: Sequence[str]) -> None:
        """Says `texts` through the sound card, with the microphone deaf meanwhile.

        Deaf for exactly as long as there is something for it to mishear -
        on a microphone that needs it - and in a `finally`, because a
        sentence that failed halfway through must not leave the assistant
        unable to hear at all. The voice's stream is closed on the way out,
        so that a sentence cut short does not leave its reader open.
        """
        if self._playing == 0:
            self._capture.mute()
        self._playing += 1
        try:
            async with contextlib.aclosing(self._tts.stream(texts)) as buffers:
                await self._speaker.play(buffers, sample_rate=self._tts.sample_rate)
        except PlaybackError as failure:
            # Only the sound of it was lost. A headset switched off between
            # two questions is not a bug, so it is a line in the log rather
            # than the program.
            logger.warning("playback failed: {problem}", problem=failure)
        finally:
            self._playing -= 1
            if self._playing == 0:
                self._capture.unmute()

    async def _sound(self, pcm16: bytes, sample_rate: int) -> None:
        """A sound of the assistant's own - the chime - through the same
        bracket as a sentence: deaf for it on a microphone that needs it."""
        if self._playing == 0:
            self._capture.mute()
        self._playing += 1
        try:
            await self._speaker.play(_one_buffer(pcm16), sample_rate=sample_rate)
        except PlaybackError as failure:
            logger.warning("playback failed: {problem}", problem=failure)
        finally:
            self._playing -= 1
            if self._playing == 0:
                self._capture.unmute()

    def _start_answer(self) -> None:
        """The model's voice - or the filler in its place - is about to play:
        deaf for it on a microphone that needs it (D18), and `SPEAKING`."""
        if not self._answering:
            self._capture.mute()
            self._answering = True
        if self._state is not State.SPEAKING:
            self._enter(State.SPEAKING)

    def _answered(self) -> None:
        """Nothing of the model's is playing any more: the microphone
        listens again, and the machine rests."""
        if self._playback.playing:
            return
        if self._answering:
            self._capture.unmute()
            self._answering = False
        if self._state is State.SPEAKING:
            self._rest()
        self._changed.set()

    # ----------------------------------------------------------------------
    # Where it is
    # ----------------------------------------------------------------------

    def _enter(self, state: State) -> None:
        self._state = state
        if self._on_state is not None:
            self._on_state(state)
        self._changed.set()

    def _rest(self) -> None:
        """Back to the door: `IDLE`, or `USER_SPEAKING` when the doorman is
        still hearing a voice; `SLEEPING` while the capture is asleep (D21).
        Off stays off."""
        if self._state is State.OFF:
            return
        if self._capture.asleep:
            resting = State.SLEEPING
        else:
            resting = State.USER_SPEAKING if self._user_speaking else State.IDLE
        if self._state is not resting:
            self._enter(resting)

    def _withdrawn(self) -> bool:
        """Whether the user has moved on in the meantime: started saying
        something else over the assistant's sentence, or switched it off."""
        return self._interrupted.is_set() or self._state is State.OFF

    def _tell_session(self, opened: bool) -> None:
        self._session_open = opened
        if self._on_session is not None:
            self._on_session(opened)

    def _active(self) -> None:
        self._last_activity = _now()

    def _handle_fresh(self) -> bool:
        return self._handle is not None and _now() - self._handle_at <= self._resume_seconds

    def _spawn(self, work: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._done)
        return task

    def _done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None and self._crash is None:
            # A bug, not a failure the rules name: kept, and raised out of
            # `run` - never swallowed into a sentence.
            self._crash = error
            self._crashed.set()


def _drop(task: asyncio.Task[None] | None) -> None:
    """Cancels `task`, when there is one still to cancel."""
    if task is not None and not task.done():
        task.cancel()


def _failure_of(error: ProviderError) -> str:
    """Which of the three sentences a provider's refusal is (rule 7)."""
    if isinstance(error, AuthenticationError):
        return "key_invalid"
    if error.kind == "timeout":
        return "took_too_long"
    return "unreachable"


def _now() -> float:
    return asyncio.get_running_loop().time()


def hear(transcript: Transcript) -> Heard:
    """What to make of a transcript: nothing, unreadable speech, or words.

    The decision rests on whether there was speech, never on how sure the
    decoder was of its words. Measured on the target machine with the local
    engine (2026-09-05, before D36): a correct single "Merhaba." scores 0.48
    confidence and silence scores up to
    0.54, so no confidence floor can tell them apart - but the engine's own
    no-speech estimate can (0.06 against 0.86). Confidence is still carried,
    for the log and the screen, because a run of low numbers is how a bad
    microphone is diagnosed afterwards.

    An engine with no opinion is believed about its words, and its silence is
    read as speech it could not make out: it cannot tell the two apart, and
    neither can this. Treating "no opinion" as "nothing was said" would make
    the assistant deaf: Google's recogniser has no opinion about its words.
    """
    no_speech = transcript.no_speech_probability
    if no_speech is not None and no_speech >= NO_SPEECH_CEILING:
        # The engine says nothing was said. Words it produced anyway are what
        # a recogniser makes of silence, and they are not answered.
        return Heard()

    text = transcript.text.strip()
    if text:
        return Heard(text=text, confidence=transcript.confidence)

    # There was speech, or an engine with no opinion, and no words came of it.
    return Heard(missed=True, confidence=transcript.confidence)


# A word, for the purpose of hearing "yes" in an answer: letters and digits in
# any script. Punctuation and the spaces between are where words end.
_WORD = re.compile(r"\w+")


def confirm_words(locale: Locale) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The yes and the no words of `locale`, or the English ones written here.

    Resolved in one place: the pack writes them, the window that judges the
    answer (`read_answer`) reads them. Until D36 the local recogniser was
    told them as well, and that is what turned a clear "Evet." into
    "iptal." (measured 2026-09-26); Google is told nothing but the language.
    """
    return (tuple(locale.yes_words) or YES_WORDS, tuple(locale.no_words) or NO_WORDS)


def read_answer(text: str, *, yes: Iterable[str], no: Iterable[str]) -> bool | None:
    """Whether `text` says yes, says no, or says neither.

    Whole words, folded the way search is (`store/normalize.py`): "Evet." and
    "EVET" are the same word, and "evetlemedim" is not it. A `no` word
    anywhere wins over a `yes` word - "yes, but no" is a no - because the
    window only ever guards something that should not happen by mistake.
    `None` means neither was heard, and is the caller's cue to ask once more.
    """
    said = _spaced(text)
    if any(phrase in said for phrase in _phrases(no)):
        return False
    if any(phrase in said for phrase in _phrases(yes)):
        return True
    return None


def _phrases(words: Iterable[str]) -> list[str]:
    """Each entry as it would appear inside `_spaced` text; blanks dropped."""
    spaced = (_spaced(word) for word in words)
    return [phrase for phrase in spaced if phrase.strip()]


def _spaced(text: str) -> str:
    """The words of `text`, folded, one space between and one either side -
    so that a phrase of one or more words can be found only at word edges."""
    return f" {' '.join(_WORD.findall(normalize_search(text)))} "


async def _one_buffer(pcm16: bytes) -> AsyncIterator[bytes]:
    """A sound of the assistant's own - the chime - as a stream of one."""
    yield pcm16
