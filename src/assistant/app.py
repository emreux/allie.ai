"""The state machine of the live product (plan.md sections 4.2-4.4).

`IDLE` at the door, `USER_SPEAKING` while the doorman hears a voice,
`SPEAKING` while the model's voice plays, and back; `CONFIRMING` is the
window of the permission gate, opened from inside a tool round; `ANNOUNCING`
the announce queue being read between turns; `RECONNECTING` a session being
reopened with the handle it left behind; `OFF` the switch. The session itself
is opened by speech and closed by silence (D5), and everything in between is
an event the session sends and this file answers.

Everything is injected - the microphone, the provider, the runner, the two
local voices, the sound card - so a whole conversation can be driven in a
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
the local voice, tells the user how to answer, and opens the local
recogniser's microphone for six seconds. A "no" anywhere in the answer wins
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
sound - and said in the local voice with the session's input paused; then
one line goes to the session so the model knows what was said. Listening
switched off does not silence it: the assistant was asked not to listen,
not to forget the dentist. A voice that starts cuts it off like an answer.

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
decoder was of its words.** Whisper answers a recording of silence with
confident looking words, and answers a correct single word with unsure ones:
measured on the owner's machine (2026-09-05), a subtitle credit over silence
scored 0.54 and "Merhaba." alone 0.48, so no confidence floor can separate
them. The engine's own estimate of whether anything was said does separate
them (0.86 against 0.06), and `hear` below rests on that and on nothing
else. In the live product this judgement serves one place: the yes-and-no
window, which is the one thing the local recogniser still hears.

**No user-facing sentence is written here** (rule 8). The pack answers first
and the English constants below are the end of the chain, exactly as in the
wizard (section 3.12).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol

from loguru import logger

from assistant.agent.core import ToolRunner
from assistant.agent.prompts import ANNOUNCED_PREFIX
from assistant.announce.queue import Announcement, AnnounceQueue
from assistant.audio.player import LivePlayback, PlaybackError, Speaker
from assistant.config import LiveSettings
from assistant.live.base import (
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
from assistant.locales import Locale
from assistant.store.normalize import normalize_search
from assistant.stt.base import NO_SPEECH_CEILING, SAMPLE_RATE, Audio, STTProvider, Transcript
from assistant.tts.base import TTSProvider, choose_voice
from assistant.usage.tracker import UsageTracker

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
    "Capture",
    "Heard",
    "LiveAssistant",
    "NoVoiceError",
    "State",
    "Turn",
    "choose_voice",
    "hear",
    "read_answer",
]


class NoVoiceError(RuntimeError):
    """No speech voice is installed, for the locale or otherwise.

    The user can install one; the program cannot. Named so that `live-assistant
    run` can say so in a sentence instead of a traceback.
    """


class State(StrEnum):
    """Where the assistant is. A string so the status line and the logs can
    print it without a table of names.

    The set of plan.md section 4.2. `USER_SPEAKING` is the local detector's
    opinion and nothing else - the server decides where a turn ends - and is
    there for the screen. `OFF` is a state rather than a mode, because a
    session is something the switch closes.
    """

    OFF = "off"
    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    SPEAKING = "speaking"
    CONFIRMING = "confirming"
    ANNOUNCING = "announcing"
    RECONNECTING = "reconnecting"


# Section 3.1 rule 6. How long the microphone stays open for a yes or a no
# after the question has been read; what comes after it is a no.
CONFIRM_WINDOW_SECONDS = 6.0

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

# The last link of the chain of section 3.12: what is said when no locale pack
# offers a translation. Keys are unique across the whole project - the pack has
# one table of sentences, and `test_locales.py` checks that no two modules
# claim the same key.
TEXT: dict[str, str] = {
    # The three failures said out loud when there is no session to say them
    # (plan.md 4.4 rule 7), by the local voice.
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
    nobody was heard to stop. `missed`, `confidence` and `intent` are the
    old pipeline's and stay for the log that reads them; a live turn never
    sets them.
    """

    heard: str = ""
    said: str = ""
    usage: Usage = field(default_factory=Usage)
    missed: bool = False
    confidence: float | None = None
    failure: str | None = None
    turn_id: str = ""
    cost_usd: float | None = None
    tool_calls: int = 0
    intent: str | None = None
    first_sound_ms: float | None = None
    audio_in_ms: int = 0
    audio_out_ms: int = 0


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

    @property
    def listening(self) -> bool: ...

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
        on_state: Callable[[State], None] | None = None,
        on_turn: Callable[[Turn], None] | None = None,
        on_mode: Callable[[bool], None] | None = None,
        on_session: Callable[[bool], None] | None = None,
    ) -> None:
        self._capture = capture
        self._provider = provider
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
        self._yes = locale.yes_words or YES_WORDS
        self._no = locale.no_words or NO_WORDS
        self._fillers = locale.fillers or FILLERS
        self._fillers_said = 0
        self._state = State.IDLE
        self._voice = ""
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

    async def begin(self) -> None:
        """Picks the voice and opens the microphone, before anything is said.

        The voice is settled here rather than at the first question: a
        machine with no voice installed can never ask one, and that is worth
        finding out at startup instead of at two in the morning.
        """
        self._voice = await self._pick_voice()
        self._capture.on_speech = self._speech
        self._capture.on_mode = self._mode_changed

        # Said out loud to whoever is watching, rather than merely being true:
        # the status line went on showing the last thing it was told until
        # something happened, and a program that looks like it never finished
        # starting is one nobody speaks to.
        self._enter(State.IDLE if self._capture.listening else State.OFF)
        self._capture.start()

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
        try:
            await self._crashed.wait()
            if self._crash is not None:
                raise self._crash
        finally:
            announcing.cancel()
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
        if self._state is State.OFF:
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
        try:
            async with self._provider.connect(config) as session:
                return await self._attend(session)
        except AuthenticationError:
            failure = "key_invalid"
        except ProviderError as refusal:
            failure = _failure_of(refusal)
        except OSError:
            # A socket refused below the adapter's transport never reaches
            # it to be translated.
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
                # dropped, what the sound card holds is aborted, and the turn
                # is over as it stands (rule 2).
                self._playback.stop()
                self._end_turn(session)
                self._turn.cut = True
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
                logger.warning("session dropped: {kind}", kind=error.kind)
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
        # A turn came through: whatever failed before, the network is fine.
        self._failures = 0
        self._quiet_at = None
        self._new_turn(session)
        self._spawn(self._settle(finished))

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
        )

    def _close_turn(self, session: LiveSession) -> None:
        """The session ended under the turn: what there is of it is written
        down as it stands, and the rest is forgotten."""
        turn = self._turn
        _drop(turn.filler)
        turn.tool_calls = self._runner.ran
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
            async for buffer in self._tts.stream(_one(filler), voice=self._voice):
                self._playback.push(buffer, sample_rate=self._tts.sample_rate)
        except ProviderError as failure:
            logger.warning("the filler could not be said: {kind}", kind=failure.kind)
        finally:
            # Also when the model's voice cut it short: what was pushed of
            # it is an answer of its own, and the model's follows.
            self._playback.flush()

    def _filler(self) -> str:
        """The next of the pack's fillers, in turn."""
        chosen = self._fillers[self._fillers_said % len(self._fillers)]
        self._fillers_said += 1
        return chosen

    async def confirm(self, question: str) -> bool:
        """Asks `question` out loud and listens for a yes (section 3.1 rule 2).

        The gate's `Confirm`, run inside the tool's own round. `question`
        already holds the real argument values; what is added is how to
        answer, since the user cannot know that only two words are being
        listened for. The model's voice is heard to its end first - it may
        have said "let me check" before it asked - and the session's input
        is paused for the whole exchange (D3), so that the yes never reaches
        the model: only the tool's result does. Everything that is not a
        clear yes is a no: silence, a no beside a yes, a voice or the switch
        while the question is still being read, and two answers with neither
        word in them. Stopped from outside - the model withdrew the call, or
        the switch went off - it lets the microphone go on the way out.
        """
        await self._playback.drained()
        self._answered()
        if self._state is State.OFF:
            return False
        self._interrupted.clear()

        self._enter(State.CONFIRMING)
        self._capture.pause()
        try:
            answer = await self._ask(f"{question} {self._said['confirm_hint']}")
            if answer is None:
                answer = await self._ask(self._said["confirm_again"])
        finally:
            self._capture.resume()
            self._active()
            if self._state is State.CONFIRMING:
                self._rest()
        return answer is True

    async def _ask(self, prompt: str) -> bool | None:
        """Reads `prompt`, opens the window, and reads the answer.

        `None` is "neither word was heard": something was said, or the
        recogniser could not read it, and it is worth one more try. `False`
        is every way of not saying yes that is not worth one: silence, a no,
        the user speaking over the question or switching the assistant off.
        """
        await self._play(_one(prompt))
        if self._withdrawn():
            return False

        # The window has to hear. A sentence under way keeps the microphone
        # deaf (`_play`); it is opened for the window and closed again after
        # it - and `unmute` is also what lets the room's echo of the
        # question pass before the window listens (`audio/capture.py`).
        if self._playing:
            self._capture.unmute()
        try:
            pcm = await self._capture.listen_for(CONFIRM_WINDOW_SECONDS)
        finally:
            if self._playing:
                self._capture.mute()
        if pcm is None or self._withdrawn():
            return False

        heard = await self._heard(pcm)
        if not heard.text:
            return None if heard.missed else False
        return read_answer(heard.text, yes=self._yes, no=self._no)

    async def _heard(self, pcm: Audio) -> Heard:
        """What the user answered, and what to make of it when they said nothing."""
        if len(pcm) < MIN_UTTERANCE_SECONDS * SAMPLE_RATE:
            # A noise rather than a word. Transcribing it costs seconds of
            # four cores, for nothing - and there is nothing here to have
            # misheard.
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
        to forget the dentist."""
        return (
            self._state in (State.IDLE, State.OFF)
            and not self._turn.open
            and self._runner.pending == 0
            and not self._playback.playing
        )

    async def _announce(self, announcement: Announcement) -> None:
        """Says one announcement in the local voice, with the session's
        input paused, and tells the session what was said (D4)."""
        self._interrupted.clear()
        self._enter(State.ANNOUNCING)
        logger.info("announcing reminder {id}", id=announcement.reminder_id)
        self._capture.pause()
        try:
            await self._play(_one(announcement.text))
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
    # Saying things in the local voice
    # ----------------------------------------------------------------------

    async def _speak(self, said: str) -> None:
        """Says one sentence of the assistant's own: a failure, a limit."""
        if not said or self._state is State.OFF:
            return
        self._interrupted.clear()
        self._enter(State.SPEAKING)
        await self._play(_one(said))

    async def _play(self, pieces: AsyncIterator[str]) -> None:
        """Says `pieces` through the sound card, with the microphone deaf meanwhile.

        Deaf for exactly as long as there is something for it to mishear -
        on a microphone that needs it - and in a `finally`, because a
        sentence that failed halfway through must not leave the assistant
        unable to hear at all.
        """
        if self._playing == 0:
            self._capture.mute()
        self._playing += 1
        try:
            buffers = self._tts.stream(pieces, voice=self._voice)
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
        still hearing a voice. Off stays off."""
        if self._state is State.OFF:
            return
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

    async def _pick_voice(self) -> str:
        preferred = self._locale.voice(self._tts.id)

        for language in (self._locale.code, None):
            # The locale's own language first. Failing that, anything installed:
            # the wrong accent is a poor answer, and no answer is worse.
            voice = choose_voice(await self._tts.list_voices(language), preferred)
            if voice:
                return voice

        raise NoVoiceError(f"no speech voice is installed, for {self._locale.code!r} or otherwise")


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
    decoder was of its words. Measured on the target machine (2026-09-05): a
    correct single "Merhaba." scores 0.48 confidence and silence scores up to
    0.54, so no confidence floor can tell them apart - but the engine's own
    no-speech estimate can (0.06 against 0.86). Confidence is still carried,
    for the log and the screen, because a run of low numbers is how a bad
    microphone is diagnosed afterwards.

    An engine with no opinion is believed about its words, and its silence is
    read as speech it could not make out: it cannot tell the two apart, and
    neither can this. Treating "no opinion" as "nothing was said" would make
    the assistant mute the day it moves to a cloud recogniser.
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


async def _one(said: str) -> AsyncIterator[str]:
    """A sentence of the assistant's own - a question, a reminder, a failure
    - as a stream of one."""
    yield said
