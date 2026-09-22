"""Being listened to: press a key once, and talk.

`HandsFree` keeps the microphone live and lets `audio/vad.py` decide where
each sentence begins and ends; one key, `Ctrl+Alt+H`, turns that listening
off and on again. Until 2026-09-11 there was a second way - holding
`Ctrl+Alt+Space` and letting go - and the first phases were built on it. The
owner removed it: one key with one meaning, and the assistant is either
listening or it is not (design.md section 12, decision 17).

Three threads meet in this file and only one of them owns the state.

* **PortAudio's callback thread** delivers the microphone blocks. The array it
  passes is its own scratch buffer and is overwritten on the next callback, so
  it is copied before it goes anywhere.
* **The keyboard listener's thread** reports the toggle. It flips a plain
  `threading.Event` so that the audio callback starts - or stops - showing
  blocks to the detector in the same instant, without waiting for the event
  loop to get around to it.
* **The event loop** owns the detector, the finished sentences and the window
  below. Both other threads reach it only through `call_soon_threadsafe`;
  touching an asyncio object from outside the loop is a race that shows up as
  a hang weeks later.

The microphone stays open for as long as the assistant runs - opening a
PortAudio stream takes long enough to swallow the first syllable. Open is not
the same as listened to: a block that arrives while listening is off is
dropped in the callback and never reaches memory. Fifty blocks a second cross
into the event loop while it is on, each costing a fraction of a millisecond,
and none at all while it is off.

The confirmation window of phase 2.3 is the second way to be listened to.
`listen_for` takes the next sentence - whether or not listening is on - for a
few seconds, and hands it back to whoever asked instead of queueing it as a
question. What is said inside it is an answer, so its beginning is not
announced: the state machine would otherwise take the user's "yes" for a new
question and withdraw the one it was asking.

`LiveCapture` is the same microphone, key and window for a live session
(plan.md section 4.1): the detector is a doorman rather than a scribe, and
the blocks themselves go to the server as a stream that starts at the door
with the pre-roll and ends when the state machine says so. Whether the
microphone stays live while the assistant speaks is decided by the way the
microphone was opened (D18), not by a setting alone.
"""

from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, Protocol, Self, runtime_checkable

import numpy as np
from loguru import logger

from assistant.audio.resample import Resampler
from assistant.audio.vad import PREROLL_SECONDS, Endpoint, Segmenter
from assistant.audio.wake import WakeDetector
from assistant.stt.base import SAMPLE_RATE, Audio, dbfs, to_pcm16

__all__ = [
    "CHUNK_FRAMES",
    "DEFAULT_TOGGLE_HOTKEY",
    "ECHO_TAIL_SECONDS",
    "FULL_DUPLEX_HOST_APIS",
    "QUIET_DBFS",
    "Duplex",
    "HandsFree",
    "Hotkey",
    "KeyCombination",
    "LiveCapture",
    "Microphone",
    "MicrophoneUnavailableError",
    "OnLevel",
    "SystemHotkey",
    "SystemMicrophone",
    "device_choice",
    "duplex_for",
]

# The one key, in `pynput`'s own spelling. Phase 4 moves it into
# `config.toml`; until then it is one constant rather than a string repeated
# in three places. A letter rather than space, so that a key held down by a
# game or a chat application cannot be the one that switches the microphone.
DEFAULT_TOGGLE_HOTKEY = "<ctrl>+<alt>+h"

# How long the microphone stays deaf after the assistant stops speaking. The
# sound card holds some of the answer and the room holds the rest of it; both
# arrive after the speaker has been told to stop, and the detector would hear
# the assistant finishing its own sentence and answer it. This is not echo
# cancellation - that is phase 5.3, and it is what barge-in needs.
ECHO_TAIL_SECONDS = 0.25

# 20 ms per block: below a syllable, and fifty callbacks a second is nothing.
# Measured at 100 ms blocks in phase 1, two seconds of speech lost its last
# block to the driver; the detector's own frame is 32 ms, so the block is
# kept small.
CHUNK_FRAMES = SAMPLE_RATE // 50

# Which ways to a microphone go through the Windows audio engine, where the
# device's own echo canceller runs. Measured 2026-09-18 (ADR-001 section 4):
# through any of these the array is clean in full duplex, and the model can
# be talked over; raw kernel streaming has no canceller, and the server hears
# its own voice, interrupts itself and answers itself. A host API not on the
# list is treated as raw: a loop that answers itself costs money every
# round, and no barge-in costs a key press.
FULL_DUPLEX_HOST_APIS = ("WASAPI", "MME", "DirectSound")

# Under this sent level the microphone is called quiet. ADR-001 section 5:
# the owner's array at -45 dBFS RMS was heard as Hindi, -35 is the target.
# No digital gain - measured, x6 made recognition worse - so the number is
# said (log, `doctor`, the status line) and the fix is Windows' own level and
# boost controls. **Measured through the Windows audio engine**, which applies
# the driver's own gain: on the raw kernel path the number means something
# else, and `LiveCapture.level_judged` is where that is decided.
QUIET_DBFS = -40.0

# Below this RMS a block is silence and does not count towards the level:
# the pauses between words would otherwise pull it down.
_SILENCE_RMS = 0.001

Duplex = Literal["full", "half"]

OnChunk = Callable[[Audio], None]
OnEvent = Callable[[], None]
OnLevel = Callable[[float], None]
OnMode = Callable[[bool], None]
OnSpeech = Callable[[bool], None]


@runtime_checkable
class Hotkey(Protocol):
    """A key combination watched everywhere, not just in our own window."""

    def watch(self, *, on_press: OnEvent, on_release: OnEvent) -> None: ...

    def stop(self) -> None: ...


@runtime_checkable
class Microphone(Protocol):
    """An input stream that hands over one block at a time."""

    # PortAudio's name for the way to the device ("Windows WASAPI", "MME",
    # "Windows WDM-KS"), known once it is open; empty before. The live
    # capture reads its duplex off it (D18).
    host_api: str

    def open(self, on_chunk: OnChunk) -> None: ...

    def close(self) -> None: ...


class KeyCombination:
    """Which of the wanted keys are down, and when that last changed.

    Separate from the keyboard for two reasons. It is the only part with rules
    worth stating - a held key repeats, an unrelated key must not interfere -
    and it is the only part that can be tested without pressing anything.
    """

    def __init__(self, keys: frozenset[Any]) -> None:
        self._wanted = keys
        self._down: set[Any] = set()
        self._complete = False

    def press(self, key: Any) -> bool:
        """Records a key going down. True if this completed the combination."""
        if key in self._wanted:
            self._down.add(key)

        # Windows repeats key-down events while a key is held; without this
        # guard every repeat would switch the mode again.
        if self._complete or self._down != self._wanted:
            return False
        self._complete = True
        return True

    def release(self, key: Any) -> bool:
        """Records a key going up. True if this broke a complete combination."""
        self._down.discard(key)

        if not self._complete or self._down == self._wanted:
            return False
        self._complete = False
        return True


class HandsFree:
    """A microphone that listens on its own, and one key that switches it.

    Three rules are worth stating, because each is about something *not*
    happening.

    **The key does not record.** Pressing it arms or disarms the mode; the
    detector decides where sentences begin and end. What the key going *off*
    does to an answer under way is the state machine's business - `app.py`
    learns of it through `on_mode` - not this file's.

    **The microphone is deaf while the assistant speaks.** `mute` is called by
    the state machine, and without it the detector hears the answer coming out
    of the speakers, the assistant answers itself, and every round of that
    costs an API call. `ECHO_TAIL_SECONDS` covers what the sound card and the
    room hand over after the speaker has already been told to stop.

    **Switching off ends whatever was half heard.** The detector is reset, so a
    sentence the room had started is thrown away; a confirmation window open
    at that moment is closed with nothing, so that the question is a no at
    once rather than six seconds later.
    """

    def __init__(
        self,
        *,
        microphone: Microphone | None = None,
        toggle: Hotkey | None = None,
        endpoint: Segmenter | None = None,
        on_listening: OnEvent | None = None,
        on_mode: OnMode | None = None,
        listening: bool = True,
    ) -> None:
        self._microphone = microphone if microphone is not None else SystemMicrophone()
        self._toggle = toggle if toggle is not None else SystemHotkey(DEFAULT_TOGGLE_HOTKEY)
        self._endpoint = endpoint if endpoint is not None else Endpoint()

        # Called on the event loop the moment the detector hears a sentence
        # begin, before there is anything to transcribe. The state machine
        # stops the speaker from it: an assistant that waited for the finished
        # sentence would talk over the user, into the microphone recording
        # them. Public because whoever builds this is rarely whoever listens.
        self.on_listening = on_listening

        # Called on the event loop when the mode is switched - and once at
        # `start`, so that the screen shows the mode the microphone is
        # actually in. The state machine is the listener and passes it on to
        # the status line: it has to act on "off" before the screen does.
        self.on_mode = on_mode

        self._loop: asyncio.AbstractEventLoop | None = None
        self._finished: asyncio.Queue[list[Audio]] = asyncio.Queue()

        # The window of `listen_for`, while one is open: the next sentence
        # goes here instead of into `_finished`, and its start is not announced.
        self._window: asyncio.Future[list[Audio]] | None = None

        # Read by the audio callback on PortAudio's thread and written by the
        # keyboard's: a plain event, so that the switch is visible to the
        # audio thread in the same instant. `_muted` is written by the event
        # loop instead, and is an event for the same reason. `_answering` is
        # set while `listen_for` is open, so that the detector is shown blocks
        # even with listening off.
        self._on = threading.Event()
        if listening:
            self._on.set()
        self._muted = threading.Event()
        self._answering = threading.Event()

        # Counted down on the event loop, in samples rather than blocks so that
        # a block of any size costs what it actually holds.
        self._deaf_samples = 0

    @property
    def listening(self) -> bool:
        """Whether the microphone is live."""
        return self._on.is_set()

    def start(self) -> None:
        """Opens the microphone, starts watching the key, and says which mode
        it is in."""
        self._loop = asyncio.get_running_loop()
        self._microphone.open(self._heard)
        self._toggle.watch(on_press=self.toggle, on_release=_nothing)
        # On the loop already, so said directly rather than posted: the status
        # line would otherwise show the paused hint over a live microphone
        # until the first press.
        self._switched(self._on.is_set())

    def stop(self) -> None:
        self._on.clear()
        self._toggle.stop()
        self._microphone.close()

    def mute(self) -> None:
        """Stops listening while the assistant speaks. Called by `app.py`."""
        self._muted.set()

    def unmute(self) -> None:
        """Listens again, once the room has stopped repeating the answer.

        The order matters: the tail and the reset are in place before the flag
        is cleared, so no block of the assistant's own voice can arrive between
        the two and be treated as the beginning of a question.
        """
        self._deaf_samples = round(ECHO_TAIL_SECONDS * SAMPLE_RATE)
        self._endpoint.reset()
        self._muted.clear()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    async def utterance(self) -> Audio:
        """Waits for the next finished sentence and returns it as one buffer."""
        return _as_audio(await self._finished.get())

    async def listen_for(self, seconds: float) -> Audio | None:
        """One sentence within `seconds`, by the detector; else `None`.

        The confirmation window of `app.py` (section 3.1 rule 2). Opened
        whether or not listening is on: the assistant asked a question, and
        hearing the answer is the point of asking. The echo tail `unmute` left
        behind still counts down first - the room repeating the question is
        not a yes. The detector starts afresh on either side of the window,
        and what it hears inside is the answer: not announced, and never a
        question. The window closes when time is up, on the first sentence,
        or when listening is switched off - then with an empty answer.
        """
        self._endpoint.reset()
        self._answering.set()
        self._window = asyncio.get_running_loop().create_future()
        try:
            chunks = await asyncio.wait_for(self._window, seconds)
        except TimeoutError:
            return None
        finally:
            self._window = None
            self._answering.clear()
            self._endpoint.reset()
        return _as_audio(chunks)

    # ----------------------------------------------------------------------
    # Called from other threads. Nothing here touches asyncio directly.
    # ----------------------------------------------------------------------

    def _heard(self, chunk: Audio) -> None:
        if self._answering.is_set() or (self._on.is_set() and not self._muted.is_set()):
            self._to_loop(self._examine, chunk)

    def toggle(self) -> None:
        """What the key does, from any thread: the other mode, at once.

        Public since the tray icon (4.3): its menu line and the key are one
        road, so that whatever "off" interrupts is interrupted the same way
        whichever of the two asked.
        """
        if self._on.is_set():
            self._on.clear()
        else:
            self._on.set()

        # Which way it went travels with the message rather than being read
        # again on the other side. Two presses in quick succession are two
        # messages that both run after both flips, and a listener that read the
        # flag would be told the mode changed to whatever it is *now*, twice.
        self._to_loop(self._switched, self._on.is_set())

    def _to_loop(self, work: Callable[..., None], *args: Any) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(work, *args)

    # ----------------------------------------------------------------------
    # Called on the event loop, in the order the threads above submitted them.
    # ----------------------------------------------------------------------

    def _switched(self, listening: bool) -> None:
        self._deaf_samples = 0
        self._endpoint.reset()
        if not listening and self._window is not None and not self._window.done():
            # The assistant was asking a question and the user switched it
            # off: the answer is nothing, now, rather than silence later.
            self._window.set_result([])
        if self.on_mode is not None:
            self.on_mode(listening)

    def _examine(self, chunk: Audio) -> None:
        """One block through the detector, and a sentence out when one ended."""
        if self._deaf_samples > 0:
            self._deaf_samples -= len(chunk)
            return

        started = self._endpoint.speaking
        finished = self._endpoint.feed(chunk)

        if self._window is not None:
            # The sentence answers the question just asked: it goes to whoever
            # asked, and the state machine is not told a new question began.
            for utterance in finished:
                if not self._window.done():
                    self._window.set_result([utterance])
            return

        # Announced before the sentence is handed over: this is the moment the
        # state machine learns that whatever it was doing has been overtaken.
        if not started and self._endpoint.speaking and self.on_listening is not None:
            self.on_listening()

        for utterance in finished:
            self._finished.put_nowait([utterance])


def duplex_for(host_api: str, *, barge_in: bool) -> Duplex:
    """Whether the microphone stays live while the assistant speaks (D18).

    "full": it does, and the server hears the user over its own voice.
    "half": it is deaf while the assistant speaks and for `ECHO_TAIL_SECONDS`
    after - the old product's rule - and the only way to interrupt is the
    key. `barge_in` is `[live] barge_in`, the user's choice over the rule.
    """
    if barge_in and any(name in host_api for name in FULL_DUPLEX_HOST_APIS):
        return "full"
    return "half"


class LiveCapture(HandsFree):
    """`HandsFree` for a live session: the blocks themselves go to the server.

    The old capture collected sentences and handed each over whole. This one
    hands over the stream, and the detector is a doorman rather than a
    scribe. Three rules, each about money or about not hearing oneself.

    **The stream starts at the door.** Nothing is taken while the room is
    quiet: a session costs money from the moment it opens (D5, D12). The
    detector's onset opens the stream with the pre-roll - `PREROLL_SECONDS`
    of what came before, the first consonant included - and says so through
    `on_speech`; every block after that is queued, silence included, since
    the server's own detector finds the end of the sentence in the silence
    after it. Queued rather than dropped, because opening a session takes
    most of a second (ADR-001) and the sentence that opened it is half over
    by then. `close()` ends the stream - the state machine's call, on
    silence or a hang-up - and drops what nobody took.

    **The gate pauses the stream, not the doorman.** While paused, blocks
    are dropped instead of queued: the yes of a confirmation window and the
    room during a reminder are not for the model, now or later. The
    detector keeps watching, so that a reminder being read out is cut by
    the user starting to talk.

    **Full or half duplex is the microphone's to say** (D18). Through the
    Windows audio engine the microphone stays live while the assistant
    speaks, and `mute`/`unmute` do nothing; through raw kernel streaming
    they do what they did in the old product, deaf while speaking and
    `ECHO_TAIL_SECONDS` after. `barge_in=False` is half duplex on any
    microphone. Decided at `start`, from the microphone actually opened.

    The level of what was sent is kept (`level_dbfs`) and written down when
    the stream closes: a quiet microphone is the first thing to check when
    the model hears another language (ADR-001 section 5). The level of every
    block sent is also handed to `on_level` as it goes, for the window's orb
    (plan.md D20).

    **Asleep, only the wake word is heard** (D21). With a detector, the
    capture starts asleep: every block goes to it and to nothing else - no
    doorman, no pre-roll, no stream - until it hears the phrase; then the
    stream opens and `on_wake` is told, and the doorman is back at the
    door. `sleep()` is the state machine's call at the idle close; the key
    from off wakes it.
    """

    def __init__(
        self,
        *,
        microphone: Microphone | None = None,
        toggle: Hotkey | None = None,
        endpoint: Segmenter | None = None,
        on_speech: OnSpeech | None = None,
        on_mode: OnMode | None = None,
        on_level: OnLevel | None = None,
        listening: bool = True,
        barge_in: bool = True,
        wake: WakeDetector | None = None,
        on_wake: OnEvent | None = None,
    ) -> None:
        super().__init__(
            microphone=microphone,
            toggle=toggle,
            endpoint=endpoint,
            on_mode=on_mode,
            listening=listening,
        )

        # Called on the event loop when the detector hears speech begin
        # (`True`) and end (`False`): the `USER_SPEAKING` state of the
        # screen, and at the door the reason to open a session. Not called
        # for what is said inside a confirmation window.
        self.on_speech = on_speech
        # Called on the event loop with the level, in dBFS, of every block
        # that went to the server (plan.md D20): the orb swells with it.
        # Not called at the door or while paused - what is not sent is not
        # heard, and the picture should say so.
        self.on_level = on_level

        # The wake word (D21): while asleep, blocks go to this detector and
        # nowhere else; when it hears the phrase, `on_wake` is called on the
        # event loop, after the stream has been opened. `None` is today's
        # product: never asleep, the doorman at the door.
        self._wake = wake
        self.on_wake = on_wake
        self._asleep = False

        self._barge_in = barge_in
        # Which of the two rules applies: known once the microphone is open,
        # and half - the safe one - until then.
        self.duplex: Duplex = "half"

        # The pre-roll: the last `PREROLL_SECONDS` of blocks while nobody is
        # talking, counted in samples so that a block of any size costs what
        # it holds. Emptied into the stream at the onset.
        self._preroll_samples = round(PREROLL_SECONDS * SAMPLE_RATE)
        self._recent: list[Audio] = []
        self._recent_samples = 0

        # The stream, while one is open; `None` at the door. A new queue per
        # stream, so that a reader of the last one ends where it ended.
        self._queue: asyncio.Queue[bytes | None] | None = None
        self._paused = False

        # What was last reported through `on_speech`, so that a sentence cut
        # by a reset is still reported as ended.
        self._speaking = False

        # The level of what was sent this stream: the sum of squares and the
        # count of the samples in the blocks that were not silence.
        self._level_squares = 0.0
        self._level_samples = 0

    @property
    def taking(self) -> bool:
        """Whether a stream is open - blocks are being queued for `chunks`."""
        return self._queue is not None

    @property
    def level_dbfs(self) -> float | None:
        """The RMS level, in dBFS, of what the current or last stream sent,
        silence left out; `None` until a block that was not silence went."""
        if not self._level_samples:
            return None
        return 20 * math.log10(math.sqrt(self._level_squares / self._level_samples))

    @property
    def level_judged(self) -> float | None:
        """`level_dbfs` where `QUIET_DBFS` means something, `None` where it
        does not.

        The threshold was calibrated through the Windows audio engine, which
        applies the driver's own gain (ADR-001 section 5). The raw kernel path
        has no such gain: measured 2026-09-19 on the owner's array, the very
        session that answered every one of 25 spoken sentences sent -52 dBFS.
        A microphone that works is not worth a warning, and a user sent to the
        sound settings by a false one loses trust in every other line, so
        nothing is judged on a path where nothing was measured.
        """
        return self.level_dbfs if self.duplex == "full" else None

    @property
    def asleep(self) -> bool:
        """Whether only the wake word is listened for."""
        return self._asleep

    @property
    def wake_word(self) -> bool:
        """Whether there is a wake word to sleep behind at all."""
        return self._wake is not None

    def start(self) -> None:
        """Opens the microphone, learns which way it was opened, starts
        watching the key, and says which mode it is in - in that order, so
        that whoever hears the mode can already read the duplex."""
        self._loop = asyncio.get_running_loop()
        self._microphone.open(self._heard)
        self.duplex = duplex_for(self._microphone.host_api, barge_in=self._barge_in)
        logger.info(
            "microphone: {duplex} duplex ({host}, barge_in={barge})",
            duplex=self.duplex,
            host=self._microphone.host_api or "unknown host API",
            barge=self._barge_in,
        )
        self._toggle.watch(on_press=self.toggle, on_release=_nothing)
        self._switched(self._on.is_set())
        # Asleep from the start, when there is a phrase to wake on and the
        # switch is on (D21); the key from off, later, wakes it (`_switched`).
        if self._wake is not None and self._on.is_set():
            self._asleep = True
            self._wake.reset()

    def mute(self) -> None:
        """Deaf while the assistant speaks - in half duplex. In full duplex
        the microphone stays live: that is what barge-in is."""
        if self.duplex == "half":
            super().mute()

    def unmute(self) -> None:
        if self.duplex == "half":
            super().unmute()

    def pause(self) -> None:
        """Stops forwarding without stopping the detector: the gate is
        asking, or a reminder is being read."""
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def sleep(self) -> None:
        """Back behind the wake word: the stream ends, the pre-roll and the
        doorman are dropped, the detector starts afresh. Nothing without a
        detector."""
        if self._wake is None:
            return
        self.close()
        self._recent.clear()
        self._recent_samples = 0
        self._speaking = False
        self._endpoint.reset()
        self._wake.reset()
        self._asleep = True

    def wake(self) -> None:
        """Awake: the stream opens now - whoever hears `on_wake` opens a
        session and reads it - and the doorman starts from silence."""
        if not self._asleep:
            return
        self._asleep = False
        self._endpoint.reset()
        self._open()

    def close(self) -> None:
        """Ends the stream. The session closed on silence or was hung up;
        what was queued and not taken is dropped, and the level the server
        got is written down."""
        queue, self._queue = self._queue, None
        if queue is None:
            return
        while not queue.empty():
            queue.get_nowait()
        queue.put_nowait(None)
        self._report_level()

    async def chunks(self) -> AsyncIterator[bytes]:
        """The stream: 16-bit PCM at `SAMPLE_RATE`, one block at a time, from
        the pre-roll on, until `close`. Ends at once when no stream is open."""
        queue = self._queue
        if queue is None:
            return
        while (chunk := await queue.get()) is not None:
            yield chunk

    # ----------------------------------------------------------------------
    # Called on the event loop.
    # ----------------------------------------------------------------------

    def _switched(self, listening: bool) -> None:
        if listening:
            # The key from off means the user is about to talk: awake, not
            # asleep (D21).
            self._asleep = False
        if not listening:
            # Off is a hang-up (section 4.4 rule 4): the half sentence in the
            # queue is not the opening of the next session, and the blocks
            # either side of the switch are not neighbours.
            self.close()
            self._recent.clear()
            self._recent_samples = 0
            self._speaking = False
        super()._switched(listening)

    def _examine(self, chunk: Audio) -> None:
        """One block: through the detector, then into the stream or the
        pre-roll, and the onset reported."""
        if self._deaf_samples > 0:
            self._deaf_samples -= len(chunk)
            return

        if self._asleep and self._wake is not None:
            # Asleep: the detector alone hears the room (D21). Nothing is
            # queued and nothing remembered - a room that is not listened to.
            if self._wake.feed(chunk):
                logger.info(
                    "woken by the wake word ({score:.2f})",
                    score=getattr(self._wake, "score", 0.0),
                )
                self.wake()
                if self.on_wake is not None:
                    self.on_wake()
            return

        finished = self._endpoint.feed(chunk)

        if self._window is not None:
            # The sentence answers the question just asked (`listen_for`).
            for utterance in finished:
                if not self._window.done():
                    self._window.set_result([utterance])

        if self._queue is not None:
            if not self._paused:
                self._send(chunk)
        else:
            self._remember(chunk)

        if self._window is None:
            self._report(self._endpoint.speaking)

    def _report(self, speaking: bool) -> None:
        if speaking == self._speaking:
            return
        self._speaking = speaking
        # Opened before it is announced: whoever hears the onset opens a
        # session and starts reading, and by then the pre-roll is queued.
        if speaking:
            self._open()
        if self.on_speech is not None:
            self.on_speech(speaking)

    def _open(self) -> None:
        if self._queue is not None:
            return
        self._queue = asyncio.Queue()
        self._level_squares = 0.0
        self._level_samples = 0
        for chunk in self._recent:
            self._send(chunk)
        self._recent.clear()
        self._recent_samples = 0

    def _remember(self, chunk: Audio) -> None:
        """Keeps the block in the pre-roll, and lets the oldest go once the
        ring is longer than it should be."""
        self._recent.append(chunk)
        self._recent_samples += len(chunk)
        while self._recent and self._recent_samples - len(self._recent[0]) >= self._preroll_samples:
            self._recent_samples -= len(self._recent.pop(0))

    def _send(self, chunk: Audio) -> None:
        if self._queue is None:
            return
        squares = float(np.dot(chunk, chunk))
        if squares > _SILENCE_RMS * _SILENCE_RMS * len(chunk):
            self._level_squares += squares
            self._level_samples += len(chunk)
        if self.on_level is not None:
            self.on_level(dbfs(chunk))
        self._queue.put_nowait(to_pcm16(chunk))

    def _report_level(self) -> None:
        level = self.level_dbfs
        if level is None:
            return
        judged = self.level_judged
        if judged is not None and judged < QUIET_DBFS:
            logger.warning(
                "microphone level: {level:.0f} dBFS sent, under {quiet:.0f} dBFS - quiet; "
                "raise the microphone level or boost in Windows' sound settings",
                level=level,
                quiet=QUIET_DBFS,
            )
        else:
            # The number, always; the verdict only where it was measured.
            logger.info(
                "microphone level: {level:.0f} dBFS sent ({duplex} duplex)",
                level=level,
                duplex=self.duplex,
            )


def _as_audio(chunks: list[Audio]) -> Audio:
    """The blocks of one sentence as a single buffer."""
    if not chunks:
        # A window closed by the switch. Still an answer - what to make of it
        # belongs to the state machine, not here.
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _nothing() -> None:
    """A key coming up means nothing to a key that toggles."""


class SystemHotkey:
    """The real keyboard, through `pynput`."""

    def __init__(self, combination: str = DEFAULT_TOGGLE_HOTKEY) -> None:
        # Deferred, and untyped: `pynput` ships no stubs, which is why the
        # listener is held as `Any`.
        from pynput import keyboard  # type: ignore[import-untyped]

        try:
            parsed = keyboard.HotKey.parse(combination)
        except ValueError as error:
            raise ValueError(f"{combination!r} is not a hotkey: {error}") from error

        self.keys = frozenset(parsed)
        self._listener: Any = None

    def watch(self, *, on_press: OnEvent, on_release: OnEvent) -> None:
        from pynput import keyboard

        combination = KeyCombination(self.keys)

        # A real press reports `ctrl_l`, while the combination is written as
        # `ctrl`. `canonical` is what makes the two the same key.
        def pressed(key: Any) -> None:
            if combination.press(self._listener.canonical(key)):
                on_press()

        def released(key: Any) -> None:
            if combination.release(self._listener.canonical(key)):
                on_release()

        self._listener = keyboard.Listener(on_press=pressed, on_release=released)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


def device_choice(text: str | None) -> int | str | None:
    """A device setting as `sounddevice` wants it: an index, words, or the default.

    An int is an index into its device list; a str is words matched, in order,
    against "<device name>, <host API>". A digit string handed over as a str
    matches no name at all - which is how `bench_mic.py --device 12` never
    found a device. Empty, or nothing, is the system default.
    """
    chosen = (text or "").strip()
    if not chosen:
        return None
    return int(chosen) if chosen.isdigit() else chosen


@dataclass(frozen=True, slots=True)
class MicrophoneInfo:
    """One line of PortAudio's table, as `live-assistant mic` offers it and as
    `sounddevice` will find it again."""

    name: str
    host_api: str

    @property
    def setting(self) -> str:
        """What goes into `[audio] input_device`: the whole "<name>, <host
        API>" line. `sounddevice` matches words in order and, when several
        entries share them, takes the one this line is exactly - so the MME
        entry, which Windows cuts to 31 characters and which is therefore a
        prefix of the WASAPI one, is never mistaken for it."""
        return f"{self.name}, {self.host_api}"

    @property
    def label(self) -> str:
        """The same on one line: kernel streaming names carry a driver path
        and a line break, and the terminal shows one line per choice."""
        return f"{' '.join(self.name.split())} - {self.host_api}"


@dataclass(frozen=True, slots=True)
class Microphones:
    """What there is to choose from: every input device, and the name of the
    one Windows has chosen right now - `None` when there is none at all."""

    default: str | None
    devices: tuple[MicrophoneInfo, ...]


# PortAudio lists the default input under two more names, one per legacy
# host API. Each is whatever the default is, and the list says that once.
_DEFAULT_ALIASES = frozenset({"Microsoft Sound Mapper - Input", "Primary Sound Capture Driver"})


def available_microphones() -> Microphones:
    """Every input device, once per host API - kept apart on purpose, since the
    same microphone reaches Whisper differently through each (README: "which
    microphone path, measured")."""
    # Imported here rather than at module scope: it loads PortAudio's native
    # library, which most of the program has no use for. It ships no type
    # information either - said once, on this first import of the module,
    # which is where mypy reports it.
    import sounddevice  # type: ignore[import-untyped]

    devices = tuple(
        MicrophoneInfo(
            name=str(facts["name"]),
            host_api=str(sounddevice.query_hostapis(facts["hostapi"])["name"]),
        )
        for facts in sounddevice.query_devices()
        if facts["max_input_channels"] > 0 and facts["name"] not in _DEFAULT_ALIASES
    )
    try:
        default = str(sounddevice.query_devices(kind="input")["name"])
    except (ValueError, sounddevice.PortAudioError):
        # No input device at all: PortAudio has no default, and asking for
        # it is an error rather than an empty answer.
        default = None
    return Microphones(default=default, devices=devices)


class MicrophoneUnavailableError(RuntimeError):
    """The input device could not be opened: nothing matched the name, or
    PortAudio refused it. Fixable by the user, so named for `run`."""


# When a dropped block is worth a line in the log: the first, so that one bad
# minute is on record, then rarely enough that a bad microphone cannot fill
# the file. Between these the count is kept and nothing is written.
_REPORT_OVERFLOWS_AT = frozenset({1, 10, 100})
_REPORT_OVERFLOWS_EVERY = 1000


class SystemMicrophone:
    """The real microphone, through `sounddevice`.

    Asked for 16 kHz mono, which is what everything downstream expects. Not
    every path to a microphone will give it: PortAudio's WASAPI runs a shared
    stream at the mixer's rate unless told to convert, and kernel streaming
    has no converter at all - measured 2026-09-05, the internal array's two
    rawer entries refused 16 kHz while its MME entry, with Windows resampling
    in between, accepted it. So a WASAPI device is told to convert, and a
    device that still refuses is opened at its own rate and brought to 16 kHz
    here, block by block (`audio/resample.py`). `rate` says which happened.
    """

    def __init__(self, *, device: int | str | None = None) -> None:
        self._device = device
        self._stream: Any = None
        self._on_chunk: OnChunk | None = None
        self._resampler: Resampler | None = None
        # The rate the device actually runs at once opened: `SAMPLE_RATE`
        # unless it would not, in which case the blocks are resampled.
        self.rate: int | None = None
        # The way to the device, in PortAudio's words, once opened.
        self.host_api = ""
        # Blocks the driver dropped before this saw them. In hands-free mode
        # each is a hole in a sentence that nothing else would notice.
        self.overflows = 0

    def open(self, on_chunk: OnChunk) -> None:
        # Local for the same reason as in `available_microphones`.
        import sounddevice

        self._on_chunk = on_chunk
        try:
            self._stream = self._open_stream(sounddevice)
            self._stream.start()
        except (ValueError, sounddevice.PortAudioError) as failure:
            # `ValueError` is a name that matched no device. `PortAudioError`
            # is a device that exists and would not open: held by another
            # program, or unplugged since the list was made.
            self.close()
            which = (
                "the default microphone"
                if self._device is None
                else f"the microphone {self._device!r}"
            )
            raise MicrophoneUnavailableError(f"{which} could not be opened: {failure}") from failure

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _open_stream(self, sounddevice: Any) -> Any:
        """16 kHz if the device will run at it; its own rate, resampled, if not."""
        facts = sounddevice.query_devices(self._device, "input")
        host = str(sounddevice.query_hostapis(facts["hostapi"])["name"])
        self.host_api = host
        native = round(float(facts["default_samplerate"]))
        # Which device this is, in the line `live-assistant mic` would store for
        # it. With the setting empty it is whatever Windows handed over, and
        # this is the only place that says which.
        logger.info("microphone: {line}", line=MicrophoneInfo(str(facts["name"]), host).setting)
        # Only WASAPI understands this, and PortAudio refuses a stream whose
        # host-specific settings belong to another host API.
        extra = sounddevice.WasapiSettings(auto_convert=True) if "WASAPI" in host else None

        try:
            stream = self._stream_at(sounddevice, SAMPLE_RATE, extra)
        except sounddevice.PortAudioError:
            if native == SAMPLE_RATE:
                raise
            # Whatever the refusal was, the one thing left to try is the rate
            # the device says it runs at. If that fails too, that error is
            # the one worth reading, and it is the one that propagates.
            stream = self._stream_at(sounddevice, native, extra)
            self._resampler = Resampler(native, SAMPLE_RATE)
            self.rate = native
            logger.info(
                "microphone {device!r} runs at {rate} Hz; resampling to {target}",
                device=self._device,
                rate=native,
                target=SAMPLE_RATE,
            )
        else:
            self._resampler = None
            self.rate = SAMPLE_RATE
        return stream

    def _stream_at(self, sounddevice: Any, rate: int, extra: Any) -> Any:
        return sounddevice.InputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            # Twenty milliseconds at whatever rate the device runs at, so a
            # resampled block is still the block size everything else counts on.
            blocksize=round(CHUNK_FRAMES * rate / SAMPLE_RATE),
            device=self._device,
            callback=self._block,
            extra_settings=extra,
        )

    def _block(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """PortAudio's callback thread. Whatever this does, it does it quickly."""
        if status and status.input_overflow:
            # Counted always, written rarely: a log line per block, on this
            # thread, would be the next thing to cause an overflow.
            self.overflows += 1
            count = self.overflows
            if count in _REPORT_OVERFLOWS_AT or count % _REPORT_OVERFLOWS_EVERY == 0:
                logger.warning("microphone overflow: {count} blocks dropped so far", count=count)
        if self._on_chunk is not None:
            # One channel, and a copy: `indata` is PortAudio's own buffer and
            # holds the next block by the time anyone reads this one.
            chunk = indata[:, 0].copy()
            if self._resampler is not None:
                chunk = self._resampler.push(chunk)
            self._on_chunk(chunk)
