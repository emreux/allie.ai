"""The state machine of the live product (plan.md sections 4.2-4.4, task L1.4).

`IDLE` at the door, `USER_SPEAKING` while the doorman hears a voice,
`SPEAKING` while the model's voice plays, and back. Everything it needs is
injected, so a whole conversation is driven here without a microphone, a
model or a sound card: a fake capture whose doorman speaks when the test
says, a fake provider that opens scripted sessions, a fake sound card that
remembers what it played.

Each rule of section 4.4 is a test here, in the old file's style, and the
confirmation window's tests are lifted from the old file whole - the window
is the one thing the old pipeline and the live product do the same way.

Four of these tests are the ones worth keeping if the rest were deleted.

**A session opens on speech and closes on silence.** Nothing is opened for a
quiet room, and a session left alone for `idle_close_seconds` is closed:
a live model bills by the minute while open.

**The model's voice stops the instant the server says so.** `Interrupted`
drops what is queued and aborts what the sound card holds.

**A tool that asks runs only on a yes heard out loud**, with the session's
input paused for the whole exchange, so that the yes never reaches the model.

**Nothing said out loud is written in this file.** The sentences come from the
locale pack, with the English constants of `app.TEXT` as the end of the
chain (section 3.12), exactly as in the setup wizard.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pytest
from loguru import logger

from assistant import app
from assistant.agent.core import Confirm, ToolRunner
from assistant.agent.limits import Limits
from assistant.agent.policy import DECLINED, dispatch
from assistant.announce.queue import AnnounceQueue
from assistant.app import (
    CONFIRM_WINDOW_SECONDS,
    IDLE_CLOSE_SECONDS,
    MAX_OPEN_FAILURES,
    RESUME_MINUTES,
    Greeting,
    LiveAssistant,
    NoVoiceError,
    State,
    Turn,
    read_answer,
)
from assistant.audio.player import PlaybackError
from assistant.audio.wake import CHIME_RATE, chime
from assistant.live.base import (
    AudioChunk,
    AuthenticationError,
    Closed,
    GoingAway,
    InputText,
    Interrupted,
    LiveEvent,
    OutputText,
    ProviderError,
    Resumable,
    SessionConfig,
    ToolCall,
    ToolCallCancelled,
    ToolCallEvent,
    TurnComplete,
    Usage,
    UsageReport,
)
from assistant.locales import Locale
from assistant.store.db import open_database
from assistant.store.repos import UsageRepo
from assistant.stt.base import SAMPLE_RATE, Audio, Transcript
from assistant.tools.registry import ToolRegistry, tool
from assistant.tts.base import VoiceInfo
from assistant.usage.tracker import Pricing, UsageTracker
from tests.live_contract import FakeLiveProvider, FakeLiveSession

TOLGA = VoiceInfo(id=r"HKLM\...\TR-TR_TOLGA", display_name="Microsoft Tolga", language="tr")
ZIRA = VoiceInfo(id=r"HKLM\...\EN-US_ZIRA", display_name="Microsoft Zira", language="en")

TURKISH = Locale(
    code="tr",
    name="Türkçe",
    stt_language="tr",
    voices={"fake": "Tolga"},
    ui={
        "key_invalid": "API anahtarın geçersiz görünüyor, yenilemen gerekiyor.",
        "unreachable": "Sağlayıcıya bağlanamadım, tekrar dener misin?",
        "took_too_long": "Bu iş uzadı, tekrar dener misin?",
        "confirm_hint": "Evet ya da hayır de.",
        "confirm_again": "Anlayamadım. Evet mi, hayır mı?",
        "daily_over": "Bugünkü harcama sınırını aştın.",
        "monthly_over": "Bu ayki harcama sınırını aştın.",
        "spend_stopped": "Harcama sınırı aşıldı, bu yüzden modele sormuyorum.",
        "wake_greeting": "Sizi dinliyorum efendim.",
    },
    yes_words=("evet", "tamam"),
    no_words=("hayır", "iptal"),
    fillers=("Bir saniye, bakıyorum.",),
)

# One 20 ms block of 16 kHz, 16-bit audio, as the capture sends it.
BLOCK = bytes(640)
RATE = 24_000


def speech(seconds: float = 2.0) -> Audio:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def voice(text: str) -> AudioChunk:
    """The model saying `text`: its bytes stand in for the sound, so that
    what the sound card played can be read back as words."""
    return AudioChunk(pcm16=text.encode("utf-8"), sample_rate=RATE)


async def until(condition: Callable[[], object], *, tries: int = 400) -> None:
    """Waits, briefly, for `condition()` to become true."""
    for _ in range(tries):
        if condition():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("the condition did not come true in time")


async def tick() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


# --------------------------------------------------------------------------
# The fakes
# --------------------------------------------------------------------------


class FakeCapture:
    """The live microphone without a room: the doorman speaks when the test
    says so, and the stream it opens holds what the test puts in it."""

    def __init__(
        self,
        *,
        answers: Sequence[Audio | None] = (),
        listening: bool = True,
        hold: float = 0.0,
        asleep: bool = False,
    ) -> None:
        self.on_speech: Callable[[bool], None] | None = None
        self.on_mode: Callable[[bool], None] | None = None
        self.started = False
        self.listening = listening
        self.taking = False
        # The wake word (D21): a capture built asleep has a detector, and
        # the state machine's hook for the phrase; how often it was put
        # back to sleep.
        self.on_wake: Callable[[], None] | None = None
        self.asleep = asleep
        self.wake_word = asleep
        self.sleeps = 0
        # Whether the state machine has told it to stop listening, and how many
        # times it has been told either thing.
        self.deaf = False
        self.switches: list[bool] = []
        # Whether the session's input is paused (D3), and every change of it.
        self.paused = False
        self.pauses: list[bool] = []
        self.closed = 0
        self._stream: asyncio.Queue[bytes | None] | None = None
        # What each confirmation window hears, in order - a window with
        # nothing scripted hears silence - and, for each one opened, how long
        # it was opened for and what the microphone and the input were doing.
        self.windows: list[float] = []
        self.deaf_windows: list[bool] = []
        self.paused_windows: list[bool] = []
        self._answers = list(answers)
        # What happens while the window is open, and how long it stays so.
        self.on_window: Callable[[], None] | None = None
        self._hold = hold

    def start(self) -> None:
        self.started = True
        # As `LiveCapture` does: asleep only with the switch on (D21).
        if not self.listening:
            self.asleep = False
        if self.on_mode is not None:
            self.on_mode(self.listening)

    def stop(self) -> None:
        self.started = False

    def toggle(self) -> None:
        self.listening = not self.listening
        if not self.listening:
            self._end()
        else:
            self.asleep = False
        if self.on_mode is not None:
            self.on_mode(self.listening)

    def mute(self) -> None:
        self.deaf = True
        self.switches.append(True)

    def unmute(self) -> None:
        self.deaf = False
        self.switches.append(False)

    def pause(self) -> None:
        self.paused = True
        self.pauses.append(True)

    def resume(self) -> None:
        self.paused = False
        self.pauses.append(False)

    def close(self) -> None:
        self.closed += 1
        self._end()

    def sleep(self) -> None:
        self.sleeps += 1
        self.asleep = True
        self._end()

    def wake(self) -> None:
        self.asleep = False
        if self._stream is None:
            self._stream = asyncio.Queue()
            self.taking = True

    def call_wake(self) -> None:
        """The room said the phrase: the stream opens, the loop is told."""
        self.wake()
        if self.on_wake is not None:
            self.on_wake()

    def _end(self) -> None:
        stream, self._stream = self._stream, None
        self.taking = False
        if stream is not None:
            stream.put_nowait(None)

    async def chunks(self) -> AsyncIterator[bytes]:
        stream = self._stream
        if stream is None:
            return
        while (chunk := await stream.get()) is not None:
            yield chunk

    async def listen_for(self, seconds: float) -> Audio | None:
        self.windows.append(seconds)
        self.deaf_windows.append(self.deaf)
        self.paused_windows.append(self.paused)
        if self.on_window is not None:
            self.on_window()
        if self._hold:
            await asyncio.sleep(self._hold)
        return self._answers.pop(0) if self._answers else None

    # The room.

    def speak(self, *blocks: bytes) -> None:
        """The doorman's onset: the stream opens with the pre-roll, and the
        state machine is told - as `LiveCapture` does, in that order."""
        if not self.listening:
            return
        if self._stream is None:
            self._stream = asyncio.Queue()
            self.taking = True
            for block in blocks or (BLOCK,):
                self._stream.put_nowait(block)
        if self.on_speech is not None:
            self.on_speech(True)

    def quiet(self) -> None:
        """The doorman's offset."""
        if self.on_speech is not None:
            self.on_speech(False)

    def heard(self, block: bytes) -> None:
        """More of the room, while the stream is open; dropped while paused."""
        if self._stream is not None and not self.paused:
            self._stream.put_nowait(block)

    def switch_off(self) -> None:
        if self.listening:
            self.toggle()

    def switch_on(self) -> None:
        if not self.listening:
            self.toggle()


class FakeSTT:
    """A recogniser that hears whatever the test says it hears."""

    id = "fake_stt"
    supports_streaming = False

    def __init__(self, *heard: Transcript) -> None:
        self.heard = list(heard) or [Transcript(text="evet", confidence=0.9)]
        self.hints: list[str | None] = []

    async def transcribe(self, pcm: Audio, *, hint: str | None = None) -> Transcript:
        self.hints.append(hint)
        return self.heard.pop(0) if len(self.heard) > 1 else self.heard[0]

    def transcribe_stream(
        self, pcm_chunks: AsyncIterator[Audio], *, hint: str | None = None
    ) -> AsyncIterator[Transcript]:
        raise NotImplementedError


class FakeTTS:
    """An engine that returns the text it was given instead of speech.

    It speaks at the model's rate rather than Windows', deliberately: a rate
    that happened to match the one the player would default to would let a
    hardcoded 16 kHz pass the test written to catch it.
    """

    id = "fake"
    sample_rate = RATE

    def __init__(self, *, installed: list[VoiceInfo] | None = None) -> None:
        self.installed = [TOLGA, ZIRA] if installed is None else installed
        self.said: list[str] = []
        self.voices_used: list[str] = []
        self.asked_for: list[str | None] = []

    async def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        self.asked_for.append(language)
        if language is None:
            return list(self.installed)
        return [voice for voice in self.installed if voice.language == language]

    async def stream(self, chunks: AsyncIterator[str], *, voice: str) -> AsyncIterator[bytes]:
        self.voices_used.append(voice)
        async for chunk in chunks:
            self.said.append(chunk)
            yield chunk.encode("utf-8")


class FakeSpeaker:
    """A sound card that remembers what it was asked to play."""

    def __init__(self, *, on_play: Callable[[], None] | None = None) -> None:
        self.played: list[bytes] = []
        self.rates: list[int] = []
        self.stopped = 0
        self.on_play = on_play

    async def play(self, buffers: AsyncIterator[bytes], *, sample_rate: int) -> None:
        self.rates.append(sample_rate)
        if self.on_play is not None:
            self.on_play()
        async for buffer in buffers:
            self.played.append(buffer)

    def stop(self) -> None:
        self.stopped += 1

    @property
    def heard(self) -> str:
        return b"".join(self.played).decode("utf-8")


class Room(FakeLiveSession):
    """A session that stays open: its events arrive when the test says, and
    it ends when the server hangs up or the assistant does. What the server
    says once a tool has answered is scripted apart, since it cannot come
    before the answer."""

    def __init__(self, *events: LiveEvent, after_result: Sequence[LiveEvent] = ()) -> None:
        super().__init__()
        self._events: asyncio.Queue[LiveEvent] = asyncio.Queue()
        self.arrives(*events)
        self._after_result = list(after_result)

    def arrives(self, *events: LiveEvent) -> None:
        for event in events:
            self._events.put_nowait(event)

    def hangs_up(self, error: ProviderError | None = None) -> None:
        self.arrives(Closed(error))

    async def send_tool_result(self, call: ToolCall, content: str) -> None:
        await super().send_tool_result(call, content)
        after, self._after_result = self._after_result, []
        self.arrives(*after)

    async def events(self) -> AsyncIterator[LiveEvent]:
        while True:
            event = await self._events.get()
            self.read += 1
            if isinstance(event, AudioChunk):
                self.audio_out_ms += len(event.pcm16) * 1000 // (event.sample_rate * 2)
            yield event
            if isinstance(event, Closed):
                return


class Provider(FakeLiveProvider):
    """Opens the sessions it was given, in order - a plain scripted one
    that closes at once when it runs out - refusing where told to."""

    def __init__(
        self,
        *sessions: FakeLiveSession,
        refuse: Sequence[ProviderError | None] = (),
        events: Sequence[LiveEvent] = (),
    ) -> None:
        super().__init__(events)
        self._sessions = list(sessions)
        self._refuse = list(refuse)

    @asynccontextmanager
    async def connect(self, config: SessionConfig) -> AsyncIterator[FakeLiveSession]:
        self.opened.append(config)
        refusal = self._refuse.pop(0) if self._refuse else None
        if refusal is not None:
            raise refusal
        session = self._sessions.pop(0) if self._sessions else FakeLiveSession(self.events)
        self.sessions.append(session)
        yield session


class FakeGate:
    """A gate that lets everything through and remembers what came."""

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []
        self.confirms: list[Confirm] = []

    async def __call__(self, call: ToolCall, *, turn_id: str, confirm: Confirm) -> str:
        self.calls.append(call)
        self.confirms.append(confirm)
        return f"{call.name}: done"


class Held(FakeGate):
    """A gate whose tool takes as long as the test says."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def __call__(self, call: ToolCall, *, turn_id: str, confirm: Confirm) -> str:
        self.calls.append(call)
        self.confirms.append(confirm)
        await self.release.wait()
        return f"{call.name}: done"


@tool(risk="safe")
async def clock() -> str:
    """Tells the time."""
    return "15:04"


def runner_with(gate: FakeGate | None = None, **limits: Any) -> ToolRunner:
    return ToolRunner(ToolRegistry([clock]), gate if gate else FakeGate(), Limits(**limits))


def calls(tool_name: str = "clock", call_id: str = "c1", **arguments: object) -> ToolCallEvent:
    """The model asking for `tool_name`, as a session delivers it."""
    return ToolCallEvent(ToolCall(id=call_id, name=tool_name, arguments=arguments))


def assistant_with(
    *,
    capture: FakeCapture | None = None,
    provider: Provider | None = None,
    events: Sequence[LiveEvent] = (),
    tts: FakeTTS | None = None,
    stt: FakeSTT | None = None,
    speaker: FakeSpeaker | None = None,
    locale: Locale = TURKISH,
    runner: ToolRunner | None = None,
    tracker: UsageTracker | None = None,
    announcements: AnnounceQueue | None = None,
    idle_close_seconds: float = IDLE_CLOSE_SECONDS,
    resume_minutes: float = RESUME_MINUTES,
    filler_delay: float = 60.0,
    on_state: Callable[[State], None] | None = None,
    on_turn: Callable[[Turn], None] | None = None,
    on_mode: Callable[[bool], None] | None = None,
    on_session: Callable[[bool], None] | None = None,
    config: Callable[[], SessionConfig] | None = None,
    greeting: Greeting = "chime",
) -> LiveAssistant:
    """An assistant over fakes. `events` scripts the one session the plain
    provider opens: the model answers with them and hangs up politely."""
    tools = runner if runner is not None else runner_with()
    return LiveAssistant(
        capture=capture if capture is not None else FakeCapture(),
        provider=provider if provider is not None else Provider(events=events or ANSWER),
        session_config=config
        if config is not None
        else lambda: SessionConfig(model="fake-1", tools=tools.specs()),
        tool_runner=tools,
        tts=tts if tts is not None else FakeTTS(),
        stt=stt if stt is not None else FakeSTT(),
        speaker=speaker if speaker is not None else FakeSpeaker(),
        locale=locale,
        tracker=tracker,
        announcements=announcements,
        idle_close_seconds=idle_close_seconds,
        resume_minutes=resume_minutes,
        filler_delay=filler_delay,
        on_state=on_state,
        on_turn=on_turn,
        on_mode=on_mode,
        on_session=on_session,
        greeting=greeting,
    )


# The model answering "saat kaç": what it heard, what it says, and that it
# is done. The plain provider's session hangs up politely after these.
ANSWER: list[LiveEvent] = [
    InputText("saat kaç"),
    voice("Üçü dört geçiyor."),
    OutputText("Üçü dört geçiyor."),
    TurnComplete(),
]


async def one_turn(assistant: LiveAssistant, capture: FakeCapture) -> None:
    """Starts the machine, has the user say one thing, and waits until
    everything that follows is over."""
    await assistant.begin()
    capture.speak()
    capture.quiet()
    await assistant.settled()


# --------------------------------------------------------------------------
# The states
# --------------------------------------------------------------------------


async def test_a_turn_walks_through_the_states_the_plan_names() -> None:
    """Section 4.2: the door, the voice, the door, the answer, the door."""
    seen: list[State] = []
    capture = FakeCapture()

    await one_turn(assistant_with(capture=capture, on_state=seen.append), capture)

    assert seen == [State.IDLE, State.USER_SPEAKING, State.IDLE, State.SPEAKING, State.IDLE]


async def test_the_assistant_says_it_is_ready_before_anybody_says_anything() -> None:
    seen: list[State] = []
    assistant = assistant_with(on_state=seen.append)

    await assistant.begin()

    assert seen == [State.IDLE]
    assert assistant.state is State.IDLE


async def test_a_switch_that_is_off_at_the_start_is_off() -> None:
    seen: list[State] = []
    assistant = assistant_with(capture=FakeCapture(listening=False), on_state=seen.append)

    await assistant.begin()

    assert seen == [State.OFF]


async def test_the_microphone_opens_when_the_assistant_does() -> None:
    capture = FakeCapture()
    assistant = assistant_with(capture=capture)

    await assistant.begin()

    assert capture.started
    assert capture.on_speech is not None and capture.on_mode is not None


async def test_the_microphone_is_closed_when_the_program_ends() -> None:
    capture = FakeCapture()
    assistant = assistant_with(capture=capture)
    task = asyncio.create_task(assistant.run())
    await until(lambda: capture.started)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not capture.started


async def test_the_doorman_s_opinion_is_shown_and_nothing_more() -> None:
    """`USER_SPEAKING` is the local detector's and is there for the screen;
    a voice while the model speaks changes nothing here - the server hears
    it too, and says so (`Interrupted`)."""
    seen: list[State] = []
    capture = FakeCapture()
    room = Room(voice("Uzun bir cevap"))
    assistant = assistant_with(capture=capture, provider=Provider(room), on_state=seen.append)
    await assistant.begin()

    capture.speak()
    capture.quiet()
    await until(lambda: assistant.state is State.SPEAKING)
    capture.speak()
    await tick()

    assert seen == [State.IDLE, State.USER_SPEAKING, State.IDLE, State.SPEAKING]
    assert assistant.state is State.SPEAKING
    room.hangs_up()
    await assistant.settled()


async def test_the_tray_learns_when_a_session_opens_and_closes() -> None:
    sessions: list[bool] = []
    capture = FakeCapture()

    await one_turn(assistant_with(capture=capture, on_session=sessions.append), capture)

    assert sessions == [True, False]


# --------------------------------------------------------------------------
# Rule 1: a session opens on speech and closes on silence (D5)
# --------------------------------------------------------------------------


async def test_nothing_is_opened_for_a_quiet_room() -> None:
    """A live session bills by the minute while open, silent or not."""
    provider = Provider()
    assistant = assistant_with(provider=provider)

    await assistant.begin()
    await asyncio.sleep(0.02)

    assert provider.opened == []
    assert not assistant.session_open


async def test_speech_opens_a_session_and_the_pre_roll_reaches_it_first() -> None:
    """The doorman's onset comes with what was said before it was sure; the
    session hears that first, then the rest as it comes."""
    capture = FakeCapture()
    room = Room()
    assistant = assistant_with(capture=capture, provider=Provider(room))
    await assistant.begin()

    capture.speak(b"pre", b"roll")
    await until(lambda: assistant.session_open)
    capture.heard(b"more")
    await until(lambda: len(room.audio) == 3)

    assert room.audio == [b"pre", b"roll", b"more"]
    room.hangs_up()
    await assistant.settled()


async def test_the_session_is_opened_with_what_the_source_says_at_that_moment() -> None:
    """The prompt carries the user's facts and the time, and both move: the
    source is read at every open, never once."""
    prompts = iter(["first", "second"])
    provider = Provider()
    runner = runner_with()
    capture = FakeCapture()
    assistant = assistant_with(
        capture=capture,
        provider=provider,
        runner=runner,
        config=lambda: SessionConfig(
            model="fake-1", system_prompt=next(prompts), tools=runner.specs()
        ),
    )

    await one_turn(assistant, capture)
    capture.speak()
    capture.quiet()
    await assistant.settled()

    assert [config.system_prompt for config in provider.opened] == ["first", "second"]
    assert provider.opened[0].tools == runner.specs()
    assert provider.opened[0].model == "fake-1"


async def test_silence_closes_the_session_after_the_idle_seconds() -> None:
    capture = FakeCapture()
    seen: list[State] = []
    assistant = assistant_with(
        capture=capture, provider=Provider(Room()), idle_close_seconds=0.03, on_state=seen.append
    )
    await assistant.begin()

    capture.speak()
    capture.quiet()
    await until(lambda: assistant.session_open)
    await until(lambda: not assistant.session_open)
    await assistant.settled()

    assert capture.closed == 1
    assert assistant.state is State.IDLE
    assert seen[-1] is State.IDLE


async def test_the_model_s_voice_keeps_the_session_open() -> None:
    """Sixty seconds of nothing *said and nothing played*: an answer under
    way is not silence."""
    capture = FakeCapture()
    room = Room()
    assistant = assistant_with(capture=capture, provider=Provider(room), idle_close_seconds=0.1)
    await assistant.begin()
    capture.speak()
    capture.quiet()
    await until(lambda: assistant.session_open)

    for _ in range(8):
        await asyncio.sleep(0.02)
        room.arrives(voice("..."))

    assert assistant.session_open
    room.hangs_up()
    await assistant.settled()


async def test_the_next_sentence_opens_a_session_of_its_own() -> None:
    capture = FakeCapture()
    provider = Provider()
    assistant = assistant_with(capture=capture, provider=provider)

    await one_turn(assistant, capture)
    capture.speak()
    capture.quiet()
    await assistant.settled()

    assert len(provider.opened) == 2
    assert capture.closed == 2


async def test_the_conversation_continues_within_the_resume_window() -> None:
    """The handle the session left behind is what the next one opens with
    (D5); after `resume_minutes` it is not."""
    capture = FakeCapture()
    provider = Provider(events=[Resumable("h-1"), *ANSWER])
    assistant = assistant_with(capture=capture, provider=provider)

    await one_turn(assistant, capture)
    capture.speak()
    capture.quiet()
    await assistant.settled()

    assert provider.opened[0].resume_handle is None
    assert provider.opened[1].resume_handle == "h-1"


async def test_the_handle_is_forgotten_after_the_resume_minutes() -> None:
    capture = FakeCapture()
    provider = Provider(events=[Resumable("h-1"), *ANSWER])
    assistant = assistant_with(capture=capture, provider=provider, resume_minutes=0.0)

    await one_turn(assistant, capture)
    capture.speak()
    capture.quiet()
    await assistant.settled()

    assert provider.opened[1].resume_handle is None


async def test_a_handle_the_provider_refuses_is_dropped_and_the_session_opened_afresh() -> None:
    """Measured 2026-09-21: Gemini takes a handle 4 minutes after the close
    and refuses one 5 minutes after it (close code 1011), for the whole two
    hours its documentation promises. Retried with the same handle, the same
    refusal came every time - the owner heard "could not reach the provider"
    on every sentence for the rest of the resume window (2026-09-19 and 20),
    and only a restart helped. A refused handle is forgotten on the spot and
    the session opened without it: nothing is said, and the turn goes on."""
    capture = FakeCapture()
    tts = FakeTTS()
    turns: list[Turn] = []
    stale = ProviderError("gemini refused the request (1011): Internal error", kind="refused")
    provider = Provider(refuse=[None, stale, None], events=[Resumable("h-1"), *ANSWER])
    assistant = assistant_with(capture=capture, provider=provider, tts=tts, on_turn=turns.append)

    await one_turn(assistant, capture)
    capture.speak()
    capture.quiet()
    await assistant.settled()

    assert [config.resume_handle for config in provider.opened] == [None, "h-1", None]
    assert tts.said == []
    assert [turn.failure for turn in turns] == [None, None]
    assert assistant.state is State.IDLE


async def test_a_fresh_open_refused_after_the_handle_is_one_failure_said_once() -> None:
    capture = FakeCapture()
    tts = FakeTTS()
    turns: list[Turn] = []
    refused = ProviderError("gemini refused the request (1011): Internal error", kind="refused")
    provider = Provider(refuse=[None, refused, refused], events=[Resumable("h-1"), *ANSWER])
    assistant = assistant_with(capture=capture, provider=provider, tts=tts, on_turn=turns.append)

    await one_turn(assistant, capture)
    capture.speak()
    capture.quiet()
    await assistant.settled()

    assert [config.resume_handle for config in provider.opened] == [None, "h-1", None]
    assert tts.said == [TURKISH.ui["unreachable"]]
    assert [turn.failure for turn in turns] == [None, "unreachable"]


async def test_a_network_that_is_down_keeps_the_handle_for_when_it_is_back() -> None:
    """Only a refusal is the handle's fault. A socket that would not open
    is said out loud as before, and the handle is still offered next time."""
    capture = FakeCapture()
    tts = FakeTTS()
    down = ProviderError("gemini could not be reached (ConnectionError)", kind="unreachable")
    provider = Provider(refuse=[None, down, None], events=[Resumable("h-1"), *ANSWER])
    assistant = assistant_with(capture=capture, provider=provider, tts=tts)

    await one_turn(assistant, capture)
    capture.speak()
    capture.quiet()
    await assistant.settled()
    capture.speak()
    capture.quiet()
    await assistant.settled()

    assert [config.resume_handle for config in provider.opened] == [None, "h-1", "h-1"]
    assert tts.said == [TURKISH.ui["unreachable"]]


async def test_a_session_that_cannot_be_opened_is_logged_with_the_providers_words() -> None:
    """The log of 2026-09-20 said "session failed: unreachable" and nothing
    else - the close code and Google's sentence, which named the cause,
    were in the exception and nowhere else."""
    lines: list[str] = []
    sink = logger.add(lines.append, format="{level} {message}")
    try:
        capture = FakeCapture()
        provider = Provider(
            refuse=[
                ProviderError("gemini refused the request (1011): Internal error", kind="refused")
            ]
        )
        assistant = assistant_with(capture=capture, provider=provider)

        await one_turn(assistant, capture)
    finally:
        logger.remove(sink)

    assert any("1011" in line and "Internal error" in line for line in lines)


async def test_a_session_that_cannot_be_opened_is_said_out_loud_and_the_door_is_kept() -> None:
    """The sentence is the pack's; the stream the doorman opened is dropped;
    the next onset tries again."""
    capture = FakeCapture()
    tts = FakeTTS()
    turns: list[Turn] = []
    provider = Provider(refuse=[ProviderError("down", kind="unreachable")])
    assistant = assistant_with(capture=capture, provider=provider, tts=tts, on_turn=turns.append)

    await one_turn(assistant, capture)

    assert tts.said == [TURKISH.ui["unreachable"]]
    assert assistant.state is State.IDLE
    assert capture.closed == 1
    assert [turn.failure for turn in turns] == ["unreachable"]

    capture.speak()
    capture.quiet()
    await assistant.settled()
    assert len(provider.opened) == 2


async def test_a_refused_key_is_said_out_loud_in_words_that_help() -> None:
    """The user has to renew it (section 3.2); "could not connect" would
    send them to check the network instead."""
    capture = FakeCapture()
    tts = FakeTTS()
    provider = Provider(refuse=[AuthenticationError("401")])

    await one_turn(assistant_with(capture=capture, provider=provider, tts=tts), capture)

    assert tts.said == [TURKISH.ui["key_invalid"]]


async def test_a_provider_that_took_too_long_is_said_so() -> None:
    capture = FakeCapture()
    tts = FakeTTS()
    provider = Provider(refuse=[ProviderError("slow", kind="timeout")])

    await one_turn(assistant_with(capture=capture, provider=provider, tts=tts), capture)

    assert tts.said == [TURKISH.ui["took_too_long"]]


async def test_a_network_that_is_not_there_is_said_out_loud_too() -> None:
    """`OSError` as well as our own: a socket refused below the adapter's
    transport never reaches it to be translated."""
    capture = FakeCapture()
    tts = FakeTTS()

    class Unplugged(Provider):
        @asynccontextmanager
        async def connect(self, config: SessionConfig) -> AsyncIterator[FakeLiveSession]:
            raise OSError("no route to host")
            yield  # pragma: no cover

    await one_turn(assistant_with(capture=capture, provider=Unplugged(), tts=tts), capture)

    assert tts.said == [TURKISH.ui["unreachable"]]


async def test_three_failures_in_a_row_switch_the_assistant_off() -> None:
    """A dead network must not cost a sentence at every onset: the third
    failure says its sentence and the switch goes off, tray and all."""
    capture = FakeCapture()
    tts = FakeTTS()
    modes: list[bool] = []
    down = ProviderError("down", kind="unreachable")
    provider = Provider(refuse=[down] * MAX_OPEN_FAILURES)
    assistant = assistant_with(capture=capture, provider=provider, tts=tts, on_mode=modes.append)
    await assistant.begin()

    for _ in range(MAX_OPEN_FAILURES):
        capture.speak()
        capture.quiet()
        await assistant.settled()

    assert len(tts.said) == MAX_OPEN_FAILURES
    assert assistant.state is State.OFF
    assert not capture.listening
    assert modes == [True, False]


async def test_a_turn_that_came_through_starts_the_count_again() -> None:
    down = ProviderError("down", kind="unreachable")
    capture = FakeCapture()
    provider = Provider(refuse=[down, down, None, down, down], events=ANSWER)
    assistant = assistant_with(capture=capture, provider=provider)
    await assistant.begin()

    for _ in range(5):
        capture.speak()
        capture.quiet()
        await assistant.settled()

    assert assistant.state is State.IDLE
    assert capture.listening


async def test_the_limit_is_the_three_failures_the_plan_names() -> None:
    assert MAX_OPEN_FAILURES == 3


async def test_the_idle_close_and_the_resume_window_are_the_config_s_defaults() -> None:
    assert IDLE_CLOSE_SECONDS == 60.0
    assert RESUME_MINUTES == 10.0


# --------------------------------------------------------------------------
# Rule 2: the model's voice plays as it arrives and stops when the server says so
# --------------------------------------------------------------------------


async def test_the_model_s_voice_is_played_at_the_rate_it_came_in() -> None:
    capture = FakeCapture()
    speaker = FakeSpeaker()

    await one_turn(assistant_with(capture=capture, speaker=speaker), capture)

    assert speaker.heard == "Üçü dört geçiyor."
    assert speaker.rates == [RATE]


async def test_the_answer_stops_the_instant_the_server_says_so() -> None:
    """`Interrupted`: the sound card is told to abort, and nothing said
    after that point is written down as said."""
    capture = FakeCapture()
    speaker = FakeSpeaker()
    turns: list[Turn] = []
    events: list[LiveEvent] = [
        voice("Uzun "),
        OutputText("Uzun "),
        Interrupted(),
        OutputText("bir cevap."),
        TurnComplete(),
    ]

    await one_turn(
        assistant_with(capture=capture, speaker=speaker, events=events, on_turn=turns.append),
        capture,
    )

    assert speaker.stopped >= 1
    assert [turn.said for turn in turns] == ["Uzun"]


async def test_the_microphone_is_deaf_for_the_answer_and_listens_again_after_it() -> None:
    """On a microphone that needs it (D18): the state machine calls `mute`
    and `unmute` around the model's voice; which microphone is deafened is
    not its business."""
    capture = FakeCapture()

    await one_turn(assistant_with(capture=capture), capture)

    assert capture.switches == [True, False]
    assert not capture.deaf


async def test_a_turn_reports_what_was_heard_and_what_was_said() -> None:
    turns: list[Turn] = []
    capture = FakeCapture()
    events: list[LiveEvent] = [
        InputText("saat "),
        InputText("kaç"),
        voice("Üç."),
        OutputText("Üç"),
        OutputText("."),
        TurnComplete(),
    ]

    await one_turn(assistant_with(capture=capture, events=events, on_turn=turns.append), capture)

    [turn] = turns
    assert (turn.heard, turn.said) == ("saat kaç", "Üç.")
    assert turn.turn_id
    assert turn.first_sound_ms is not None and turn.first_sound_ms >= 0


async def test_every_turn_is_handed_to_whoever_is_watching() -> None:
    turns: list[Turn] = []
    capture = FakeCapture()
    assistant = assistant_with(capture=capture, on_turn=turns.append)

    await one_turn(assistant, capture)
    capture.speak()
    capture.quiet()
    await assistant.settled()

    assert len(turns) == 2
    assert turns[0].turn_id != turns[1].turn_id


async def test_a_sound_card_that_fails_loses_the_answer_and_not_the_program() -> None:
    capture = FakeCapture()

    class Deaf(FakeSpeaker):
        async def play(self, buffers: AsyncIterator[bytes], *, sample_rate: int) -> None:
            raise PlaybackError("device unavailable")

    assistant = assistant_with(capture=capture, speaker=Deaf())

    await one_turn(assistant, capture)

    assert assistant.state is State.IDLE
    assert not capture.deaf


# --------------------------------------------------------------------------
# Rule 3: a tool call runs through the gate and nowhere else
# --------------------------------------------------------------------------


async def test_a_tool_call_runs_through_the_gate_and_its_result_goes_back() -> None:
    gate = FakeGate()
    capture = FakeCapture()
    turns: list[Turn] = []
    room = Room(calls(), TurnComplete(), after_result=[voice("Üç."), TurnComplete(), Closed()])
    assistant = assistant_with(
        capture=capture, provider=Provider(room), runner=runner_with(gate), on_turn=turns.append
    )

    await one_turn(assistant, capture)

    assert [call.name for call in gate.calls] == ["clock"]
    assert [content for _, content in room.results] == ["clock: done"]
    assert [turn.tool_calls for turn in turns] == [1]


async def test_the_turn_is_one_utterance_until_the_model_is_done_with_it() -> None:
    """ADR-001 section 8: Gemini ends its turn at the tool call and speaks
    the answer as a new one. Two server turns, one turn here."""
    capture = FakeCapture()
    turns: list[Turn] = []
    seen: list[State] = []
    room = Room(
        InputText("saat kaç"),
        calls(),
        TurnComplete(),
        after_result=[voice("Üç."), OutputText("Üç."), TurnComplete(), Closed()],
    )
    assistant = assistant_with(
        capture=capture, provider=Provider(room), on_turn=turns.append, on_state=seen.append
    )

    await one_turn(assistant, capture)

    [turn] = turns
    assert (turn.heard, turn.said, turn.tool_calls) == ("saat kaç", "Üç.", 1)
    assert seen == [State.IDLE, State.USER_SPEAKING, State.IDLE, State.SPEAKING, State.IDLE]


async def test_a_call_the_model_withdrew_is_not_answered() -> None:
    gate = Held()
    capture = FakeCapture()
    room = Room(calls(), TurnComplete())
    assistant = assistant_with(capture=capture, provider=Provider(room), runner=runner_with(gate))
    await assistant.begin()

    capture.speak()
    capture.quiet()
    await until(lambda: len(gate.calls) == 1)
    room.arrives(ToolCallCancelled(("c1",)))
    await tick()
    room.hangs_up()
    await assistant.settled()

    assert room.results == []


async def test_the_tools_of_a_turn_are_counted_from_zero_with_the_next() -> None:
    """D9: the guard is the turn's. Two calls allowed per turn, two turns
    of two calls each: all four run."""
    gate = FakeGate()
    capture = FakeCapture()
    room = Room(calls("clock", "c1", n=1), calls("clock", "c2", n=2), TurnComplete())
    room2 = Room(calls("clock", "c3", n=3), calls("clock", "c4", n=4), TurnComplete())
    assistant = assistant_with(
        capture=capture,
        provider=Provider(room, room2),
        runner=runner_with(gate, tool_calls_per_turn=2),
    )
    await assistant.begin()

    for room_ in (room, room2):
        capture.speak()
        capture.quiet()
        await until(lambda room_=room_: len(room_.results) == 2)
        room_.arrives(voice("."), TurnComplete())
        room_.hangs_up()
        await assistant.settled()

    assert len(gate.calls) == 4


async def test_the_filler_is_said_when_a_tool_round_goes_quiet() -> None:
    """D19: the model is silent for as long as the tool takes; after the
    delay the pack's filler is said, and the answer follows it rather than
    talking over it."""
    gate = Held()
    capture = FakeCapture()
    tts = FakeTTS()
    speaker = FakeSpeaker()
    room = Room(calls(), TurnComplete(), after_result=[voice("Üç."), TurnComplete(), Closed()])
    assistant = assistant_with(
        capture=capture,
        provider=Provider(room),
        runner=runner_with(gate),
        tts=tts,
        speaker=speaker,
        filler_delay=0.01,
    )
    await assistant.begin()

    capture.speak()
    capture.quiet()
    await until(lambda: tts.said == [TURKISH.fillers[0]])
    gate.release.set()
    await assistant.settled()

    assert speaker.heard == "Bir saniye, bakıyorum.Üç."


async def test_a_tool_that_answers_at_once_needs_no_filler() -> None:
    capture = FakeCapture()
    tts = FakeTTS()
    room = Room(calls(), TurnComplete(), after_result=[voice("Üç."), TurnComplete(), Closed()])
    assistant = assistant_with(capture=capture, provider=Provider(room), tts=tts, filler_delay=0.02)

    await one_turn(assistant, capture)
    await asyncio.sleep(0.04)

    assert tts.said == []


async def test_the_filler_comes_from_the_code_when_the_pack_has_none() -> None:
    english = Locale(code="en", name="English", stt_language="en", voices={}, ui={})
    gate = Held()
    capture = FakeCapture()
    tts = FakeTTS()
    room = Room(calls(), TurnComplete(), after_result=[TurnComplete(), Closed()])
    assistant = assistant_with(
        capture=capture,
        provider=Provider(room),
        runner=runner_with(gate),
        tts=tts,
        locale=english,
        filler_delay=0.01,
    )
    await assistant.begin()

    capture.speak()
    capture.quiet()
    await until(lambda: tts.said == [app.FILLERS[0]])
    gate.release.set()
    await assistant.settled()


# --------------------------------------------------------------------------
# The confirmation window (lifted from the old file: the one thing done the same way)
# --------------------------------------------------------------------------


@tool(risk="confirm", confirm_prompt="{name} will be opened.")
async def open_app(name: str) -> str:
    """Opens an application, once the user has said yes."""
    ran.append(f"open_app:{name}")
    return f"{name} opened"


# What ran, in order. The body of the one risky tool above writes here, so
# "did not run" is something a test sees rather than assumes.
ran: list[str] = []


@pytest.fixture(autouse=True)
def _nothing_ran_before() -> Iterator[None]:
    ran.clear()
    yield
    ran.clear()


async def through_the_gate(call: ToolCall, *, turn_id: str, confirm: Confirm) -> str:
    """The real gate of section 3.9 over this file's one risky tool."""
    return await dispatch(call, turn_id=turn_id, registry=ToolRegistry([open_app]), confirm=confirm)


def wants_spotify(*before: LiveEvent) -> Room:
    """A model that asks to open Spotify, then says whatever it has to say."""
    return Room(
        *before,
        calls("open_app", "c1", name="Spotify"),
        TurnComplete(),
        after_result=[voice("Tamam."), OutputText("Tamam."), TurnComplete(), Closed()],
    )


def asking(room: Room | None = None, **parts: Any) -> LiveAssistant:
    """An assistant whose gate is the real one and whose model wants Spotify."""
    runner = ToolRunner(ToolRegistry([open_app]), through_the_gate)
    return assistant_with(
        provider=Provider(room if room is not None else wants_spotify()), runner=runner, **parts
    )


def says(*answers: str) -> FakeSTT:
    """A recogniser that hears each answer in turn."""
    return FakeSTT(*(Transcript(text=answer, confidence=0.9) for answer in answers))


async def test_a_tool_that_asks_runs_when_the_user_says_yes() -> None:
    capture = FakeCapture(answers=[speech()])

    await one_turn(asking(capture=capture, stt=says("Evet.")), capture)

    assert ran == ["open_app:Spotify"]
    assert capture.windows == [CONFIRM_WINDOW_SECONDS]


async def test_the_question_says_how_to_answer() -> None:
    """The user cannot know only two words are being listened for, so the
    gate's sentence - real argument values and all - is followed by the
    pack's hint on how to answer it."""
    capture = FakeCapture(answers=[speech()])
    tts = FakeTTS()

    await one_turn(asking(capture=capture, stt=says("evet"), tts=tts), capture)

    assert tts.said[0] == "Spotify will be opened. Evet ya da hayır de."


async def test_a_tool_that_asks_does_not_run_when_the_user_says_no() -> None:
    capture = FakeCapture(answers=[speech()])

    await one_turn(asking(capture=capture, stt=says("Hayır.")), capture)

    assert ran == []


async def test_the_model_is_told_the_user_declined() -> None:
    """In the tool's own channel, so the model can say so rather than pretend."""
    capture = FakeCapture(answers=[speech()])
    room = wants_spotify()

    await one_turn(asking(room, capture=capture, stt=says("hayır")), capture)

    assert [content for _, content in room.results] == [DECLINED]


async def test_silence_in_the_window_is_a_no() -> None:
    """Six seconds of nothing is the safe side (section 3.1 rule 2), and
    not worth asking again: nobody was there to hear the second question."""
    capture = FakeCapture()
    tts = FakeTTS()

    await one_turn(asking(capture=capture, stt=says(), tts=tts), capture)

    assert ran == []
    assert capture.windows == [CONFIRM_WINDOW_SECONDS]
    assert TURKISH.ui["confirm_again"] not in tts.said


async def test_an_answer_with_neither_word_is_asked_about_once_more() -> None:
    capture = FakeCapture(answers=[speech(), speech()])
    tts = FakeTTS()

    await one_turn(asking(capture=capture, stt=says("belki", "evet"), tts=tts), capture)

    assert ran == ["open_app:Spotify"]
    assert len(capture.windows) == 2
    assert tts.said[1] == TURKISH.ui["confirm_again"]


async def test_two_answers_with_neither_word_are_a_no() -> None:
    capture = FakeCapture(answers=[speech(), speech()])

    await one_turn(asking(capture=capture, stt=says("belki", "olabilir")), capture)

    assert ran == []
    assert len(capture.windows) == 2


async def test_speech_that_could_not_be_read_is_asked_about_once_more() -> None:
    """The engine says there was speech and gives no words for it: worth a
    second question, unlike silence."""
    capture = FakeCapture(answers=[speech(), speech()])
    stt = FakeSTT(Transcript(text="", no_speech_probability=0.0), Transcript(text="evet"))

    await one_turn(asking(capture=capture, stt=stt), capture)

    assert ran == ["open_app:Spotify"]
    assert len(capture.windows) == 2


async def test_a_recording_the_engine_calls_silence_is_silence() -> None:
    """The detector fired on a cough; the engine says nothing was said. Words
    it produced anyway are not an answer, and silence is not a reason to ask
    again."""
    capture = FakeCapture(answers=[speech(), speech()])
    stt = FakeSTT(Transcript(text="evet", no_speech_probability=1.0))

    await one_turn(asking(capture=capture, stt=stt), capture)

    assert ran == []
    assert len(capture.windows) == 1


async def test_a_noise_too_short_to_be_a_word_is_not_an_answer() -> None:
    capture = FakeCapture(answers=[speech(0.1)])
    stt = FakeSTT(Transcript(text="evet"))

    await one_turn(asking(capture=capture, stt=stt), capture)

    assert ran == []
    assert stt.hints == []


async def test_the_recogniser_is_told_which_language_to_expect() -> None:
    capture = FakeCapture(answers=[speech()])
    stt = says("evet")

    await one_turn(asking(capture=capture, stt=stt), capture)

    assert stt.hints == ["tr"]


async def test_a_no_beside_a_yes_is_a_no() -> None:
    capture = FakeCapture(answers=[speech()])

    await one_turn(asking(capture=capture, stt=says("evet, yok hayır")), capture)

    assert ran == []


async def test_the_turn_passes_through_confirming_and_back_to_the_door() -> None:
    """The tool round that asked is still running; the window is a detour
    from the door, not a state the turn ends in."""
    seen: list[State] = []
    capture = FakeCapture(answers=[speech()])

    await one_turn(asking(capture=capture, stt=says("evet"), on_state=seen.append), capture)

    assert seen == [
        State.IDLE,
        State.USER_SPEAKING,
        State.IDLE,
        State.CONFIRMING,
        State.IDLE,
        State.SPEAKING,
        State.IDLE,
    ]


async def test_the_session_s_input_is_paused_for_the_window_and_not_after() -> None:
    """D3: the yes never reaches the model. Nothing the room says while the
    question is asked is forwarded; only the tool's result is."""
    capture = FakeCapture(answers=[speech()])
    capture.on_window = lambda: capture.heard(b"evet")
    room = wants_spotify()

    await one_turn(asking(room, capture=capture, stt=says("evet")), capture)

    assert capture.pauses == [True, False]
    assert capture.paused_windows == [True]
    assert b"evet" not in room.audio
    assert room.texts == []


async def test_the_microphone_is_deaf_while_the_question_is_read_and_not_after() -> None:
    """A question is something a half-duplex microphone could mishear like
    any answer, and the window right after it is the one moment it has to
    hear."""
    capture = FakeCapture(answers=[speech()])
    during: list[bool] = []
    speaker = FakeSpeaker(on_play=lambda: during.append(capture.deaf))

    await one_turn(asking(capture=capture, stt=says("evet"), speaker=speaker), capture)

    assert during == [True, True], "the question, then the answer"
    assert capture.deaf_windows == [False]


async def test_the_model_is_heard_to_its_end_before_the_question_is_asked() -> None:
    """The model said "let me check" before it asked: that plays to its end,
    then the question - never the two over each other."""
    capture = FakeCapture(answers=[speech()])
    speaker = FakeSpeaker()
    room = wants_spotify(voice("Bakayım. "))

    await one_turn(asking(room, capture=capture, stt=says("evet"), speaker=speaker), capture)

    assert speaker.heard.startswith("Bakayım. Spotify will be opened.")


async def test_a_voice_while_the_question_is_read_is_a_no() -> None:
    """The user is talking over the question: the window is never opened,
    nothing runs, and what the model says about being refused is not said
    over them either."""
    capture = FakeCapture()
    speaker = FakeSpeaker()
    speaker.on_play = capture.speak

    await one_turn(asking(capture=capture, stt=says("evet"), speaker=speaker), capture)

    assert ran == []
    assert capture.windows == []
    assert speaker.stopped >= 1


async def test_switching_off_in_the_window_is_a_no() -> None:
    capture = FakeCapture(answers=[speech()])
    capture.on_window = capture.switch_off
    assistant = asking(capture=capture, stt=says("evet"))

    await one_turn(assistant, capture)

    assert ran == []
    assert assistant.state is State.OFF


async def test_the_words_that_count_come_from_the_pack() -> None:
    """A pack that names no words gets the English ones beside the code."""
    english = Locale(code="en", name="English", stt_language="en", voices={}, ui={})
    capture = FakeCapture(answers=[speech()])

    await one_turn(asking(capture=capture, stt=says("Yes."), locale=english), capture)

    assert ran == ["open_app:Spotify"]


async def test_the_pack_s_words_replace_the_english_ones_rather_than_adding_to_them() -> None:
    """The Turkish pack says nothing about "yes", so "yes" is not a yes."""
    capture = FakeCapture(answers=[speech()])

    await one_turn(asking(capture=capture, stt=says("yes")), capture)

    assert ran == []


async def test_the_hint_falls_back_to_english_together_with_the_words() -> None:
    """Whichever words are listened for, the hint names them: the two fall
    back as one, or the user would be told to say words nobody hears."""
    english = Locale(code="en", name="English", stt_language="en", voices={}, ui={})
    capture = FakeCapture(answers=[speech()])
    tts = FakeTTS()

    await one_turn(asking(capture=capture, stt=says("no"), tts=tts, locale=english), capture)

    assert tts.said[0] == "Spotify will be opened. " + app.TEXT["confirm_hint"]


def test_the_window_is_the_six_seconds_section_3_1_allows() -> None:
    assert CONFIRM_WINDOW_SECONDS == 6


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("evet", True),
        ("Evet.", True),
        ("EVET", True),
        ("tamam, olur", True),
        ("hayır", False),
        ("Hayır!", False),
        ("HAYIR", False),
        ("evet ama hayır", False),
        ("belki", None),
        ("", None),
        ("evetlemedim", None),
        ("hayırlısı olsun", None),
    ],
)
def test_read_answer_hears_whole_words_in_any_case(text: str, expected: bool | None) -> None:
    """Folded the way search is: "HAYIR" is "hayır" once the dotless i is
    folded, which `casefold` alone gets wrong. Whole words, so that a word
    that merely begins with "evet" is not a yes."""
    assert read_answer(text, yes=("evet", "tamam", "olur"), no=("hayır", "iptal")) is expected


def test_read_answer_hears_a_phrase_of_more_than_one_word() -> None:
    assert read_answer("boş ver artık", yes=("evet",), no=("boş ver",)) is False
    assert read_answer("boş verme, evet", yes=("evet",), no=("boş ver",)) is True


def test_a_blank_entry_in_the_pack_matches_nothing() -> None:
    """An empty string is inside every string; it must not be a yes."""
    assert read_answer("belki", yes=("", " "), no=("",)) is None


# --------------------------------------------------------------------------
# Rule 4: switching off is an interruption and a hang-up
# --------------------------------------------------------------------------


async def test_switching_off_while_the_model_talks_stops_it_and_hangs_up() -> None:
    capture = FakeCapture()
    speaker = FakeSpeaker()
    seen: list[State] = []
    room = Room(voice("Uzun bir cevap"))
    assistant = assistant_with(
        capture=capture, provider=Provider(room), speaker=speaker, on_state=seen.append
    )
    await assistant.begin()
    capture.speak()
    capture.quiet()
    await until(lambda: assistant.state is State.SPEAKING)

    capture.switch_off()
    await assistant.settled()

    assert speaker.stopped >= 1
    assert not assistant.session_open
    assert seen[-1] is State.OFF
    assert capture.closed >= 1


async def test_switching_off_while_idle_with_a_session_open_hangs_up() -> None:
    capture = FakeCapture()
    assistant = assistant_with(capture=capture, provider=Provider(Room()))
    await assistant.begin()
    capture.speak()
    capture.quiet()
    await until(lambda: assistant.session_open)

    capture.switch_off()
    await assistant.settled()

    assert not assistant.session_open
    assert assistant.state is State.OFF


async def test_switching_off_while_idle_at_the_door_changes_only_the_state() -> None:
    capture = FakeCapture()
    provider = Provider()
    assistant = assistant_with(capture=capture, provider=provider)
    await assistant.begin()

    capture.switch_off()
    capture.speak()
    await assistant.settled()

    assert assistant.state is State.OFF
    assert provider.opened == []


async def test_switching_on_again_opens_the_door() -> None:
    capture = FakeCapture()
    provider = Provider()
    seen: list[State] = []
    assistant = assistant_with(capture=capture, provider=provider, on_state=seen.append)
    await assistant.begin()
    capture.switch_off()

    capture.switch_on()
    capture.speak()
    capture.quiet()
    await assistant.settled()

    assert seen[:3] == [State.IDLE, State.OFF, State.IDLE]
    assert len(provider.opened) == 1


async def test_the_screen_hears_of_the_mode_after_the_machine_acted_on_it() -> None:
    """The line must not say "off" over an assistant still talking."""
    capture = FakeCapture()
    states_at_mode: list[State] = []
    room = Room(voice("Uzun bir cevap"))
    assistant: LiveAssistant | None = None

    def mode(listening: bool) -> None:
        assert assistant is not None
        states_at_mode.append(assistant.state)

    assistant = assistant_with(capture=capture, provider=Provider(room), on_mode=mode)
    await assistant.begin()
    capture.speak()
    capture.quiet()
    await until(lambda: assistant is not None and assistant.state is State.SPEAKING)

    capture.switch_off()
    await assistant.settled()

    assert states_at_mode == [State.IDLE, State.OFF]


# --------------------------------------------------------------------------
# Rule 6: what a turn cost is written down
# --------------------------------------------------------------------------

# A round price for the fake model: a token in is a millionth of a dollar and
# a token out ten of them, so a turn's cost can be read off its counts.
PRICING = Pricing.from_toml('[fake."fake-1"]\ninput_per_mtok = 1.0\noutput_per_mtok = 10.0\n')


@pytest.fixture
def ledger() -> Iterator[sqlite3.Connection]:
    connection = open_database(":memory:")
    try:
        yield connection
    finally:
        connection.close()


def tracker_over(ledger: sqlite3.Connection, **limits: Any) -> UsageTracker:
    return UsageTracker(
        UsageRepo(ledger), PRICING, provider="fake", model="fake-1", limits=Limits(**limits)
    )


async def test_a_turn_that_reached_the_model_is_priced_and_written_down(
    ledger: sqlite3.Connection,
) -> None:
    """The tokens when the provider reports them, the audio minutes both
    ways from the bytes (D8), and the price from the tracker."""
    turns: list[Turn] = []
    capture = FakeCapture()
    room = Room()
    assistant = assistant_with(
        capture=capture, provider=Provider(room), tracker=tracker_over(ledger), on_turn=turns.append
    )
    await assistant.begin()

    capture.speak()
    capture.quiet()
    await until(lambda: room.audio_in_ms == 20)
    room.arrives(UsageReport(Usage(100, 5)), voice("Üç."), TurnComplete())
    room.hangs_up()
    await assistant.settled()

    [turn] = turns
    assert turn.usage == Usage(100, 5)
    assert turn.cost_usd == pytest.approx(0.00015)
    assert turn.audio_in_ms == 20
    assert turn.audio_out_ms == len("Üç.".encode()) * 1000 // (RATE * 2)
    assert UsageRepo(ledger).sum_since(0) == pytest.approx(0.00015)


async def test_two_server_turns_with_a_tool_between_them_are_billed_as_one(
    ledger: sqlite3.Connection,
) -> None:
    turns: list[Turn] = []
    capture = FakeCapture()
    room = Room(
        UsageReport(Usage(100, 5)),
        calls(),
        TurnComplete(),
        after_result=[UsageReport(Usage(150, 20)), voice("Üç."), TurnComplete(), Closed()],
    )
    assistant = assistant_with(
        capture=capture, provider=Provider(room), tracker=tracker_over(ledger), on_turn=turns.append
    )

    await one_turn(assistant, capture)

    [turn] = turns
    assert turn.usage == Usage(250, 25)


async def test_without_a_tracker_nothing_costs_anything() -> None:
    turns: list[Turn] = []
    capture = FakeCapture()

    await one_turn(assistant_with(capture=capture, on_turn=turns.append), capture)

    assert turns[0].cost_usd is None


async def test_past_the_day_s_limit_a_session_opens_with_the_warning(
    ledger: sqlite3.Connection,
) -> None:
    """Every new session, until the day turns: the one sentence the user
    has to act on, before the session that costs more."""
    tracker = tracker_over(ledger, daily_usd=0.001)
    tracker.record("earlier", Usage(2_000, 0))
    capture = FakeCapture()
    tts = FakeTTS()
    provider = Provider()

    assistant = assistant_with(capture=capture, provider=provider, tracker=tracker, tts=tts)
    await one_turn(assistant, capture)

    assert tts.said == [TURKISH.ui["daily_over"]]
    assert len(provider.opened) == 1


async def test_past_the_month_s_limit_the_warning_is_the_month_s(
    ledger: sqlite3.Connection,
) -> None:
    tracker = tracker_over(ledger, daily_usd=0.001, monthly_usd=0.001)
    tracker.record("earlier", Usage(2_000, 0))
    capture = FakeCapture()
    tts = FakeTTS()

    await one_turn(assistant_with(capture=capture, tracker=tracker, tts=tts), capture)

    assert tts.said == [TURKISH.ui["monthly_over"]]


async def test_with_hard_stop_on_no_session_is_opened_past_the_limit(
    ledger: sqlite3.Connection,
) -> None:
    tracker = tracker_over(ledger, daily_usd=0.001, hard_stop=True)
    tracker.record("earlier", Usage(2_000, 0))
    capture = FakeCapture()
    tts = FakeTTS()
    provider = Provider()
    turns: list[Turn] = []
    assistant = assistant_with(
        capture=capture, provider=provider, tracker=tracker, tts=tts, on_turn=turns.append
    )

    await one_turn(assistant, capture)

    assert tts.said == [TURKISH.ui["spend_stopped"]]
    assert provider.opened == []
    assert assistant.state is State.IDLE
    assert capture.closed == 1
    assert [turn.failure for turn in turns] == ["spend_stopped"]


async def test_under_the_limit_nothing_is_said_about_money(ledger: sqlite3.Connection) -> None:
    capture = FakeCapture()
    tts = FakeTTS()

    await one_turn(assistant_with(capture=capture, tracker=tracker_over(ledger), tts=tts), capture)

    assert tts.said == []


# --------------------------------------------------------------------------
# Rule 7: a session that drops, and the failures that are said out loud
# --------------------------------------------------------------------------


async def test_a_session_that_dies_is_reopened_with_its_handle() -> None:
    """Dropped with an error: reopened at once, with the handle, and the
    stream the doorman opened goes on into the new session."""
    capture = FakeCapture()
    seen: list[State] = []
    tts = FakeTTS()
    first = Room(Resumable("h-1"))
    second = Room()
    provider = Provider(first, second)
    assistant = assistant_with(capture=capture, provider=provider, tts=tts, on_state=seen.append)
    await assistant.begin()
    capture.speak()
    capture.quiet()
    await until(lambda: assistant.session_open)

    first.hangs_up(ProviderError("reset", kind="unreachable"))
    await until(lambda: len(provider.opened) == 2)
    await until(lambda: assistant.state is State.IDLE)
    capture.heard(b"more")
    await until(lambda: second.audio == [b"more"])

    assert provider.opened[1].resume_handle == "h-1"
    assert State.RECONNECTING in seen
    assert tts.said == []
    second.hangs_up()
    await assistant.settled()


async def test_a_reopen_that_fails_is_said_out_loud() -> None:
    capture = FakeCapture()
    tts = FakeTTS()
    first = Room()
    provider = Provider(first, refuse=[None, ProviderError("down", kind="unreachable")])
    assistant = assistant_with(capture=capture, provider=provider, tts=tts)
    await assistant.begin()
    capture.speak()
    capture.quiet()
    await until(lambda: assistant.session_open)

    first.hangs_up(ProviderError("reset", kind="unreachable"))
    await assistant.settled()

    assert tts.said == [TURKISH.ui["unreachable"]]
    assert assistant.state is State.IDLE


async def test_the_server_about_to_hang_up_is_left_before_it_does() -> None:
    """`GoingAway`: the session's hard cap is near. Reopened with the
    handle, and nothing is said about it."""
    capture = FakeCapture()
    tts = FakeTTS()
    first = Room(Resumable("h-1"), GoingAway("10s"))
    second = Room()
    provider = Provider(first, second)
    assistant = assistant_with(capture=capture, provider=provider, tts=tts)
    await assistant.begin()

    capture.speak()
    capture.quiet()
    await until(lambda: len(provider.opened) == 2)

    assert provider.opened[1].resume_handle == "h-1"
    assert tts.said == []
    second.hangs_up()
    await assistant.settled()


async def test_a_polite_hang_up_ends_the_session_and_the_next_sentence_opens_one() -> None:
    capture = FakeCapture()
    provider = Provider()
    assistant = assistant_with(capture=capture, provider=provider)

    await one_turn(assistant, capture)

    assert not assistant.session_open
    assert assistant.state is State.IDLE
    assert len(provider.opened) == 1


async def test_the_sentences_it_says_are_the_ones_in_the_locale_pack() -> None:
    """Nothing said out loud is written in this file: the pack answers
    first, `TEXT` last."""
    english = Locale(code="en", name="English", stt_language="en", voices={}, ui={})
    capture = FakeCapture()
    tts = FakeTTS()
    provider = Provider(refuse=[ProviderError("down", kind="unreachable")])

    await one_turn(
        assistant_with(capture=capture, provider=provider, tts=tts, locale=english), capture
    )

    assert tts.said == [app.TEXT["unreachable"]]


async def test_a_bug_in_our_own_code_is_not_reported_as_a_network_problem() -> None:
    """Anything but the three named failures is a bug, and comes out as one."""
    capture = FakeCapture()

    class Broken(Provider):
        @asynccontextmanager
        async def connect(self, config: SessionConfig) -> AsyncIterator[FakeLiveSession]:
            raise KeyError("a bug")
            yield  # pragma: no cover

    assistant = assistant_with(capture=capture, provider=Broken())
    await assistant.begin()
    capture.speak()

    with pytest.raises(KeyError):
        await assistant.settled()


async def test_the_bug_comes_out_of_run_as_well() -> None:
    capture = FakeCapture()

    class Broken(Provider):
        @asynccontextmanager
        async def connect(self, config: SessionConfig) -> AsyncIterator[FakeLiveSession]:
            raise KeyError("a bug")
            yield  # pragma: no cover

    assistant = assistant_with(capture=capture, provider=Broken())
    task = asyncio.create_task(assistant.run())
    await until(lambda: capture.started)
    capture.speak()

    with pytest.raises(KeyError):
        await task


# --------------------------------------------------------------------------
# The voice the questions are asked in
# --------------------------------------------------------------------------


async def test_the_question_is_read_in_the_voice_the_locale_asks_for() -> None:
    capture = FakeCapture(answers=[speech()])
    tts = FakeTTS()

    await one_turn(asking(capture=capture, stt=says("evet"), tts=tts), capture)

    assert tts.voices_used == [TOLGA.id]


async def test_a_locale_with_no_voice_of_its_own_still_gets_a_voice() -> None:
    capture = FakeCapture(answers=[speech()])
    tts = FakeTTS(installed=[ZIRA])

    await one_turn(asking(capture=capture, stt=says("evet"), tts=tts), capture)

    assert tts.voices_used == [ZIRA.id]


async def test_a_machine_with_no_voice_at_all_says_so_before_it_listens() -> None:
    capture = FakeCapture()

    with pytest.raises(NoVoiceError):
        await assistant_with(capture=capture, tts=FakeTTS(installed=[])).begin()

    assert not capture.started


# --------------------------------------------------------------------------
# Asleep behind the wake word (D21)
# --------------------------------------------------------------------------


async def test_a_sleeping_capture_starts_the_machine_asleep() -> None:
    capture = FakeCapture(asleep=True)
    seen: list[State] = []
    assistant = assistant_with(capture=capture, on_state=seen.append)

    await assistant.begin()

    assert assistant.state is State.SLEEPING
    assert seen[-1] is State.SLEEPING
    assert capture.on_wake is not None


async def test_the_wake_word_chimes_and_opens_a_session_at_once() -> None:
    """ "Hey Friday" is the moment the assistant really listens: the chime
    says so, and the session is opened now rather than at the first
    sentence, so that "hey Friday, saat kaç" is not answered a socket late."""
    capture = FakeCapture(asleep=True)
    speaker = FakeSpeaker()
    assistant = assistant_with(capture=capture, speaker=speaker, provider=Provider(Room()))
    await assistant.begin()

    capture.call_wake()
    await until(lambda: assistant.session_open)

    assert speaker.rates == [CHIME_RATE]
    assert b"".join(speaker.played) == chime()
    assert capture.switches == [True, False]  # deaf for the chime, then listening
    assert assistant.state is State.IDLE
    assert not capture.asleep


async def test_the_greeting_can_be_a_sentence_in_the_local_voice() -> None:
    capture = FakeCapture(asleep=True)
    speaker = FakeSpeaker()
    assistant = assistant_with(
        capture=capture, speaker=speaker, provider=Provider(Room()), greeting="sentence"
    )
    await assistant.begin()

    capture.call_wake()
    await until(lambda: assistant.session_open)

    assert speaker.heard == TURKISH.ui["wake_greeting"]
    assert assistant.state is State.IDLE


async def test_the_greeting_can_be_nothing() -> None:
    capture = FakeCapture(asleep=True)
    speaker = FakeSpeaker()
    assistant = assistant_with(
        capture=capture, speaker=speaker, provider=Provider(Room()), greeting="none"
    )
    await assistant.begin()

    capture.call_wake()
    await until(lambda: assistant.session_open)

    assert speaker.played == []


async def test_silence_after_waking_puts_it_back_to_sleep() -> None:
    capture = FakeCapture(asleep=True)
    seen: list[State] = []
    assistant = assistant_with(
        capture=capture, provider=Provider(Room()), idle_close_seconds=0.03, on_state=seen.append
    )
    await assistant.begin()

    capture.call_wake()
    await until(lambda: assistant.session_open)
    await until(lambda: not assistant.session_open)
    await assistant.settled()

    assert capture.sleeps == 1
    assert capture.asleep
    assert assistant.state is State.SLEEPING
    assert seen[-1] is State.SLEEPING


async def test_switching_on_from_off_wakes_it() -> None:
    capture = FakeCapture(asleep=True, listening=False)
    assistant = assistant_with(capture=capture)
    await assistant.begin()
    assert assistant.state is State.OFF

    capture.toggle()

    assert assistant.state is State.IDLE
    assert not capture.asleep


async def test_switching_off_while_asleep_is_off_and_on_again_is_awake() -> None:
    capture = FakeCapture(asleep=True)
    assistant = assistant_with(capture=capture)
    await assistant.begin()

    capture.toggle()
    assert assistant.state is State.OFF
    capture.toggle()
    assert assistant.state is State.IDLE


async def test_the_wake_word_is_ignored_unless_asleep() -> None:
    """A detection that arrives after the key already woke it changes nothing."""
    capture = FakeCapture(asleep=True)
    speaker = FakeSpeaker()
    assistant = assistant_with(capture=capture, speaker=speaker)
    await assistant.begin()
    capture.toggle()
    capture.toggle()
    assert assistant.state is State.IDLE

    capture.call_wake()
    await assistant.settled()

    assert speaker.played == []


async def test_without_a_wake_word_the_idle_close_does_not_sleep() -> None:
    """Today's product, unchanged: the session closes and the door stays open."""
    capture = FakeCapture()
    assistant = assistant_with(capture=capture, provider=Provider(Room()), idle_close_seconds=0.03)
    await assistant.begin()

    capture.speak()
    capture.quiet()
    await until(lambda: assistant.session_open)
    await until(lambda: not assistant.session_open)
    await assistant.settled()

    assert capture.sleeps == 0
    assert assistant.state is State.IDLE
