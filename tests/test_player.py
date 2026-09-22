"""Playing the answer out loud (design.md item 1.10).

The TTS layer yields PCM and stops there, deliberately: what plays it is the
same for Windows' own voice and for a cloud engine. This is that piece.

Two rules are worth more than the rest of the file. **PortAudio's write blocks
until the buffer drains** - the length of the sentence - so it happens in a
worker thread; rule 4 of section 3.1 gives anything awaited in `app.py` fifty
milliseconds. And **speech has to stop when the user presses the key**, which
is why the audio goes out in blocks rather than in one write: a sentence handed
over whole cannot be interrupted at all.

No device is opened here. The stream is injected, which is how the interruption
tests can be about what was written rather than about what was heard.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable

import numpy as np
import pytest
from loguru import logger

from allie.audio.player import (
    BLOCK_FRAMES,
    BYTES_PER_FRAME,
    LivePlayback,
    PlaybackError,
    Speaker,
    SystemSpeaker,
)
from allie.stt.base import LEVEL_FLOOR_DBFS

RATE = 16_000
BLOCK = BLOCK_FRAMES * BYTES_PER_FRAME


def silence(blocks: float = 1) -> bytes:
    """A buffer of `blocks` worth of sixteen bit silence."""
    return bytes(int(BLOCK * blocks))


async def spoken(*buffers: bytes) -> AsyncIterator[bytes]:
    for buffer in buffers:
        yield buffer


class FakeStream:
    """An open output device that remembers what was done to it."""

    def __init__(self, on_write: Callable[[], None] | None = None) -> None:
        self.written: list[bytes] = []
        self.finished = False
        self.aborted = False
        self.closed = False
        self.on_write = on_write

    def write(self, data: bytes) -> None:
        if self.on_write is not None:
            self.on_write()
        self.written.append(bytes(data))

    def stop(self) -> None:
        self.finished = True

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        self.closed = True

    @property
    def heard(self) -> bytes:
        return b"".join(self.written)


class FakeDevice:
    """Stands in for the sound card: hands out one stream and records the rate."""

    def __init__(self, stream: FakeStream | None = None) -> None:
        self.stream = stream if stream is not None else FakeStream()
        self.rates: list[int] = []

    def __call__(self, sample_rate: int) -> FakeStream:
        self.rates.append(sample_rate)
        return self.stream


def speaker_on(device: FakeDevice) -> SystemSpeaker:
    return SystemSpeaker(open_stream=device)


# --------------------------------------------------------------------------
# Playing
# --------------------------------------------------------------------------


def test_the_speaker_is_what_the_state_machine_expects() -> None:
    player: Speaker = SystemSpeaker()

    assert isinstance(player, Speaker)


async def test_every_byte_of_the_answer_reaches_the_device() -> None:
    device = FakeDevice()

    await speaker_on(device).play(spoken(b"first", b"second"), sample_rate=RATE)

    assert device.stream.heard == b"firstsecond"


async def test_the_device_is_opened_at_the_rate_the_engine_declares() -> None:
    """Windows speaks at 16 kHz and Azure at 24. Playing one at the other's
    rate is the chipmunk bug, and nothing resamples on the way here."""
    device = FakeDevice()

    await speaker_on(device).play(spoken(silence()), sample_rate=24_000)

    assert device.rates == [24_000]


async def test_an_answer_with_nothing_in_it_opens_no_device() -> None:
    """Opening a stream costs a tenth of a second and makes an audible click."""
    device = FakeDevice()

    await speaker_on(device).play(spoken(b"", b""), sample_rate=RATE)

    assert device.rates == []


async def test_the_audio_goes_out_in_blocks_that_can_be_interrupted() -> None:
    """One write of a whole sentence cannot be cut off part way through."""
    device = FakeDevice()

    await speaker_on(device).play(spoken(silence(blocks=3)), sample_rate=RATE)

    assert [len(block) for block in device.stream.written] == [BLOCK, BLOCK, BLOCK]


def test_a_block_is_short_enough_that_being_told_to_stop_is_obeyed() -> None:
    """This is exactly how long the assistant keeps talking after the user has
    pressed the key. A tenth of a second nobody notices; a sentence they do.

    Measured against the lowest rate the project speaks at, which is the one
    where a block lasts longest."""
    assert BLOCK_FRAMES / 16_000 <= 0.15


async def test_a_finished_answer_is_played_to_its_end() -> None:
    """Closing an active stream discards whatever is still queued; stopping it
    waits for the last block to be heard, which is the last word."""
    device = FakeDevice()

    await speaker_on(device).play(spoken(silence()), sample_rate=RATE)

    assert (device.stream.finished, device.stream.aborted) == (True, False)
    assert device.stream.closed


async def test_the_event_loop_keeps_running_while_audio_plays() -> None:
    """Rule 4 of section 3.1. A write blocks for as long as the audio lasts, so
    on the loop it would freeze the hotkey, the scheduler and everything else
    for the length of the answer."""
    device = FakeDevice(FakeStream(on_write=lambda: time.sleep(0.03)))
    beats = 0

    async def heartbeat() -> None:
        nonlocal beats
        while True:
            await asyncio.sleep(0.001)
            beats += 1

    pulse = asyncio.create_task(heartbeat())
    await speaker_on(device).play(spoken(silence(blocks=2)), sample_rate=RATE)
    pulse.cancel()

    assert beats > 0


# --------------------------------------------------------------------------
# Being interrupted
# --------------------------------------------------------------------------


async def test_speech_stops_within_one_block_of_being_told_to() -> None:
    """The user pressed the key: they are talking now, and the assistant
    talking over them is both rude and something the microphone hears."""
    device = FakeDevice()
    player = speaker_on(device)
    device.stream.on_write = player.stop  # cut it off during the first block

    await player.play(spoken(silence(blocks=4)), sample_rate=RATE)

    assert len(device.stream.written) == 1


async def test_what_is_already_queued_is_dropped_when_speech_is_cut() -> None:
    """Sound cards hold a second of audio. Waiting for it to drain would mean
    the assistant keeps talking after being told to stop."""
    device = FakeDevice()
    player = speaker_on(device)
    device.stream.on_write = player.stop

    await player.play(spoken(silence(blocks=2)), sample_rate=RATE)

    assert (device.stream.aborted, device.stream.finished) == (True, False)
    assert device.stream.closed


async def test_nothing_more_is_asked_of_the_engine_once_speech_is_cut() -> None:
    """The sentences arrive from the model as it writes them. Being cut off
    means the rest of the answer is never synthesised at all."""
    device = FakeDevice()
    player = speaker_on(device)
    synthesised = 0

    async def sentences() -> AsyncIterator[bytes]:
        nonlocal synthesised
        for _ in range(5):
            synthesised += 1
            yield silence()

    device.stream.on_write = player.stop
    await player.play(sentences(), sample_rate=RATE)

    assert synthesised == 1


async def test_a_stop_from_a_previous_answer_does_not_silence_the_next_one() -> None:
    """Being interrupted is the normal end of an answer, not a state to leave
    the speaker in."""
    device = FakeDevice()
    player = speaker_on(device)
    player.stop()

    await player.play(spoken(b"answer"), sample_rate=RATE)

    assert device.stream.heard == b"answer"


# --------------------------------------------------------------------------
# A device that fails
# --------------------------------------------------------------------------


def _dies() -> None:
    raise RuntimeError("stream closed")


class DiesWhileDraining(FakeStream):
    """A device that goes away while the end of the answer is still queued."""

    def stop(self) -> None:
        _dies()


async def test_a_device_that_cannot_be_opened_is_a_playback_error() -> None:
    """A Bluetooth headset switched off between two answers. The answer is
    lost; the program is not."""

    def refuses(sample_rate: int) -> FakeStream:
        raise RuntimeError("Error opening RawOutputStream: Device unavailable")

    with pytest.raises(PlaybackError, match="unavailable"):
        await SystemSpeaker(open_stream=refuses).play(spoken(silence()), sample_rate=RATE)


async def test_a_device_that_dies_while_writing_is_a_playback_error() -> None:
    """The same headset, switched off in the middle of a sentence."""
    device = FakeDevice()
    device.stream.on_write = _dies

    with pytest.raises(PlaybackError, match="stream closed"):
        await speaker_on(device).play(spoken(silence()), sample_rate=RATE)


async def test_a_device_that_died_while_writing_is_still_closed() -> None:
    """Whatever PortAudio thinks of the stream now, its handle is given back."""
    device = FakeDevice()
    device.stream.on_write = _dies

    with pytest.raises(PlaybackError):
        await speaker_on(device).play(spoken(silence()), sample_rate=RATE)

    assert device.stream.closed


async def test_a_device_that_dies_while_draining_is_a_playback_error_and_closed() -> None:
    device = FakeDevice(DiesWhileDraining())

    with pytest.raises(PlaybackError, match="stream closed"):
        await speaker_on(device).play(spoken(silence()), sample_rate=RATE)

    assert device.stream.closed


async def test_a_failure_in_the_engine_is_not_dressed_up_as_the_device() -> None:
    """A voice that raises while synthesising is a bug in an adapter or a
    cloud that went away - each worth its own traceback, neither the sound
    card's fault. The device is still closed on the way out."""
    device = FakeDevice()

    async def breaks() -> AsyncIterator[bytes]:
        yield silence()
        raise ValueError("the engine broke")

    with pytest.raises(ValueError, match="engine"):
        await speaker_on(device).play(breaks(), sample_rate=RATE)

    assert device.stream.closed


# --------------------------------------------------------------------------
# The level hook (plan.md D20: the orb swells with what is heard)
# --------------------------------------------------------------------------


def loud(blocks: float = 1) -> bytes:
    """A buffer of full-scale sixteen bit samples, `blocks` blocks long."""
    frames = int(BLOCK_FRAMES * blocks)
    return np.full(frames, 32767, dtype="<i2").tobytes()


async def test_every_block_written_is_reported_to_the_level_hook() -> None:
    """One number per block, measured where the block goes to the device -
    that is when it is heard, near enough; the chunks arrive faster."""
    levels: list[float] = []
    device = FakeDevice()

    await SystemSpeaker(open_stream=device, on_level=levels.append).play(
        spoken(loud(2)), sample_rate=RATE
    )

    assert len(levels) == 3
    assert levels[0] == pytest.approx(0.0, abs=0.01)
    assert levels[1] == pytest.approx(0.0, abs=0.01)
    assert levels[2] == LEVEL_FLOOR_DBFS


async def test_the_hook_hears_silence_at_the_floor_and_the_floor_again_when_done() -> None:
    levels: list[float] = []
    device = FakeDevice()

    await SystemSpeaker(open_stream=device, on_level=levels.append).play(
        spoken(silence()), sample_rate=RATE
    )

    assert levels == [LEVEL_FLOOR_DBFS, LEVEL_FLOOR_DBFS]


async def test_the_hook_is_called_on_the_worker_thread_not_the_loop() -> None:
    """Rule 4 of section 3.1: the measurement is beside the blocking write,
    where blocking is allowed; whoever is hooked must not touch the loop."""
    threads: set[int] = set()
    device = FakeDevice()

    await SystemSpeaker(
        open_stream=device, on_level=lambda _: threads.add(threading.get_ident())
    ).play(spoken(loud()), sample_rate=RATE)

    assert threads and threading.get_ident() not in threads


async def test_without_a_hook_nothing_changes() -> None:
    device = FakeDevice()

    await speaker_on(device).play(spoken(loud()), sample_rate=RATE)

    assert device.stream.heard == loud()


# --------------------------------------------------------------------------
# Playing what the model says as it arrives (plan.md section 4.4 rule 2, L1.2)
# --------------------------------------------------------------------------


class Recorded(FakeStream):
    """A stream that also writes every call into the device's journal."""

    def __init__(self, journal: Journal, on_write: Callable[[], None] | None = None) -> None:
        super().__init__(on_write)
        self._journal = journal

    def write(self, data: bytes) -> None:
        super().write(data)
        self._journal.events.append(f"write {len(data)}")

    def stop(self) -> None:
        super().stop()
        self._journal.events.append("stop")

    def abort(self) -> None:
        super().abort()
        self._journal.events.append("abort")

    def close(self) -> None:
        super().close()
        self._journal.events.append("close")


class Journal:
    """A sound card that hands out a fresh stream per answer and writes down
    everything done to any of them, in order - the tests about two answers
    are about that order."""

    def __init__(self, *, on_write: Callable[[], None] | None = None, refuse: int = 0) -> None:
        self.events: list[str] = []
        self.streams: list[FakeStream] = []
        self.on_write = on_write
        # How many opens are refused before one succeeds.
        self.refuse = refuse

    def __call__(self, sample_rate: int) -> FakeStream:
        if self.refuse:
            self.refuse -= 1
            raise RuntimeError("Error opening RawOutputStream: Device unavailable")
        stream = Recorded(self, on_write=self.on_write)
        self.streams.append(stream)
        self.events.append(f"open {sample_rate}")
        return stream

    @property
    def rates(self) -> list[int]:
        return [int(event.split()[1]) for event in self.events if event.startswith("open")]


class Held:
    """A write that waits to be let go, so that a test can act mid-block the
    way the state machine does: from the event loop, while the worker
    thread is inside the device."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self) -> None:
        self.started.set()
        self.release.wait(1)

    async def begun(self) -> None:
        await asyncio.to_thread(self.started.wait, 1)


def playback_on(device: Journal) -> LivePlayback:
    return LivePlayback(SystemSpeaker(open_stream=device))


async def test_the_playback_makes_no_sound_until_something_is_pushed() -> None:
    device = Journal()
    playback = playback_on(device)

    assert playback.playing is False
    await playback.drained()

    assert device.events == []


async def test_what_is_pushed_is_played_in_the_order_it_came() -> None:
    """The model's audio arrives faster than it plays; the queue is what
    keeps the sentence in order while the sound card catches up."""
    device = Journal()
    playback = playback_on(device)

    playback.push(b"first", sample_rate=RATE)
    playback.push(b"second", sample_rate=RATE)
    await playback.drained()

    assert device.streams[0].heard == b"firstsecond"


async def test_the_device_is_opened_at_the_rate_of_the_audio() -> None:
    """Gemini speaks at 24 kHz and says so on every blob (`AudioChunk`);
    the rate is the audio's, not a constant of this file."""
    device = Journal()
    playback = playback_on(device)

    playback.push(silence(), sample_rate=24_000)
    await playback.drained()

    assert device.rates == [24_000]


async def test_flushing_lets_the_answer_finish_and_gives_the_device_back() -> None:
    """`TurnComplete`: nothing more is coming for now. What is queued is
    still heard to its last word - stopped, not aborted - and then the
    device is free for the local voice of a confirmation or a reminder."""
    device = Journal()
    playback = playback_on(device)

    playback.push(silence(), sample_rate=RATE)
    playback.flush()
    await playback.drained()

    [stream] = device.streams
    assert (stream.finished, stream.aborted, stream.closed) == (True, False, True)


async def test_draining_waits_for_the_last_block_to_be_heard() -> None:
    """The confirm tool asks its question after the answer, not over it."""
    hold = Held()
    device = Journal(on_write=hold)
    playback = playback_on(device)
    playback.push(silence(blocks=3), sample_rate=RATE)
    playback.flush()

    waiting = asyncio.ensure_future(playback.drained())
    await hold.begun()
    assert waiting.done() is False, "drained before the first block was written"

    hold.release.set()
    await waiting

    assert [len(block) for block in device.streams[0].written] == [BLOCK, BLOCK, BLOCK]
    assert device.streams[0].finished


async def test_draining_ends_the_answer_under_way_if_nobody_flushed() -> None:
    """Waiting for an answer that was never told it was over would wait for
    ever; `drained` means "play what you have, then tell me"."""
    device = Journal()
    playback = playback_on(device)

    playback.push(silence(), sample_rate=RATE)
    await asyncio.wait_for(playback.drained(), 1)

    assert device.streams[0].finished


async def test_audio_pushed_after_a_flush_is_a_new_answer() -> None:
    """A model turn, a tool call, another turn (ADR-001 section 8): each
    server turn is one stretch of sound with a beginning and an end."""
    device = Journal()
    playback = playback_on(device)

    playback.push(b"one", sample_rate=RATE)
    playback.flush()
    playback.push(b"two", sample_rate=RATE)
    await playback.drained()

    assert [stream.heard for stream in device.streams] == [b"one", b"two"]


async def test_two_answers_never_hold_the_device_at_once() -> None:
    """The second answer arrives while the first is still draining out of
    the sound card. It waits: two streams on one device is two answers
    talking over each other."""
    device = Journal()
    playback = playback_on(device)

    playback.push(b"one", sample_rate=RATE)
    playback.flush()
    playback.push(b"two", sample_rate=RATE)
    await playback.drained()

    assert device.events == [
        f"open {RATE}",
        "write 3",
        "stop",
        "close",
        f"open {RATE}",
        "write 3",
        "stop",
        "close",
    ]


async def test_a_new_rate_in_the_middle_of_an_answer_starts_a_new_answer() -> None:
    """A device opened at one rate cannot play the other; the chipmunk bug
    is playing it anyway. The answer under way is finished at its rate and
    the rest is played at its own."""
    device = Journal()
    playback = playback_on(device)

    playback.push(b"one", sample_rate=16_000)
    playback.push(b"two", sample_rate=24_000)
    await playback.drained()

    assert device.rates == [16_000, 24_000]
    assert [stream.heard for stream in device.streams] == [b"one", b"two"]


async def test_playing_says_whether_there_is_sound_still_to_come() -> None:
    """`SPEAKING` (section 4.2) ends when the queue is drained *and* the
    turn is complete; this is the first half of that."""
    device = Journal()
    playback = playback_on(device)

    playback.push(silence(), sample_rate=RATE)
    assert playback.playing is True

    await playback.drained()
    assert playback.playing is False


async def test_an_empty_chunk_is_not_a_sound() -> None:
    """A device opened for nothing is a click and a tenth of a second."""
    device = Journal()
    playback = playback_on(device)

    playback.push(b"", sample_rate=RATE)
    await playback.drained()

    assert device.events == []


# --------------------------------------------------------------------------
# Being interrupted
# --------------------------------------------------------------------------


async def test_stopping_drops_what_is_queued_and_cuts_the_sound_within_a_block() -> None:
    """`Interrupted`: the user is talking. The rest of the answer is not
    played later, and what the sound card holds is aborted, not drained -
    a second of talking over the user is what draining would be."""
    hold = Held()
    device = Journal(on_write=hold)
    playback = playback_on(device)
    for _ in range(4):
        playback.push(silence(), sample_rate=RATE)

    await hold.begun()
    playback.stop()
    hold.release.set()
    await playback.drained()

    [stream] = device.streams
    assert len(stream.written) == 1
    assert (stream.aborted, stream.finished) == (True, False)
    assert playback.playing is False


async def test_stopping_when_nothing_plays_is_harmless() -> None:
    device = Journal()
    playback = playback_on(device)

    playback.stop()
    await playback.drained()

    assert device.events == []


async def test_a_stop_does_not_silence_the_next_answer() -> None:
    """Being interrupted is the normal end of an answer, not a state to
    leave the playback in."""
    device = Journal()
    playback = playback_on(device)
    playback.stop()

    playback.push(b"answer", sample_rate=RATE)
    await playback.drained()

    assert device.streams[0].heard == b"answer"


async def test_what_is_pushed_after_a_stop_is_a_new_answer() -> None:
    """Audio still in flight from the interrupted turn is dropped by the
    state machine (rule 2); what comes next is the model's next turn."""
    hold = Held()
    device = Journal(on_write=hold)
    playback = playback_on(device)
    playback.push(silence(blocks=2), sample_rate=RATE)

    await hold.begun()
    playback.stop()
    hold.release.set()
    playback.push(b"next", sample_rate=RATE)
    await playback.drained()

    assert device.streams[0].aborted
    assert device.streams[1].heard == b"next"


async def test_stopping_an_answer_that_has_not_started_playing_drops_it() -> None:
    """Interrupted between the first chunk and the first write: nothing is
    heard at all, and no device is opened for it."""
    device = Journal()
    playback = playback_on(device)

    playback.push(silence(), sample_rate=RATE)
    playback.stop()
    await playback.drained()

    assert device.events == []


# --------------------------------------------------------------------------
# A device that fails
# --------------------------------------------------------------------------


async def test_a_device_that_fails_loses_the_answer_and_not_the_program() -> None:
    """A Bluetooth headset switched off mid-conversation. The words are in
    the transcript; the session goes on; the failure is in the log."""
    device = Journal(refuse=1)
    playback = playback_on(device)
    lines: list[str] = []
    sink = logger.add(lines.append, format="{message}")
    try:
        playback.push(silence(), sample_rate=RATE)
        await playback.drained()
    finally:
        logger.remove(sink)

    assert playback.playing is False
    assert any("unavailable" in line for line in lines)


async def test_the_next_answer_tries_the_device_again() -> None:
    device = Journal(refuse=1)
    playback = playback_on(device)

    playback.push(silence(), sample_rate=RATE)
    await playback.drained()
    playback.push(b"again", sample_rate=RATE)
    await playback.drained()

    assert [stream.heard for stream in device.streams] == [b"again"]


async def test_a_failure_that_is_not_the_device_is_not_swallowed() -> None:
    """A bug in the speaker is a traceback, not a log line."""

    class Broken:
        async def play(self, buffers: AsyncIterator[bytes], *, sample_rate: int) -> None:
            raise ValueError("a bug")

        def stop(self) -> None:
            pass

    playback = LivePlayback(Broken())
    playback.push(silence(), sample_rate=RATE)

    with pytest.raises(ValueError, match="a bug"):
        await playback.drained()
