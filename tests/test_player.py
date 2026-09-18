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
import time
from collections.abc import AsyncIterator, Callable

import pytest

from assistant.audio.player import (
    BLOCK_FRAMES,
    BYTES_PER_FRAME,
    PlaybackError,
    Speaker,
    SystemSpeaker,
)

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
