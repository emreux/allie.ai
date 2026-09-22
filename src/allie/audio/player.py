"""Playing the answer out loud (design.md item 1.10).

The TTS layer yields raw PCM and deliberately stops there: sixteen bit signed
mono at the rate the engine declares, whether it came from Windows or from a
cloud voice. This file is the other end of that - the only place in the project
that opens an output device.

Two things shape it.

**Writing audio blocks for as long as the audio lasts.** PortAudio's `write`
returns when the sound card has room, which for a five second answer means five
seconds. Rule 4 of section 3.1 gives anything awaited in `app.py` fifty
milliseconds, so all of it happens in a worker thread. Every block's level
goes to `on_level` beside the write, from that thread (plan.md D20): the
window's orb swells with what is heard, when it is heard.

**Speech has to stop the instant the user presses the key.** That is why the
audio goes out in blocks rather than in one write, and why an interrupted
answer is *aborted* rather than stopped: a sound card holds up to a second of
audio, and draining it politely would mean talking over the user for a second
after being told not to.

**The device failing is the device's problem, not the program's.** A Bluetooth
headset switched off between two answers raises out of PortAudio; here that
becomes `PlaybackError`, which `app.py` writes down and moves on from - the
words are already on the screen. Only the device's own calls are wrapped: an
engine that fails while producing the audio is a bug or an outage, and keeps
its own traceback.

The device itself is injected, so the tests are about what was written rather
than about what was heard.

`LivePlayback` is the live product's way in (plan.md section 4.4 rule 2): the
model's audio arrives in blobs, faster than it plays and with no warning of
the last one, so it is pushed into a queue that feeds `play`, and the state
machine says when an answer is over (`flush`), waits for it (`drained`) or
cuts it (`stop`). The device is opened and closed per answer, as before: the
local voice of a confirmation or a reminder needs it in between.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from allie.stt.base import LEVEL_FLOOR_DBFS, dbfs, from_pcm16

__all__ = [
    "BLOCK_FRAMES",
    "BYTES_PER_FRAME",
    "CHANNELS",
    "LivePlayback",
    "PlaybackError",
    "Speaker",
    "SystemSpeaker",
]

# Sixteen bit signed mono, the format `tts/base.py` promises.
BYTES_PER_FRAME = 2
CHANNELS = 1

# 100 ms at 16 kHz. This is how long an interruption can take to be obeyed, and
# also how often a thread that is otherwise blocked gets to look at the flag.
BLOCK_FRAMES = 1_600


# What an open sound device is to this module: `write`, `stop`, `abort`,
# `close`, as `sounddevice.RawOutputStream` has them. Typed as `Any` rather
# than as a protocol - the old repository declared one and nothing referred
# to it, because the only thing that opens a stream here is `_open_output`
# and the tests pass their own object in its place.
StreamFactory = Callable[[int], Any]


class PlaybackError(RuntimeError):
    """The output device failed: gone, busy, or refusing the format.

    Raised for the device and for nothing else. A failure in the engine that
    produces the audio is a different thing and is not dressed up as this.
    """


@runtime_checkable
class Speaker(Protocol):
    """What the state machine needs to make a sound and to stop making one."""

    async def play(self, buffers: AsyncIterator[bytes], *, sample_rate: int) -> None:
        """Plays PCM as it arrives, returning when the last of it was heard."""
        ...

    def stop(self) -> None:
        """Cuts the current answer short. Safe to call when nothing is playing."""
        ...


class SystemSpeaker:
    """The real sound card, through `sounddevice`."""

    def __init__(
        self,
        *,
        open_stream: StreamFactory | None = None,
        on_level: Callable[[float], None] | None = None,
    ) -> None:
        self._open = open_stream if open_stream is not None else _open_output
        # Told the level, in dBFS, of every block as it goes to the device,
        # and the floor once the answer is over (plan.md D20) - from the
        # worker thread, so whoever listens must not touch the loop.
        self._on_level = on_level

        # Written from the event loop, read in the worker thread between
        # blocks. A plain event rather than an asyncio one for exactly that
        # reason: the thread cannot wait on the loop's.
        self._stopped = threading.Event()

    @property
    def on_level(self) -> Callable[[float], None] | None:
        return self._on_level

    async def play(self, buffers: AsyncIterator[bytes], *, sample_rate: int) -> None:
        # An interruption belongs to the answer it cut short. Left set, it
        # would silence the next one before it began.
        self._stopped.clear()
        stream: Any = None

        try:
            # Pulled one at a time, and only while nothing has said stop. An
            # `async for` would ask the engine for the next sentence before it
            # got to look at the flag, and synthesising a sentence nobody will
            # ever hear is a fifth of a second of the four cores Whisper wants.
            while not self._stopped.is_set():
                buffer = await anext(buffers, None)
                if buffer is None:
                    break
                if not buffer:
                    continue
                if stream is None:
                    # Opened on the first sound there is to make: an answer of
                    # nothing should not cost a tenth of a second and a click.
                    stream = await asyncio.to_thread(self._open_device, sample_rate)
                await asyncio.to_thread(self._write_device, stream, buffer)
        finally:
            if stream is not None:
                await asyncio.to_thread(self._finish_device, stream)

    def stop(self) -> None:
        self._stopped.set()

    # ----------------------------------------------------------------------
    # In a worker thread, where blocking is allowed.
    # ----------------------------------------------------------------------

    def _open_device(self, sample_rate: int) -> Any:
        try:
            return self._open(sample_rate)
        except Exception as failure:
            raise PlaybackError(f"the sound device could not be opened: {failure}") from failure

    def _write_device(self, stream: Any, buffer: bytes) -> None:
        block = BLOCK_FRAMES * BYTES_PER_FRAME
        try:
            for start in range(0, len(buffer), block):
                if self._stopped.is_set():
                    return
                piece = buffer[start : start + block]
                if self._on_level is not None:
                    self._on_level(dbfs(from_pcm16(piece)))
                stream.write(piece)
        except Exception as failure:
            raise PlaybackError(f"the sound device failed while playing: {failure}") from failure

    def _finish_device(self, stream: Any) -> None:
        # Whatever happens, the handle goes back; whatever was raised, it was
        # the device's. Interrupted answers are aborted rather than drained -
        # see the module docstring.
        if self._on_level is not None:
            self._on_level(LEVEL_FLOOR_DBFS)
        try:
            try:
                if self._stopped.is_set():
                    stream.abort()
                else:
                    stream.stop()
            finally:
                stream.close()
        except Exception as failure:
            raise PlaybackError(f"the sound device failed while finishing: {failure}") from failure


class LivePlayback:
    """A queue in front of a `Speaker`, for audio that arrives as it is said.

    `push` queues a chunk and starts the sound if none is under way. `flush`
    says the answer is over: what is queued is still heard to its last word,
    then the device is given back. `drained` waits for exactly that - and
    ends the answer itself if nobody has, since waiting for an answer never
    told it was over would wait for ever. `stop` is the interruption: the
    queue is dropped and what the sound card holds is aborted, not drained,
    within one block (`BLOCK_FRAMES`).

    An answer is one open-to-close of the device, at one rate. The next one
    waits for the device to be given back - two streams on one device is
    two answers talking over each other - and a chunk at another rate is
    another answer: a device opened at one rate cannot play the other.

    The device failing loses the answer, not the program: `PlaybackError`
    goes to the log, the words are in the transcript, and the next answer
    tries the device again. Anything else raised in the speaker is a bug,
    and comes out of `drained`.

    Called from the event loop only. `stop` is the one that is urgent, and
    it does nothing that blocks.
    """

    def __init__(self, speaker: Speaker) -> None:
        self._speaker = speaker
        # The answer under way, while chunks may still be pushed into it;
        # `None` between answers, and once one has been flushed.
        self._queue: asyncio.Queue[bytes | None] | None = None
        self._rate = 0
        # The task playing the latest answer, which waits for the one before.
        self._task: asyncio.Task[None] | None = None

    @property
    def playing(self) -> bool:
        """Whether there is sound still to come: queued, or in the device."""
        return self._task is not None and not self._task.done()

    def push(self, pcm16: bytes, *, sample_rate: int) -> None:
        """Queues a chunk of the answer, starting the sound if none is under
        way. `sample_rate` is the audio's own (`AudioChunk.sample_rate`)."""
        if self._queue is None or sample_rate != self._rate:
            self.flush()
            self._queue = asyncio.Queue()
            self._rate = sample_rate
            self._task = asyncio.create_task(
                self._play(self._queue, sample_rate, after=self._task),
            )
        self._queue.put_nowait(pcm16)

    def flush(self) -> None:
        """The answer is over (`TurnComplete`). What is queued is still
        played; pushing more starts the next answer."""
        queue, self._queue = self._queue, None
        if queue is not None:
            queue.put_nowait(None)

    async def drained(self) -> None:
        """Returns once everything pushed so far has been heard - or dropped
        by `stop`. The answer under way is ended first, as `flush` would."""
        self.flush()
        task = self._task
        if task is not None:
            await task

    def stop(self) -> None:
        """Cuts the sound short: what is queued is dropped, what the device
        holds is aborted. Safe to call when nothing is playing."""
        queue, self._queue = self._queue, None
        if queue is not None:
            while not queue.empty():
                queue.get_nowait()
            queue.put_nowait(None)
        self._speaker.stop()

    async def _play(
        self,
        queue: asyncio.Queue[bytes | None],
        sample_rate: int,
        *,
        after: asyncio.Task[None] | None,
    ) -> None:
        if after is not None:
            # The previous answer is still in the device: one at a time.
            await after
        try:
            await self._speaker.play(_queued(queue), sample_rate=sample_rate)
        except PlaybackError as failure:
            logger.warning("playback failed, the answer was lost: {failure}", failure=failure)


async def _queued(queue: asyncio.Queue[bytes | None]) -> AsyncIterator[bytes]:
    """The chunks of one answer, as they come, until the answer is over."""
    while (chunk := await queue.get()) is not None:
        yield chunk


def _open_output(sample_rate: int) -> Any:
    """Opens the default output device, already running."""
    # Imported here rather than at module scope: it loads PortAudio, and
    # `allie --help` has no use for a sound card. `sounddevice` ships no
    # type information, which is why the stream is `Any` throughout.
    import sounddevice  # type: ignore[import-untyped]

    stream = sounddevice.RawOutputStream(samplerate=sample_rate, channels=CHANNELS, dtype="int16")
    stream.start()
    return stream
