"""Playing the answer out loud (design.md item 1.10).

The TTS layer yields raw PCM and deliberately stops there: sixteen bit signed
mono at the rate the engine declares, whether it came from Windows or from a
cloud voice. This file is the other end of that - the only place in the project
that opens an output device.

Two things shape it.

**Writing audio blocks for as long as the audio lasts.** PortAudio's `write`
returns when the sound card has room, which for a five second answer means five
seconds. Rule 4 of section 3.1 gives anything awaited in `app.py` fifty
milliseconds, so all of it happens in a worker thread.

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
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "BLOCK_FRAMES",
    "BYTES_PER_FRAME",
    "CHANNELS",
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


class OutputStream(Protocol):
    """The four things this needs an open sound device to do."""

    def write(self, data: bytes) -> None: ...

    def stop(self) -> None:
        """Waits for what is queued to finish playing."""
        ...

    def abort(self) -> None:
        """Drops what is queued instead of playing it."""
        ...

    def close(self) -> None: ...


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

    def __init__(self, *, open_stream: StreamFactory | None = None) -> None:
        self._open = open_stream if open_stream is not None else _open_output

        # Written from the event loop, read in the worker thread between
        # blocks. A plain event rather than an asyncio one for exactly that
        # reason: the thread cannot wait on the loop's.
        self._stopped = threading.Event()

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
                stream.write(buffer[start : start + block])
        except Exception as failure:
            raise PlaybackError(f"the sound device failed while playing: {failure}") from failure

    def _finish_device(self, stream: Any) -> None:
        # Whatever happens, the handle goes back; whatever was raised, it was
        # the device's. Interrupted answers are aborted rather than drained -
        # see the module docstring.
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


def _open_output(sample_rate: int) -> Any:
    """Opens the default output device, already running."""
    # Imported here rather than at module scope: it loads PortAudio, and
    # `live-assistant --help` has no use for a sound card. `sounddevice` ships no
    # type information, which is why the stream is `Any` throughout.
    import sounddevice  # type: ignore[import-untyped]

    stream = sounddevice.RawOutputStream(samplerate=sample_rate, channels=CHANNELS, dtype="int16")
    stream.start()
    return stream
