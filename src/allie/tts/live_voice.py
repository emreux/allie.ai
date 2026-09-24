"""The assistant's own voice for the program's own sentences (plan.md D32).

Until 2026-09-24 a gate's question, a reminder, the filler and the three
failure sentences were read by Windows' Tolga, or by Google's separate
synthesiser with Tolga behind it - a man's voice between the words of
Friday's, and a different one again when the synthesiser's small free quota
ran out. The owner asked for one voice. So the live model reads them itself,
in the voice the conversation speaks in (`[live] voice`), from a session of
its own.

**Not the conversation.** The gate's question must reach the user as the
gate wrote it, and the user's yes must never reach the model (D3); a
session opened with `READER_PROMPT`, no tools and no history is a reader
and nothing else. Measured at the desk (docs/specs/2026-09-24-settings-and-
one-voice-design.md section 1): it read every sentence it was given - a
question, a request, a message telling it to say yes - and answered none.

**One session a sentence, stopped at the first `TurnComplete`.** Kept open
past it the model started a second turn and read the sentence again. A
read that runs far longer than its text can take is cut (`MAX_STRETCH`):
the settings the reader was not given produced fifteen to fifty seconds of
rambling at the desk, and nothing may hold the speaker that long.

**A question is heard whole, or read again.** The model sometimes stopped
early - a question that began with a quoted part was read only as far as
the quote, "Akşam yemeğe geliyorum" and not to whom. Three things answer
that: the quote marks are taken off before reading (they are silent
anyway), the prompt says to read to the last word, and a reading that ends
much shorter than its text should take is read once more from the start
(`too_short`). The user may hear a fragment and then the whole; never only
the fragment.

**The sentences the program always says are kept on disk.** The fillers,
the hint after a question, the failures, the greeting: read once per voice
at a start (`prepare`), kept as WAV files when the reading is as long as its
text should be, and played from there ever after - at once, with no request
and no network, which is exactly when the three failure sentences are
needed. A reading that is not kept is never fatal: the sentence is read
live when it is needed. What cannot be read - the provider refused, the
network is down - is handed to `on_unsaid`, which puts it on the screen.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import re
import wave
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
from loguru import logger

from allie.agent.prompts import READER_PROMPT
from allie.live.base import (
    AudioChunk,
    Closed,
    LiveEvent,
    LiveProvider,
    ProviderError,
    SessionConfig,
    TurnComplete,
)

__all__ = [
    "CHARS_PER_SECOND",
    "MAX_STRETCH",
    "OUTPUT_RATE",
    "READ_SECONDS",
    "LiveVoice",
    "plausible",
    "speech_seconds",
    "spoken",
    "too_short",
]

# What both live providers speak at (`live/base.py::AudioChunk`).
OUTPUT_RATE = 24_000

# How fast the reader speaks: 13-16 characters of text a second of speech,
# spaces and punctuation counted, measured on Kore and Orus (2026-09-24).
CHARS_PER_SECOND = 14.5

# A reading whose speech is shorter than this multiple of what its text
# should take, less a little, stopped early. The desk's whole readings lay
# at 0.64-1.3 of it (the lowest a short sentence, where the little counts);
# the ones cut off at 0.19-0.59.
SHORT_BELOW = 0.72
SHORT_SLACK_SECONDS = 0.15

# One longer than this multiple, plus a little, was read twice or rambled:
# it is not kept, and read again at the next start.
LONG_ABOVE = 1.6
LONG_SLACK_SECONDS = 0.4

# How many readings a sentence read live gets: the second only when the
# first stopped short of its text.
LIVE_TRIES = 2

# A live reading is cut past this multiple of its expected length, plus a
# second - room for the trailing silence and a slow reading, none for a
# second turn or a ramble.
MAX_STRETCH = 1.8
STRETCH_SLACK_SECONDS = 1.0

# How long a whole reading may take, open to last sound, before it is given
# up as not answered. The desk's longest was under ten seconds.
READ_SECONDS = 20.0

# How many readings a sentence to be kept gets at a start.
TRIES = 2

# A 20 ms block whose loudest sample passes this is speech (of 32 767).
SPEECH_PEAK = 800
BLOCK_SECONDS = 0.02


class LiveVoice:
    """The program's sentences in the conversation's voice, read by the
    live model from a session of their own and kept on disk when they are
    always the same."""

    id = "live"
    sample_rate = OUTPUT_RATE

    def __init__(
        self,
        provider: LiveProvider,
        *,
        model: str,
        voice: str = "",
        language_code: str = "",
        directory: Path | None = None,
        on_unsaid: Callable[[str], None] | None = None,
        read_seconds: float = READ_SECONDS,
    ) -> None:
        self._provider = provider
        self._model = model
        self._voice = voice
        self._language_code = language_code
        # Without one nothing is kept: every sentence is read live.
        self._directory = directory
        self._on_unsaid = on_unsaid
        self._read_seconds = read_seconds
        # The sentences worth keeping, told by `prepare`.
        self._fixed: set[str] = set()

    async def stream(self, texts: Sequence[str]) -> AsyncGenerator[bytes]:
        for said in texts:
            text = spoken(said)
            if not text:
                continue
            kept = self._kept(text)
            if kept is not None:
                yield kept
                continue
            # Closed the moment the speaker stops asking - the user talked
            # over it - so the reader's session goes with it.
            async with contextlib.aclosing(self._live(text)) as reading:
                async for pcm in reading:
                    yield pcm

    async def prepare(self, texts: Iterable[str]) -> None:
        """Reads the sentences given that are not on disk yet, one at a
        time, and keeps each whose reading is plausible; lets go of the
        files no sentence given names any more - another voice's, an older
        wording's."""
        fixed = {spoken(text) for text in texts} - {""}
        self._fixed = fixed
        if self._directory is None:
            return
        wanted = {self._path(self._directory, text): text for text in fixed}
        self._forget_all_but(set(wanted))
        for path, text in wanted.items():
            if path.is_file():
                continue
            for _ in range(TRIES):
                try:
                    pcm = await self._read_whole(text)
                except ProviderError as failure:
                    logger.warning(
                        "voice: a sentence could not be prepared ({kind}); it is read when needed",
                        kind=failure.kind,
                    )
                    return
                if self._keep(text, pcm):
                    break

    # ----------------------------------------------------------------------

    async def _live(self, text: str) -> AsyncGenerator[bytes]:
        """One sentence read by the model as it comes - once more from the
        start when the reading stopped short of its text - and kept
        afterwards when it is one of the fixed ones and came out whole."""
        reading = b""
        for _ in range(LIVE_TRIES):
            heard = bytearray()
            try:
                async with contextlib.aclosing(self._read(text)) as pieces:
                    async for pcm in pieces:
                        heard.extend(pcm)
                        yield pcm
            except ProviderError as failure:
                logger.warning("voice: a sentence could not be read ({kind})", kind=failure.kind)
            reading = bytes(heard)
            if not reading or not too_short(text, reading):
                break
            logger.warning(
                "voice: a reading stopped at {seconds:.1f} s of a sentence of {chars} "
                "characters; reading it again",
                seconds=speech_seconds(reading),
                chars=len(text),
            )
        if not reading:
            self._unsaid(text)
        elif text in self._fixed:
            self._keep(text, reading)

    async def _read_whole(self, text: str) -> bytes:
        async with contextlib.aclosing(self._read(text)) as reading:
            return b"".join([pcm async for pcm in reading])

    async def _read(self, text: str) -> AsyncGenerator[bytes]:
        """The model's reading of `text`, piece by piece, from a session of
        its own; over at the first `TurnComplete`, or at the stretch cap."""
        config = SessionConfig(
            model=self._model,
            voice=self._voice,
            system_prompt=READER_PROMPT,
            language_code=self._language_code,
            transcripts=False,
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._read_seconds
        room = _bytes_for(MAX_STRETCH * _expected(text) + STRETCH_SLACK_SECONDS)
        async with self._provider.connect(config) as session:
            await session.send_text(text, role="user", turn_complete=True)
            events = aiter(session.events())
            while True:
                event = await _next(events, deadline - loop.time())
                if event is None or isinstance(event, TurnComplete):
                    return
                if isinstance(event, Closed):
                    if event.error is not None:
                        raise event.error
                    return
                if isinstance(event, AudioChunk) and event.pcm16:
                    if event.sample_rate != OUTPUT_RATE:
                        # Neither live provider does this; if one ever does,
                        # the sentence is lost rather than played at the
                        # wrong speed.
                        raise ProviderError(
                            f"the voice came at {event.sample_rate} Hz, not {OUTPUT_RATE}",
                            kind="refused",
                        )
                    piece = event.pcm16[:room]
                    room -= len(piece)
                    if piece:
                        yield piece
                    if room <= 0:
                        logger.warning("voice: a reading ran past its text's length and was cut")
                        return

    def _kept(self, text: str) -> bytes | None:
        if self._directory is None:
            return None
        path = self._path(self._directory, text)
        try:
            with wave.open(str(path), "rb") as file:
                return file.readframes(file.getnframes())
        except (OSError, EOFError, wave.Error):
            return None

    def _keep(self, text: str, pcm: bytes) -> bool:
        """Writes the reading down when it is as long as its text should
        take; says whether it did."""
        if self._directory is None:
            return False
        if not plausible(text, pcm):
            logger.info(
                "voice: a reading of {chars} characters lasted {seconds:.1f} s and was not kept",
                chars=len(text),
                seconds=speech_seconds(pcm),
            )
            return False
        path = self._path(self._directory, text)
        partial = path.with_suffix(".part")
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            with wave.open(str(partial), "wb") as file:
                file.setnchannels(1)
                file.setsampwidth(2)
                file.setframerate(OUTPUT_RATE)
                file.writeframes(pcm)
            partial.replace(path)
        except OSError as failure:
            logger.warning("voice: a reading could not be kept: {problem}", problem=failure)
            return False
        return True

    def _forget_all_but(self, wanted: set[Path]) -> None:
        if self._directory is None or not self._directory.is_dir():
            return
        for path in self._directory.iterdir():
            if path.suffix in (".wav", ".part") and path not in wanted:
                with contextlib.suppress(OSError):
                    path.unlink()

    def _path(self, directory: Path, text: str) -> Path:
        # Everything the sound depends on: the same words in another voice,
        # model or language are another file.
        key = "\n".join((self._model, self._voice, self._language_code, text))
        name = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return directory / f"{name}.wav"

    def _unsaid(self, text: str) -> None:
        if self._on_unsaid is not None:
            self._on_unsaid(text)


async def _next(events: AsyncIterator[LiveEvent], seconds: float) -> LiveEvent | None:
    """The next event, `None` when the session has no more, and a timeout
    in the protocol's own currency when none came in time."""
    try:
        return await asyncio.wait_for(anext(events), max(0.0, seconds))
    except StopAsyncIteration:
        return None
    except TimeoutError:
        raise ProviderError("the voice did not finish in time", kind="timeout") from None


def _expected(text: str) -> float:
    """How long reading `text` should take, in seconds of speech."""
    return len(text) / CHARS_PER_SECOND


def _bytes_for(seconds: float) -> int:
    return int(seconds * OUTPUT_RATE) * 2


def speech_seconds(pcm: bytes) -> float:
    """How long the speech in `pcm` lasts, first loud block to last: the
    silence the model leaves before and after does not count."""
    samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2")
    block = int(OUTPUT_RATE * BLOCK_SECONDS)
    whole = len(samples) // block
    if whole == 0:
        return 0.0
    peaks = np.abs(samples[: whole * block].astype(np.int32)).reshape(whole, block).max(axis=1)
    loud = np.flatnonzero(peaks > SPEECH_PEAK)
    if loud.size == 0:
        return 0.0
    return float((loud[-1] - loud[0] + 1) * BLOCK_SECONDS)


def too_short(text: str, pcm: bytes) -> bool:
    """Whether a reading of `text` stopped well before its end."""
    return speech_seconds(pcm) < SHORT_BELOW * _expected(text) - SHORT_SLACK_SECONDS


def plausible(text: str, pcm: bytes) -> bool:
    """Whether a reading lasts about as long as `text` should take: not cut
    short, not read twice."""
    too_long = speech_seconds(pcm) > LONG_ABOVE * _expected(text) + LONG_SLACK_SECONDS
    return not too_short(text, pcm) and not too_long


# A quote mark that is not an apostrophe: one with no letter or digit on
# one of its sides. `Store'dan` and `14:30'da` keep theirs.
_QUOTE = re.compile(r"(?<![\w])['’‘]|['’‘](?![\w])|[\"“”«»]")


def spoken(text: str) -> str:
    """The words to be read, as the reader is given them: the quote marks
    the screen needs taken off - they are silent, and a question that began
    with a quoted part was read only as far as the quote - and the spaces
    evened out."""
    return " ".join(_QUOTE.sub("", text).split())
