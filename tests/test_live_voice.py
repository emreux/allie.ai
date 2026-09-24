"""The program's own sentences in the assistant's voice (plan.md D32).

`LiveVoice` opens a session of its own for each sentence, reads what the
model says until the first `TurnComplete`, and keeps the sentences the
program always says on disk. The provider here is a reader that speaks at
the rate the desk measured - so many seconds of "speech" per character -
and can be told to read twice, ramble, say nothing, hang or refuse, which
are the ways the real one was seen to go wrong or might.
"""

from __future__ import annotations

import asyncio
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pytest

from allie.agent.prompts import READER_PROMPT
from allie.live.base import (
    AudioChunk,
    Closed,
    LiveEvent,
    ProviderError,
    SessionConfig,
    TurnComplete,
)
from allie.tts.base import TTSProvider
from allie.tts.live_voice import (
    CHARS_PER_SECOND,
    MAX_STRETCH,
    OUTPUT_RATE,
    LiveVoice,
    plausible,
    speech_seconds,
    spoken,
    too_short,
)
from tests.live_contract import FakeLiveSession

HINT = "Evet ya da hayır de."
QUESTION = "Süt al notu silinecek."
# Pieces of a reading: a tenth of a second each.
PIECE = OUTPUT_RATE // 10


def speech(seconds: float, *, silence: float = 0.3) -> bytes:
    """A loud tone for `seconds`, then `silence` - how a reading sounds to
    a meter that only asks whether a block is loud."""
    loud = np.full(int(seconds * OUTPUT_RATE), 6000, dtype="<i2")
    loud[1::2] = -6000
    quiet = np.zeros(int(silence * OUTPUT_RATE), dtype="<i2")
    return np.concatenate([loud, quiet]).tobytes()


def reading_of(text: str) -> bytes:
    return speech(len(text) / CHARS_PER_SECOND)


class Reader(FakeLiveSession):
    """A session that reads whatever text it is sent, in pieces, as the
    reader does - and then misbehaves as it was told to."""

    def __init__(self, how: str) -> None:
        super().__init__()
        self.how = how
        self.pieces_read = 0

    async def events(self) -> AsyncIterator[LiveEvent]:
        text = self.texts[0][0] if self.texts else ""
        if self.how == "refuse_midway":
            yield AudioChunk(reading_of(text)[: PIECE * 2], OUTPUT_RATE)
            yield Closed(ProviderError("the socket dropped", kind="unreachable"))
            return
        if self.how == "hang":
            await asyncio.Event().wait()
        if self.how == "silent":
            yield TurnComplete()
            yield Closed()
            return
        pcm = reading_of(text)
        if self.how == "ramble":
            pcm = speech(60.0)
        if self.how == "short":
            pcm = speech(len(text) / CHARS_PER_SECOND / 3)
        for start in range(0, len(pcm), PIECE * 2):
            self.pieces_read += 1
            yield AudioChunk(pcm[start : start + PIECE * 2], OUTPUT_RATE)
        yield TurnComplete()
        if self.how == "twice":
            # The second turn the model started when the loop kept listening.
            for start in range(0, len(pcm), PIECE * 2):
                yield AudioChunk(pcm[start : start + PIECE * 2], OUTPUT_RATE)
            yield TurnComplete()
        yield Closed()


class Voices:
    """The live provider as the reader sees it: a session per sentence,
    each behaving as `how` says, or a refusal at the open."""

    id = "fake"
    capabilities: frozenset[str] = frozenset()

    def __init__(self, *hows: str, refuse: ProviderError | None = None) -> None:
        self.hows = list(hows)
        self.refuse = refuse
        self.opened: list[SessionConfig] = []
        self.sessions: list[Reader] = []
        self.closed = 0

    async def validate_credentials(self) -> bool:
        return True

    async def list_models(self) -> list[object]:
        return []

    @asynccontextmanager
    async def connect(self, config: SessionConfig) -> AsyncIterator[Reader]:
        self.opened.append(config)
        if self.refuse is not None:
            raise self.refuse
        session = Reader(self.hows.pop(0) if self.hows else "well")
        self.sessions.append(session)
        try:
            yield session
        finally:
            self.closed += 1

    @property
    def read(self) -> list[str]:
        return [session.texts[0][0] for session in self.sessions]


def voice_with(
    provider: Voices,
    directory: Path | None = None,
    *,
    unsaid: list[str] | None = None,
    read_seconds: float = 5.0,
) -> LiveVoice:
    return LiveVoice(
        provider,  # type: ignore[arg-type]
        model="gemini-3.8-live",
        voice="Kore",
        language_code="tr-TR",
        directory=directory,
        on_unsaid=unsaid.append if unsaid is not None else None,
        read_seconds=read_seconds,
    )


async def said(voice: LiveVoice, *texts: str) -> bytes:
    return b"".join([pcm async for pcm in voice.stream(texts)])


def wavs(directory: Path) -> list[Path]:
    """What the voice kept, from outside the loop's tests."""
    return sorted(directory.glob("*.wav"))


def spoil(path: Path) -> None:
    path.write_bytes(b"not a wave")


# --------------------------------------------------------------------------
# Reading live
# --------------------------------------------------------------------------


def test_it_is_a_voice_the_state_machine_can_use() -> None:
    assert isinstance(voice_with(Voices()), TTSProvider)


async def test_a_sentence_is_read_by_the_model_in_the_assistant_s_voice() -> None:
    provider = Voices()

    pcm = await said(voice_with(provider), QUESTION)

    assert pcm == reading_of(QUESTION)
    (config,) = provider.opened
    assert config.model == "gemini-3.8-live"
    assert config.voice == "Kore"
    assert config.language_code == "tr-TR"


async def test_the_reader_is_not_the_conversation() -> None:
    """D3: no tools, no history - the prompt that only reads, and nothing
    of the conversation's."""
    provider = Voices()

    await said(voice_with(provider), QUESTION)

    (config,) = provider.opened
    assert config.system_prompt == READER_PROMPT
    assert config.tools == ()
    assert config.resume_handle is None
    assert not config.transcripts
    assert provider.sessions[0].texts == [(QUESTION, "user", True)]


async def test_each_sentence_gets_a_session_of_its_own_and_it_is_closed() -> None:
    provider = Voices()

    await said(voice_with(provider), QUESTION, HINT)

    assert provider.read == [QUESTION, HINT]
    assert provider.closed == 2


async def test_the_reading_stops_at_the_first_turn_complete() -> None:
    """Measured: listening past it, the model read the sentence again."""
    pcm = await said(voice_with(Voices("twice")), QUESTION)

    assert pcm == reading_of(QUESTION)


async def test_a_reading_that_stops_short_is_read_again_whole() -> None:
    """Measured: a question read only as far as its quoted part. The user
    hears the fragment and then all of it - never only the fragment."""
    provider = Voices("short")

    pcm = await said(voice_with(provider), QUESTION)

    assert provider.read == [QUESTION, QUESTION]
    assert pcm.endswith(reading_of(QUESTION))


async def test_a_sentence_is_read_again_only_once() -> None:
    provider = Voices("short", "short", "short")

    await said(voice_with(provider), QUESTION)

    assert provider.read == [QUESTION, QUESTION]


async def test_the_quote_marks_are_not_read() -> None:
    """They are for the screen, silent aloud - and a question that began
    with one was read only as far as the closing one."""
    provider = Voices()

    await said(voice_with(provider), "'Akşam yemeğe geliyorum' mesajı Ahmet kişisine gidecek.")

    assert provider.read == ["Akşam yemeğe geliyorum mesajı Ahmet kişisine gidecek."]


def test_an_apostrophe_is_part_of_the_word() -> None:
    assert spoken("'Spotify' Store'dan 14:30'da, “hemen”.") == "Spotify Store'dan 14:30'da, hemen."


async def test_a_ramble_is_cut_at_the_stretch() -> None:
    pcm = await said(voice_with(Voices("ramble")), HINT)

    longest = (MAX_STRETCH * len(HINT) / CHARS_PER_SECOND + 1.0) * OUTPUT_RATE * 2
    assert 0 < len(pcm) <= longest


async def test_the_first_piece_is_handed_on_before_the_reading_is_over() -> None:
    provider = Voices()
    stream = voice_with(provider).stream([QUESTION])

    first = await anext(stream)
    reading = provider.sessions[0]
    await stream.aclose()

    assert first
    assert reading.pieces_read < len(reading_of(QUESTION)) // (PIECE * 2)


async def test_stopping_halfway_closes_the_reader_s_session() -> None:
    """The user talked over it: the speaker stops asking and closes the
    stream, and the session goes with it at once."""
    provider = Voices()
    stream = voice_with(provider).stream([QUESTION])

    await anext(stream)
    await stream.aclose()

    assert provider.closed == 1


# --------------------------------------------------------------------------
# What could not be read
# --------------------------------------------------------------------------


async def test_a_refused_reading_is_handed_to_the_screen_and_the_next_goes_on() -> None:
    unsaid: list[str] = []
    provider = Voices(refuse=ProviderError("no network", kind="unreachable"))

    pcm = await said(voice_with(provider, unsaid=unsaid), QUESTION, HINT)

    assert pcm == b""
    assert unsaid == [QUESTION, HINT]


async def test_a_reading_that_never_ends_is_given_up_in_time() -> None:
    unsaid: list[str] = []

    pcm = await said(voice_with(Voices("hang"), unsaid=unsaid, read_seconds=0.05), QUESTION)

    assert pcm == b""
    assert unsaid == [QUESTION]


async def test_a_reading_with_no_sound_is_unsaid() -> None:
    unsaid: list[str] = []

    await said(voice_with(Voices("silent"), unsaid=unsaid), QUESTION)

    assert unsaid == [QUESTION]


async def test_a_socket_that_drops_midway_keeps_what_was_heard() -> None:
    unsaid: list[str] = []

    pcm = await said(voice_with(Voices("refuse_midway"), unsaid=unsaid), QUESTION)

    assert pcm
    assert unsaid == []


async def test_a_voice_at_another_rate_is_not_played_at_the_wrong_speed() -> None:
    class Other(Reader):
        async def events(self) -> AsyncIterator[LiveEvent]:
            yield AudioChunk(b"\x01\x00" * 100, 16_000)
            yield TurnComplete()

    class OtherVoices(Voices):
        @asynccontextmanager
        async def connect(self, config: SessionConfig) -> AsyncIterator[Reader]:
            yield Other("well")

    unsaid: list[str] = []

    pcm = await said(voice_with(OtherVoices(), unsaid=unsaid), QUESTION)

    assert pcm == b""
    assert unsaid == [QUESTION]


# --------------------------------------------------------------------------
# Kept on disk
# --------------------------------------------------------------------------


async def test_a_prepared_sentence_is_played_from_disk_without_a_session(tmp_path: Path) -> None:
    provider = Voices()
    voice = voice_with(provider, tmp_path)

    await voice.prepare([HINT])
    opened = len(provider.opened)
    pcm = await said(voice, HINT)

    assert opened == 1
    assert len(provider.opened) == 1
    assert pcm == reading_of(HINT)


async def test_the_kept_file_is_a_wav_at_the_voice_s_rate(tmp_path: Path) -> None:
    await voice_with(Voices(), tmp_path).prepare([HINT])

    (path,) = wavs(tmp_path)
    with wave.open(str(path), "rb") as file:
        assert (file.getnchannels(), file.getsampwidth(), file.getframerate()) == (
            1,
            2,
            OUTPUT_RATE,
        )


async def test_preparing_again_reads_nothing_that_is_kept(tmp_path: Path) -> None:
    await voice_with(Voices(), tmp_path).prepare([HINT, QUESTION])
    provider = Voices()

    await voice_with(provider, tmp_path).prepare([HINT, QUESTION])

    assert provider.opened == []


async def test_a_sentence_not_prepared_is_read_live_every_time(tmp_path: Path) -> None:
    """A gate's question carries the real values; it is never kept."""
    provider = Voices()
    voice = voice_with(provider, tmp_path)
    await voice.prepare([HINT])

    await said(voice, QUESTION)
    await said(voice, QUESTION)

    assert provider.read == [HINT, QUESTION, QUESTION]
    assert len(wavs(tmp_path)) == 1


async def test_a_reading_of_the_wrong_length_is_not_kept_and_is_read_again(
    tmp_path: Path,
) -> None:
    provider = Voices("short")

    await voice_with(provider, tmp_path).prepare([QUESTION])

    assert provider.read == [QUESTION, QUESTION]
    assert len(wavs(tmp_path)) == 1


async def test_a_sentence_never_read_whole_is_not_kept_and_not_fatal(tmp_path: Path) -> None:
    provider = Voices("short", "short")

    await voice_with(provider, tmp_path).prepare([QUESTION])

    assert wavs(tmp_path) == []


async def test_preparing_stops_at_a_provider_that_cannot_be_reached(tmp_path: Path) -> None:
    provider = Voices(refuse=ProviderError("no network", kind="unreachable"))

    await voice_with(provider, tmp_path).prepare([HINT, QUESTION])

    assert len(provider.opened) == 1


async def test_a_fixed_sentence_read_live_is_kept_for_next_time(tmp_path: Path) -> None:
    """Prepared at a start that could not reach the provider, then said
    once the network was back: kept then."""
    provider = Voices(refuse=ProviderError("no network", kind="unreachable"))
    voice = voice_with(provider, tmp_path)
    await voice.prepare([HINT])
    provider.refuse = None

    await said(voice, HINT)
    await said(voice, HINT)

    assert provider.read == [HINT]


async def test_another_voice_is_another_file_and_the_old_ones_go(tmp_path: Path) -> None:
    await voice_with(Voices(), tmp_path).prepare([HINT])
    (kore,) = wavs(tmp_path)
    orus = LiveVoice(Voices(), model="gemini-3.8-live", voice="Orus", directory=tmp_path)  # type: ignore[arg-type]

    await orus.prepare([HINT])

    (kept,) = wavs(tmp_path)
    assert kept != kore


async def test_a_broken_file_is_read_live_instead(tmp_path: Path) -> None:
    provider = Voices()
    voice = voice_with(provider, tmp_path)
    await voice.prepare([HINT])
    (path,) = wavs(tmp_path)
    spoil(path)

    pcm = await said(voice, HINT)

    assert pcm == reading_of(HINT)
    assert provider.read == [HINT, HINT]


async def test_without_a_directory_everything_is_read_live() -> None:
    provider = Voices()
    voice = voice_with(provider)

    await voice.prepare([HINT])
    await said(voice, HINT)

    assert provider.read == [HINT]


# --------------------------------------------------------------------------
# How long a reading should be
# --------------------------------------------------------------------------


def test_the_silence_around_speech_does_not_count() -> None:
    pcm = bytes(OUTPUT_RATE) + speech(1.0, silence=0.5)

    assert speech_seconds(pcm) == pytest.approx(1.0, abs=0.03)


def test_nothing_is_no_speech() -> None:
    assert speech_seconds(b"") == 0.0
    assert speech_seconds(bytes(OUTPUT_RATE * 2)) == 0.0


@pytest.mark.parametrize(
    ("factor", "kept"),
    [(1.0, True), (0.85, True), (1.3, True), (2.0, False), (0.55, False), (0.2, False)],
)
def test_a_reading_is_plausible_when_its_length_fits_its_text(factor: float, kept: bool) -> None:
    text = "Dişçi randevusu hatırlatıcısı iptal edilecek."
    pcm = speech(len(text) / CHARS_PER_SECOND * factor)

    assert plausible(text, pcm) is kept


@pytest.mark.parametrize(
    ("text", "seconds", "short"),
    [
        # The desk's: whole readings, the slowest and the fastest...
        ("Süt al notu silinecek.", 1.42, False),
        ("Hemen bakıyorum...", 0.80, False),
        ("Sizi dinliyorum efendim.", 1.34, False),
        # ...and the ones the model cut off.
        ("Süt al notu silinecek.", 0.90, True),
        ("Annemin doğum günü haziranda kaydı unutulacak.", 1.76, True),
        ("Spotify (Spotify AB) Microsoft Store'dan indirilecek.", 0.74, True),
    ],
)
def test_a_reading_cut_off_is_told_from_a_quick_one(text: str, seconds: float, short: bool) -> None:
    assert too_short(text, speech(seconds)) is short
