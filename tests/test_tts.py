"""Speaking: when a sentence is ready, and which voice reads it.

The interesting part is not the synthesis - Windows does that - but the
regrouping in front of it. The model sends text in fragments; waiting for all
of them before making a sound would add the whole generation time to the
latency of section 4. So a sentence is spoken as soon as it is whole, and the
test below proves it happens before the next fragment is even read.

The COM calls are behind one small interface. What cannot be faked - that a
voice Windows only lists in the OneCore hive can still be used, and that a
worker thread has to initialise COM before it may - is proved by
`scripts/smoke_tts.py` against the real engine.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Coroutine, Sequence
from typing import Any

from assistant.tts.base import MIN_SENTENCE_CHARS, TTSProvider, VoiceInfo, sentences
from assistant.tts.sapi import AUDIO_FORMAT, SAMPLE_RATE, SapiTTS, language_code

TOLGA = VoiceInfo(
    id=r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens\MSTTS_V110_trTR_Tolga",
    display_name="Microsoft Tolga - Turkish (Turkey)",
    language="tr",
)
DAVID = VoiceInfo(
    id=r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0",
    display_name="Microsoft David Desktop - English (United States)",
    language="en",
)


class FakeSpeech:
    """Stands in for the COM engine, and remembers what it was asked to say."""

    def __init__(self, *, voices: Sequence[VoiceInfo] = (TOLGA, DAVID), delay: float = 0.0) -> None:
        self._voices = list(voices)
        self.delay = delay
        self.said: list[tuple[str, str]] = []

    def voices(self) -> list[VoiceInfo]:
        if self.delay:
            time.sleep(self.delay)
        return list(self._voices)

    def speak(self, text: str, voice: str) -> bytes:
        self.said.append((text, voice))
        if self.delay:
            time.sleep(self.delay)
        return text.encode("utf-8")  # stands in for the audio


async def fragments(*pieces: str, seen: list[str] | None = None) -> AsyncIterator[str]:
    for piece in pieces:
        if seen is not None:
            seen.append(piece)
        yield piece


async def spoken(*pieces: str) -> list[str]:
    return [sentence async for sentence in sentences(fragments(*pieces))]


async def ticks_during(work: Coroutine[Any, Any, Any]) -> int:
    """How many turns the event loop got while `work` ran."""
    ticks = 0

    async def clock() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticking = asyncio.create_task(clock())
    await work
    ticking.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await ticking
    return ticks


# --------------------------------------------------------------------------
# Regrouping fragments into sentences
# --------------------------------------------------------------------------


async def test_fragments_become_whole_sentences() -> None:
    assert await spoken("Bugun hava ", "guzel olacak. ", "Yarin yagmur var.") == [
        "Bugun hava guzel olacak.",
        "Yarin yagmur var.",
    ]


async def test_a_sentence_is_spoken_before_the_next_fragment_is_read() -> None:
    """The whole point of streaming: section 4's optimisation is to start
    making sound while the model is still writing."""
    seen: list[str] = []
    when: list[int] = []

    async for _ in sentences(fragments("Bir tam cumle burada. ", "Ikinci cumle.", seen=seen)):
        when.append(len(seen))

    assert when == [1, 2], "the first sentence waited for the second fragment"


async def test_text_with_no_full_stop_is_still_spoken() -> None:
    """Models end a turn without punctuation more often than one would like."""
    assert await spoken("Bunun sonunda nokta yok") == ["Bunun sonunda nokta yok"]


async def test_questions_and_exclamations_end_a_sentence_too() -> None:
    assert await spoken("Bunu yapayim mi? Tabii ki yaparim!") == [
        "Bunu yapayim mi?",
        "Tabii ki yaparim!",
    ]


async def test_a_line_break_ends_a_sentence() -> None:
    assert await spoken("Birinci satir burada\nIkinci satir burada") == [
        "Birinci satir burada",
        "Ikinci satir burada",
    ]


async def test_an_abbreviation_does_not_cut_the_sentence_in_half() -> None:
    """ "Dr." ends in a full stop followed by a space, and is not a sentence.
    Anything too short to be one is kept and read with what follows."""
    assert await spoken("Dr. Mehmet Bey aradi ve mesaj birakti.") == [
        "Dr. Mehmet Bey aradi ve mesaj birakti."
    ]


async def test_a_decimal_point_is_not_a_full_stop() -> None:
    assert await spoken("Sicaklik 21.5 derece olacak.") == ["Sicaklik 21.5 derece olacak."]


async def test_nothing_at_all_is_nothing_to_say() -> None:
    assert await spoken() == []
    assert await spoken("   ", "\n") == []


async def test_the_threshold_is_shorter_than_a_sentence_and_longer_than_a_title() -> None:
    """The number decides both cases above; it is not free to drift."""
    assert 4 < MIN_SENTENCE_CHARS < 25


# --------------------------------------------------------------------------
# Choosing a voice
# --------------------------------------------------------------------------


async def test_every_voice_windows_knows_is_offered() -> None:
    assert await SapiTTS(engine=FakeSpeech()).list_voices() == [TOLGA, DAVID]


async def test_asking_for_a_language_gives_only_that_language() -> None:
    assert await SapiTTS(engine=FakeSpeech()).list_voices("tr") == [TOLGA]


async def test_the_region_is_not_part_of_the_question() -> None:
    """The locale of section 3.12 is `tr`; the voice claims `tr-TR`."""
    assert await SapiTTS(engine=FakeSpeech()).list_voices("tr-TR") == [TOLGA]


async def test_the_case_of_the_language_code_does_not_matter() -> None:
    """It comes out of a settings file somebody typed by hand."""
    assert await SapiTTS(engine=FakeSpeech()).list_voices("TR") == [TOLGA]


async def test_a_language_with_no_voice_is_an_empty_list_not_an_error() -> None:
    """Most languages have no voice installed; the fallback belongs to the
    caller, and it needs to be told plainly."""
    assert await SapiTTS(engine=FakeSpeech()).list_voices("el") == []


def test_windows_language_ids_are_read_with_the_table_python_already_has() -> None:
    """A hand-written LCID table would be a language constant in code, and
    wrong for the two hundred locales nobody here will ever check."""
    assert language_code("041f") == "tr"
    assert language_code("409") == "en"


def test_a_voice_that_claims_several_languages_is_filed_under_the_first() -> None:
    assert language_code("409;41f") == "en"


def test_a_language_windows_will_not_name_is_left_blank() -> None:
    assert language_code("") == ""
    assert language_code("ffff") == ""
    assert language_code("not a number") == ""


# --------------------------------------------------------------------------
# Speaking
# --------------------------------------------------------------------------


async def test_each_sentence_comes_back_as_its_own_audio() -> None:
    """One buffer per sentence is what lets the player start on the first one."""
    engine = FakeSpeech()
    tts = SapiTTS(engine=engine)

    audio = [
        buffer
        async for buffer in tts.stream(
            fragments("Birinci cumle burada. ", "Ikinci cumle burada."), voice=TOLGA.id
        )
    ]

    assert audio == [b"Birinci cumle burada.", b"Ikinci cumle burada."]


async def test_the_voice_that_was_asked_for_is_the_one_that_speaks() -> None:
    engine = FakeSpeech()

    async for _ in SapiTTS(engine=engine).stream(
        fragments("Bir sey soyleyeyim mi?"), voice=DAVID.id
    ):
        pass

    assert engine.said == [("Bir sey soyleyeyim mi?", DAVID.id)]


async def test_a_turn_with_nothing_in_it_never_reaches_the_engine() -> None:
    engine = FakeSpeech()

    audio = [
        buffer async for buffer in SapiTTS(engine=engine).stream(fragments(" "), voice=TOLGA.id)
    ]

    assert (audio, engine.said) == ([], [])


async def test_speaking_leaves_the_event_loop_free() -> None:
    """Synthesis is a blocking COM call. On the loop it would stop the
    scheduler and the announce queue for the length of every sentence."""

    async def speak_it() -> None:
        async for _ in SapiTTS(engine=FakeSpeech(delay=0.3)).stream(
            fragments("Bu cumle uzun surecek."), voice=TOLGA.id
        ):
            pass

    assert await ticks_during(speak_it()) >= 5, "the event loop was blocked while Windows spoke"


async def test_listing_voices_leaves_the_event_loop_free() -> None:
    """Walking two registry hives through COM blocks as well, and the wizard
    of phase 4.5 does it while the rest of the assistant is running."""
    ticks = await ticks_during(SapiTTS(engine=FakeSpeech(delay=0.3)).list_voices())

    assert ticks >= 5, "the event loop was blocked while Windows was asked"


def test_the_declared_rate_and_the_format_asked_of_windows_agree() -> None:
    """The SAPI format enumeration counts two entries per sample rate, so 22
    means 22 kHz and not 16. That mistake has already been made once here."""
    assert (SAMPLE_RATE, AUDIO_FORMAT) == (16_000, 18)


def test_sapi_is_a_tts_provider() -> None:
    tts = SapiTTS(engine=FakeSpeech())

    assert isinstance(tts, TTSProvider)
    assert tts.sample_rate == SAMPLE_RATE
