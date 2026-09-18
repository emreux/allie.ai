"""What every speech engine is reduced to (design.md section 3.5).

The contract is one sentence long: **a provider yields 16-bit signed mono PCM,
little endian, at the rate it declares in `sample_rate`.** Windows speaks at
16 kHz, Azure at 24; both are asked for raw PCM rather than a compressed
format, so whatever plays the audio never has to decode anything.

Streaming is not optional here, and `sentences` below is why. The model writes
a reply in fragments over a second or two. Waiting for the last one before
making a sound would add the whole generation time to the latency of section
4; instead a sentence is handed to the engine the moment it is whole, and the
first one is usually speaking while the model is still writing the second.

Text normalisation - reading `25.08.2026` and `%14` the way a person would -
is phase 3.4 and lives in `tts/normalize.py` when it arrives. It belongs in
front of this module, not inside it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "MIN_SENTENCE_CHARS",
    "TERMINATORS",
    "TTSProvider",
    "VoiceInfo",
    "choose_voice",
    "sentences",
]

# What ends a sentence in the scripts this project has voices for. It is a
# fallback in code, the last link of the chain in section 3.12: a language
# whose script ends sentences differently - Greek with `;`, Devanagari with
# `|` - says so in its locale pack when item 1.8 gives it somewhere to say it.
TERMINATORS = ".!?…"

# A full stop after four characters is far more likely to be a title or an
# abbreviation than the end of a thought. Anything shorter than this is kept
# and read together with what follows, which is also what a person would do
# with "Dr. Mehmet Bey aradi."
MIN_SENTENCE_CHARS = 12


@dataclass(frozen=True, slots=True)
class VoiceInfo:
    """One voice the engine can read with, as the setup wizard lists them.

    `language` is ISO 639-1 without the region: the product locale of section
    3.12 is `tr`, and which of `tr-TR` or `tr-CY` a voice claims is not a
    choice anyone wants to make in a menu.
    """

    id: str
    display_name: str
    language: str


@runtime_checkable
class TTSProvider(Protocol):
    """A speech engine, local or hosted."""

    id: str

    # Of the PCM `stream` yields. Providers differ and none of them should be
    # resampled on the way to the speaker.
    sample_rate: int

    async def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        """The voices installed or offered, optionally for one language only."""
        ...

    def stream(self, chunks: AsyncIterator[str], *, voice: str) -> AsyncIterator[bytes]:
        """Speaks text as it arrives, yielding one buffer per sentence.

        Declared `def` rather than `async def` for the same reason as
        `LLMProvider.stream`: implementations are async generators.
        """
        ...


async def sentences(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """Regroups a stream of text fragments into whole sentences.

    Shared by every provider: the rule about when a sentence is safe to speak
    has nothing to do with which engine speaks it, and three copies of it
    would drift apart.
    """
    buffer = ""

    async for fragment in chunks:
        buffer += fragment
        while (cut := _end_of_sentence(buffer)) is not None:
            head, buffer = buffer[:cut], buffer[cut:]
            # Never blank: a cut is only offered once what precedes it holds
            # `MIN_SENTENCE_CHARS` of something other than space.
            yield head.strip()

    # Models end a turn without punctuation more often than one would like,
    # and the last sentence of one that does not is still worth speaking.
    if buffer.strip():
        yield buffer.strip()


def choose_voice(voices: Sequence[VoiceInfo], preferred: str | None) -> str:
    """Which of the installed voices to speak with, given the pack's preference.

    The pack names a preference rather than an identifier (item 1.8): `tr.toml`
    says `Tolga`, and what is installed is `Microsoft Tolga` under a registry
    path nobody would put in a TOML file. A preference that matches nothing is
    not an error - it is a machine where that voice was never installed.
    Shared by the state machine, which picks the voice an answer is read in,
    and by a hosted engine picking the local voice it falls back to.
    """
    if not voices:
        return ""

    if preferred:
        wanted = preferred.casefold()
        for voice in voices:
            if wanted in voice.display_name.casefold():
                return voice.id

    return voices[0].id


def _end_of_sentence(text: str) -> int | None:
    """Where the first sentence that is safe to speak ends, if there is one."""
    for index, character in enumerate(text):
        if character == "\n":
            ends_here = True
        elif character in TERMINATORS:
            # Only when something follows it: a full stop at the end of the
            # buffer may yet turn out to be `21.5`, and one inside a number
            # has a digit after it rather than a space.
            ends_here = index + 1 < len(text) and text[index + 1].isspace()
        else:
            continue

        if ends_here and len(text[: index + 1].strip()) >= MIN_SENTENCE_CHARS:
            return index + 1

    return None
