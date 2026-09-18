"""Windows' own speech engine (design.md section 3.5, item 1.7).

The voice of phases 1 and 2, and the offline fallback for ever after: no key,
no network, no cost, installed on every Windows machine. It is not a pleasant
voice; Azure Neural arrives in phase 3.4 and this stays as the thing that
still works when the network does not.

**The Turkish voice lives in the other registry hive.** Windows keeps modern
voices under `Speech_OneCore` and `SAPI.SpVoice.GetVoices()` reads only
`Speech`, which is why `Microsoft Tolga` is installed on the owner's machine
and invisible to SAPI. The usual advice is to copy the registry key across,
which needs an administrator and edits a machine-wide setting. It is not
necessary: a token can be named by its full registry path, and both hives can
be enumerated through `SpObjectTokenCategory`. Measured on the target machine -
Tolga reads a Turkish sentence this way with nothing installed and nothing
changed.

Two traps sit behind the COM calls. Synthesis blocks, so it runs in a worker
thread (rule 4 of section 3.1); and a thread that has not initialised COM
cannot create a SAPI object at all, which is why every worker does so once,
the first time it is handed a sentence, and keeps the `SpVoice` it made for
the sentences after that. The pool has a handful of threads and an answer has
a handful of sentences; making a fresh engine for each was work for nothing.
"""

from __future__ import annotations

import asyncio
import locale
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any, Protocol

from assistant.tts.base import VoiceInfo, sentences

__all__ = ["AUDIO_FORMAT", "SAMPLE_RATE", "SapiTTS", "WindowsSpeech", "language_code"]

SAMPLE_RATE = 16_000

# SPSF_16kHz16BitMono. The enumeration counts two entries per sample rate - 8
# bit, then 16 bit - so 22 means 22 kHz and not 16. This project has already
# written the wrong file once by assuming otherwise.
AUDIO_FORMAT = 18

# Both places Windows keeps voices. The second is where every voice installed
# through Settings since Windows 10 has gone.
VOICE_CATEGORIES = (
    r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices",
    r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices",
)

# What each worker thread keeps between sentences: whether it has initialised
# COM, and the `SpVoice` it made. COM objects belong to the thread that made
# them, which is why this is thread-local rather than a module attribute.
_local = threading.local()


class SpeechEngine(Protocol):
    """The two things this provider needs Windows to do."""

    def voices(self) -> list[VoiceInfo]: ...

    def speak(self, text: str, voice: str) -> bytes: ...


class SapiTTS:
    """Speaks through Windows, one sentence at a time."""

    id = "sapi"
    sample_rate = SAMPLE_RATE

    def __init__(self, *, engine: SpeechEngine | None = None) -> None:
        self._engine = engine if engine is not None else WindowsSpeech()

    async def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        installed = await asyncio.to_thread(self._engine.voices)
        if language is None:
            return installed

        # `tr-TR` and `tr` are the same question; the region is nobody's choice.
        wanted = language.partition("-")[0].casefold()
        return [voice for voice in installed if voice.language == wanted]

    async def stream(self, chunks: AsyncIterator[str], *, voice: str) -> AsyncIterator[bytes]:
        async for sentence in sentences(chunks):
            yield await asyncio.to_thread(self._engine.speak, sentence, voice)


class WindowsSpeech:
    """The COM engine itself. Everything here runs in a worker thread."""

    def voices(self) -> list[VoiceInfo]:
        found: dict[str, VoiceInfo] = {}

        for category in VOICE_CATEGORIES:
            for token in _tokens(category):
                # A voice present in both hives is one voice; the first hive
                # wins because that is the one SAPI itself would have used.
                found.setdefault(
                    str(token.Id),
                    VoiceInfo(
                        id=str(token.Id),
                        display_name=str(token.GetDescription()),
                        language=language_code(_attribute(token, "Language")),
                    ),
                )

        return list(found.values())

    def speak(self, text: str, voice: str) -> bytes:
        # `pywin32` ships no type information, which is why every COM handle
        # in this file is `Any`.
        import win32com.client  # type: ignore[import-untyped]

        # This thread's own engine, first: making it initialises COM when this
        # is the thread's first sentence, and no other COM object can be made
        # before that.
        engine = _engine()

        # A fresh stream per sentence: it is the sentence's buffer.
        stream = win32com.client.Dispatch("SAPI.SpMemoryStream")
        stream.Format.Type = AUDIO_FORMAT

        engine.Voice = _token(voice)
        engine.AudioOutputStream = stream
        engine.Speak(text)

        return bytes(stream.GetData())


def language_code(attribute: str) -> str:
    """Turns a SAPI `Language` attribute into an ISO 639-1 code.

    The attribute is one or more Windows locale identifiers in hexadecimal,
    separated by semicolons. Python already ships the table that names them,
    which is the whole reason not to write one here: a hand-made list would be
    a language constant in code (section 3.12) and wrong for the two hundred
    locales nobody in this project will ever check.
    """
    first = attribute.split(";")[0].strip()
    if not first:
        return ""

    try:
        identifier = int(first, 16)
    except ValueError:
        return ""

    return locale.windows_locale.get(identifier, "").partition("_")[0]


def _tokens(category_id: str) -> Iterator[Any]:
    """Every voice token in one registry category, or none if it is absent."""
    import win32com.client

    _initialise_com()

    category = win32com.client.Dispatch("SAPI.SpObjectTokenCategory")
    try:
        category.SetId(category_id, False)
        tokens = list(category.EnumerateTokens())
    except Exception:
        # A hive that does not exist on this edition of Windows is not an
        # error: it is one of two places to look, and the other one answered.
        return

    yield from tokens


def _attribute(token: Any, name: str) -> str:
    """One attribute of a voice token, or empty if it does not declare it."""
    try:
        return str(token.GetAttribute(name))
    except Exception:
        # `GetAttribute` raises rather than returning nothing for an attribute
        # a voice never set, and a voice with no declared language is still a
        # voice somebody may want.
        return ""


def _token(voice_id: str) -> Any:
    """Opens one voice by its full registry path, whichever hive it is in."""
    import win32com.client

    handle = win32com.client.Dispatch("SAPI.SpObjectToken")
    handle.SetId(voice_id)
    return handle


def _initialise_com() -> None:
    """COM has to be initialised on the thread that uses it, once.

    `asyncio.to_thread` hands the work to whichever pool thread is free, and a
    thread that has not done this cannot create a SAPI object at all. Done
    once per thread and remembered: `CoInitialize` counts its calls and
    expects each to be paired with an uninitialise that nothing here makes.
    """
    if getattr(_local, "com_ready", False):
        return

    import pythoncom  # type: ignore[import-untyped]

    pythoncom.CoInitialize()
    _local.com_ready = True


def _engine() -> Any:
    """This thread's `SpVoice`, made on the first sentence it is handed."""
    import win32com.client

    _initialise_com()

    engine = getattr(_local, "engine", None)
    if engine is None:
        engine = win32com.client.Dispatch("SAPI.SpVoice")
        _local.engine = engine
    return engine
