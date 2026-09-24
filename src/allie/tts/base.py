"""What the program's own voice is reduced to (plan.md D32).

The contract is one sentence long: **a voice yields 16-bit signed mono PCM,
little endian, at the rate it declares in `sample_rate`.** Raw PCM rather
than a compressed format, so whatever plays it never decodes anything.

What it says is short and already written: a gate's question and the hint
after it, a reminder, the filler while a tool takes its time, one of three
failure sentences (D3, D4, D10). Each item handed to `stream` is one whole
utterance - the pipeline's regrouping of a model's fragments into sentences
went with the pipeline, since nothing streams fragments here any more - and
the streaming shape is what lets `app.py` cut any of them off mid-word when
the user talks.

Since 2026-09-24 there is one voice and it is the assistant's: the live
model reads the sentences itself, in the voice the conversation speaks in
(`tts/live_voice.py`). Windows' voice and Google's separate synthesiser are
gone, and with them the choosing of a voice - it is `[live] voice`.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterable, Sequence
from typing import Protocol, runtime_checkable

__all__ = ["TTSProvider"]


@runtime_checkable
class TTSProvider(Protocol):
    """The program's own voice."""

    id: str

    # Of the PCM `stream` yields.
    sample_rate: int

    def stream(self, texts: Sequence[str]) -> AsyncGenerator[bytes]:
        """Says each of `texts` in turn, yielding PCM as it comes.

        Declared `def` rather than `async def` for the same reason as
        `LiveSession.events`: implementations are async generators. A
        generator rather than any iterator, so that whoever stops listening
        halfway can close it - and with it whatever it had open to speak.
        """
        ...

    async def prepare(self, texts: Iterable[str]) -> None:
        """Makes the sentences the program always says ready to be said at
        once and without a network. Never raises for a sentence it could
        not prepare: that one is read when it is needed."""
        ...
