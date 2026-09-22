"""The wake word: asleep until called by name (plan.md D21, spec 2026-09-21
section 7).

Until now hands-free meant the doorman: any voice in the room opened a
session, and a session costs money from the moment it opens (D5, D12). The
owner wanted the film version instead - the assistant sleeps, "hey Friday"
wakes it, and only then is anyone listened to. This module is the ear that
stays open while it sleeps.

**A small model, on this machine, and nothing leaves.** `livekit-wakeword`
runs three ONNX graphs on the CPU: a mel spectrogram over the last two
seconds, Google's speech embedding over that, and the classifier trained
for the phrase (`allie/wake/hey_friday.onnx`, F6a). It answers with a
score from 0 to 1 and understands no words. The model is stateless - the
caller keeps the window - so `LiveKitWakeWord` keeps a two-second ring of
the microphone's blocks and asks once every `HOP_SECONDS` of new audio;
asking at every 20 ms block would cost sixteen times the CPU for the same
answer. The toolkit's own `predict` runs the sixteen embeddings of a window
as sixteen ONNX calls (~75 ms on the owner's machine, F6a); `BatchedModel`
runs them as one (~13 ms, bit-identical), through the toolkit's own
frontends, which is what keeps the sleeping assistant under a twentieth
of a core.

**What "heard" means.** A score at or over the threshold, and not within
`DEBOUNCE_SECONDS` of the last one - the phrase sits in the window for a
few hops and would otherwise wake the assistant three times. The threshold
is the owner's (`[wake] threshold`), set from their own recordings with
`scripts/wake_eval.py`.

**The chime** is the greeting of D21: two short notes at the speaker's
rate, generated here so that no file ships for it, faded at both ends so
that it does not click. A sentence in the local voice is the other choice
(`[wake] greeting`), the state machine's business.

Fed on the event loop, in order, like the doorman; the model's load is the
only slow part (three sessions, about a tenth of a second) and goes to a
worker thread.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from allie.stt.base import SAMPLE_RATE, Audio

__all__ = [
    "CHIME_RATE",
    "DEBOUNCE_SECONDS",
    "HOP_SECONDS",
    "WINDOW_SECONDS",
    "BatchedModel",
    "LiveKitWakeWord",
    "WakeDetector",
    "WakeModelMissingError",
    "chime",
    "wake_model_path",
]

# The model judges the last two seconds (16 embeddings of 80 ms, LiveKit's
# fixed input) and is asked every 320 ms of new audio (four of its 80 ms
# frames; measured in F6a at ~16 ms per batched call on this machine, a
# twentieth of a core).
WINDOW_SECONDS = 2.0
HOP_SECONDS = 0.32
# One wake per phrase: the phrase stays in the window for several hops.
DEBOUNCE_SECONDS = 2.0

# The classifier reads the last sixteen embeddings (the toolkit's
# MIN_EMBEDDINGS), each over 76 mel frames, eight frames apart.
_EMBEDDINGS = 16

# The chime: two notes a fifth apart, short, quiet, faded.
CHIME_RATE = 24_000
CHIME_NOTES: tuple[tuple[float, float], ...] = ((880.0, 0.09), (1320.0, 0.09))
CHIME_DBFS = -12.0
CHIME_FADE_SECONDS = 0.01


class WakeModelMissingError(RuntimeError):
    """`[wake] model` names a file that is not there. The user can fix it;
    named so that `run` says so in a sentence."""


class WakeDetector(Protocol):
    """What the capture needs while it sleeps."""

    name: str

    async def load(self) -> None: ...

    def feed(self, chunk: Audio) -> bool:
        """One block of the microphone; `True` on the block that completed the phrase."""
        ...

    def reset(self) -> None: ...


def wake_model_path(name_or_path: str) -> Path:
    """The model file: a stem names one this product ships (`allie/wake/`),
    a path ending in `.onnx` names the user's own."""
    given = Path(name_or_path)
    if given.suffix.casefold() == ".onnx":
        if not given.is_file():
            raise WakeModelMissingError(f"the wake-word model {given} is not there")
        return given
    shipped = Path(str(resources.files("allie.wake") / f"{name_or_path}.onnx"))
    if not shipped.is_file():
        raise WakeModelMissingError(
            f"no wake-word model called {name_or_path!r} ships with this program "
            f"(looked for {shipped.name} beside allie/wake/__init__.py)"
        )
    return shipped


def _wake_word_model() -> Any:
    """Deferred: `setup` has no use for it, and importing pulls in onnxruntime."""
    from livekit.wakeword import WakeWordModel  # type: ignore[import-untyped]

    return WakeWordModel


class BatchedModel:
    """The toolkit's model with its sixteen embeddings in one ONNX call.

    `WakeWordModel.predict` is correct and slow: one session run per
    embedding, sixteen per window. The frontends it owns take a batch, so
    this asks them once - the same mel, the same windows, the same
    classifier, one run (F6a: 12.9 ms against 71.7, max difference 0.0).
    It reaches into the toolkit's attributes to do so, pinned to 0.2.1;
    should a later version rename them, `predict` falls back to the
    toolkit's own path rather than fail - slower, never wrong.
    """

    def __init__(self, model: Any, name: str) -> None:
        self._model = model
        self._name = name

    def predict(self, window: Audio) -> dict[str, float]:
        mel_frontend = getattr(self._model, "_mel_frontend", None)
        embedding = getattr(self._model, "_speech_embedding", None)
        classifiers = getattr(self._model, "_classifiers", None)
        if mel_frontend is None or embedding is None or not isinstance(classifiers, dict):
            scores: dict[str, float] = self._model.predict(window)
            return scores
        session, input_name = classifiers[self._name]
        mel = mel_frontend(np.asarray(window, dtype=np.float32))
        embeddings = embedding.extract_embeddings(mel)
        if embeddings.shape[1] < _EMBEDDINGS:
            return {self._name: 0.0}
        sequence = embeddings[:, -_EMBEDDINGS:, :].astype(np.float32)
        return {self._name: float(session.run(None, {input_name: sequence})[0][0, 0])}


class LiveKitWakeWord:
    """The detector over one `livekit-wakeword` classifier."""

    def __init__(
        self,
        model_path: Path,
        *,
        threshold: float,
        hop_seconds: float = HOP_SECONDS,
        debounce_seconds: float = DEBOUNCE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        model: Any | None = None,
    ) -> None:
        self._path = model_path
        self.name = model_path.stem
        self._threshold = threshold
        self._hop = round(hop_seconds * SAMPLE_RATE)
        self._debounce = debounce_seconds
        self._clock = clock
        self._model = model
        self._ring = np.zeros(round(WINDOW_SECONDS * SAMPLE_RATE), dtype=np.float32)
        self._since_asked = 0
        self._last_wake = -math.inf
        # The last score, for the log and the tests.
        self.score = 0.0

    async def load(self) -> None:
        await asyncio.to_thread(self._load)

    def _load(self) -> None:
        if self._model is None:
            self._built()

    def _built(self) -> Any:
        """The model, built now if `load` was never awaited."""
        if self._model is None:
            self._model = BatchedModel(_wake_word_model()(models=[str(self._path)]), self.name)
        return self._model

    def feed(self, chunk: Audio) -> bool:
        count = len(chunk)
        if count >= len(self._ring):
            self._ring[:] = chunk[-len(self._ring) :]
        else:
            self._ring = np.roll(self._ring, -count)
            self._ring[-count:] = chunk
        self._since_asked += count
        if self._since_asked < self._hop:
            return False
        self._since_asked = 0
        model = self._model if self._model is not None else self._built()
        self.score = float(model.predict(self._ring).get(self.name, 0.0))
        now = self._clock()
        if self.score >= self._threshold and now - self._last_wake >= self._debounce:
            self._last_wake = now
            return True
        return False

    def reset(self) -> None:
        """A new stretch of listening: the window and the score are dropped,
        the model that took time to build is kept."""
        self._ring[:] = 0.0
        self._since_asked = 0
        self.score = 0.0


def chime() -> bytes:
    """The two notes, as 16-bit PCM at `CHIME_RATE`."""
    gain = 10 ** (CHIME_DBFS / 20)
    fade = round(CHIME_FADE_SECONDS * CHIME_RATE)
    pieces: list[np.ndarray[Any, Any]] = []
    for hertz, seconds in CHIME_NOTES:
        count = round(seconds * CHIME_RATE)
        t = np.arange(count) / CHIME_RATE
        tone = np.sin(2 * np.pi * hertz * t) * gain
        ramp = np.minimum(1.0, np.minimum(np.arange(count), count - 1 - np.arange(count)) / fade)
        pieces.append(tone * ramp)
    return (np.concatenate(pieces) * 32767).astype("<i2").tobytes()
