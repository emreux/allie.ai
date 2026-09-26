"""Live end-of-speech detection: where a sentence began and where it ended.

This is what turns an open microphone into turns. Push to talk asks the user to
say where the sentence ends by letting go of a key; here nobody is holding
anything, so the audio has to answer the question itself (design.md section
3.4, item 2.9).

**The model ships inside this package.** Silero VAD v6 (`MODEL_FILE`, MIT,
its notice in `silero_vad.LICENSE` beside it) - the very file `faster-whisper`
carried, copied over on 2026-09-26 when the local recogniser left and took
that package with it (plan.md D36). It runs on `onnxruntime`, which the wake
word needs anyway, so detection costs neither a dependency of its own nor a
download that can fail on a machine with no network. One session for the
whole program, on one thread: `audio/wake.py::_on_one_thread` measured what
ONNX Runtime's default spinning pool costs.

**It is driven frame by frame with the recurrent state kept, and that is the
whole difference.** A batch call builds fresh LSTM state and consumes a whole
array at once - exactly right for a finished recording, and wrong for a
microphone that never stops. The session takes `h` and `c` and hands back
`hn` and `cn`; keeping them between frames is what makes the answer about
*now* rather than about 32 milliseconds in isolation.

**Measured on this machine: 0.14 ms per 32 ms frame.** That is 0.4% of one core
to listen continuously, so rule 4 of section 3.1 is in no danger and the energy
pre-filter section 3.4 allows is not written: it would buy back a fraction of a
percent in exchange for a second threshold, and a threshold that is wrong cuts
quiet speech instead of saving power.

**The frames before the onset are kept.** A detector confident enough to fire is
already a few frames late, and those frames hold the first consonant of the
sentence. `PREROLL_SECONDS` is the ring buffer that gives them back.

**An utterance has a ceiling.** A microphone left open in a room with a
television would otherwise collect until the machine runs out of memory, and
then hand over an hour of it. At the ceiling what has been collected is
handed over and the detector goes back to waiting for a new sentence.
"""

from __future__ import annotations

import functools
from importlib import resources
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from allie.stt.base import SAMPLE_RATE, Audio

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "CONTEXT_SAMPLES",
    "FRAME_SAMPLES",
    "MAX_UTTERANCE_SECONDS",
    "MODEL_FILE",
    "ONSET_FRAMES",
    "PREROLL_SECONDS",
    "SILENCE_SECONDS",
    "SPEECH_THRESHOLD",
    "Endpoint",
    "Segmenter",
    "SileroVAD",
    "VoiceDetector",
]

# Silero VAD v6, beside this module (module docstring).
MODEL_FILE = "silero_vad_v6.onnx"

# What the model takes at 16 kHz: 512 new samples with the 64 before them for
# context. Both are the network's own shape, not a choice this project makes.
FRAME_SAMPLES = 512
CONTEXT_SAMPLES = 64

# Above this a frame counts as speech. Silero's own default, and the number
# phase 2.9 re-measures against real recordings (`scripts/bench_mic.py`).
SPEECH_THRESHOLD = 0.5

# How many frames in a row have to be speech before a sentence is declared
# started. One frame is 32 ms, so a door closing or a single key press does not
# open a turn - and nothing is lost by waiting, because the pre-roll below
# still holds those frames.
ONSET_FRAMES = 3

# How long a silence ends the sentence. Section 4 budgets 600 ms for phase 2
# and aims at 350 ms once it has been measured; lower than that and the pause
# in the middle of a thought becomes the end of a question.
SILENCE_SECONDS = 0.6

# How much of what came before the onset is kept. Three hundred milliseconds is
# longer than the detector's lag and shorter than a syllable of the sentence
# before it.
PREROLL_SECONDS = 0.3

# The ceiling. Long enough for any sentence anybody says to an assistant, short
# enough that nobody is handed minutes of a television programme.
MAX_UTTERANCE_SECONDS = 30.0


@runtime_checkable
class VoiceDetector(Protocol):
    """How likely it is that a frame of audio is somebody speaking."""

    # How many samples one frame is. `Endpoint` reads it rather than assuming,
    # so a detector with a different window needs no change anywhere else.
    frame_samples: int

    def probability(self, frame: Audio) -> float:
        """How likely `frame` is speech, given everything fed before it."""
        ...

    def reset(self) -> None:
        """Forgets what came before: a new stream, not a later part of this one."""
        ...


@runtime_checkable
class Segmenter(Protocol):
    """What `audio/capture.py` needs of this module, and nothing more.

    Here for the same reason `Hotkey` and `Microphone` are there: the tests for
    the capture are about which blocks reach the detector and when the state
    machine is told, not about where a threshold sits. Those belong to
    `Endpoint` and are tested against it directly.
    """

    @property
    def speaking(self) -> bool:
        """Whether a sentence is being collected right now."""
        ...

    def feed(self, chunk: Audio) -> list[Audio]:
        """Takes one block of audio; returns the sentences it ended."""
        ...

    def reset(self) -> None:
        """Throws away the sentence in progress and the stream behind it."""
        ...


class SileroVAD:
    """Silero's voice activity detector, run one frame at a time.

    Not thread safe, and it does not need to be: it is fed from the event loop
    only, in the order the microphone's blocks arrived.
    """

    frame_samples = FRAME_SAMPLES

    def __init__(self) -> None:
        self._session: Any = None
        self._h = _state()
        self._c = _state()
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    async def load(self) -> None:
        """Builds the inference session, off the event loop.

        Roughly a tenth of a second - under the eye but over the 50 ms of rule
        4, and paid at startup rather than inside the first sentence somebody
        says. `run` calls it at startup.
        """
        import asyncio

        await asyncio.to_thread(self._load)

    def probability(self, frame: Audio) -> float:
        if len(frame) != FRAME_SAMPLES:
            raise ValueError(f"a frame is {FRAME_SAMPLES} samples, not {len(frame)}")

        if self._session is None:
            self._load()

        # The context is the tail of the previous frame; the model is given
        # both as one row and hands back the state to carry into the next.
        window = np.concatenate((self._context, frame)).astype(np.float32, copy=False)
        speech, self._h, self._c = self._session.run(
            None, {"input": window.reshape(1, -1), "h": self._h, "c": self._c}
        )
        self._context = np.asarray(frame[-CONTEXT_SAMPLES:], dtype=np.float32).copy()
        return float(np.ravel(speech)[0])

    def reset(self) -> None:
        """Starts a new stream, keeping the session that took time to build.

        The state carries the room and the speaker, which is worth keeping
        between one sentence and the next. It is worth dropping when the audio
        itself was interrupted - the assistant spoke, or the mode was switched
        on - because the frames either side of the gap are not neighbours.
        """
        self._h = _state()
        self._c = _state()
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def _load(self) -> None:
        self._session = _session()


@functools.cache
def _session() -> Any:
    """The one inference session, built on first use and shared: the state
    lives in each detector, the weights do not need to.

    Deferred: `allie setup` has no use for an inference session, and
    importing onnxruntime takes a moment. The options are the ones the
    session was built with before D36 - one thread each way, no arena, no
    chatter on stderr.
    """
    import onnxruntime  # type: ignore[import-untyped]

    options = onnxruntime.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.log_severity_level = 4
    return onnxruntime.InferenceSession(
        str(resources.files("allie.audio") / MODEL_FILE),
        providers=["CPUExecutionProvider"],
        sess_options=options,
    )


class Endpoint:
    """Where each sentence starts and stops, in a stream that never stops.

    A pure state machine over probabilities: it holds no device, no thread and
    no clock, which is why it can be driven frame by frame in a test with no
    model behind it at all.
    """

    def __init__(
        self,
        detector: VoiceDetector | None = None,
        *,
        threshold: float = SPEECH_THRESHOLD,
        onset_frames: int = ONSET_FRAMES,
        silence_seconds: float = SILENCE_SECONDS,
        preroll_seconds: float = PREROLL_SECONDS,
        max_seconds: float = MAX_UTTERANCE_SECONDS,
    ) -> None:
        self._detector = detector if detector is not None else SileroVAD()
        self._threshold = threshold
        self._onset_frames = onset_frames

        frame = self._detector.frame_samples
        # Everything is counted in frames rather than in seconds: the frame is
        # the only clock this module has, and it is the one that matters.
        self._silence_frames = max(1, round(silence_seconds * SAMPLE_RATE / frame))
        self._preroll_frames = max(1, round(preroll_seconds * SAMPLE_RATE / frame))
        self._max_frames = max(1, round(max_seconds * SAMPLE_RATE / frame))

        self._pending = _empty()
        self._recent: list[Audio] = []
        self._take: list[Audio] = []
        self._speech_run = 0
        self._silent_run = 0
        self._speaking = False

    @property
    def speaking(self) -> bool:
        """Whether a sentence is being collected right now.

        `audio/capture.py` watches this for the moment it turns true: that is
        when the state machine is told somebody started talking, and it is the
        hands-free equivalent of the key going down.
        """
        return self._speaking

    def feed(self, chunk: Audio) -> list[Audio]:
        """Takes one block of microphone audio; returns the sentences it ended.

        A list, and almost always an empty one. Two sentences can only end in
        the same block if the block is longer than the silence that separates
        them, which a 20 ms block never is - but returning the one that
        happened to be last would be a silent way to lose a question.
        """
        self._pending = np.concatenate((self._pending, chunk)) if len(self._pending) else chunk

        finished: list[Audio] = []
        for frame in self._frames():
            ended = self._frame(frame)
            if ended is not None:
                finished.append(ended)
        return finished

    def reset(self) -> None:
        """Throws away the sentence in progress and the stream behind it."""
        self._pending = _empty()
        self._recent.clear()
        self._take.clear()
        self._speech_run = 0
        self._silent_run = 0
        self._speaking = False
        self._detector.reset()

    # ----------------------------------------------------------------------

    def _frames(self) -> Iterator[Audio]:
        """Whole frames out of the blocks that have arrived, and the rest kept.

        The microphone's block size and the model's frame size have no reason
        to agree, and making them agree would tie a capture constant to a
        network's input shape.
        """
        size = self._detector.frame_samples
        while len(self._pending) >= size:
            frame, self._pending = self._pending[:size], self._pending[size:]
            yield frame

    def _frame(self, frame: Audio) -> Audio | None:
        """One frame in; a finished utterance out, when this frame ended one."""
        speech = self._detector.probability(frame) >= self._threshold

        if not self._speaking:
            self._wait(frame, speech=speech)
            return None

        self._take.append(frame)
        self._silent_run = 0 if speech else self._silent_run + 1

        if self._silent_run >= self._silence_frames:
            return self._finish()
        if len(self._take) >= self._max_frames:
            # Not a sentence anybody finished - a room that has been making
            # noise for half a minute. What was collected is still handed over;
            # `app.py` decides whether it was worth answering.
            return self._finish()
        return None

    def _wait(self, frame: Audio, *, speech: bool) -> None:
        """Nobody is talking yet: keep the frame in case they just started."""
        self._recent.append(frame)
        if len(self._recent) > self._preroll_frames:
            self._recent.pop(0)

        self._speech_run = self._speech_run + 1 if speech else 0
        if self._speech_run < self._onset_frames:
            return

        # The sentence starts at the pre-roll, not at the frame that convinced
        # the detector: the frames it took to be convinced are the ones that
        # hold the beginning of the word.
        self._speaking = True
        self._silent_run = 0
        self._take = list(self._recent)
        self._recent = []

    def _finish(self) -> Audio:
        take = np.concatenate(self._take) if self._take else _empty()
        self._take = []
        self._speech_run = 0
        self._silent_run = 0
        self._speaking = False
        return take.astype(np.float32, copy=False)


def _empty() -> Audio:
    return np.empty(0, dtype=np.float32)


def _state() -> Any:
    """One of the two recurrent state tensors the session carries between frames."""
    return np.zeros((1, 1, 128), dtype=np.float32)
