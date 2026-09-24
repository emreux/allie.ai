"""The wake word (plan.md D21): the shipped model, the detector around it,
the chime, and where a model is looked for."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from allie.assistants import load_assistants
from allie.audio import wake as module
from allie.audio.wake import (
    CHIME_RATE,
    DEBOUNCE_SECONDS,
    HOP_SECONDS,
    WINDOW_SECONDS,
    BatchedModel,
    LiveKitWakeWord,
    WakeModelMissingError,
    chime,
    wake_model_path,
)
from allie.stt.base import SAMPLE_RATE, Audio

HOP = round(HOP_SECONDS * SAMPLE_RATE)
WINDOW = round(WINDOW_SECONDS * SAMPLE_RATE)
# One model per assistant (plan.md D31); each test below runs once per model.
MODELS = [assistant.wake for assistant in load_assistants().values()]


def shipped(name: str) -> Path:
    return Path(str(resources.files("allie.wake") / f"{name}.onnx"))


# --------------------------------------------------------------------------
# The shipped models (F6a)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_a_shipped_model_loads_and_hears_nothing_in_silence(name: str) -> None:
    """The tests that run the real classifier: two seconds of nothing must
    score under any threshold the owner would set."""
    wakeword = pytest.importorskip("livekit.wakeword")
    model = wakeword.WakeWordModel(models=[str(shipped(name))])
    silence = np.zeros(32_000, dtype=np.float32)

    assert model.predict(silence)[name] < 0.3


@pytest.mark.parametrize("name", MODELS)
def test_the_batched_path_scores_exactly_as_the_toolkit_s_predict(name: str) -> None:
    """`BatchedModel` runs the sixteen embeddings in one ONNX call (measured
    in F6a: 13 ms instead of 75) and must answer what `predict` answers."""
    wakeword = pytest.importorskip("livekit.wakeword")
    model = wakeword.WakeWordModel(models=[str(shipped(name))])
    window = (np.random.default_rng(0).standard_normal(WINDOW) * 0.02).astype(np.float32)

    theirs = model.predict(window)[name]
    ours = BatchedModel(model, name).predict(window)[name]

    assert ours == pytest.approx(theirs, abs=1e-5)


@pytest.mark.parametrize("name", MODELS)
def test_the_detector_runs_its_three_onnx_sessions_on_one_thread(name: str) -> None:
    """ONNX Runtime left to itself gives every session a pool of one thread
    per core, and the pool spins between calls: measured 2026-09-24 on the
    owner's four-core laptop at 2.4 cores - about 30 % of the machine and
    the fans at full - for a detector asked three times a second. On one
    thread each it was 1.4 %."""
    pytest.importorskip("livekit.wakeword")
    toolkit = LiveKitWakeWord(shipped(name), threshold=0.5)._built()._model

    sessions = [
        toolkit._mel_frontend._onnx_session,
        toolkit._speech_embedding._session,
        toolkit._classifiers[name][0],
    ]

    for session in sessions:
        options = session.get_session_options()
        assert options.intra_op_num_threads == 1
        assert options.inter_op_num_threads == 1


@pytest.mark.parametrize("name", MODELS)
def test_on_one_thread_the_detector_scores_as_the_toolkit_does(name: str) -> None:
    """Fewer threads is less CPU, not another answer: the threshold each
    model ships with was measured on the toolkit's own path."""
    wakeword = pytest.importorskip("livekit.wakeword")
    window = (np.random.default_rng(1).standard_normal(WINDOW) * 0.02).astype(np.float32)

    theirs = wakeword.WakeWordModel(models=[str(shipped(name))]).predict(window)[name]
    ours = LiveKitWakeWord(shipped(name), threshold=0.5)._built().predict(window)[name]

    assert ours == pytest.approx(theirs, abs=1e-5)


@pytest.mark.parametrize("name", MODELS)
def test_a_shipped_model_is_found_by_its_stem(name: str) -> None:
    path = shipped(name)

    assert wake_model_path(name) == path


# --------------------------------------------------------------------------
# The detector around a model
# --------------------------------------------------------------------------


class FakeModel:
    """`WakeWordModel` as the detector calls it: remembers every window
    and answers the scores it was given, in order."""

    def __init__(self, *scores: float) -> None:
        self.windows: list[Audio] = []
        self.scores = list(scores)

    def predict(self, window: Audio) -> dict[str, float]:
        self.windows.append(np.array(window, copy=True))
        return {"hey_friday": self.scores.pop(0) if self.scores else 0.0}


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def block(value: float, frames: int = 320) -> Audio:
    """One 20 ms block, as the capture sends them."""
    return np.full(frames, value, dtype=np.float32)


def detector(
    *scores: float, threshold: float = 0.5, clock: Clock | None = None
) -> tuple[LiveKitWakeWord, FakeModel, Clock]:
    model = FakeModel(*scores)
    ticking = clock if clock is not None else Clock()
    heard = LiveKitWakeWord(
        Path("x/hey_friday.onnx"), threshold=threshold, clock=ticking, model=model
    )
    return heard, model, ticking


def test_the_model_is_asked_once_per_hop_over_the_last_two_seconds() -> None:
    heard, model, _ = detector(0.1, 0.1)
    blocks = HOP // 320

    for index in range(blocks * 2):
        heard.feed(block(float(index + 1) / 100))

    assert len(model.windows) == 2
    assert len(model.windows[-1]) == WINDOW
    # The newest block is at the end of the window, the oldest samples first.
    assert model.windows[-1][-1] == pytest.approx(blocks * 2 / 100)
    assert model.windows[-1][0] == 0.0


def test_a_score_over_the_threshold_wakes_it_once_per_debounce() -> None:
    heard, _, clock = detector(0.9, 0.9, 0.9, threshold=0.5)
    blocks = HOP // 320

    def hop() -> bool:
        woke = False
        for _ in range(blocks):
            woke = heard.feed(block(0.2)) or woke
        return woke

    assert hop() is True
    assert heard.score == pytest.approx(0.9)
    assert hop() is False  # inside the debounce
    clock.now += DEBOUNCE_SECONDS
    assert hop() is True


def test_under_the_threshold_nothing_wakes() -> None:
    heard, _, _ = detector(0.49, threshold=0.5)

    assert not any(heard.feed(block(0.2)) for _ in range(HOP // 320))


def test_reset_forgets_the_window_and_the_score() -> None:
    heard, model, _ = detector(0.9, 0.1)
    for _ in range(HOP // 320):
        heard.feed(block(0.5))

    heard.reset()
    for _ in range(HOP // 320):
        heard.feed(block(0.0))

    assert heard.score == pytest.approx(0.1)
    assert float(np.abs(model.windows[-1]).max()) == 0.0


def test_the_detector_is_named_after_its_file() -> None:
    heard, _, _ = detector()

    assert heard.name == "hey_friday"


def test_the_chime_is_two_short_notes_at_the_speaker_s_rate() -> None:
    pcm = chime()
    samples = np.frombuffer(pcm, dtype="<i2")

    assert CHIME_RATE == 24_000
    assert 0.15 <= len(samples) / CHIME_RATE <= 0.25
    peak = float(np.abs(samples).max()) / 32767
    assert 0.15 <= peak <= 0.4  # around -12 dBFS, not a shock
    assert samples[0] == 0 and samples[-1] == 0  # faded, no click


def test_the_user_s_own_model_is_found_by_its_path(tmp_path: Path) -> None:
    own = tmp_path / "mine.onnx"
    own.write_bytes(b"\x00")

    assert wake_model_path(str(own)) == own


def test_a_missing_model_is_a_sentence_not_a_traceback(tmp_path: Path) -> None:
    with pytest.raises(WakeModelMissingError, match="hey_nobody"):
        wake_model_path("hey_nobody")
    with pytest.raises(WakeModelMissingError, match=r"nope\.onnx"):
        wake_model_path(str(tmp_path / "nope.onnx"))


async def test_load_builds_the_model_off_the_loop_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    built: list[tuple[list[str], int]] = []

    class Model:
        def __init__(self, *, models: list[str]) -> None:
            built.append((models, threading.get_ident()))

        def predict(self, window: Audio) -> dict[str, float]:
            return {"hey_friday": 0.0}

    monkeypatch.setattr(module, "_wake_word_model", lambda: Model)
    heard = LiveKitWakeWord(Path("a/hey_friday.onnx"), threshold=0.5)

    await heard.load()
    await heard.load()

    assert len(built) == 1
    assert built[0][0] == ["a\\hey_friday.onnx"] or built[0][0] == ["a/hey_friday.onnx"]
    assert built[0][1] != threading.get_ident()
