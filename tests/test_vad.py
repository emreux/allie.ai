"""Where a sentence starts and stops, when nobody is holding a key.

Two halves, tested two ways.

`Endpoint` is a state machine over probabilities and holds no model, so it is
driven here by a detector that says whatever the test tells it to. That is what
makes the rules readable: how long a silence ends a sentence, how much of the
run-up is kept, and what happens to a room that never stops talking are each
one short test rather than a recording somebody has to listen to.

`SileroVAD` is the real network. Three claims about it are worth the tenth of a
second it costs to load: that it can tell a voice from a silent room, that the
recurrent state really is carried from frame to frame - which is the whole
difference between this and the batch call `faster-whisper` offers - and that
the frame size is the one the model was built for rather than a plausible
looking number.
"""

from __future__ import annotations

import numpy as np
import pytest

from assistant.audio.vad import (
    CONTEXT_SAMPLES,
    FRAME_SAMPLES,
    Endpoint,
    SileroVAD,
    VoiceDetector,
)
from assistant.stt.base import SAMPLE_RATE, Audio

# A frame of four samples, so that a test can spell out a whole conversation as
# a list of probabilities. The real one is 512; nothing here depends on which.
FRAME = 4


class ScriptedVAD:
    """A detector that hears what the test says it hears, frame by frame."""

    frame_samples = FRAME

    def __init__(self, *probabilities: float) -> None:
        self.seen: list[Audio] = []
        self.resets = 0
        self._script = list(probabilities)

    def probability(self, frame: Audio) -> float:
        self.seen.append(frame)
        # Running out means the test said less than it fed; the room going
        # quiet is the least surprising thing for it to mean.
        return self._script.pop(0) if self._script else 0.0

    def reset(self) -> None:
        self.resets += 1


def seconds(frames: int) -> float:
    """`frames` frames, expressed the way `Endpoint` is configured."""
    return frames * FRAME / SAMPLE_RATE


def endpoint(
    *probabilities: float,
    onset: int = 1,
    silence: int = 2,
    preroll: int = 2,
    longest: int = 1000,
) -> tuple[Endpoint, ScriptedVAD]:
    detector = ScriptedVAD(*probabilities)
    return (
        Endpoint(
            detector,
            onset_frames=onset,
            silence_seconds=seconds(silence),
            preroll_seconds=seconds(preroll),
            max_seconds=seconds(longest),
        ),
        detector,
    )


def stream(frames: int) -> Audio:
    """Audio whose every frame says which frame it is."""
    return np.concatenate([np.full(FRAME, index, dtype=np.float32) for index in range(frames)])


def numbers(audio: Audio) -> list[int]:
    """Which frames of `stream` an utterance turned out to be made of."""
    return [int(audio[index]) for index in range(0, len(audio), FRAME)]


# --------------------------------------------------------------------------
# Where a sentence ends
# --------------------------------------------------------------------------


def test_a_sentence_is_handed_over_once_the_silence_after_it_is_long_enough() -> None:
    ends, _ = endpoint(0.0, 0.9, 0.9, 0.0, 0.0, silence=2, preroll=2)

    finished = ends.feed(stream(5))

    assert [numbers(one) for one in finished] == [[0, 1, 2, 3, 4]]


def test_a_pause_in_the_middle_of_a_sentence_does_not_end_it() -> None:
    """Somebody thinking is not somebody who has finished. One frame under the
    threshold has to be survivable or every sentence is cut in half."""
    ends, _ = endpoint(0.9, 0.0, 0.9, 0.9, silence=2)

    assert ends.feed(stream(4)) == []
    assert ends.speaking is True


def test_nothing_comes_back_while_nobody_is_talking() -> None:
    ends, _ = endpoint(0.0, 0.1, 0.0, 0.2)

    assert ends.feed(stream(4)) == []
    assert ends.speaking is False


def test_the_silence_has_to_be_the_length_it_was_asked_for() -> None:
    """One frame short of the threshold is still the middle of a sentence."""
    ends, _ = endpoint(0.9, 0.0, 0.0, silence=3, preroll=1)

    assert ends.feed(stream(3)) == []


# --------------------------------------------------------------------------
# Where a sentence starts
# --------------------------------------------------------------------------


def test_a_single_loud_frame_does_not_open_a_turn() -> None:
    """A door closing, a key pressed, a cough. Answering one costs four cores
    of Whisper and an API call."""
    ends, _ = endpoint(0.9, 0.0, 0.0, 0.0, onset=3)

    assert ends.feed(stream(4)) == []
    assert ends.speaking is False


def test_a_run_of_loud_frames_does_open_one() -> None:
    ends, _ = endpoint(0.9, 0.9, 0.9, onset=3)

    ends.feed(stream(3))

    assert ends.speaking is True


def test_the_run_has_to_be_unbroken() -> None:
    ends, _ = endpoint(0.9, 0.0, 0.9, 0.9, onset=3)

    ends.feed(stream(4))

    assert ends.speaking is False, "two runs of two are not a run of three"


def test_the_frames_before_the_detector_was_sure_are_kept() -> None:
    """A detector confident enough to fire is already late, and the frames it
    took to be convinced are the ones holding the first consonant."""
    ends, _ = endpoint(0.0, 0.9, 0.9, 0.9, 0.0, 0.0, onset=3, silence=2, preroll=3)

    finished = ends.feed(stream(6))

    # Frame 3 is the one that convinced it; 1 and 2 are the ones it took to be
    # convinced, and they are in the sentence rather than in front of it.
    assert [numbers(one) for one in finished] == [[1, 2, 3, 4, 5]]


def test_the_run_up_that_is_kept_has_a_length() -> None:
    """The alternative is a ring buffer that is not a ring: a microphone open
    all morning would hand Whisper the whole morning."""
    ends, _ = endpoint(0.0, 0.0, 0.0, 0.9, 0.0, 0.0, onset=1, silence=2, preroll=2)

    finished = ends.feed(stream(6))

    assert [numbers(one) for one in finished] == [[2, 3, 4, 5]], "frames 0 and 1 fell off"


# --------------------------------------------------------------------------
# The room that never stops
# --------------------------------------------------------------------------


def test_a_room_that_never_stops_is_cut_at_the_ceiling() -> None:
    """A television, and nobody in the chair. Without this the buffer grows
    until the machine runs out of memory."""
    ends, _ = endpoint(*[0.9] * 10, onset=1, silence=2, preroll=1, longest=4)

    finished = ends.feed(stream(10))

    assert [numbers(one) for one in finished] == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_what_the_ceiling_cut_still_has_to_be_started_again() -> None:
    """The cut is not a sentence anybody finished, so what follows it is a new
    one - and the onset rule gets to refuse it."""
    ends, _ = endpoint(0.9, 0.9, 0.0, 0.0, onset=2, silence=2, preroll=1, longest=2)

    ends.feed(stream(4))

    assert ends.speaking is False


# --------------------------------------------------------------------------
# Frames, blocks, and the difference
# --------------------------------------------------------------------------


def test_the_microphone_s_block_size_and_the_model_s_frame_size_need_not_agree() -> None:
    """Making them agree would tie a capture constant to a network's input
    shape, and one of the two would move."""
    ends, detector = endpoint(0.0, 0.9, 0.9, 0.0, 0.0, silence=2, preroll=2)
    audio = stream(5)

    finished: list[Audio] = []
    for start in range(0, len(audio), 3):  # blocks of three, frames of four
        finished += ends.feed(audio[start : start + 3])

    assert [numbers(one) for one in finished] == [[0, 1, 2, 3, 4]]
    assert all(len(frame) == FRAME for frame in detector.seen)


def test_a_block_that_holds_no_whole_frame_is_kept_for_the_next_one() -> None:
    ends, detector = endpoint(0.9)

    assert ends.feed(np.zeros(FRAME - 1, dtype=np.float32)) == []
    assert detector.seen == []


def test_the_audio_that_comes_back_is_what_the_speech_layer_expects() -> None:
    ends, _ = endpoint(0.9, 0.0, 0.0, silence=2, preroll=1)

    finished = ends.feed(stream(3))

    assert finished[0].dtype == np.float32


# --------------------------------------------------------------------------
# Starting over
# --------------------------------------------------------------------------


def test_speaking_says_whether_a_sentence_is_being_collected() -> None:
    """`audio/capture.py` watches this edge for the moment to tell the state
    machine that somebody started talking."""
    ends, _ = endpoint(0.0, 0.9, 0.0, 0.0, silence=2, preroll=1)

    before = ends.speaking
    ends.feed(stream(2))
    during = ends.speaking
    ends.feed(stream(4)[FRAME * 2 :])
    after = ends.speaking

    assert (before, during, after) == (False, True, False)


def test_resetting_throws_away_the_sentence_in_progress() -> None:
    """The assistant spoke, or the mode was switched on. Either way the frames
    either side of the gap are not neighbours."""
    ends, _ = endpoint(0.9, 0.0, 0.0, 0.0, silence=2, preroll=1)
    ends.feed(stream(1))

    ends.reset()
    finished = ends.feed(stream(4)[FRAME:])

    assert ends.speaking is False
    assert finished == [], "the frame collected before the interruption came back"


def test_resetting_tells_the_detector_as_well() -> None:
    """Its recurrent state is part of the stream that was interrupted."""
    ends, detector = endpoint()

    ends.reset()

    assert detector.resets == 1


def test_a_half_filled_block_does_not_survive_a_reset() -> None:
    ends, detector = endpoint(0.9)
    ends.feed(np.zeros(FRAME - 1, dtype=np.float32))

    ends.reset()
    ends.feed(np.zeros(1, dtype=np.float32))

    assert detector.seen == [], "the leftover was glued to audio from another stream"


# --------------------------------------------------------------------------
# The real detector
# --------------------------------------------------------------------------


def voiced(frames: int = 8) -> Audio:
    """Something with a voice's shape in it: a harmonic stack, not a tone."""
    time = np.arange(frames * FRAME_SAMPLES) / SAMPLE_RATE
    stack = sum(np.sin(2 * np.pi * 120 * n * time) / n for n in range(1, 20))
    return np.asarray(stack * 0.3, dtype=np.float32)


def test_the_detector_answers_the_protocol_the_endpoint_asks_for() -> None:
    assert isinstance(SileroVAD(), VoiceDetector)


def test_a_voice_and_a_silent_room_are_not_the_same_answer() -> None:
    """The one claim the whole mode rests on. Everything else here is about
    what happens once this is true."""
    speech, quiet = SileroVAD(), SileroVAD()
    sound = voiced()
    silence = np.zeros(len(sound), dtype=np.float32)

    loudest = max(
        speech.probability(sound[at : at + FRAME_SAMPLES])
        for at in range(0, len(sound), FRAME_SAMPLES)
    )
    emptiest = max(
        quiet.probability(silence[at : at + FRAME_SAMPLES])
        for at in range(0, len(silence), FRAME_SAMPLES)
    )

    assert loudest > 0.5 > emptiest


def test_what_came_before_a_frame_changes_the_answer_to_it() -> None:
    """The reason this module runs the session itself rather than calling
    `SileroVADModel`: that builds fresh state per call, which is right for a
    finished recording and wrong for a microphone that never stops."""
    detector = SileroVAD()
    frame = voiced(1)

    first = detector.probability(frame)
    second = detector.probability(frame)

    assert second != pytest.approx(first), "the state between frames is not being carried"


def test_resetting_the_detector_puts_the_stream_back_to_its_beginning() -> None:
    detector = SileroVAD()
    frame = voiced(1)
    first = detector.probability(frame)
    detector.probability(frame)

    detector.reset()

    assert detector.probability(frame) == pytest.approx(first)


def test_a_frame_is_the_size_the_model_was_built_for() -> None:
    """A plausible looking window is a silent bug: the session would refuse it,
    or worse, accept it and answer about the wrong span of time."""
    assert SileroVAD.frame_samples == FRAME_SAMPLES == 512
    assert CONTEXT_SAMPLES == 64

    with pytest.raises(ValueError, match="samples"):
        SileroVAD().probability(np.zeros(FRAME_SAMPLES - 1, dtype=np.float32))


def test_one_frame_costs_less_than_the_event_loop_can_spare() -> None:
    """Rule 4 of section 3.1 gives anything awaited in `app.py` 50 ms, and this
    runs fifty times a second while hands-free is on. Measured at 0.14 ms; the
    assertion is loose because a busy CI machine is not the target machine -
    what it catches is a change of order of magnitude."""
    import time

    detector = SileroVAD()
    frame = voiced(1)
    detector.probability(frame)  # the session is built on the first call

    start = time.perf_counter()
    for _ in range(20):
        detector.probability(frame)
    each = (time.perf_counter() - start) / 20

    assert each < 0.005, f"{each * 1000:.2f} ms a frame"
