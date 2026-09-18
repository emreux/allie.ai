"""Bringing a microphone that will not run at 16 kHz to 16 kHz (audio/resample.py).

Measured 2026-09-05 on the target machine: the internal array's WASAPI entry
refuses 16 kHz in shared mode, and its kernel-streaming entry runs at 48 kHz or
not at all. Both are the same physical microphone as the MME entry that does
open at 16 kHz - Windows resamples on that path - and the whole point of
opening them was to hear the microphone *without* Windows in between. So the
conversion has to happen here, one 20 ms block at a time, on PortAudio's own
thread, and it has to be honest about two things a quick `pcm[::3]` is not:
what lies above 8 kHz is removed rather than folded into the speech band, and
the block boundaries leave no seam.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from assistant.audio.resample import Resampler
from assistant.stt.base import SAMPLE_RATE, Audio


def tone(hertz: float, rate: int, seconds: float = 1.0, amplitude: float = 0.5) -> Audio:
    moments = np.arange(int(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * hertz * moments)).astype(np.float32)


def blocks(pcm: Audio, size: int) -> Iterator[Audio]:
    for start in range(0, len(pcm), size):
        yield pcm[start : start + size]


def through(resampler: Resampler, pcm: Audio, size: int) -> Audio:
    """The whole signal, pushed through in blocks of `size` and joined back up."""
    return np.concatenate([resampler.push(block) for block in blocks(pcm, size)])


def loudest_frequency(pcm: Audio, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(pcm * np.hanning(len(pcm))))
    return float(np.fft.rfftfreq(len(pcm), 1 / rate)[int(np.argmax(spectrum))])


def rms(pcm: Audio) -> float:
    return float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64))))


SETTLED = SAMPLE_RATE // 10  # the filter's first few milliseconds are its ramp-in


# --------------------------------------------------------------------------
# 48 kHz down to 16 kHz: the internal array's raw paths
# --------------------------------------------------------------------------


def test_a_second_of_48_khz_is_a_second_of_16_khz() -> None:
    out = through(Resampler(48_000, SAMPLE_RATE), tone(440, 48_000), 960)

    assert abs(len(out) - SAMPLE_RATE) <= 1


def test_every_20_ms_block_in_is_a_20_ms_block_out() -> None:
    """The state machine's timing - the echo tail, the endpoint's silence
    window - is counted in samples, so a block must not shrink or swell."""
    resampler = Resampler(48_000, SAMPLE_RATE)

    lengths = [len(resampler.push(block)) for block in blocks(tone(440, 48_000), 960)]

    assert lengths == [320] * len(lengths)


def test_the_pitch_survives_the_way_down() -> None:
    out = through(Resampler(48_000, SAMPLE_RATE), tone(440, 48_000), 960)

    assert loudest_frequency(out[SETTLED:], SAMPLE_RATE) == pytest.approx(440, abs=2)


def test_the_level_survives_the_way_down() -> None:
    out = through(Resampler(48_000, SAMPLE_RATE), tone(440, 48_000), 960)

    assert float(np.abs(out[SETTLED:]).max()) == pytest.approx(0.5, rel=0.05)


def test_what_16_khz_cannot_hold_is_removed_rather_than_folded() -> None:
    """A 20 kHz tone has no place at 16 kHz. Taking every third sample would
    fold it to 4 kHz - in the middle of the speech band, as a whistle the
    recogniser hears and the user does not."""
    out = through(Resampler(48_000, SAMPLE_RATE), tone(20_000, 48_000), 960)

    assert rms(out[SETTLED:]) < rms(tone(20_000, 48_000)) * 0.02


def test_block_boundaries_leave_no_seam() -> None:
    """The filter's history and the fractional position carry across blocks:
    a stream resampled in 20 ms pieces is the stream resampled in one piece."""
    pcm = tone(440, 48_000)

    whole = Resampler(48_000, SAMPLE_RATE).push(pcm)
    pieces = through(Resampler(48_000, SAMPLE_RATE), pcm, 960)

    assert len(whole) == len(pieces)
    assert float(np.abs(whole - pieces).max()) < 1e-5


def test_the_output_is_the_format_the_recogniser_expects() -> None:
    out = Resampler(48_000, SAMPLE_RATE).push(tone(440, 48_000, seconds=0.02))

    assert out.dtype == np.float32
    assert out.ndim == 1


# --------------------------------------------------------------------------
# Rates that do not divide, and rates that are too slow
# --------------------------------------------------------------------------


def test_a_rate_that_does_not_divide_evenly_still_keeps_time() -> None:
    """44.1 kHz into 16 kHz is 441 samples in for 160 out. Rounding the
    position once per block would lose a sample every few blocks, and be a
    semitone sharp within a minute."""
    out = through(Resampler(44_100, SAMPLE_RATE), tone(440, 44_100, seconds=5.0), 882)

    assert abs(len(out) - 5 * SAMPLE_RATE) <= 2
    assert loudest_frequency(out[SAMPLE_RATE:], SAMPLE_RATE) == pytest.approx(440, abs=1)


def test_a_slow_device_is_brought_up_too() -> None:
    """A Bluetooth hands-free microphone offers 8 kHz and nothing else."""
    out = through(Resampler(8_000, SAMPLE_RATE), tone(440, 8_000), 160)

    assert abs(len(out) - SAMPLE_RATE) <= 1
    assert loudest_frequency(out[SETTLED:], SAMPLE_RATE) == pytest.approx(440, abs=2)


def test_the_same_rate_passes_straight_through() -> None:
    pcm = tone(440, SAMPLE_RATE, seconds=0.1)

    out = Resampler(SAMPLE_RATE, SAMPLE_RATE).push(pcm)

    assert np.array_equal(out, pcm)


def test_an_empty_block_is_an_empty_block() -> None:
    resampler = Resampler(48_000, SAMPLE_RATE)

    assert len(resampler.push(np.empty(0, dtype=np.float32))) == 0
