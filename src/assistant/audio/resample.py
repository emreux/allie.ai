"""One sample rate into another, one block at a time (used by `audio/capture.py`).

Everything downstream of the microphone counts in 16 kHz samples: the
detector's frames, the echo tail, the endpoint's silence window, the model
itself. Not every path to a microphone runs at 16 kHz. Measured 2026-09-05 on
the target machine: the internal array's MME entry accepts it because Windows
resamples in between, its WASAPI entry refuses it in shared mode, and its
kernel-streaming entry runs at 48 kHz or not at all. Those two rawer paths were
opened precisely to hear the microphone *without* Windows in between, so the
conversion happens here.

Two decisions.

**What lies above 8 kHz is removed before the rate drops, never folded.** A
signal at 16 kHz can hold nothing above 8 kHz; taking every third sample of a
48 kHz stream would fold whatever was up there into the speech band, as a
whistle the recogniser hears and the user does not. A windowed-sinc low-pass
runs first - a Hamming window, which reaches about 53 dB of attenuation with a
transition of 3.3 / taps cycles per sample.

**The state crosses block boundaries.** The filter keeps the tail of the
previous block, and where the next output sample falls is kept as a fraction
of an input sample. So 44.1 kHz, which is 441 samples in for every 160 out,
keeps time indefinitely instead of drifting a sample every few blocks - and a
stream resampled in 20 ms pieces is the stream resampled in one piece.

Pure `numpy`: a hundred taps over a 20 ms block is microseconds on PortAudio's
thread, and the alternative was a dependency for one function.
"""

from __future__ import annotations

import numpy as np

from assistant.stt.base import Audio

__all__ = ["Resampler"]

# Where the low-pass lets everything through and where it is gone, as fractions
# of the target rate. Half the rate is the most a signal can hold; the band
# between is the transition, and speech has little there to lose.
PASS_EDGE = 0.42
STOP_EDGE = 0.5

# A Hamming window needs about this many cycles-per-sample-taps of transition.
HAMMING_WIDTH = 3.3


class Resampler:
    """Streams audio from `source_rate` to `target_rate`, block by block."""

    def __init__(self, source_rate: int, target_rate: int) -> None:
        self.source_rate = source_rate
        self.target_rate = target_rate
        # Input samples per output sample: 3 for 48 kHz into 16, 0.5 for 8.
        self._step = source_rate / target_rate
        self._taps: Audio | None = _low_pass(source_rate, target_rate)
        self._history: Audio = np.zeros(
            0 if self._taps is None else len(self._taps) - 1, dtype=np.float32
        )
        # The last sample already seen, so that the first output of the next
        # block can be interpolated against it - and where the next output
        # falls, counted in input samples from that sample.
        self._last: Audio = np.zeros(1, dtype=np.float32)
        self._position = 1.0

    def push(self, block: Audio) -> Audio:
        """The block's worth of output: `len(block) / step` samples, give or take one."""
        if len(block) == 0:
            return np.empty(0, dtype=np.float32)

        signal: Audio = np.concatenate((self._last, self._filtered(block)))
        last = len(signal) - 1
        count = 0 if self._position > last else int((last - self._position) // self._step) + 1
        positions = self._position + self._step * np.arange(count)
        out: Audio = np.interp(positions, np.arange(len(signal)), signal).astype(np.float32)

        self._position = self._position + self._step * count - last
        self._last = signal[-1:]
        return out

    def _filtered(self, block: Audio) -> Audio:
        if self._taps is None:
            return np.asarray(block, dtype=np.float32)
        padded: Audio = np.concatenate((self._history, block))
        self._history = padded[-(len(self._taps) - 1) :]
        filtered: Audio = np.convolve(padded, self._taps, mode="valid").astype(np.float32)
        return filtered


def _low_pass(source_rate: int, target_rate: int) -> Audio | None:
    """The anti-aliasing filter for a drop in rate, or none for a rise."""
    if source_rate <= target_rate:
        return None

    pass_edge = PASS_EDGE * target_rate / source_rate
    stop_edge = STOP_EDGE * target_rate / source_rate
    half = int(np.ceil(HAMMING_WIDTH / (stop_edge - pass_edge) / 2))
    cutoff = (pass_edge + stop_edge) / 2

    offsets = np.arange(-half, half + 1)
    taps = 2 * cutoff * np.sinc(2 * cutoff * offsets) * np.hamming(2 * half + 1)
    normalised: Audio = (taps / taps.sum()).astype(np.float32)
    return normalised
