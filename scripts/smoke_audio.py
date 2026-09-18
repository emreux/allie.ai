"""Records a few seconds from the microphone and plays them back (design.md section 8, phase 0.6).

This is the one phase 0 check a machine cannot make on its own: whether the
microphone picks you up and the speaker is audible. It records at the format
phase 1 uses - 16 kHz, mono - reports the signal level, then plays the take
back so you can hear it.

    uv run python scripts/smoke_audio.py
    uv run python scripts/smoke_audio.py --seconds 5 --keep take.wav

Nothing is written to disk unless `--keep` is given, and the audio never
leaves the machine.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

SAMPLE_RATE = 16_000
CHANNELS = 1
SILENCE_THRESHOLD = 0.01
CLIPPING_THRESHOLD = 0.99


def _describe_level(peak: float, rms: float) -> str:
    """Turns two numbers into the sentence you actually need."""
    if peak >= CLIPPING_THRESHOLD:
        return "clipping - move further from the microphone or lower the input gain"
    if peak < SILENCE_THRESHOLD:
        return "silence - wrong input device, muted microphone, or nothing was said"
    if rms < SILENCE_THRESHOLD:
        return "very quiet - speech may be lost; try speaking closer"
    return "healthy level"


def main() -> int:
    """Records, reports the level, plays back."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--keep", type=Path, default=None, help="Write the take to this WAV file.")
    arguments = parser.parse_args()

    import numpy
    import sounddevice

    source = sounddevice.query_devices(kind="input")
    sink = sounddevice.query_devices(kind="output")
    print(f"recording from : {source['name']}")
    print(f"playing back to: {sink['name']}")
    print(f"\nSay something - recording {arguments.seconds:.0f} seconds...")

    frames = int(arguments.seconds * SAMPLE_RATE)
    take = sounddevice.rec(frames, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32")
    sounddevice.wait()

    samples = take.reshape(-1)
    peak = float(numpy.max(numpy.abs(samples))) if samples.size else 0.0
    rms = float(math.sqrt(float(numpy.mean(numpy.square(samples))))) if samples.size else 0.0

    print(f"\npeak {peak:.3f}   rms {rms:.3f}   -> {_describe_level(peak, rms)}")

    if arguments.keep is not None:
        import soundfile

        soundfile.write(str(arguments.keep), take, SAMPLE_RATE)
        print(f"written: {arguments.keep}")

    print("\nPlaying it back...")
    sounddevice.play(take, SAMPLE_RATE)
    sounddevice.wait()
    time.sleep(0.2)

    print("\nDid you hear yourself? If yes, phase 0.6 is done.")
    return 0 if peak >= SILENCE_THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())
