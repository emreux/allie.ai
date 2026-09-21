"""The wake word against the owner's own voice and room (plan.md D21).

The toolkit trains on synthetic speech and cannot take a recording as a
positive; what a recording can do is set the threshold. So: `record` a
clip at the pipeline's 16 kHz (say "hey Friday" once per clip, at the
desk, across the room, over music), `score` a folder of them and one long
recording of ordinary talk, and read off the threshold where every clip
wakes it and the long one does not. `bench` times one prediction over a
two-second window, both the toolkit's own `predict` (one embedding call
per window, sixteen calls) and the batched path `audio/wake.py` takes (one
call for all sixteen) - the number the detector's hop is sized against.

    uv run python scripts/wake_eval.py record clips/desk-01.wav --seconds 3
    uv run python scripts/wake_eval.py record noise/talk-10min.wav --seconds 600
    uv run python scripts/wake_eval.py score --positives clips --negative noise/talk-10min.wav
    uv run python scripts/wake_eval.py bench
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from importlib import resources
from pathlib import Path

import numpy as np

RATE = 16_000
WINDOW = 2 * RATE
HOP = 1280 * 4  # LiveKit's own frame is 1280 samples; four of them: 320 ms
DEBOUNCE_SECONDS = 2.0
SHIPPED = Path(str(resources.files("assistant.wake") / "hey_friday.onnx"))


def record(path: Path, seconds: float) -> None:
    import sounddevice  # type: ignore[import-untyped]

    print(f"recording {seconds:g} s into {path} ...")
    audio = sounddevice.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="int16")
    sounddevice.wait()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(audio.tobytes())
    print("done")


def read(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as clip:
        if clip.getframerate() != RATE or clip.getnchannels() != 1 or clip.getsampwidth() != 2:
            raise SystemExit(f"{path}: record it with this script (16 kHz, mono, 16-bit)")
        return np.frombuffer(clip.readframes(clip.getnframes()), dtype="<i2")


def scores(model: object, audio: np.ndarray, name: str) -> list[float]:
    """The score at every hop, the way the product will see it."""
    padded = np.concatenate([np.zeros(WINDOW, dtype="<i2"), audio])
    out: list[float] = []
    for end in range(WINDOW, len(padded) + 1, HOP):
        out.append(float(model.predict(padded[end - WINDOW : end])[name]))  # type: ignore[attr-defined]
    return out


def score(model_path: Path, positives: Path, negative: Path | None, threshold: float) -> None:
    from livekit.wakeword import WakeWordModel

    model = WakeWordModel(models=[str(model_path)])
    name = model_path.stem
    print(f"model {model_path.name}, threshold {threshold}")
    woke = 0
    clips = sorted(positives.glob("*.wav"))
    for clip in clips:
        peak = max(scores(model, read(clip), name), default=0.0)
        woke += peak >= threshold
        print(f"  {clip.name:28s} peak {peak:.3f} {'WAKE' if peak >= threshold else '-'}")
    if clips:
        print(f"recall {woke}/{len(clips)} = {100 * woke / len(clips):.0f} %")
    if negative is not None:
        audio = read(negative)
        hops = scores(model, audio, name)
        wakes, last = 0, -DEBOUNCE_SECONDS
        for index, value in enumerate(hops):
            at = index * HOP / RATE
            if value >= threshold and at - last >= DEBOUNCE_SECONDS:
                wakes += 1
                last = at
        hours = len(audio) / RATE / 3600
        print(
            f"false wakes in {negative.name}: {wakes} in {hours * 60:.1f} min "
            f"= {wakes / hours:.2f} per hour"
        )


def batched(model: object, window: np.ndarray, name: str) -> float:
    """`predict` with the sixteen embeddings in one call (what `audio/wake.py`
    does): the same mel frontend, the same windows, one ONNX run."""
    mel = model._mel_frontend(window.astype(np.float32) / 32768.0)  # type: ignore[attr-defined]
    embeddings = model._speech_embedding.extract_embeddings(mel)  # type: ignore[attr-defined]
    session, input_name = model._classifiers[name]  # type: ignore[attr-defined]
    sequence = embeddings[:, -16:, :].astype(np.float32)
    return float(session.run(None, {input_name: sequence})[0][0, 0])


def timed(call: object, rounds: int = 40) -> float:
    """Milliseconds per call, the best of three rounds: the machine is
    never idle and the worst round says more about it than about the model."""
    for _ in range(5):
        call()  # type: ignore[operator]
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        for _ in range(rounds):
            call()  # type: ignore[operator]
        best = min(best, (time.perf_counter() - started) / rounds * 1000)
    return best


def bench(model_path: Path) -> None:
    from livekit.wakeword import WakeWordModel

    model = WakeWordModel(models=[str(model_path)])
    name = model_path.stem
    window = (np.random.default_rng(0).standard_normal(WINDOW) * 300).astype("<i2")
    per_call = timed(lambda: model.predict(window))
    per_batched = timed(lambda: batched(model, window, name))
    print(f"toolkit predict over 2 s: {per_call:.1f} ms per call (sixteen embedding calls)")
    print(f"batched over 2 s:         {per_batched:.1f} ms per call (one embedding call)")
    print(
        f"at one call per {HOP / RATE * 1000:.0f} ms the batched path is "
        f"{100 * per_batched / (HOP / RATE * 1000):.1f} % of one core"
    )
    theirs, ours = model.predict(window)[name], batched(model, window, name)
    print(f"same score both ways: {theirs:.4f} / {ours:.4f}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=SHIPPED)
    commands = parser.add_subparsers(dest="command", required=True)
    rec = commands.add_parser("record")
    rec.add_argument("path", type=Path)
    rec.add_argument("--seconds", type=float, default=3.0)
    sc = commands.add_parser("score")
    sc.add_argument("--positives", type=Path, required=True)
    sc.add_argument("--negative", type=Path)
    sc.add_argument("--threshold", type=float, default=0.5)
    commands.add_parser("bench")
    args = parser.parse_args(argv)
    if args.command == "record":
        record(args.path, args.seconds)
    elif args.command == "score":
        score(args.model, args.positives, args.negative, args.threshold)
    else:
        bench(args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
