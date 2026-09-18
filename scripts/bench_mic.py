"""What the microphone actually hears, and from how far (design.md item 2.9).

Hands-free listening rests on two things this machine has never been asked
about. Whether the microphone picks a sentence up from where the user sits -
the risk table calls insufficient range likely, and no amount of tuning fixes
a signal that is not there - and whether the assistant's own voice comes back
into the microphone loudly enough to be taken for a question.

    uv run python scripts/bench_mic.py --all      # all of it, in order
    uv run python scripts/bench_mic.py --at "2 m" # one distance
    uv run python scripts/bench_mic.py --quiet    # the room alone
    uv run python scripts/bench_mic.py --echo     # what the speakers put back
    uv run python scripts/bench_mic.py --list-devices   # which microphones there are
    uv run python scripts/bench_mic.py --fixtures       # the endpoint over the recordings (2.9)
    uv run python scripts/bench_mic.py --takes          # record the fixtures, a sentence at a time

`--all` is the one to run. It walks through the room, one distance, another
distance and the echo, waits for you between them, loads Whisper once, and
prints the four side by side at the end - which is the only way any of these
numbers mean anything.

`--device` picks the microphone the way `live-assistant run --device` does - words
from its name, or an index, defaulting to `[audio] input_device` in the
settings - so what is measured here is the microphone the assistant will
actually listen through. `--list-devices` prints the choices.

**A level is not an answer.** The detector saying "speech" and Whisper reading
the words are two different claims, and it is the second one that decides
whether the assistant is usable from across the room. So a take is transcribed
as well as measured, unless `--no-read` says otherwise, and the recording is
judged the way `app.py` judges one: by whether anything was said in it, not by
how sure the decoder was of the words.

**Nothing heard is not the same as nothing to hear.** `--echo` asks the
detector, not the average level: an answer is a few seconds of sound inside a
window that also holds the silence around it, and a mean over the whole window
buries the part that matters. If no frame reads as speech, the speakers were
not audible and the run measured nothing - saying so is the point, because a
silent take reported as a pass closes a risk that was never opened.

**The keypress is not part of the take.** Every recording starts a moment after
you have finished pressing anything: a room measured while somebody is typing
into it is not a room floor, and the first attempt at this measured one.

**The two numbers of hands-free are measured, not inherited** (`--fixtures`,
2.9). `SPEECH_THRESHOLD` and `SILENCE_SECONDS` in `audio/vad.py` were
Silero's defaults, left at that on 2026-08-31 with the note "not measured".
This mode runs the endpoint over the recorded fixtures of `fixtures/audio/`
- each of them one sentence - with a grid of thresholds and silence windows,
and says for each pair how many sentences came out whole, how many were cut
in two, and how many never came out at all. A second of silence is appended
to every recording, because a microphone goes on listening after the sentence
and a file does not. The constants change only when a pair beats them on
these numbers; otherwise the measurement is the reason they stay.

**The recordings come from here too** (`--takes`). ADR-001's table was taken
with a synthetic voice, and its accuracy column is a floor until the owner's
own sentences replace it. This mode records them the way `bench_stt.py` wants
them: you type the sentence first, exactly as you are about to say it, then
say it; the text file is written from what was typed and the recording is
cut where the detector last heard speech, plus the tail the endpoint would
keep. Through the same microphone `live-assistant run` opens - a fixture taken
through Windows' effects path measures a signal the assistant never hears.

Nothing is written to disk - except by `--takes`, which writes only into the
directory you name - and no audio leaves the machine.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from assistant.app import Heard, hear
from assistant.audio.capture import (
    ECHO_TAIL_SECONDS,
    MicrophoneUnavailableError,
    SystemMicrophone,
    device_choice,
)
from assistant.audio.vad import (
    FRAME_SAMPLES,
    SILENCE_SECONDS,
    SPEECH_THRESHOLD,
    Endpoint,
    SileroVAD,
)
from assistant.stt.base import NO_SPEECH_CEILING, SAMPLE_RATE, Audio, Transcript

if TYPE_CHECKING:
    from assistant.stt.local_whisper import LocalWhisper

SPOKEN = "Bir, iki, üç. Bu cümle mikrofonun ne duyduğunu ölçmek için okunuyor."

# Long enough for the keyboard to stop and the room to settle, short enough
# that nobody wonders whether the program has hung.
SETTLE_SECONDS = 1.0

# The grid `--fixtures` tries: the constants of `audio/vad.py` in the middle
# of each, one step either way. Section 4 aims at a 350 ms window one day.
THRESHOLDS = (0.4, 0.5, 0.6)
SILENCES = (0.4, 0.6, 0.8)

# What is appended to every fixture before the endpoint hears it: the
# silence a microphone would go on delivering after the sentence.
TAIL_SECONDS = 1.0


@dataclass
class Take:
    """One recording and everything measured about it."""

    label: str
    pcm: Audio
    peak: float = 0.0
    rms: float = 0.0
    loudest: float = 0.0
    speaking: int = 0
    frames: int = 0
    transcript: str = ""
    confidence: float | None = None
    no_speech: float | None = None
    read: bool = False
    heard: list[float] = field(default_factory=list)

    @property
    def verdict(self) -> Heard:
        """What `app.py` would have made of this take - the same function, not a copy."""
        return hear(
            Transcript(
                text=self.transcript,
                confidence=self.confidence,
                no_speech_probability=self.no_speech,
            )
        )

    @property
    def answered(self) -> bool:
        """Whether `app.py` would have answered this turn rather than dropping it."""
        return bool(self.verdict.text)


def record(seconds: float, *, device: int | str | None = None) -> Audio:
    """Everything the microphone hears for `seconds`, at the rate phase 1 uses."""
    blocks: list[Audio] = []
    microphone = SystemMicrophone(device=device)
    microphone.open(blocks.append)
    if microphone.rate != SAMPLE_RATE:
        # Worth knowing when reading the numbers: this path has no Windows
        # resampler in front of it, which is usually why it was chosen.
        print(f"  (the device runs at {microphone.rate} Hz; resampled to {SAMPLE_RATE} here)")
    try:
        time.sleep(seconds)
    finally:
        microphone.close()
    return np.concatenate(blocks) if blocks else np.empty(0, dtype=np.float32)


def loudness(pcm: Audio) -> tuple[float, float]:
    """Peak and RMS, the two numbers a level is ever described by."""
    if not len(pcm):
        return 0.0, 0.0
    return float(np.max(np.abs(pcm))), float(np.sqrt(np.mean(np.square(pcm))))


def probabilities(pcm: Audio) -> list[float]:
    """What the detector made of each 32 ms frame, in order."""
    detector = SileroVAD()
    return [
        detector.probability(pcm[at : at + FRAME_SAMPLES])
        for at in range(0, len(pcm) - FRAME_SAMPLES + 1, FRAME_SAMPLES)
    ]


def measure(pcm: Audio, *, label: str) -> Take:
    """One line per second, then what the detector made of the whole take."""
    peak, rms = loudness(pcm)
    heard = probabilities(pcm)
    take = Take(
        label=label,
        pcm=pcm,
        peak=peak,
        rms=rms,
        loudest=max(heard, default=0.0),
        speaking=sum(1 for one in heard if one >= SPEECH_THRESHOLD),
        frames=len(heard),
        heard=heard,
    )

    print(f"\n{label} - {len(pcm) / SAMPLE_RATE:.1f} s")
    print(f"{'second':>7}  {'peak':>7}  {'rms':>8}  {'loudest frame':>14}")
    for second in range(0, len(pcm), SAMPLE_RATE):
        window = pcm[second : second + SAMPLE_RATE]
        window_peak, window_rms = loudness(window)
        frames = probabilities(window)
        print(
            f"{second // SAMPLE_RATE:>7}  {window_peak:>7.3f}  "
            f"{window_rms:>8.5f}  {max(frames, default=0.0):>14.3f}"
        )

    print(f"\n  peak {take.peak:.3f}   rms {take.rms:.5f}")
    print(f"  frames called speech: {take.speaking} of {take.frames}")
    print(f"  loudest frame: {take.loudest:.3f} (threshold {SPEECH_THRESHOLD})")
    return take


def verdict(take: Take, *, quiet: bool) -> None:
    """What the levels mean for the detector. Not what they mean for Whisper."""
    if quiet:
        if take.loudest >= SPEECH_THRESHOLD:
            print("\n  The empty room already reads as speech. Hands-free would open")
            print("  turns nobody asked for; raise SPEECH_THRESHOLD or move the microphone.")
        else:
            print(f"\n  The empty room reads {take.loudest:.3f} against a threshold of")
            print(f"  {SPEECH_THRESHOLD}. Nothing here would open a turn.")
        return

    if take.speaking == 0:
        print("\n  Nothing in this take reads as speech. From this distance hands-free")
        print("  will not trigger at all - this is a microphone question, not a threshold one.")
    elif take.loudest < 0.8:
        print("\n  It triggers, but with no margin. Expect missed sentences from here.")
    else:
        print("\n  The detector is comfortable from this distance.")


async def whisper() -> LocalWhisper:
    """The recogniser, loaded once however many takes are read back."""
    from assistant.stt.local_whisper import LocalWhisper

    speech = LocalWhisper()
    await speech.load()
    return speech


async def read_back(take: Take, *, language: str, speech: LocalWhisper | None = None) -> None:
    """What Whisper makes of the take - the claim the levels cannot make.

    A signal the detector is sure about is not a signal the recogniser can
    read. Section 11's risk is "the microphone does not reach", and reaching
    means the words come back, not that something was loud enough to notice.
    """
    if speech is None:
        print("\n  Reading it back (Whisper is loading, this takes a few seconds)...")
        speech = await whisper()
    else:
        print("\n  Reading it back...")

    heard = await speech.transcribe(take.pcm, hint=language)
    take.transcript = heard.text.strip()
    take.confidence = heard.confidence
    take.no_speech = heard.no_speech_probability
    take.read = True

    confidence = "-" if heard.confidence is None else f"{heard.confidence:.2f}"
    no_speech = "-" if take.no_speech is None else f"{take.no_speech:.2f}"
    print(f'  transcript: "{take.transcript}"' if take.transcript else "  transcript: (nothing)")
    language_heard = heard.language or "-"
    print(f"  confidence: {confidence}   no-speech: {no_speech}   language: {language_heard}")

    # What `app.py` would do, said out loud rather than left for the reader to
    # work out. The decision is the engine's no-speech estimate against the
    # ceiling, never the confidence - that number is shown for the microphone
    # comparison, because a run of low ones is how a bad path is recognised.
    verdict = take.verdict
    if verdict.text:
        print("  The assistant would answer this turn.")
    elif verdict.missed:
        print("  Speech was heard and no words came of it: the assistant would say it")
        print("  did not understand. Not usable from here.")
    else:
        print(f"  The engine calls this silence (no-speech {no_speech} against the ceiling")
        print(f"  of {NO_SPEECH_CEILING}): the assistant would say nothing at all.")


def stt_language() -> str:
    """The language to expect, from the settings if there are any.

    The same chain `live-assistant run` uses. A script that hardcoded one would be
    the language constant section 3.12 exists to prevent.
    """
    from assistant import locales
    from assistant.config import is_configured, load_settings

    settings = load_settings()
    code = settings.locale.code if is_configured() else locales.system_code()
    return locales.load(code).stt_language


def configured_device() -> str:
    """The microphone `live-assistant run` would open, from the settings if any."""
    from assistant.config import load_settings

    return load_settings().audio.input_device


def ready(message: str) -> None:
    """Waits for the user, then lets the room go quiet before recording."""
    input(f"\n{message}\n  Press Enter when you are in place...")
    print(f"  Settling for {SETTLE_SECONDS:.0f} second, then recording. Go.")
    time.sleep(SETTLE_SECONDS)


async def echo(*, device: int | str | None = None, floor: Take | None = None) -> Take | None:
    """How much of the assistant's own voice comes back into the microphone.

    The tail is what `ECHO_TAIL_SECONDS` has to cover: the sound card holds
    part of the answer and the room holds the rest, and both arrive after the
    speaker has already been told it is finished.
    """
    from assistant.audio.player import SystemSpeaker
    from assistant.tts.sapi import SapiTTS

    tts = SapiTTS()
    voices = await tts.list_voices(None)
    if not voices:
        print("No speech voice is installed; there is nothing to echo.", file=sys.stderr)
        return None

    # The room before anything is said, as the control. Without it a take of
    # silence and a take with the speakers turned off are the same number.
    if floor is None:
        ready("The room first: say nothing, and do not type, for two seconds.")
        floor = measure(record(2.0, device=device), label="the room, as the control")
    if floor.loudest >= SPEECH_THRESHOLD:
        print("\n  Something was making a noise during the control. Nothing after it would")
        print("  mean anything - run it again and let those seconds pass in silence.")
        return None

    ready("Now the echo. Turn the speakers up to the volume you listen at, and say nothing.")
    blocks: list[Audio] = []
    microphone = SystemMicrophone(device=device)
    microphone.open(blocks.append)
    try:
        print(f"  Speaking, and listening to itself: {SPOKEN}")
        await SystemSpeaker().play(
            tts.stream(_one(SPOKEN), voice=voices[0].id), sample_rate=tts.sample_rate
        )
        spoken_until = sum(len(block) for block in blocks)
        # A second of room after the speaker fell silent: that is the part the
        # detector must not be shown.
        await asyncio.sleep(1.0)
    finally:
        microphone.close()

    pcm = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.float32)
    during = measure(pcm[:spoken_until], label="the assistant's own voice")

    # The detector is the instrument, not the average level. An answer is a few
    # seconds of sound inside a window that also holds the silence before and
    # after it, and a mean over the whole window buries exactly the part that
    # matters. What is being asked is what the detector would have done.
    if during.loudest < SPEECH_THRESHOLD:
        print(f"\n  The detector never heard the speakers: loudest frame {during.loudest:.3f}")
        print(f"  against a threshold of {SPEECH_THRESHOLD}. This run measured nothing.")
        print("  Check the speakers are on, audible, and not headphones, then run it again.")
        return during

    print(f"\n  The assistant's own voice reads {during.loudest:.3f} at its own microphone -")
    print("  as loud as a person in the room. This is why it is deafened while it speaks.")

    tail = measure(pcm[spoken_until:], label="after it stopped")
    over = [at for at, one in enumerate(tail.heard) if one >= SPEECH_THRESHOLD]
    lasted = (max(over) + 1) * FRAME_SAMPLES / SAMPLE_RATE if over else 0.0
    print(f"\n  The room read as speech for {lasted:.2f} s after the answer ended.")
    print(f"  ECHO_TAIL_SECONDS is {ECHO_TAIL_SECONDS}.")
    if lasted > ECHO_TAIL_SECONDS:
        print("  Too short: the assistant would hear the end of its own sentence.")
    else:
        print("  Long enough on this machine, at this volume.")
    return during


def summary(takes: list[Take]) -> None:
    """The takes side by side, which is the only way any of them mean anything."""
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'take':<28} {'peak':>7} {'rms':>9} {'loudest':>8} {'speech':>9} {'answered':>9}")
    for take in takes:
        share = f"{take.speaking}/{take.frames}"
        answered = "-" if not take.read else ("yes" if take.answered else "NO")
        print(
            f"{take.label:<28} {take.peak:>7.3f} {take.rms:>9.5f} "
            f"{take.loudest:>8.3f} {share:>9} {answered:>9}"
        )

    read = [take for take in takes if take.read]
    if read:
        print(f"\nWhat was read out: {SPOKEN}")
        print(f"  {'':<26} {'conf':>5} {'nsp':>5}")
        for take in read:
            confidence = "-" if take.confidence is None else f"{take.confidence:.2f}"
            no_speech = "-" if take.no_speech is None else f"{take.no_speech:.2f}"
            print(f'  {take.label:<26} {confidence:>5} {no_speech:>5}  "{take.transcript}"')

    print(f"\nThresholds: detector {SPEECH_THRESHOLD}, no-speech ceiling {NO_SPEECH_CEILING}.")
    print("A take answered NO is one app.py would not have sent to the model: silence")
    print("gets nothing back, speech that made no words gets 'I did not catch that'.")


async def guided(*, seconds: float, device: int | str | None, no_read: bool) -> None:
    """The room, two distances and the echo, in one sitting."""
    language = stt_language()
    takes: list[Take] = []

    ready("First the room: say nothing, and do not type, for two seconds.")
    floor = measure(record(2.0, device=device), label="the room, nobody talking")
    verdict(floor, quiet=True)
    takes.append(floor)

    # Loaded once, before the first distance, so that the pause between the two
    # takes is the user walking rather than a gigabyte of weights.
    speech = None
    if not no_read:
        print("\nLoading Whisper once, for both distances...")
        speech = await whisper()

    for distance in ("1 m", "2 m"):
        ready(
            f"Now from {distance}. Read this out, exactly as written, for about "
            f"{seconds:.0f} seconds:\n\n    {SPOKEN}\n"
        )
        take = measure(record(seconds, device=device), label=f"speaking, {distance}")
        verdict(take, quiet=False)
        if speech is not None:
            await read_back(take, language=language, speech=speech)
        takes.append(take)

    heard = await echo(device=device, floor=floor)
    if heard is not None:
        takes.append(heard)

    summary(takes)


async def _one(said: str) -> AsyncIterator[str]:
    yield said


# --------------------------------------------------------------------------
# The endpoint over the recordings (2.9)
# --------------------------------------------------------------------------


@dataclass
class Split:
    """What the endpoint made of one fixture at one setting: the sentences
    it handed over, as (start, end) seconds within the recording."""

    name: str
    seconds: float
    spans: list[tuple[float, float]] = field(default_factory=list)

    @property
    def whole(self) -> bool:
        return len(self.spans) == 1

    @property
    def kept(self) -> float:
        """How much of the recording came out inside a sentence."""
        return min(1.0, sum(end - start for start, end in self.spans) / self.seconds)


def split(pcm: Audio, *, endpoint: Endpoint, silence: float, name: str) -> Split:
    """Feeds one recording to the endpoint the way the microphone would -
    20 ms blocks, then the silence after it - and notes what came out."""
    from assistant.audio.capture import CHUNK_FRAMES

    endpoint.reset()
    audio = np.concatenate((pcm, np.zeros(round(TAIL_SECONDS * SAMPLE_RATE), dtype=np.float32)))
    result = Split(name=name, seconds=len(pcm) / SAMPLE_RATE)
    fed = 0
    for start in range(0, len(audio), CHUNK_FRAMES):
        block = audio[start : start + CHUNK_FRAMES]
        fed += len(block)
        for sentence in endpoint.feed(block):
            # The sentence closed once the silence after it was long enough,
            # so its end is that long before the block that closed it.
            end = fed / SAMPLE_RATE - silence
            result.spans.append((max(0.0, end - len(sentence) / SAMPLE_RATE), end))
    return result


# What `--takes` keeps after the last frame the detector called speech: the
# window the endpoint waits before it hands a sentence over, so that a
# fixture ends where a live recording would.
TAKE_TAIL_SECONDS = SILENCE_SECONDS


def trimmed(pcm: Audio) -> Audio | None:
    """The take up to its last speech frame plus the tail, or `None` when no
    frame read as speech - a fixture with nothing in it measures nothing."""
    spoken = [at for at, p in enumerate(probabilities(pcm)) if p >= SPEECH_THRESHOLD]
    if not spoken:
        return None
    end = (spoken[-1] + 1) * FRAME_SAMPLES + int(TAKE_TAIL_SECONDS * SAMPLE_RATE)
    return pcm[:end]


def slug(said: str) -> str:
    """A file stem from the first words of the sentence: folded, ASCII, short."""
    from assistant.store.normalize import normalize_search

    words = re.sub(r"[^a-z0-9]+", " ", normalize_search(said)).split()
    return "-".join(words[:3]) or "take"


def next_number(directory: Path) -> int:
    """One past the highest `NN-` prefix already in the directory."""
    numbers = [
        int(path.stem[:2])
        for path in directory.iterdir()
        if path.stem[:2].isdigit() and path.stem[2:3] == "-"
    ]
    return max(numbers, default=0) + 1


def takes_mode(directory: Path, *, seconds: float, device: int | str | None) -> int:
    """Records fixtures until an empty line: typed sentence, then spoken."""
    import soundfile  # type: ignore[import-untyped]

    directory.mkdir(parents=True, exist_ok=True)
    print(f"Recording into {directory}. Each take is {seconds:.0f} s; speak once, then wait.")
    print("Type each sentence exactly as you will say it - not corrected afterwards.")
    written = 0
    while True:
        said = input("\nSentence to record (empty line to finish): ").strip()
        if not said:
            break
        stem = f"{next_number(directory):02d}-{slug(said)}"
        ready(f"Say: {said}")
        pcm = trimmed(record(seconds, device=device))
        if pcm is None:
            print("  Nothing in that take reads as speech - not saved. Try it again.")
            continue
        measure(pcm, label=stem)
        print(f"  {len(pcm) / SAMPLE_RATE:.1f} s kept")
        soundfile.write(str(directory / f"{stem}.wav"), pcm, SAMPLE_RATE, subtype="PCM_16")
        (directory / f"{stem}.txt").write_text(said + "\n", encoding="utf-8")
        written += 1
        print(f"  written: {stem}.wav and {stem}.txt")
    print(f"\n{written} recordings written.")
    print("Measure them: uv run python scripts/bench_stt.py --sizes small")
    return 0


def fixtures_mode(directory: Path) -> int:
    """The grid over every recording, and where the current pair stands."""
    from bench_stt import load_fixtures

    fixtures = load_fixtures(directory)
    if not fixtures:
        print(f"nothing to measure: no .wav with a .txt beside it in {directory}")
        return 2

    detector = SileroVAD()
    print(f"{len(fixtures)} recordings, {TAIL_SECONDS:.0f} s of silence appended to each\n")
    print(f"{'threshold':>9} {'silence':>8} {'whole':>6} {'split':>6} {'none':>5} {'kept':>6}")
    current: list[Split] = []
    for threshold in THRESHOLDS:
        for silence in SILENCES:
            endpoint = Endpoint(detector, threshold=threshold, silence_seconds=silence)
            splits = [
                split(fixture.pcm, endpoint=endpoint, silence=silence, name=fixture.wav.name)
                for fixture in fixtures
            ]
            whole = sum(1 for one in splits if one.whole)
            none = sum(1 for one in splits if not one.spans)
            cut = len(splits) - whole - none
            kept = sum(one.kept for one in splits) / len(splits)
            current_pair = (threshold, silence) == (SPEECH_THRESHOLD, SILENCE_SECONDS)
            marker = "  <- audio/vad.py" if current_pair else ""
            print(
                f"{threshold:>9.1f} {silence:>7.1f}s {whole:>6} {cut:>6} {none:>5} "
                f"{kept:>6.0%}{marker}"
            )
            if marker:
                current = splits

    print(f"\nAt the current pair ({SPEECH_THRESHOLD}, {SILENCE_SECONDS} s), sentence by sentence:")
    for one in current:
        spans = ", ".join(f"{start:.2f}-{end:.2f} s" for start, end in one.spans)
        spans = spans or "nothing came out"
        print(f"  {one.name:<22} {one.seconds:>4.1f} s  ->  {spans}")

    print(
        "\nA pair is better than the current one only if it turns more recordings into"
        " exactly one sentence, and no fewer into none. The silence window is also the"
        " wait after every sentence (section 4): shorter is felt, longer is safe."
    )
    return 0


def main() -> int:
    # The transcript below is in whatever language was spoken, and Windows
    # hands a redirected stream its legacy code page - which has no `ğ` in it.
    # The same fix `live-assistant run` makes, through the same function.
    from assistant.__main__ import use_utf8

    use_utf8(sys.stdout)
    use_utf8(sys.stderr)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="Room, 1 m, 2 m and echo, in order.")
    parser.add_argument("--seconds", type=float, default=6.0, help="How long to record.")
    parser.add_argument("--at", default="", help="What to call this take, e.g. '2 m'.")
    parser.add_argument("--quiet", action="store_true", help="Measure the room, saying nothing.")
    parser.add_argument("--echo", action="store_true", help="Measure what the speakers put back.")
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Input device: an index, or words from its name as --list-devices prints them. "
            "Default: [audio] input_device from the settings, else the system default."
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print the devices sounddevice can see, with their indices, and exit.",
    )
    parser.add_argument(
        "--no-read", action="store_true", help="Skip the transcript, and measure levels only."
    )
    parser.add_argument(
        "--fixtures",
        nargs="?",
        const="",
        default=None,
        metavar="DIR",
        help=(
            "Run the endpoint over the recorded fixtures with a grid of thresholds and "
            "silence windows (2.9); DIR defaults to fixtures/audio."
        ),
    )
    parser.add_argument(
        "--takes",
        nargs="?",
        const="",
        default=None,
        metavar="DIR",
        help=(
            "Record fixtures for bench_stt.py, one typed-then-spoken sentence at a time, "
            "through the configured microphone; DIR defaults to fixtures/audio."
        ),
    )
    args = parser.parse_args()

    if args.fixtures is not None:
        from bench_stt import FIXTURES

        return fixtures_mode(Path(args.fixtures) if args.fixtures else FIXTURES)

    if args.list_devices:
        import sounddevice  # type: ignore[import-untyped]

        print(sounddevice.query_devices())
        print()
        print('Name a device by words from its line, e.g. --device "Microphone Array WASAPI",')
        print("or by its index. Words survive a Bluetooth headset connecting; an index does not.")
        return 0

    # The same choice `live-assistant run` makes, so that what is measured here is
    # the microphone the assistant will actually listen through.
    device = device_choice(args.device if args.device is not None else configured_device())

    try:
        if args.takes is not None:
            from bench_stt import FIXTURES

            return takes_mode(
                Path(args.takes) if args.takes else FIXTURES, seconds=args.seconds, device=device
            )
        return _measure(args, device)
    except MicrophoneUnavailableError as problem:
        # The same sentence `live-assistant run` would print, for the same reason:
        # a name that matched nothing, or a device that would not open, is the
        # user's to fix, and a traceback says nothing about how.
        print(f"Cannot record: {problem}")
        return 1


def _measure(args: argparse.Namespace, device: int | str | None) -> int:
    if args.all:
        asyncio.run(guided(seconds=args.seconds, device=device, no_read=args.no_read))
        return 0

    if args.echo:
        asyncio.run(echo(device=device))
        return 0

    label = args.at or ("the room, with nobody talking" if args.quiet else "speaking")
    if args.quiet:
        print(f"Say nothing for {args.seconds:.0f} seconds.")
    else:
        print(f"Speak normally for {args.seconds:.0f} seconds, from where you would sit.")
        print(f"Read this out: {SPOKEN}")

    take = measure(record(args.seconds, device=device), label=label)
    verdict(take, quiet=args.quiet)

    # The levels answered the detector's question. Whisper answers the one the
    # risk table actually asks, and only a take with words in it has one.
    if not args.quiet and not args.no_read:
        asyncio.run(read_back(take, language=stt_language()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
