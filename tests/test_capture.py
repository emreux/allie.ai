"""Being listened to: what reaches the detector, and what does not.

Two threads that are not the event loop reach into this module - the keyboard
listener and PortAudio's callback - so the tests drive it the same way, and
three specific traps are pinned:

* audio that arrives while listening is off belongs to nobody,
* the buffer PortAudio hands over is reused, so it has to be copied,
* the stream is opened at the rate the speech layer declares, not a plausible
  looking one. That last mistake has already been made once in this project.

No microphone and no keyboard are touched here. The devices are behind two
small interfaces; the real ones are exercised through a fake `sounddevice`
module, which is also what proves the import stays deferred.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from loguru import logger

from assistant.audio.capture import (
    CHUNK_FRAMES,
    DEFAULT_TOGGLE_HOTKEY,
    ECHO_TAIL_SECONDS,
    QUIET_DBFS,
    HandsFree,
    KeyCombination,
    LiveCapture,
    MicrophoneInfo,
    MicrophoneUnavailableError,
    SystemHotkey,
    SystemMicrophone,
    available_microphones,
    device_choice,
    duplex_for,
)
from assistant.audio.vad import PREROLL_SECONDS
from assistant.stt.base import LEVEL_FLOOR_DBFS, SAMPLE_RATE, Audio, to_pcm16

CTRL, ALT, SPACE, SHIFT = "ctrl", "alt", "space", "shift"


def combination() -> KeyCombination:
    return KeyCombination(frozenset({CTRL, ALT, SPACE}))


def tone(value: float, frames: int = 4) -> Audio:
    return np.full(frames, value, dtype=np.float32)


class FakeHotkey:
    """A key combination nobody has to press."""

    def __init__(self) -> None:
        self.watching = False
        self._on_press: Any = None
        self._on_release: Any = None

    def watch(self, *, on_press: Any, on_release: Any) -> None:
        self._on_press, self._on_release = on_press, on_release
        self.watching = True

    def stop(self) -> None:
        self.watching = False

    def press(self) -> None:
        self._on_press()

    def release(self) -> None:
        self._on_release()


class FakeMicrophone:
    """An input stream that hears exactly what a test tells it to."""

    def __init__(self, host_api: str = "Windows WASAPI") -> None:
        self.opened = False
        self.host_api = host_api
        self._on_chunk: Any = None

    def open(self, on_chunk: Any) -> None:
        self._on_chunk = on_chunk
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def hear(self, chunk: Audio) -> None:
        self._on_chunk(chunk)


def fake_sounddevice(
    *,
    host_api: str = "MME",
    native_rate: int = SAMPLE_RATE,
    accepts: set[int] | None = None,
    devices: list[tuple[str, str, int]] | None = None,
) -> tuple[SimpleNamespace, list[SimpleNamespace]]:
    """Stands in for the module, and records how the stream was opened.

    One input device, behind `host_api`, whose own rate is `native_rate`.
    `accepts` is the set of rates it opens at - `None` for any, the way MME
    behaves; `{48_000}` for a device that refuses 16 kHz, the way WASAPI and
    kernel streaming did on 2026-09-05. A WASAPI device told to convert opens
    at any rate. A device called `nope` is refused the way `sounddevice`
    refuses a name that matches nothing: a `ValueError` that quotes the name.

    `devices` is what `query_devices()` lists when asked for everything, as
    (name, host API, input channels) - the shape of PortAudio's table, where
    one microphone appears once per host API and outputs sit in between.
    """
    streams: list[SimpleNamespace] = []
    hosts = sorted({host_api, *(host for _, host, _ in devices or [])})

    class PortAudioError(Exception):
        pass

    class WasapiSettings:
        def __init__(self, *, auto_convert: bool = False) -> None:
            self.auto_convert = auto_convert

    def query_devices(device: Any = None, kind: str | None = None) -> Any:
        if device == "nope":
            raise ValueError("No input device matching 'nope'")
        if device is None and kind is None:
            return [
                {
                    "index": index,
                    "name": name,
                    "hostapi": hosts.index(host),
                    "max_input_channels": inputs,
                    "default_samplerate": float(native_rate),
                }
                for index, (name, host, inputs) in enumerate(devices or [])
            ]
        return {
            "index": 0,
            "name": "Microphone Array",
            "hostapi": hosts.index(host_api),
            "max_input_channels": 2,
            "default_samplerate": float(native_rate),
        }

    def query_hostapis(index: int | None = None) -> dict[str, Any]:
        return {"name": host_api if index is None else hosts[index]}

    def input_stream(**options: Any) -> SimpleNamespace:
        if options.get("device") == "nope":
            raise ValueError("No input device matching 'nope'")
        extra = options.get("extra_settings")
        converts = isinstance(extra, WasapiSettings) and extra.auto_convert
        if accepts is not None and options["samplerate"] not in accepts and not converts:
            raise PortAudioError("Error opening InputStream: Invalid sample rate", -9997)
        stream = SimpleNamespace(options=options, started=False, stopped=False, closed=False)
        stream.start = lambda: setattr(stream, "started", True)
        stream.stop = lambda: setattr(stream, "stopped", True)
        stream.close = lambda: setattr(stream, "closed", True)
        streams.append(stream)
        return stream

    module = SimpleNamespace(
        InputStream=input_stream,
        PortAudioError=PortAudioError,
        WasapiSettings=WasapiSettings,
        query_devices=query_devices,
        query_hostapis=query_hostapis,
    )
    return module, streams


# --------------------------------------------------------------------------
# The key combination - no keyboard involved
# --------------------------------------------------------------------------


def test_the_combination_fires_once_all_of_it_is_down() -> None:
    keys = combination()

    assert keys.press(CTRL) is False
    assert keys.press(ALT) is False
    assert keys.press(SPACE) is True


def test_the_order_the_keys_go_down_does_not_matter() -> None:
    keys = combination()

    assert [keys.press(SPACE), keys.press(ALT), keys.press(CTRL)] == [False, False, True]


def test_holding_a_key_does_not_start_a_second_recording() -> None:
    """Windows repeats key-down events while a key is held; each repeat would
    otherwise look like a new press."""
    keys = combination()
    keys.press(CTRL)
    keys.press(ALT)
    keys.press(SPACE)

    assert keys.press(SPACE) is False
    assert keys.press(CTRL) is False


def test_a_key_outside_the_combination_starts_nothing() -> None:
    keys = combination()
    keys.press(CTRL)
    keys.press(ALT)

    assert keys.press(SHIFT) is False


def test_letting_go_of_any_one_key_ends_it() -> None:
    keys = combination()
    keys.press(CTRL)
    keys.press(ALT)
    keys.press(SPACE)

    assert keys.release(ALT) is True


def test_letting_go_of_an_unrelated_key_does_not_end_it() -> None:
    """Typing while holding the combination must not cut the recording."""
    keys = combination()
    keys.press(CTRL)
    keys.press(ALT)
    keys.press(SPACE)

    assert keys.release(SHIFT) is False


def test_a_key_already_held_does_not_prevent_the_combination() -> None:
    """Somebody holding shift, or a game holding a key down, must not make the
    assistant unreachable."""
    keys = combination()
    keys.press(SHIFT)
    keys.press(CTRL)
    keys.press(ALT)

    assert keys.press(SPACE) is True


def test_the_combination_can_be_used_again() -> None:
    keys = combination()
    for key in (CTRL, ALT, SPACE):
        keys.press(key)
    keys.release(SPACE)

    assert keys.press(SPACE) is True


def test_a_release_that_never_had_a_press_is_harmless() -> None:
    """The combination may be completed while another window had focus."""
    keys = combination()

    assert keys.release(SPACE) is False


def test_letting_go_twice_only_ends_it_once() -> None:
    keys = combination()
    for key in (CTRL, ALT, SPACE):
        keys.press(key)
    keys.release(SPACE)

    assert keys.release(CTRL) is False


# --------------------------------------------------------------------------
# Listening without a key
# --------------------------------------------------------------------------


class FakeEndpoint:
    """A detector with no patience: a loud block is a sentence, a quiet one
    ends it. Where the real thresholds sit is `test_vad.py`'s business, and
    these tests are about which blocks reach a detector at all."""

    LOUD = 0.5

    def __init__(self) -> None:
        self.speaking = False
        self.heard: list[Audio] = []
        self.resets = 0
        self._take: list[Audio] = []

    def feed(self, chunk: Audio) -> list[Audio]:
        self.heard.append(chunk)
        if float(chunk[0]) >= self.LOUD:
            self.speaking = True
            self._take.append(chunk)
            return []
        if not self._take:
            return []

        take, self._take = self._take, []
        self.speaking = False
        return [np.concatenate(take)]

    def reset(self) -> None:
        self.resets += 1
        self.speaking = False
        self._take = []


def wired(**extra: Any) -> tuple[HandsFree, FakeHotkey, FakeMicrophone, FakeEndpoint]:
    """A capture with no keyboard, no microphone and no model."""
    toggle, microphone, endpoint = FakeHotkey(), FakeMicrophone(), FakeEndpoint()
    talk = HandsFree(microphone=microphone, toggle=toggle, endpoint=endpoint, **extra)
    return talk, toggle, microphone, endpoint


async def test_it_listens_from_the_moment_it_starts() -> None:
    """`live-assistant run` means run: there is no other key to press, and a program
    that starts deaf looks like one that failed to start (owner's decision,
    2026-09-11)."""
    talk, _, microphone, endpoint = wired()

    with talk:
        microphone.hear(tone(0.9))
        await asyncio.sleep(0)

        assert talk.listening is True
        assert len(endpoint.heard) == 1


async def test_it_can_be_started_deaf_for_whoever_wants_that() -> None:
    talk, _, microphone, endpoint = wired(listening=False)

    with talk:
        microphone.hear(tone(0.9))
        await asyncio.sleep(0)

        assert talk.listening is False
        assert endpoint.heard == []


async def test_a_sentence_comes_back_as_one_buffer() -> None:
    talk, _, microphone, _ = wired()

    with talk:
        microphone.hear(tone(0.9))
        microphone.hear(tone(0.8))
        microphone.hear(tone(0.1))  # the sentence ended

        pcm = await talk.utterance()

    assert np.array_equal(pcm, np.concatenate([tone(0.9), tone(0.8)]))
    assert pcm.dtype == np.float32


async def test_two_sentences_are_two_utterances() -> None:
    talk, _, microphone, _ = wired()

    with talk:
        microphone.hear(tone(0.9))
        microphone.hear(tone(0.1))
        microphone.hear(tone(0.7))
        microphone.hear(tone(0.1))

        first = await talk.utterance()
        second = await talk.utterance()

    assert np.array_equal(first, tone(0.9))
    assert np.array_equal(second, tone(0.7))


async def test_an_utterance_survives_until_somebody_asks_for_it() -> None:
    """The state machine is busy with the last turn when the next sentence
    ends. It is kept, not dropped."""
    talk, _, microphone, _ = wired()

    with talk:
        microphone.hear(tone(0.9))
        microphone.hear(tone(0.1))
        await asyncio.sleep(0)

        pcm = await asyncio.wait_for(talk.utterance(), timeout=0.5)

    assert np.array_equal(pcm, tone(0.9))


async def test_the_toggle_turns_it_off() -> None:
    talk, toggle, microphone, endpoint = wired()

    with talk:
        toggle.press()
        await asyncio.sleep(0)
        microphone.hear(tone(0.9))
        await asyncio.sleep(0)

        assert talk.listening is False
        assert endpoint.heard == []


async def test_the_toggle_turns_it_on_again() -> None:
    talk, toggle, microphone, endpoint = wired()

    with talk:
        toggle.press()
        await asyncio.sleep(0)
        toggle.press()
        await asyncio.sleep(0)
        microphone.hear(tone(0.9))
        await asyncio.sleep(0)

        assert talk.listening is True
        assert len(endpoint.heard) == 1


async def test_idle_audio_never_reaches_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The microphone is open all day.

    Handing every block to the loop while listening is off wakes it fifty
    times a second to throw the block away again. The tests around this one
    show the block does not end up in an utterance; this one shows it does
    not cross the thread boundary in the first place - a claim with no public
    face, which is why it reaches for the method behind it.
    """
    talk, _, microphone, _ = wired(listening=False)
    crossed: list[Audio] = []
    monkeypatch.setattr(talk, "_examine", crossed.append)

    with talk:
        microphone.hear(tone(0.9))
        await asyncio.sleep(0)

    assert crossed == []


async def test_the_mode_is_said_out_where_it_can_be_shown() -> None:
    """The status line is the only thing on screen that answers whether the
    microphone is live, and a mode nobody can see the state of is one left on
    in a room with other people in it. Starting counts: the line would
    otherwise show the paused hint over a microphone that is listening."""
    seen: list[bool] = []
    talk, toggle, _, _ = wired(on_mode=seen.append)

    with talk:
        toggle.press()
        toggle.press()
        await asyncio.sleep(0)

    assert seen == [True, False, True]


async def test_the_mode_is_reported_on_the_event_loop() -> None:
    """It is posted from the keyboard's thread and runs on the event loop,
    which is what makes it safe for `app.py` to stop the speaker from it."""
    loops: list[object] = []
    talk, toggle, _, _ = wired(on_mode=lambda _: loops.append(asyncio.get_running_loop()))

    with talk:
        toggle.press()
        await asyncio.sleep(0)

    assert loops == [asyncio.get_running_loop()] * 2


async def test_two_quick_presses_report_two_changes_and_not_one_twice() -> None:
    """Which way it went travels with the message: a listener that read the
    flag on arrival would be told the mode is whatever it is *now*, twice."""
    seen: list[bool] = []
    talk, toggle, _, _ = wired(on_mode=seen.append)

    with talk:
        toggle.press()
        toggle.press()
        toggle.press()
        await asyncio.sleep(0)

    assert seen == [True, False, True, False]


async def test_switching_the_mode_starts_a_new_stream() -> None:
    """Whatever the detector was in the middle of belongs to before the switch."""
    talk, toggle, _, endpoint = wired()

    with talk:
        before = endpoint.resets
        toggle.press()
        await asyncio.sleep(0)

    assert endpoint.resets == before + 1


async def test_the_moment_the_detector_hears_a_voice_is_announced() -> None:
    """`app.py` learns from it that whatever it was doing has been overtaken,
    and stops talking before the sentence is even over."""
    started: list[str] = []
    talk, _, microphone, _ = wired(on_listening=lambda: started.append("now"))

    with talk:
        microphone.hear(tone(0.1))
        await asyncio.sleep(0)
        assert started == []

        microphone.hear(tone(0.9))
        await asyncio.sleep(0)

    assert started == ["now"]


async def test_the_announcement_arrives_where_asyncio_can_be_touched() -> None:
    loops: list[object] = []
    talk, _, microphone, _ = wired(on_listening=lambda: loops.append(asyncio.get_running_loop()))

    with talk:
        microphone.hear(tone(0.9))
        await asyncio.sleep(0)

    assert loops == [asyncio.get_running_loop()]


async def test_a_sentence_is_announced_once_and_not_once_a_block() -> None:
    started: list[str] = []
    talk, _, microphone, _ = wired(on_listening=lambda: started.append("now"))

    with talk:
        for _ in range(3):
            microphone.hear(tone(0.9))
        await asyncio.sleep(0)

    assert started == ["now"]


async def test_listening_works_whether_or_not_anybody_listens_for_the_announcement() -> None:
    talk, _, microphone, _ = wired()

    with talk:
        microphone.hear(tone(0.9))
        microphone.hear(tone(0.1))

        pcm = await talk.utterance()

    assert np.array_equal(pcm, tone(0.9))


# --------------------------------------------------------------------------
# Not hearing the assistant's own voice
# --------------------------------------------------------------------------


async def test_nothing_is_listened_to_while_the_assistant_speaks() -> None:
    """Without this the detector hears the answer come out of the speakers,
    takes it for a question, and answers it - once per API call."""
    talk, _, microphone, endpoint = wired()

    with talk:
        talk.mute()
        microphone.hear(tone(0.9))
        await asyncio.sleep(0)

    assert endpoint.heard == []


async def test_the_room_still_repeating_the_answer_is_not_a_question() -> None:
    """The sound card holds some of the answer and the room holds the rest;
    both arrive after the speaker has been told to stop."""
    talk, _, microphone, endpoint = wired()
    tail = round(ECHO_TAIL_SECONDS * SAMPLE_RATE)

    with talk:
        talk.mute()
        talk.unmute()

        microphone.hear(tone(0.9, frames=tail))
        await asyncio.sleep(0)
        assert endpoint.heard == [], "the assistant's own echo reached the detector"

        microphone.hear(tone(0.9))
        await asyncio.sleep(0)

    assert len(endpoint.heard) == 1


async def test_listening_again_starts_a_new_stream() -> None:
    """The frames either side of an answer are not neighbours."""
    talk, _, _, endpoint = wired()

    with talk:
        before = endpoint.resets
        talk.mute()
        talk.unmute()

    assert endpoint.resets == before + 1


# --------------------------------------------------------------------------
# Devices are opened and closed
# --------------------------------------------------------------------------


async def test_starting_watches_the_key_and_opens_the_microphone() -> None:
    talk, toggle, microphone, _ = wired()

    with talk:
        assert (toggle.watching, microphone.opened) == (True, True)


async def test_leaving_lets_go_of_both_even_after_a_failure() -> None:
    """A microphone left open is a light that stays on and a device another
    application cannot have."""
    talk, toggle, microphone, _ = wired()

    with pytest.raises(RuntimeError), talk:
        raise RuntimeError("the turn failed")

    assert (toggle.watching, microphone.opened) == (False, False)
    assert talk.listening is False


# --------------------------------------------------------------------------
# The real devices
# --------------------------------------------------------------------------


def test_the_stream_is_opened_at_the_rate_the_speech_layer_declares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plausible looking sample rate is a silent bug: the model assumes the
    buffer is at 16 kHz and simply hears everything at the wrong speed."""
    module, streams = fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    SystemMicrophone().open(lambda chunk: None)

    assert streams[0].options["samplerate"] == SAMPLE_RATE
    assert streams[0].options["channels"] == 1
    assert streams[0].options["dtype"] == "float32"
    assert streams[0].started is True
    # The block being filled when the key comes up never arrives, so the block
    # size is also how much of the end of a sentence can be lost.
    assert streams[0].options["blocksize"] == CHUNK_FRAMES
    assert CHUNK_FRAMES / SAMPLE_RATE <= 0.02


def test_closing_the_microphone_releases_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    module, streams = fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    microphone = SystemMicrophone()
    microphone.open(lambda chunk: None)
    microphone.close()

    assert (streams[0].stopped, streams[0].closed) == (True, True)


def test_the_device_that_was_opened_is_on_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the setting empty, which microphone Windows handed over is not
    written anywhere else - and "which device did it open" is the first
    question when the assistant mishears (2026-09-06)."""
    module, _ = fake_sounddevice(host_api="Windows WDM-KS")
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    lines: list[str] = []
    sink = logger.add(lines.append, format="{message}")
    try:
        SystemMicrophone().open(lambda chunk: None)
    finally:
        logger.remove(sink)

    assert any("Microphone Array, Windows WDM-KS" in line for line in lines)


def test_a_microphone_that_cannot_be_opened_fails_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name that matches no device, or a device another program is holding:
    the user can fix either, so `live-assistant run` says which in a sentence."""
    module, _ = fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    with pytest.raises(MicrophoneUnavailableError, match="nope"):
        SystemMicrophone(device="nope").open(lambda chunk: None)


def test_a_wasapi_device_is_asked_to_convert_the_rate_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PortAudio runs a shared WASAPI stream at the mixer's rate - 48 kHz on
    the target machine - and refuses 16 kHz unless told to convert. Measured
    2026-09-05: `--device "Microphone Array WASAPI"` died on exactly this."""
    module, streams = fake_sounddevice(
        host_api="Windows WASAPI", native_rate=48_000, accepts={48_000}
    )
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    microphone = SystemMicrophone(device="Microphone Array WASAPI")
    microphone.open(lambda chunk: None)

    assert streams[0].options["samplerate"] == SAMPLE_RATE
    assert streams[0].options["extra_settings"].auto_convert is True
    assert microphone.rate == SAMPLE_RATE


def test_other_host_apis_are_not_handed_wasapi_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """PortAudio rejects a stream whose host-specific settings belong to
    another host API, so the setting goes only where it is understood."""
    module, streams = fake_sounddevice(host_api="MME")
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    SystemMicrophone().open(lambda chunk: None)

    assert streams[0].options.get("extra_settings") is None


def test_a_device_that_will_not_run_at_16_khz_is_opened_at_its_own_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kernel streaming has no converter: `Microphone Array 1` runs at 48 kHz
    or not at all (2026-09-05, "Invalid device"). The stream is opened at the
    rate the device offers, still in 20 ms blocks, and brought to 16 kHz here."""
    module, streams = fake_sounddevice(
        host_api="Windows WDM-KS", native_rate=48_000, accepts={48_000}
    )
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    microphone = SystemMicrophone(device="Microphone Array 1")
    microphone.open(lambda chunk: None)

    assert streams[0].options["samplerate"] == 48_000
    assert streams[0].options["blocksize"] == 960
    assert microphone.rate == 48_000


def test_blocks_from_a_48_khz_device_arrive_at_16_khz(monkeypatch: pytest.MonkeyPatch) -> None:
    """What the state machine receives is what it always received: 16 kHz,
    mono, twenty milliseconds at a time."""
    module, streams = fake_sounddevice(
        host_api="Windows WDM-KS", native_rate=48_000, accepts={48_000}
    )
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    heard: list[Audio] = []

    SystemMicrophone(device="Microphone Array 1").open(heard.append)
    callback = streams[0].options["callback"]
    block = np.full((960, 1), 0.25, dtype=np.float32)
    for _ in range(10):
        callback(block, 960, None, None)

    assert [len(chunk) for chunk in heard] == [CHUNK_FRAMES] * 10
    assert float(heard[-1].mean()) == pytest.approx(0.25, abs=0.01)


def test_a_device_that_runs_at_16_khz_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    module, streams = fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    microphone = SystemMicrophone()
    microphone.open(lambda chunk: None)

    assert microphone.rate == SAMPLE_RATE
    assert streams[0].options["samplerate"] == SAMPLE_RATE


def test_a_device_that_refuses_every_rate_is_still_a_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = fake_sounddevice(host_api="Windows WDM-KS", native_rate=48_000, accepts=set())
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    with pytest.raises(MicrophoneUnavailableError, match="could not be opened"):
        SystemMicrophone(device="Microphone Array 1").open(lambda chunk: None)


def test_the_buffer_portaudio_hands_over_is_copied(monkeypatch: pytest.MonkeyPatch) -> None:
    """PortAudio writes the next block into the same array. Keeping a
    reference instead of a copy turns the recording into the last block,
    repeated."""
    module, streams = fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    heard: list[Audio] = []

    SystemMicrophone().open(heard.append)
    callback = streams[0].options["callback"]

    reused = np.zeros((3, 1), dtype=np.float32)
    reused[:] = 0.1
    callback(reused, 3, None, None)
    reused[:] = 0.9  # exactly what PortAudio does next
    callback(reused, 3, None, None)

    assert [float(chunk[0]) for chunk in heard] == [pytest.approx(0.1), pytest.approx(0.9)]
    assert heard[0].shape == (3,), "the stream is mono; the channel axis is dropped"


def overflowing(microphone: SystemMicrophone, callback: Any, times: int) -> list[str]:
    """Drives the callback with `times` blocks the driver marked as overrun,
    and returns what was logged about them."""
    lines: list[str] = []
    sink = logger.add(lines.append, format="{message}")
    try:
        block = np.zeros((3, 1), dtype=np.float32)
        for _ in range(times):
            callback(block, 3, None, SimpleNamespace(input_overflow=True))
        callback(block, 3, None, None)
    finally:
        logger.remove(sink)
    return [line for line in lines if "overflow" in line]


def test_blocks_the_driver_dropped_are_counted_and_the_first_is_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PortAudio reports an overrun in `status`; until now it was thrown away
    with the block. In hands-free mode that is a hole in the sentence with no
    trace of it anywhere."""
    module, streams = fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    microphone = SystemMicrophone()
    microphone.open(lambda chunk: None)

    logged = overflowing(microphone, streams[0].options["callback"], times=2)

    assert microphone.overflows == 2
    assert len(logged) == 1


def test_dropped_blocks_are_written_down_rarely_after_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One line per dropped block, on PortAudio's own thread, would be the
    next cause of dropped blocks. The first, the tenth and the hundredth are
    enough to tell a bad minute from a bad microphone."""
    module, streams = fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    microphone = SystemMicrophone()
    microphone.open(lambda chunk: None)

    logged = overflowing(microphone, streams[0].options["callback"], times=100)

    assert microphone.overflows == 100
    assert len(logged) == 3
    assert "100" in logged[-1]


def test_a_block_that_arrived_whole_is_not_an_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    module, streams = fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    microphone = SystemMicrophone()
    microphone.open(lambda chunk: None)

    logged = overflowing(microphone, streams[0].options["callback"], times=0)

    assert microphone.overflows == 0
    assert logged == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", None),
        (None, None),
        ("   ", None),
        ("9", 9),
        ("  12 ", 12),
        ("Microphone Array WASAPI", "Microphone Array WASAPI"),
        (" Microphone Array 1 ", "Microphone Array 1"),
    ],
)
def test_a_device_choice_is_an_index_a_name_or_nothing(text: str | None, expected: object) -> None:
    """`sounddevice` takes an int as an index and a str as words to match. A
    digit string handed to it as a str matches no name and fails - which is
    exactly how `bench_mic.py --device 12` never worked (2026-09-05)."""
    assert device_choice(text) == expected


def test_the_default_combination_is_three_keys_the_system_knows() -> None:
    """A typo in the combination string would only be found by pressing it."""
    keys = SystemHotkey().keys

    assert len(keys) == 3
    assert DEFAULT_TOGGLE_HOTKEY == "<ctrl>+<alt>+h"


def test_a_combination_that_makes_no_sense_is_refused_at_once() -> None:
    with pytest.raises(ValueError, match="hotkey"):
        SystemHotkey("<ctrl>+<nonsense>")


# --------------------------------------------------------------------------
# The confirmation window (2.3): listening for an answer
# --------------------------------------------------------------------------


async def opened(talk: HandsFree, seconds: float = 1.0) -> asyncio.Task[Audio | None]:
    """`listen_for`, running, with its window already open."""
    window = asyncio.create_task(talk.listen_for(seconds))
    await asyncio.sleep(0)
    return window


async def test_a_window_nobody_answers_in_closes_with_nothing() -> None:
    talk, _, _, _ = wired()

    with talk:
        assert await talk.listen_for(0.02) is None


async def test_with_listening_off_the_window_still_hears_a_sentence() -> None:
    """The assistant asked, so the answer is heard even with the mode off -
    and the mode is as it was afterwards."""
    started: list[str] = []
    talk, _, microphone, endpoint = wired(
        listening=False, on_listening=lambda: started.append("now")
    )

    with talk:
        window = await opened(talk)
        microphone.hear(tone(0.9))
        microphone.hear(tone(0.8))
        microphone.hear(tone(0.1))  # the sentence ended

        answer = await window

        assert talk.listening is False
        microphone.hear(tone(0.9))  # the window is closed: nobody is listening
        await asyncio.sleep(0)

    assert answer is not None
    assert np.array_equal(answer, np.concatenate([tone(0.9), tone(0.8)]))
    assert started == []
    assert len(endpoint.heard) == 3


async def test_the_answer_does_not_become_a_question_as_well() -> None:
    started: list[str] = []
    talk, _, microphone, _ = wired(on_listening=lambda: started.append("now"))

    with talk:
        window = await opened(talk)
        microphone.hear(tone(0.9))
        microphone.hear(tone(0.1))

        answer = await window

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(talk.utterance(), timeout=0.05)

    assert answer is not None
    assert started == []


async def test_after_the_window_it_listens_for_questions_again() -> None:
    started: list[str] = []
    talk, _, microphone, _ = wired(on_listening=lambda: started.append("now"))

    with talk:
        assert await talk.listen_for(0.02) is None

        microphone.hear(tone(0.9))
        microphone.hear(tone(0.1))
        pcm = await talk.utterance()

    assert started == ["now"]
    assert np.array_equal(pcm, tone(0.9))


async def test_the_room_repeating_the_question_is_not_an_answer() -> None:
    """`unmute` leaves the echo tail behind, and the window counts it down
    before it listens - the question coming back off the walls is not a yes."""
    talk, _, microphone, endpoint = wired()
    tail = round(ECHO_TAIL_SECONDS * SAMPLE_RATE)

    with talk:
        talk.mute()
        talk.unmute()
        window = await opened(talk)
        microphone.hear(tone(0.9, frames=tail))
        await asyncio.sleep(0)
        assert endpoint.heard == [], "the assistant's own question reached the detector"

        microphone.hear(tone(0.9))
        microphone.hear(tone(0.1))
        answer = await window

    assert answer is not None
    assert np.array_equal(answer, tone(0.9))


async def test_the_window_starts_a_fresh_sentence_and_leaves_none_behind() -> None:
    """Whatever the detector had half collected before the question is not
    the answer, and whatever it had when time ran out is not the next question."""
    talk, _, _, endpoint = wired()

    with talk:
        before = endpoint.resets
        await talk.listen_for(0.02)

    assert endpoint.resets == before + 2


async def test_switching_off_closes_an_open_window_with_nothing() -> None:
    """The user turned the assistant off while it was asking. The question is
    a no at once - not six seconds later, and not whatever the room says next."""
    talk, toggle, microphone, _ = wired()

    with talk:
        window = await opened(talk, seconds=5.0)
        toggle.press()
        await asyncio.sleep(0)

        answer = await asyncio.wait_for(window, timeout=0.5)

        microphone.hear(tone(0.9))
        microphone.hear(tone(0.1))
        await asyncio.sleep(0)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(talk.utterance(), timeout=0.05)

    assert answer is not None
    assert len(answer) == 0


# --------------------------------------------------------------------------
# The list `live-assistant mic` offers
# --------------------------------------------------------------------------

# PortAudio's table on the development laptop, abridged: the one array behind
# four host APIs, the two entries that only mean "whatever the default is",
# an output, and a headset whose kernel-streaming name spans two lines.
KS_HEADSET = "Headset (@System32\\drivers\\bthhfenum.sys,#2;%1 Hands-Free%0\n;(Buds3))"
LAPTOP_DEVICES = [
    ("Microsoft Sound Mapper - Input", "MME", 2),
    ("Microphone Array (Intel\u00ae Smart ", "MME", 4),
    ("Speakers (Realtek HD Audio)", "MME", 0),
    ("Primary Sound Capture Driver", "Windows DirectSound", 2),
    ("Microphone Array (Intel\u00ae Smart Sound Technology)", "Windows WASAPI", 2),
    ("Microphone Array 1 ()", "Windows WDM-KS", 2),
    (KS_HEADSET, "Windows WDM-KS", 1),
]


def test_every_input_device_is_listed_once_per_host_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same microphone reaches the assistant through several Windows paths,
    and they are not alike (README, "which microphone path, measured"), so the
    list keeps them apart rather than folding them into one entry."""
    module, _ = fake_sounddevice(devices=LAPTOP_DEVICES)
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    found = available_microphones()

    assert [(m.name, m.host_api) for m in found.devices] == [
        ("Microphone Array (Intel\u00ae Smart ", "MME"),
        ("Microphone Array (Intel\u00ae Smart Sound Technology)", "Windows WASAPI"),
        ("Microphone Array 1 ()", "Windows WDM-KS"),
        (KS_HEADSET, "Windows WDM-KS"),
    ]


def test_the_entries_that_only_mean_the_default_are_left_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mapper and DirectSound's "primary" driver are the default under
    another name; the list offers the default once, as itself."""
    module, _ = fake_sounddevice(devices=LAPTOP_DEVICES)
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    names = [m.name for m in available_microphones().devices]

    assert "Microsoft Sound Mapper - Input" not in names
    assert "Primary Sound Capture Driver" not in names


def test_the_list_names_what_windows_has_chosen_right_now(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = fake_sounddevice(devices=LAPTOP_DEVICES)
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    assert available_microphones().default == "Microphone Array"


def test_a_machine_without_a_microphone_lists_nothing_and_has_no_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PortAudio has no default input then, and asking for it raises rather
    than returning an empty entry."""
    module, _ = fake_sounddevice(devices=[("Speakers (Realtek HD Audio)", "MME", 0)])
    listing = module.query_devices

    def query_devices(device: Any = None, kind: str | None = None) -> Any:
        if device is None and kind is None:
            return listing()
        raise module.PortAudioError("Error querying device -1")

    module.query_devices = query_devices
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    found = available_microphones()

    assert found.devices == ()
    assert found.default is None


def test_the_setting_is_the_line_sounddevice_matches_exactly() -> None:
    """`sounddevice` matches words in order and, when several entries share
    them, prefers the one whose "<name>, <host API>" is the whole query. The
    stored value is that whole line, so that the MME entry - a prefix of the
    WASAPI one - is not mistaken for it."""
    entry = MicrophoneInfo(name="Microphone Array (Intel\u00ae Smart ", host_api="MME")

    assert entry.setting == "Microphone Array (Intel\u00ae Smart , MME"


def test_the_label_is_the_name_on_one_line() -> None:
    """Kernel streaming names carry a driver path and a line break; the
    terminal shows one line per choice."""
    entry = MicrophoneInfo(name=KS_HEADSET, host_api="Windows WDM-KS")

    assert entry.label == (
        "Headset (@System32\\drivers\\bthhfenum.sys,#2;%1 Hands-Free%0 ;(Buds3)) - Windows WDM-KS"
    )


# --------------------------------------------------------------------------
# Streaming to a live session (plan.md section 4.1, L1.2)
# --------------------------------------------------------------------------

# One block of the size the microphone really delivers, so that the pre-roll
# counts in the same units the ring buffer does.
BLOCK = CHUNK_FRAMES
PREROLL_BLOCKS = round(PREROLL_SECONDS * SAMPLE_RATE / BLOCK)


def block(value: float) -> Audio:
    return tone(value, frames=BLOCK)


def live(**extra: Any) -> tuple[LiveCapture, FakeHotkey, FakeMicrophone, FakeEndpoint]:
    """A live capture with no keyboard, no microphone and no model. The fake
    microphone sits behind WASAPI unless a test says otherwise, so that the
    duplex rule is the full-duplex one the owner's machine takes (D18)."""
    toggle, microphone, endpoint = FakeHotkey(), FakeMicrophone(), FakeEndpoint()
    talk = LiveCapture(microphone=microphone, toggle=toggle, endpoint=endpoint, **extra)
    return talk, toggle, microphone, endpoint


async def heard_by_loop(microphone: FakeMicrophone, *blocks: Audio) -> None:
    """Blocks from the audio thread, each examined by the loop before the next."""
    for chunk in blocks:
        microphone.hear(chunk)
        await asyncio.sleep(0)


async def taken(talk: LiveCapture) -> list[bytes]:
    """What `chunks()` hands over right now, without waiting for more."""
    got: list[bytes] = []
    try:
        async with asyncio.timeout(0.02):
            async for chunk in talk.chunks():
                got.append(chunk)
    except TimeoutError:
        pass
    return got


def pcm(*values: float) -> list[bytes]:
    return [to_pcm16(block(value)) for value in values]


def test_the_live_capture_is_the_hands_free_capture_with_a_stream() -> None:
    """The key, the confirmation window and the mode report are the old
    ones; only what happens to a block once the detector has seen it differs."""
    talk, _, _, _ = live()

    assert isinstance(talk, HandsFree)
    assert talk.taking is False


async def test_nothing_is_taken_before_anybody_speaks() -> None:
    """A session costs money from the moment it opens (D5, D12): the room
    being quiet is not a reason to have one."""
    talk, _, microphone, _ = live()

    with talk:
        await heard_by_loop(microphone, block(0.1), block(0.2))

        assert talk.taking is False
        assert await taken(talk) == []


async def test_the_moment_speech_starts_is_reported_and_so_is_its_end() -> None:
    """`USER_SPEAKING` is a state of the screen (section 4.2): on at the
    onset, off when the sentence ends. The server decides the turn."""
    talk, _, microphone, _ = live()
    reports: list[bool] = []
    talk.on_speech = reports.append

    with talk:
        await heard_by_loop(microphone, block(0.9), block(0.9), block(0.1))

    assert reports == [True, False]


async def test_speech_at_the_door_starts_the_stream_from_before_it_began() -> None:
    """The detector is a few frames late by the time it is sure, and those
    frames hold the first consonant: what goes to the session begins with
    the pre-roll, then the block that convinced the detector."""
    talk, _, microphone, _ = live()

    with talk:
        await heard_by_loop(microphone, block(0.1), block(0.2), block(0.9))

        assert talk.taking is True
        assert await taken(talk) == pcm(0.1, 0.2, 0.9)


async def test_the_pre_roll_is_no_longer_than_it_says() -> None:
    """A ring buffer, not a recording: the quiet minute before the sentence
    is not sent to the server at the sentence's price. The stream begins
    `PREROLL_SECONDS` before the block that convinced the detector, that
    block being the last of them - the old `Endpoint` counted the same way."""
    talk, _, microphone, _ = live()
    before = [block(0.01 * index) for index in range(1, PREROLL_BLOCKS + 6)]

    with talk:
        await heard_by_loop(microphone, *before, block(0.9))

        got = await taken(talk)

    assert len(got) == PREROLL_BLOCKS
    assert got[0] == to_pcm16(before[-(PREROLL_BLOCKS - 1)])
    assert got[-1] == to_pcm16(block(0.9))


async def test_what_is_said_while_the_session_opens_is_not_lost() -> None:
    """Opening a session takes 0.6-0.8 s (ADR-001): the sentence that opened
    it is half over before anything can be sent. It waits here, in order."""
    talk, _, microphone, _ = live()

    with talk:
        await heard_by_loop(microphone, block(0.9), block(0.8), block(0.7), block(0.6))

        assert await taken(talk) == pcm(0.9, 0.8, 0.7, 0.6)


async def test_the_stream_is_sixteen_bit_pcm_at_the_capture_rate() -> None:
    """What the session protocol takes (`LiveSession.send_audio`): the same
    conversion the recogniser uses, so clipping happens the same way."""
    talk, _, microphone, _ = live()

    with talk:
        await heard_by_loop(microphone, block(0.9))

        [chunk] = await taken(talk)

    assert chunk == to_pcm16(block(0.9))
    assert len(chunk) == BLOCK * 2


async def test_silence_keeps_flowing_once_the_stream_is_open() -> None:
    """The server's own detector finds the end of the sentence in the
    silence after it. A stream that stops when the room goes quiet would
    never let the model answer."""
    talk, _, microphone, _ = live()

    with talk:
        await heard_by_loop(microphone, block(0.9), block(0.1), block(0.2))

        assert await taken(talk) == pcm(0.9, 0.1, 0.2)


async def test_closing_ends_the_stream_and_drops_what_was_not_sent() -> None:
    """The session closed on silence (D5) or was hung up: whatever was
    queued behind a slow socket has nobody to go to."""
    talk, _, microphone, _ = live()

    with talk:
        await heard_by_loop(microphone, block(0.9), block(0.8))
        talk.close()

        assert talk.taking is False
        assert await taken(talk) == []


async def test_a_consumer_already_reading_learns_that_the_stream_ended() -> None:
    """The forwarding task of `app.py` sits in `chunks()`; closing has to
    return it, not leave it waiting for a block that will never come."""
    talk, _, microphone, _ = live()

    with talk:
        await heard_by_loop(microphone, block(0.9))
        reader = asyncio.ensure_future(taken(talk))
        await asyncio.sleep(0)
        talk.close()

        assert await asyncio.wait_for(reader, 1) == pcm(0.9)


async def test_after_closing_the_next_sentence_starts_a_fresh_stream() -> None:
    """Nothing of the last conversation leaks into the next one: a block
    goes to the server once, so the pre-roll of the next stream holds only
    what came after the close."""
    talk, _, microphone, _ = live()

    with talk:
        await heard_by_loop(microphone, block(0.9), block(0.1))
        talk.close()
        await heard_by_loop(microphone, block(0.2), block(0.8))

        assert await taken(talk) == pcm(0.2, 0.8)


async def test_a_second_sentence_while_the_stream_is_open_changes_nothing() -> None:
    """The onset matters at the door. Inside, the blocks were flowing anyway,
    and sending the pre-roll again would send the last 300 ms twice."""
    talk, _, microphone, _ = live()
    reports: list[bool] = []
    talk.on_speech = reports.append

    with talk:
        await heard_by_loop(microphone, block(0.9), block(0.1), block(0.8))

        assert await taken(talk) == pcm(0.9, 0.1, 0.8)
        assert reports == [True, False, True]


async def test_the_stream_of_a_door_nobody_has_come_through_is_empty() -> None:
    """Asked for the stream with nobody speaking, `chunks()` ends at once
    rather than waiting for an onset it was never meant to announce."""
    talk, _, _, _ = live()

    with talk:
        assert [chunk async for chunk in talk.chunks()] == []


# --------------------------------------------------------------------------
# The gate: paused input
# --------------------------------------------------------------------------


async def test_paused_input_is_dropped_and_not_kept_for_later() -> None:
    """The confirmation window (section 4.4 rule 3) and the announcement
    (D4): what the room says meanwhile is for the local recogniser or for
    nobody, never for the model - not now and not after."""
    talk, _, microphone, _ = live()

    with talk:
        await heard_by_loop(microphone, block(0.9))
        talk.pause()
        await heard_by_loop(microphone, block(0.8), block(0.7))
        talk.resume()
        await heard_by_loop(microphone, block(0.6))

        assert await taken(talk) == pcm(0.9, 0.6)


async def test_the_detector_keeps_watching_while_input_is_paused() -> None:
    """A reminder being read out is cut by the user starting to talk
    (section 4.2, `ANNOUNCING`), which only works if the detector still
    hears the room while nothing is forwarded."""
    talk, _, microphone, _ = live()
    reports: list[bool] = []
    talk.on_speech = reports.append

    with talk:
        talk.pause()
        await heard_by_loop(microphone, block(0.9))

    assert reports == [True]


async def test_a_sentence_at_the_door_while_paused_still_opens_the_stream() -> None:
    """The announcement was cut by the onset; the state machine resumes and
    opens a session for what is being said, pre-roll included."""
    talk, _, microphone, _ = live()

    with talk:
        talk.pause()
        await heard_by_loop(microphone, block(0.1), block(0.9))
        talk.resume()
        await heard_by_loop(microphone, block(0.8))

        assert talk.taking is True
        assert await taken(talk) == pcm(0.1, 0.9, 0.8)


async def test_resuming_what_was_never_paused_is_harmless() -> None:
    talk, _, microphone, _ = live()

    with talk:
        talk.resume()
        await heard_by_loop(microphone, block(0.9))

        assert await taken(talk) == pcm(0.9)


# --------------------------------------------------------------------------
# The key, the window
# --------------------------------------------------------------------------


async def test_switching_off_ends_the_stream() -> None:
    """Rule 4 of section 4.4: off is a hang-up. The half sentence in the
    queue does not wait for the next session to be sent as its opening."""
    talk, toggle, microphone, _ = live()

    with talk:
        await heard_by_loop(microphone, block(0.9))
        toggle.press()
        await asyncio.sleep(0)

        assert talk.taking is False
        assert await taken(talk) == []


async def test_switched_back_on_it_starts_afresh_at_the_door() -> None:
    talk, toggle, microphone, _ = live()

    with talk:
        await heard_by_loop(microphone, block(0.9))
        toggle.press()
        toggle.press()
        await asyncio.sleep(0)
        await heard_by_loop(microphone, block(0.1), block(0.8))

        assert await taken(talk) == pcm(0.1, 0.8)


async def test_the_confirmation_window_hears_the_answer_and_says_nothing_of_it() -> None:
    """The window of the old product, unchanged: the sentence goes to
    whoever asked, and the state machine is not told a new sentence began -
    it would take the user's yes for a new question."""
    talk, _, microphone, _ = live()
    reports: list[bool] = []
    talk.on_speech = reports.append

    with talk:
        talk.pause()
        window = asyncio.ensure_future(talk.listen_for(1))
        await asyncio.sleep(0)
        await heard_by_loop(microphone, block(0.9), block(0.1))
        answer = await window

    assert answer is not None and float(answer[0]) == pytest.approx(0.9)
    assert reports == []


# --------------------------------------------------------------------------
# Full and half duplex (D18)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("host_api", ["Windows WASAPI", "MME", "Windows DirectSound"])
def test_a_microphone_behind_the_windows_audio_engine_is_full_duplex(host_api: str) -> None:
    """Measured 2026-09-18 (ADR-001 section 4): through the audio engine the
    array's own echo canceller removes the speakers, and the model can be
    talked over."""
    assert duplex_for(host_api, barge_in=True) == "full"


def test_kernel_streaming_is_half_duplex() -> None:
    """The same array, raw: the server hears its own voice, interrupts itself
    and answers itself. Measured the same day, 16 interruptions in 17 turns."""
    assert duplex_for("Windows WDM-KS", barge_in=True) == "half"


def test_a_host_api_nobody_measured_is_half_duplex() -> None:
    """A loop that answers itself costs money every round; no barge-in costs
    a key press."""
    assert duplex_for("ASIO", barge_in=True) == "half"
    assert duplex_for("", barge_in=True) == "half"


def test_barge_in_off_forces_half_duplex_on_any_microphone() -> None:
    """`[live] barge_in = false` is the user's choice over the rule."""
    assert duplex_for("Windows WASAPI", barge_in=False) == "half"


async def test_the_capture_reads_the_duplex_off_the_microphone_it_opened() -> None:
    """Not off the setting: with `[audio] input_device` empty the host API is
    whatever Windows handed over, and only the opened stream knows."""
    toggle, endpoint = FakeHotkey(), FakeEndpoint()
    raw = LiveCapture(microphone=FakeMicrophone("Windows WDM-KS"), toggle=toggle, endpoint=endpoint)
    engine = LiveCapture(microphone=FakeMicrophone("MME"), toggle=toggle, endpoint=endpoint)

    with raw:
        assert raw.duplex == "half"
    with engine:
        assert engine.duplex == "full"


async def test_the_duplex_is_known_by_the_time_the_mode_is_reported() -> None:
    """The status line reads it in `on_mode`, which `start` calls."""
    talk, _, _, _ = live(barge_in=False)
    seen: list[str] = []
    talk.on_mode = lambda listening: seen.append(talk.duplex)

    with talk:
        pass

    assert seen == ["half"]


async def test_in_full_duplex_the_microphone_stays_live_while_the_assistant_speaks() -> None:
    """Barge-in as section 4.1: the server hears the user over its own voice
    and says `Interrupted`. Deaf, it could not."""
    talk, _, microphone, _ = live()

    with talk:
        await heard_by_loop(microphone, block(0.9))
        talk.mute()
        await heard_by_loop(microphone, block(0.8))
        talk.unmute()
        await heard_by_loop(microphone, block(0.7))

        assert await taken(talk) == pcm(0.9, 0.8, 0.7)


async def test_in_half_duplex_the_microphone_is_deaf_while_the_assistant_speaks() -> None:
    """The old product's rule, copied: without it the detector hears the
    answer come out of the speakers and the assistant answers itself."""
    talk, _, microphone, endpoint = live(barge_in=False)

    with talk:
        await heard_by_loop(microphone, block(0.9))
        talk.mute()
        await heard_by_loop(microphone, block(0.8))
        talk.unmute()
        await heard_by_loop(microphone, tone(0.9, frames=round(ECHO_TAIL_SECONDS * SAMPLE_RATE)))
        await heard_by_loop(microphone, block(0.7))

        assert await taken(talk) == pcm(0.9, 0.7)
        assert len(endpoint.heard) == 2, "the echo tail reached the detector"


async def test_in_full_duplex_unmuting_does_not_restart_the_detector() -> None:
    """The frames either side of the answer *are* neighbours here: the
    microphone never stopped."""
    talk, _, _, endpoint = live()

    with talk:
        before = endpoint.resets
        talk.mute()
        talk.unmute()

    assert endpoint.resets == before


# --------------------------------------------------------------------------
# The sent level (D18: no digital gain; a quiet microphone is said)
# --------------------------------------------------------------------------


async def test_the_level_of_what_was_sent_is_measured_in_dbfs() -> None:
    """Measured on the owner's machine: speech at -45 dBFS RMS is heard as
    Hindi. The number is the RMS of the blocks that were not silence, so
    that the pauses between words do not pull it down."""
    talk, _, microphone, _ = live()

    with talk:
        await heard_by_loop(microphone, block(0.9), block(0.1))

    assert talk.level_dbfs == pytest.approx(20 * np.log10(np.sqrt((0.9**2 + 0.1**2) / 2)), abs=0.01)


async def test_silence_does_not_count_towards_the_level() -> None:
    talk, _, microphone, _ = live()

    with talk:
        await heard_by_loop(microphone, block(0.9), block(0.0), block(0.0))

    assert talk.level_dbfs == pytest.approx(20 * np.log10(0.9), abs=0.01)


async def test_with_nothing_sent_there_is_no_level() -> None:
    talk, _, microphone, _ = live()

    with talk:
        await heard_by_loop(microphone, block(0.1))

    assert talk.level_dbfs is None


async def test_only_what_the_session_gets_is_measured() -> None:
    """The level answers "what does the server hear": a block dropped at the
    gate or before the onset was not heard by anybody."""
    talk, _, microphone, _ = live()

    with talk:
        await heard_by_loop(microphone, block(0.9))
        talk.pause()
        await heard_by_loop(microphone, block(0.001))
        talk.resume()

    assert talk.level_dbfs == pytest.approx(20 * np.log10(0.9), abs=0.01)


class Sensitive(FakeEndpoint):
    """A detector that hears a whisper, for the tests about quiet speech."""

    LOUD = 0.0015


async def test_a_quiet_microphone_is_written_down_when_the_stream_closes() -> None:
    """The line the owner needed on 2026-09-18 and did not have: the level
    the server got, once per session, and a warning under the threshold."""
    microphone = FakeMicrophone()
    talk = LiveCapture(microphone=microphone, toggle=FakeHotkey(), endpoint=Sensitive())
    lines: list[str] = []
    sink = logger.add(lines.append, format="{level} {message}")
    try:
        with talk:
            await heard_by_loop(microphone, block(0.002))
            talk.close()
    finally:
        logger.remove(sink)

    [line] = [line for line in lines if "dBFS" in line]
    assert line.startswith("WARNING")
    assert "-54 dBFS" in line


async def test_a_microphone_at_a_good_level_is_written_down_quietly() -> None:
    talk, _, microphone, _ = live()
    lines: list[str] = []
    sink = logger.add(lines.append, format="{level} {message}")
    try:
        with talk:
            await heard_by_loop(microphone, block(0.5))
            talk.close()
    finally:
        logger.remove(sink)

    [line] = [line for line in lines if "dBFS" in line]
    assert line.startswith("INFO")
    assert "-6 dBFS" in line


async def test_a_stream_that_sent_nothing_writes_no_level() -> None:
    talk, _, microphone, _ = live()
    lines: list[str] = []
    sink = logger.add(lines.append, format="{message}")
    try:
        with talk:
            await heard_by_loop(microphone, block(0.1))
            talk.close()
    finally:
        logger.remove(sink)

    assert not any("dBFS" in line for line in lines)


def test_the_quiet_threshold_is_the_one_measured() -> None:
    """ADR-001 section 5: -45 dBFS was too quiet, -35 is the target; the
    warning sits between them."""
    assert QUIET_DBFS == -40.0


# --------------------------------------------------------------------------
# The real microphone knows which way it was opened
# --------------------------------------------------------------------------


def test_the_real_microphone_reports_its_host_api_once_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = fake_sounddevice(host_api="Windows WDM-KS")
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    microphone = SystemMicrophone()

    assert microphone.host_api == ""
    microphone.open(lambda chunk: None)

    assert microphone.host_api == "Windows WDM-KS"


# --------------------------------------------------------------------------
# The level hook (plan.md D20: the orb swells with the microphone)
# --------------------------------------------------------------------------


async def test_every_block_sent_is_reported_to_the_level_hook_in_dbfs() -> None:
    """The window's orb hears the same blocks the server does, one number
    each, on the loop - never a block that was not sent."""
    levels: list[float] = []
    talk, _, microphone, _ = live(on_level=levels.append)

    with talk:
        await heard_by_loop(microphone, block(0.5), block(0.1))

    assert levels == pytest.approx([20 * np.log10(0.5), 20 * np.log10(0.1)], abs=0.01)


async def test_a_block_of_silence_is_reported_as_the_floor() -> None:
    levels: list[float] = []
    talk, _, microphone, _ = live(on_level=levels.append)

    with talk:
        await heard_by_loop(microphone, block(0.9), block(0.0))

    assert levels[-1] == LEVEL_FLOOR_DBFS


async def test_paused_input_reaches_neither_the_server_nor_the_hook() -> None:
    levels: list[float] = []
    talk, _, microphone, _ = live(on_level=levels.append)

    with talk:
        await heard_by_loop(microphone, block(0.9))
        talk.pause()
        await heard_by_loop(microphone, block(0.8))
        talk.resume()

    assert len(levels) == 1


async def test_at_the_door_nothing_is_reported() -> None:
    """A quiet room while no stream is open is the pre-roll's business."""
    levels: list[float] = []
    talk, _, microphone, _ = live(on_level=levels.append)

    with talk:
        await heard_by_loop(microphone, block(0.1), block(0.1))

    assert levels == []


async def test_without_a_hook_the_stream_is_what_it_was() -> None:
    talk, _, microphone, _ = live()

    with talk:
        await heard_by_loop(microphone, block(0.9))
        assert await taken(talk) == pcm(0.9)


# --------------------------------------------------------------------------
# Asleep behind the wake word (plan.md D21)
# --------------------------------------------------------------------------


class FakeWake:
    """A detector that wakes on the block the test names."""

    name = "hey_friday"

    def __init__(self, wake_on: int | None = None) -> None:
        self.heard: list[Audio] = []
        self.resets = 0
        self.wake_on = wake_on

    async def load(self) -> None:
        return

    def feed(self, chunk: Audio) -> bool:
        self.heard.append(chunk)
        return self.wake_on is not None and len(self.heard) == self.wake_on

    def reset(self) -> None:
        self.resets += 1


async def test_with_a_detector_the_capture_starts_asleep_and_feeds_only_it() -> None:
    """Asleep: the detector hears every block, the doorman none, nothing
    is queued and no pre-roll is kept - a room that is not listened to."""
    wake = FakeWake()
    talk, _, microphone, endpoint = live(wake=wake)

    with talk:
        assert talk.asleep
        assert talk.wake_word
        await heard_by_loop(microphone, block(0.5), block(0.5))

        assert len(wake.heard) == 2
        assert endpoint.heard == []
        assert not talk.taking


async def test_the_phrase_wakes_it_opens_the_stream_and_tells_the_loop() -> None:
    wake = FakeWake(wake_on=2)
    woken: list[bool] = []
    talk, _, microphone, endpoint = live(wake=wake, on_wake=lambda: woken.append(True))

    with talk:
        await heard_by_loop(microphone, block(0.5), block(0.5))

        assert woken == [True]
        assert not talk.asleep
        assert talk.taking
        assert endpoint.resets >= 1

        await heard_by_loop(microphone, block(0.5))
        assert len(endpoint.heard) == 1  # awake: the doorman hears now
        assert len(wake.heard) == 2  # and the detector no longer does


async def test_sleep_closes_the_stream_and_forgets_what_was_heard() -> None:
    wake = FakeWake(wake_on=1)
    talk, _, microphone, _ = live(wake=wake)

    with talk:
        await heard_by_loop(microphone, block(0.5), block(0.5))
        assert talk.taking

        talk.sleep()

        assert talk.asleep
        assert not talk.taking
        assert wake.resets >= 1
        assert [chunk async for chunk in talk.chunks()] == []


async def test_the_key_wakes_it_and_off_is_still_off() -> None:
    """Pressed on from off, the user is about to talk: awake, not asleep."""
    talk, toggle, microphone, endpoint = live(wake=FakeWake())

    with talk:
        toggle.press()
        await asyncio.sleep(0)
        assert not talk.listening
        await heard_by_loop(microphone, block(0.5))
        assert endpoint.heard == []

        toggle.press()
        await asyncio.sleep(0)
        assert talk.listening
        assert not talk.asleep
        await heard_by_loop(microphone, block(0.5))
        assert len(endpoint.heard) == 1


async def test_without_a_detector_nothing_sleeps() -> None:
    talk, _, microphone, endpoint = live()

    with talk:
        assert not talk.asleep
        assert not talk.wake_word
        talk.sleep()
        assert not talk.asleep
        await heard_by_loop(microphone, block(0.5))
        assert len(endpoint.heard) == 1
