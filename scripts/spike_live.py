"""Half a day with Gemini Live on the owner's key (plan section 7, L1.0; the numbers go to ADR-001).

Throwaway, kept for the record. It opens one live session, streams the
microphone into it, plays what comes back, and writes down every number the
plan asks for: how long the session takes to open, how long after you stop
talking the first sound arrives, how long after you start talking over it the
model falls silent, what a tool call costs in silence, and what the free tier
does after a while. Nothing in `src/` changes; three of its pieces are
borrowed - the microphone, the speaker and the voice detector - so what is
measured here is what the product will feel.

    uv run python scripts/spike_live.py
    uv run python scripts/spike_live.py --voice Kore --language tr-TR
    uv run python scripts/spike_live.py --non-blocking        # O5: the tool declared NON_BLOCKING
    uv run python scripts/spike_live.py --half-duplex         # the microphone deaf while it speaks
    uv run python scripts/spike_live.py --text "saat kac"     # a typed turn before the microphone
    uv run python scripts/spike_live.py --log spike.jsonl     # every server message, stamped

Press Enter to hang up; the summary is printed whatever ended the session. The
key comes from the Credential Manager - the `allie` entry, or the old
product's `assistant` entry until `import` exists - and is never printed.

Two clocks are compared for "when did you stop talking": the local detector
(`audio/vad.py`, the product's own numbers) and the server's own end-of-speech
signal, when it sends one. The local one is what the product can act on; the
server's says how much of the wait is the model and how much is the network.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import tomllib
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL = "gemini-3.8-live"
INPUT_RATE = 16_000
INPUT_MIME = f"audio/pcm;rate={INPUT_RATE}"
# What the documentation promises for the way back; the first chunk's mime
# type is checked against it and the summary says what was actually used.
OUTPUT_RATE = 24_000
BYTES_PER_SAMPLE = 2

# The local detector's word on when you started and stopped, with the
# product's own numbers so the latency measured here is the one it will feel.
# Talking over the model needs more frames: the room carries its voice into
# the microphone, and three frames of that would count as you.
ONSET_FRAMES = 3
BARGE_FRAMES = 8
SILENCE_SECONDS = 0.6

# In half-duplex mode, how long the microphone stays deaf after the last
# sound: the sound card and the room hold the answer after the queue is empty.
ECHO_TAIL_SECONDS = 0.25

# How much of a transcript one console line carries.
SHOWN_CHARS = 72

# Below this RMS a 20 ms block is silence and does not count in the level.
SILENCE_RMS = 0.001

# How long a sentence from a file waits for its answer before the next one.
ANSWER_PATIENCE_SECONDS = 10.0

# The prompt: the product's brevity and language rules as they are, and a
# personality line written for a model that hears rather than reads.
SPIKE_PERSONALITY = (
    "You are a voice assistant running on the user's own computer, and you hear the user "
    "rather than read them. You are calm, direct and unhurried: you do what was asked and "
    "say plainly when something cannot be done or when you do not know. You do not open "
    "with pleasantries, praise the question, or announce what you are about to do."
)

# Exit codes: the key was refused or missing; the session could not be opened.
REFUSED = 2
UNREACHABLE = 3


# --------------------------------------------------------------------------
# What is written down
# --------------------------------------------------------------------------


@dataclass
class Turn:
    """One exchange: what was heard, what was said, and when."""

    index: int
    started_at: float
    heard: str = ""
    said: str = ""
    # The moment the first sound of the answer arrived, and the two opinions
    # on when you had stopped talking before it.
    first_audio_at: float | None = None
    eos_local_at: float | None = None
    eos_server_at: float | None = None
    # A tool round: the call's arrival, our answer, the first sound after it,
    # and the last sound before it (what the silence is measured against).
    tool_name: str = ""
    tool_called_at: float | None = None
    tool_answered_at: float | None = None
    audio_before_tool_at: float | None = None
    audio_after_tool_at: float | None = None
    # Talking over it: when the local detector heard you, when the server
    # said it had stopped, when the speaker was actually silent.
    barge_at: float | None = None
    interrupted_at: float | None = None
    silent_at: float | None = None
    # The end: the server's word, and the last sample heard.
    complete_at: float | None = None
    heard_done_at: float | None = None
    complete_reason: str = ""
    audio_out_bytes: int = 0
    # What the model said between the tool call and our answer: nothing when
    # the tool blocks, the O5 question when it does not.
    audio_during_tool_bytes: int = 0
    said_during_tool: str = ""
    tokens: dict[str, int] = field(default_factory=dict)

    @property
    def interrupted(self) -> bool:
        return self.interrupted_at is not None

    def wait_local_ms(self) -> float | None:
        return _ms(self.eos_local_at, self.first_audio_at)

    def wait_server_ms(self) -> float | None:
        return _ms(self.eos_server_at, self.first_audio_at)

    def tool_ms(self) -> float | None:
        return _ms(self.tool_called_at, self.audio_after_tool_at)

    def answer_to_sound_ms(self) -> float | None:
        return _ms(self.tool_answered_at, self.audio_after_tool_at)

    def tool_silence_ms(self) -> float | None:
        return _ms(self.audio_before_tool_at, self.audio_after_tool_at)

    def tool_ours_ms(self) -> float | None:
        return _ms(self.tool_called_at, self.tool_answered_at)

    def interrupt_ms(self) -> float | None:
        return _ms(self.barge_at, self.interrupted_at)

    def silence_ms(self) -> float | None:
        return _ms(self.barge_at, self.silent_at)


def _ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or end < start:
        return None
    return (end - start) * 1000.0


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    at = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[at]


def _stat(values: list[float | None]) -> str:
    known = [value for value in values if value is not None]
    if not known:
        return "-"
    if len(known) == 1:
        return f"{known[0]:.0f} ms (n=1)"
    return (
        f"p50 {_percentile(known, 0.5):.0f} ms, p95 {_percentile(known, 0.95):.0f} ms, "
        f"max {max(known):.0f} ms (n={len(known)})"
    )


def _clock(seconds: float) -> str:
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes} m {rest:02d} s"


def _shown(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SHOWN_CHARS else text[: SHOWN_CHARS - 1] + "…"


# --------------------------------------------------------------------------
# The machine's side: key, microphone, prompt
# --------------------------------------------------------------------------


def find_key() -> tuple[str, str]:
    """The Gemini key and where it came from. The key itself is never printed."""
    from allie.config import KEYRING_SERVICE, load_api_key

    key = load_api_key("gemini")
    if key:
        return key, f"Credential Manager, service {KEYRING_SERVICE!r}"
    import keyring

    key = keyring.get_password("assistant", "gemini")
    if key:
        return key, "Credential Manager, service 'assistant' (the old product's entry)"
    raise SystemExit(
        "no Gemini key in the Credential Manager: run the old product's setup, or `allie setup`"
    )


def default_device() -> str:
    """The microphone the product would use: this product's setting, else the old one's."""
    appdata = Path(os.environ.get("APPDATA", ""))
    for product in ("allie", "assistant"):
        path = appdata / product / "config.toml"
        if not path.exists():
            continue
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        device = str(data.get("audio", {}).get("input_device", "")).strip()
        if device:
            return device
    return ""


def system_prompt(user_language: str = "") -> str:
    from allie.agent.prompts import BREVITY, LANGUAGE_RULE

    parts = [SPIKE_PERSONALITY, BREVITY, LANGUAGE_RULE]
    if user_language:
        # In the product this sentence would be a locale-pack value (the
        # product-locale axis), never a constant in code; here it is a flag.
        parts.append(
            f"The user speaks {user_language}. A short or quiet utterance is {user_language} "
            f"unless it is unmistakably another language; never switch languages on a guess."
        )
    return "\n\n".join(parts)


def load_sentence(path: Path) -> Any:
    """A WAV file as 16 kHz mono float32, whatever it was written as."""
    import soundfile  # type: ignore[import-untyped]

    from allie.audio.resample import Resampler

    data, rate = soundfile.read(str(path), dtype="float32")
    samples = np.asarray(data, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples[:, 0]
    if rate != INPUT_RATE:
        samples = Resampler(rate, INPUT_RATE).push(samples)
    return np.ascontiguousarray(samples)


# --------------------------------------------------------------------------
# The spike
# --------------------------------------------------------------------------


class Spike:
    def __init__(self, arguments: argparse.Namespace) -> None:
        self.a = arguments
        self.t0 = time.perf_counter()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stop = asyncio.Event()
        self.turn_done = asyncio.Event()
        self.turns: list[Turn] = []
        self.current: Turn | None = None
        self.log_file: Any = None

        # The microphone's blocks, stamped on arrival, crossing to the loop.
        self.mic_queue: asyncio.Queue[tuple[float, Any]] = asyncio.Queue()
        self.audio_in_bytes = 0
        # What was sent, after the gain: the loudest sample, and the mean
        # RMS of the blocks that held anything (silence would drown it).
        self.in_peak = 0.0
        self.in_rms_sum = 0.0
        self.in_rms_blocks = 0
        self.audio_out_bytes = 0
        self.output_rate = OUTPUT_RATE
        self.output_mime = ""

        # Playback: one queue and one task per answer.
        self.speaker: Any = None
        self.play_queue: asyncio.Queue[bytes | None] | None = None
        self.play_task: asyncio.Task[None] | None = None
        self.playing = False
        self.play_ended_at = -1.0

        # The local detector's state.
        self.vad: Any = None
        self.vad_buffer = np.zeros(0, dtype=np.float32)
        self.speech_frames = 0
        self.user_speaking = False
        self.last_speech_at: float | None = None
        self.server_eos_at: float | None = None

        # Session facts for the summary.
        self.open_ms: float | None = None
        self.opened_at: float | None = None
        self.closed_at: float | None = None
        self.closed_by = "?"
        self.session_id = ""
        self.tokens: dict[str, int] = {}
        self.usage_reports = 0
        self.resumption_updates = 0
        self.go_away: list[str] = []
        self.cancellations = 0
        self.server_signals: dict[str, int] = {}
        self.notes: list[str] = []
        self.tools: dict[str, Any] = {}
        self.tool_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Time and output
    # ------------------------------------------------------------------

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def say(self, line: str) -> None:
        print(f"[{self.now():7.2f}] {line}", flush=True)

    def log(self, kind: str, **facts: Any) -> None:
        if self.log_file is None:
            return
        record = {"t": round(self.now(), 4), "kind": kind, **facts}
        self.log_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def config(self) -> Any:
        from google.genai import types

        a = self.a
        detection = None
        if a.silence_ms or a.prefix_ms or a.start_sensitivity or a.end_sensitivity:
            detection = types.AutomaticActivityDetection(
                silence_duration_ms=a.silence_ms,
                prefix_padding_ms=a.prefix_ms,
                start_of_speech_sensitivity=(
                    types.StartSensitivity[f"START_SENSITIVITY_{a.start_sensitivity}"]
                    if a.start_sensitivity
                    else None
                ),
                end_of_speech_sensitivity=(
                    types.EndSensitivity[f"END_SENSITIVITY_{a.end_sensitivity}"]
                    if a.end_sensitivity
                    else None
                ),
            )
        speech = None
        if a.voice or a.language:
            speech = types.SpeechConfig(
                voice_config=(
                    types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=a.voice)
                    )
                    if a.voice
                    else None
                ),
                language_code=a.language or None,
            )
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=system_prompt(a.user_language),
            tools=[types.Tool(function_declarations=self.declarations())],
            speech_config=speech,
            thinking_config=(
                # A number is a budget, a word a level; the server takes one
                # of them on this model, if either (measured: see ADR-001).
                types.ThinkingConfig(thinking_budget=int(a.thinking))
                if a.thinking.isdigit()
                else types.ThinkingConfig(thinking_level=types.ThinkingLevel[a.thinking])
                if a.thinking
                else None
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=[a.input_language] if a.input_language else None
            ),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=(
                types.RealtimeInputConfig(automatic_activity_detection=detection)
                if detection is not None
                else None
            ),
            # Asked for so the handles come; nothing is resumed here. Whether a
            # resumed session remembers the conversation is L1.1's question.
            session_resumption=types.SessionResumptionConfig(),
        )

    def declarations(self) -> list[Any]:
        """The one tool, in Gemini's envelope, from the product's own `Tool`."""
        from google.genai import types

        from allie.tools.system import get_current_time

        declared = []
        for tool in (get_current_time,):
            spec = tool.spec
            self.tools[spec.name] = tool
            declared.append(
                types.FunctionDeclaration(
                    name=spec.name,
                    description=spec.description,
                    parameters_json_schema=(
                        dict(spec.parameters) if spec.parameters.get("properties") else None
                    ),
                    behavior=types.Behavior.NON_BLOCKING if self.a.non_blocking else None,
                )
            )
        return declared

    # ------------------------------------------------------------------
    # The microphone (PortAudio's thread) and the local detector (the loop)
    # ------------------------------------------------------------------

    def on_chunk(self, chunk: Any) -> None:
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.mic_queue.put_nowait, (self.now(), chunk))

    def judge(self, stamp: float, chunk: Any) -> None:
        """Feeds the detector and keeps two facts: are you talking, and when did you last."""
        from allie.audio.vad import FRAME_SAMPLES, SPEECH_THRESHOLD

        self.vad_buffer = np.concatenate((self.vad_buffer, chunk))
        while len(self.vad_buffer) >= FRAME_SAMPLES:
            frame = self.vad_buffer[:FRAME_SAMPLES]
            self.vad_buffer = self.vad_buffer[FRAME_SAMPLES:]
            speech = self.vad.probability(frame) >= SPEECH_THRESHOLD
            if not speech:
                self.speech_frames = 0
                if (
                    self.user_speaking
                    and self.last_speech_at is not None
                    and stamp - self.last_speech_at >= SILENCE_SECONDS
                ):
                    self.user_speaking = False
                continue
            self.speech_frames += 1
            self.last_speech_at = stamp
            needed = BARGE_FRAMES if self.playing else ONSET_FRAMES
            if not self.user_speaking and self.speech_frames >= needed:
                self.user_speaking = True
                self.log("local_onset", playing=self.playing)
                if self.playing and self.current is not None and self.current.barge_at is None:
                    self.current.barge_at = stamp
                    self.say("      (you started talking over it)")

    async def send(self, session: Any) -> None:
        """Microphone blocks to the server, batched when the socket is slower than 20 ms."""
        from google.genai import types

        from allie.stt.gemini_stt import to_pcm16

        while True:
            stamp, chunk = await self.mic_queue.get()
            chunks = [chunk]
            while not self.mic_queue.empty():
                stamp, more = self.mic_queue.get_nowait()
                chunks.append(more)
            audio = np.concatenate(chunks)
            if self.a.gain != 1.0:
                audio = np.clip(audio * self.a.gain, -1.0, 1.0).astype(np.float32)
            for piece in chunks:
                self.judge(stamp, piece * self.a.gain if self.a.gain != 1.0 else piece)
            deaf = self.a.half_duplex and (
                self.playing or stamp < self.play_ended_at + ECHO_TAIL_SECONDS
            )
            if deaf:
                continue
            self.in_peak = max(self.in_peak, float(np.abs(audio).max()))
            rms = float(np.sqrt(np.mean(np.square(audio))))
            if rms > SILENCE_RMS:
                self.in_rms_sum += rms
                self.in_rms_blocks += 1
            data = to_pcm16(audio)
            self.audio_in_bytes += len(data)
            await session.send_realtime_input(audio=types.Blob(data=data, mime_type=INPUT_MIME))

    async def file_microphone(self) -> None:
        """The sentences of `--wav`, one at a time, at the microphone's pace.

        Silence between them, a wait for the answer after each, and silence
        for good after the last - the server's detector needs the quiet to
        end a turn, and the session needs to stay open for the answer.
        """
        sentences = [load_sentence(path) for path in self.a.wav]
        step = INPUT_RATE // 50
        next_at = time.perf_counter()

        async def tick(chunk: Any) -> None:
            nonlocal next_at
            next_at += step / INPUT_RATE
            await asyncio.sleep(max(0.0, next_at - time.perf_counter()))
            self.mic_queue.put_nowait((self.now(), chunk))

        silence = np.zeros(step, dtype=np.float32)
        for index, sentence in enumerate(sentences):
            for _ in range(50):  # a second of quiet before each sentence
                await tick(silence)
            self.turn_done.clear()
            self.say(f"file mic : {self.a.wav[index].name} ({len(sentence) / INPUT_RATE:.1f} s)")
            self.log("file_sentence", name=self.a.wav[index].name)
            for at in range(0, len(sentence), step):
                piece = sentence[at : at + step]
                if len(piece) < step:
                    piece = np.concatenate((piece, np.zeros(step - len(piece), dtype=np.float32)))
                await tick(piece)
            # The answer, or the server's silence: whichever comes first.
            waited = 0.0
            while not self.turn_done.is_set() and waited < ANSWER_PATIENCE_SECONDS:
                await tick(silence)
                waited += step / INPUT_RATE
            if not self.turn_done.is_set():
                self.say("      [no turn came for this sentence]")
                self.log("file_sentence_unanswered", name=self.a.wav[index].name)
        while True:
            await tick(silence)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    async def play(self, turn: Turn, queue: asyncio.Queue[bytes | None]) -> None:
        async def drain() -> AsyncIterator[bytes]:
            while True:
                item = await queue.get()
                if item is None:
                    return
                yield item

        try:
            await self.speaker.play(drain(), sample_rate=self.output_rate)
        finally:
            turn.heard_done_at = self.now()
            if turn.interrupted and turn.silent_at is None:
                turn.silent_at = turn.heard_done_at
            self.playing = False
            self.play_ended_at = turn.heard_done_at

    async def start_playback(self, turn: Turn) -> asyncio.Queue[bytes | None]:
        if self.play_task is not None and not self.play_task.done():
            # The previous answer is still in the sound card and the next has
            # begun; the speaker plays one at a time, so the old one is cut.
            self.speaker.stop()
            await self.play_task
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.play_queue = queue
        self.playing = True
        self.play_task = asyncio.create_task(self.play(turn, queue))
        return queue

    def flush_playback(self) -> None:
        self.speaker.stop()
        if self.play_queue is not None:
            while not self.play_queue.empty():
                self.play_queue.get_nowait()
            self.play_queue.put_nowait(None)
            self.play_queue = None

    def end_playback(self) -> None:
        if self.play_queue is not None:
            self.play_queue.put_nowait(None)
            self.play_queue = None

    # ------------------------------------------------------------------
    # The server's messages
    # ------------------------------------------------------------------

    def turn(self) -> Turn:
        if self.current is None:
            self.current = Turn(index=len(self.turns) + 1, started_at=self.now())
            self.turns.append(self.current)
        return self.current

    async def receive(self, session: Any) -> None:
        # The SDK's `receive()` ends at each `turn_complete`; the session does not.
        while True:
            async for message in session.receive():
                await self.on_message(session, message)

    async def on_message(self, session: Any, message: Any) -> None:
        now = self.now()
        if message.setup_complete is not None:
            self.session_id = message.setup_complete.session_id or ""
            self.log("setup_complete", session_id=self.session_id)
        if message.session_resumption_update is not None:
            update = message.session_resumption_update
            self.resumption_updates += 1
            self.log(
                "resumption",
                resumable=update.resumable,
                handle_chars=len(update.new_handle or ""),
                last_index=update.last_consumed_client_message_index,
            )
        if message.go_away is not None:
            left = str(message.go_away.time_left)
            self.go_away.append(f"{now:.0f} s: {left} left")
            self.log("go_away", time_left=left)
            self.say(f"      [server: going away, {left} left]")
        if message.voice_activity is not None:
            self.signal(now, str(message.voice_activity.voice_activity_type))
        if message.voice_activity_detection_signal is not None:
            self.signal(now, str(message.voice_activity_detection_signal.vad_signal_type))
        if message.usage_metadata is not None:
            self.usage(message.usage_metadata)
        if message.tool_call_cancellation is not None:
            self.cancellations += 1
            self.log("tool_call_cancellation", ids=message.tool_call_cancellation.ids)
            self.say(f"      [tool call cancelled: {message.tool_call_cancellation.ids}]")
        if message.tool_call is not None:
            await self.on_tool_call(session, message.tool_call, now)
        if message.server_content is not None:
            await self.on_content(message.server_content, now)

    def signal(self, now: float, kind: str) -> None:
        name = kind.rsplit(".", 1)[-1]
        self.server_signals[name] = self.server_signals.get(name, 0) + 1
        self.log("server_signal", signal=name)
        if name.endswith(("EOS", "ACTIVITY_END")):
            self.server_eos_at = now

    def usage(self, usage: Any) -> None:
        self.usage_reports += 1
        counted: dict[str, int] = {}
        for prefix, details in (
            ("in", usage.prompt_tokens_details),
            ("out", usage.response_tokens_details),
            ("tool", usage.tool_use_prompt_tokens_details),
        ):
            for detail in details or ():
                modality = str(detail.modality).rsplit(".", 1)[-1].lower()
                counted[f"{prefix}_{modality}"] = detail.token_count or 0
        counted["total"] = usage.total_token_count or 0
        if usage.thoughts_token_count:
            counted["thoughts"] = usage.thoughts_token_count
        self.log("usage", **counted)
        target = (
            self.current if self.current is not None else (self.turns[-1] if self.turns else None)
        )
        for key, value in counted.items():
            self.tokens[key] = self.tokens.get(key, 0) + value
            if target is not None:
                target.tokens[key] = target.tokens.get(key, 0) + value

    async def on_tool_call(self, session: Any, call_message: Any, now: float) -> None:
        turn = self.turn()
        for call in call_message.function_calls or ():
            turn.tool_name = call.name or "?"
            turn.tool_called_at = now
            turn.tool_answered_at = None
            turn.audio_after_tool_at = None
            self.log("tool_call", name=call.name, args=call.args, id=call.id)
            self.say(f"      [tool call: {call.name}({call.args or ''})]")
            # Run apart from the receive loop: what the server sends while the
            # tool works - nothing when blocking, speech when NON_BLOCKING - is
            # exactly what O5 wants to see, and it can only be seen if the
            # loop keeps receiving.
            self.tool_tasks.add(asyncio.create_task(self.answer_tool(session, turn, call)))

    async def answer_tool(self, session: Any, turn: Turn, call: Any) -> None:
        from google.genai import types

        tool = self.tools.get(call.name or "")
        if tool is None:
            result = f"unknown tool {call.name!r}"
        else:
            result = await tool.run(**(call.args or {}))
        if self.a.tool_delay:
            # A page, the Store, the weather: the seconds a real tool takes.
            await asyncio.sleep(self.a.tool_delay)
        scheduling = (
            types.FunctionResponseScheduling[self.a.scheduling] if self.a.non_blocking else None
        )
        await session.send_tool_response(
            function_responses=[
                types.FunctionResponse(
                    id=call.id,
                    name=call.name,
                    response={"result": result},
                    scheduling=scheduling,
                )
            ]
        )
        turn.tool_answered_at = self.now()
        self.log("tool_response", name=call.name, result=result, scheduling=str(scheduling))
        during = turn.audio_during_tool_bytes / (self.output_rate * BYTES_PER_SAMPLE)
        self.say(
            f"      [tool answered after {turn.tool_ours_ms() or 0:.0f} ms: {result};"
            f" it spoke {during:.1f} s meanwhile]"
        )

    async def on_content(self, content: Any, now: float) -> None:
        turn = self.turn()
        facts: dict[str, Any] = {}

        heard = content.input_transcription
        if heard is not None and heard.text:
            turn.heard += heard.text
            facts["heard"] = heard.text
            facts["heard_finished"] = heard.finished
        interim = content.interim_input_transcription
        if interim is not None and interim.text:
            facts["interim"] = interim.text
        said = content.output_transcription
        if said is not None and said.text:
            turn.said += said.text
            facts["said"] = said.text
            if turn.tool_called_at is not None and turn.tool_answered_at is None:
                turn.said_during_tool += said.text

        audio_bytes = 0
        if content.model_turn is not None:
            for part in content.model_turn.parts or ():
                blob = part.inline_data
                if blob is None or not blob.data:
                    if part.text:
                        facts["text_part"] = part.text
                    continue
                audio_bytes += len(blob.data)
                if not self.output_mime:
                    self.output_mime = blob.mime_type or ""
                    self.output_rate = _rate_of(self.output_mime, OUTPUT_RATE)
                    self.log("output_format", mime=self.output_mime, rate=self.output_rate)
                if turn.first_audio_at is None:
                    turn.first_audio_at = now
                    turn.eos_local_at = self.last_speech_at
                    turn.eos_server_at = self.server_eos_at
                    self.say(
                        f"      [first sound: {_n(turn.wait_local_ms())} ms after the local"
                        f" end of speech, {_n(turn.wait_server_ms())} ms after the server's]"
                    )
                if turn.tool_called_at is not None and turn.audio_after_tool_at is None:
                    turn.audio_after_tool_at = now
                    self.say(
                        f"      [sound again {turn.tool_ms() or 0:.0f} ms after the call,"
                        f" silence {turn.tool_silence_ms() or 0:.0f} ms]"
                    )
                turn.audio_out_bytes += len(blob.data)
                self.audio_out_bytes += len(blob.data)
                if turn.tool_called_at is not None and turn.tool_answered_at is None:
                    turn.audio_during_tool_bytes += len(blob.data)
                if turn.tool_called_at is None or turn.audio_after_tool_at is not None:
                    turn.audio_before_tool_at = now
                if self.play_queue is None:
                    queue = await self.start_playback(turn)
                else:
                    queue = self.play_queue
                queue.put_nowait(blob.data)
        if audio_bytes:
            facts["audio_bytes"] = audio_bytes

        if content.interrupted:
            turn.interrupted_at = now
            facts["interrupted"] = True
            self.flush_playback()
            if turn.barge_at is None:
                self.say("      [interrupted by the server; the local detector heard no onset]")
            else:
                self.say(f"      [interrupted {turn.interrupt_ms() or 0:.0f} ms after you started]")
        if content.generation_complete:
            facts["generation_complete"] = True
        if content.turn_complete:
            facts["turn_complete"] = True
            reason = str(content.turn_complete_reason or "").rsplit(".", 1)[-1]
            if reason:
                facts["reason"] = reason
            if turn.tool_called_at is not None and turn.audio_after_tool_at is None:
                # The server ends the model's turn at the tool call (measured
                # 2026-09-18: generation_complete, usage, turn_complete, in
                # that order, before our answer is even sent); the spoken
                # answer comes as the next turn. Here a turn is the whole
                # exchange, so this one stays open.
                facts["tool_half"] = True
                self.say("      [turn_complete at the tool call; waiting for the answer]")
            else:
                turn.complete_at = now
                turn.complete_reason = reason
                self.end_playback()
                self.server_eos_at = None
                self.current = None
                self.turn_done.set()
                self.report_turn(turn)
        if facts:
            self.log("content", turn=turn.index, **facts)

    def report_turn(self, turn: Turn) -> None:
        heard = _shown(turn.heard) or "(nothing transcribed)"
        said = _shown(turn.said) or "(no words)"
        self.say(f"turn {turn.index}: you  : {heard}")
        self.say(f"        it   : {said}")
        seconds = turn.audio_out_bytes / (self.output_rate * BYTES_PER_SAMPLE)
        pieces = [f"{seconds:.1f} s of speech"]
        if turn.tool_name:
            pieces.append(f"tool {turn.tool_name}")
        if turn.interrupted:
            pieces.append("interrupted")
        if turn.complete_reason and turn.complete_reason != "TURN_COMPLETE_REASON_UNSPECIFIED":
            pieces.append(turn.complete_reason)
        if turn.tokens:
            pieces.append(
                "tokens "
                + " ".join(f"{k}={v}" for k, v in sorted(turn.tokens.items()) if k != "total")
            )
        self.say("        " + ", ".join(pieces))

    # ------------------------------------------------------------------
    # Typed turns, the keyboard, the run
    # ------------------------------------------------------------------

    async def typed(self, session: Any) -> None:
        from google.genai import types

        for text in self.a.text or ():
            self.turn_done.clear()
            self.say(f"typed: {text}")
            self.log("typed", text=text)
            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=text)]), turn_complete=True
            )
            await self.turn_done.wait()

    def watch_keyboard(self) -> None:
        def wait_for_enter() -> None:
            # End of file is a missing terminal, not a key press.
            line = sys.stdin.readline() if sys.stdin is not None else ""
            if line and self.loop is not None:
                self.loop.call_soon_threadsafe(self.hang_up, "you (Enter)")

        threading.Thread(target=wait_for_enter, daemon=True).start()

    def hang_up(self, who: str) -> None:
        if not self.stop.is_set():
            self.closed_by = who
            self.stop.set()

    async def run(self) -> int:
        from google import genai
        from google.genai import errors
        from websockets.exceptions import ConnectionClosed, WebSocketException

        from allie.audio.capture import SystemMicrophone, device_choice
        from allie.audio.player import SystemSpeaker
        from allie.audio.vad import SileroVAD

        self.loop = asyncio.get_running_loop()
        if self.a.log is not None:
            self.a.log.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = self.a.log.open("w", encoding="utf-8")

        key, source = find_key()
        client = genai.Client(api_key=key)
        self.say(f"key      : {source}")
        self.say(
            f"model    : {self.a.model}, voice {self.a.voice or '(default)'}, "
            f"language {self.a.language or '(default)'}, thinking {self.a.thinking or '(default)'}"
        )
        self.say(
            f"tool     : get_current_time, "
            f"{'NON_BLOCKING, ' + self.a.scheduling if self.a.non_blocking else 'blocking'}"
        )
        self.say(f"duplex   : {'half (deaf while it speaks)' if self.a.half_duplex else 'full'}")
        self.say(
            f"hints    : input transcription {self.a.input_language or '(none)'}, "
            f"user language {self.a.user_language or '(none)'}, gain x{self.a.gain:g}"
        )

        self.vad = SileroVAD()
        await self.vad.load()
        self.speaker = SystemSpeaker()
        microphone = None
        file_task: asyncio.Task[None] | None = None
        if self.a.wav:
            self.say(f"mic      : {len(self.a.wav)} WAV sentence(s) played into the session")
        elif self.a.no_mic:
            self.say("mic      : not opened (--no-mic); the typed turns are the whole session")
        else:
            microphone = SystemMicrophone(device=device_choice(self.a.device))
            microphone.open(self.on_chunk)
            host = _host_api(device_choice(self.a.device))
            self.say(
                f"mic      : {self.a.device or '(system default)'} at {microphone.rate} Hz, {host}"
            )
            if "WDM-KS" in host and not self.a.half_duplex:
                # Measured 2026-09-18: kernel streaming bypasses the Windows
                # audio engine and with it the driver's echo cancellation, so
                # the speaker feeds the microphone, the server hears its own
                # voice, interrupts itself and answers itself - in French.
                self.say(
                    "WARNING  : a WDM-KS device has no echo cancellation; on speakers the model "
                    "will interrupt itself. Use the WASAPI or MME entry, or --half-duplex."
                )
        self.say("say something; press Enter to hang up")
        self.watch_keyboard()

        code = 0
        started = self.now()
        try:
            async with client.aio.live.connect(model=self.a.model, config=self.config()) as session:
                self.opened_at = self.now()
                self.open_ms = (self.opened_at - started) * 1000.0
                self.log("open", ms=round(self.open_ms))
                self.say(f"session  : open after {self.open_ms:.0f} ms")
                tasks = {
                    asyncio.create_task(self.receive(session), name="receive"),
                    asyncio.create_task(self.typed(session), name="typed"),
                    asyncio.create_task(self.until_hung_up(), name="stop"),
                }
                if microphone is not None or self.a.wav:
                    tasks.add(asyncio.create_task(self.send(session), name="send"))
                if self.a.wav:
                    file_task = asyncio.create_task(self.file_microphone(), name="file")
                    tasks.add(file_task)
                try:
                    while True:
                        done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            failure = task.exception()
                            if failure is not None:
                                raise failure
                        if any(task.get_name() == "stop" for task in done):
                            break
                        # The typed turns ending is the end only without a
                        # microphone; otherwise the conversation goes on.
                        if (
                            microphone is None
                            and not self.a.wav
                            and any(t.get_name() == "typed" for t in done)
                        ):
                            self.hang_up("the typed turns ending (--no-mic)")
                finally:
                    for task in tasks | self.tool_tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, *self.tool_tasks, return_exceptions=True)
        except errors.APIError as failure:
            message = " ".join(line.strip() for line in (failure.message or "").splitlines())
            self.closed_by = f"the server refused: {failure.code} {message[:240]}"
            code = REFUSED if failure.code in (400, 401, 403) else UNREACHABLE
        except ConnectionClosed as failure:
            self.closed_by = f"the server closed the socket: {failure.code} {failure.reason!r}"
        except (WebSocketException, OSError) as failure:
            self.closed_by = f"{type(failure).__name__}: {failure}"[:240]
            code = UNREACHABLE
        except (KeyboardInterrupt, asyncio.CancelledError):
            # `asyncio.run` turns Ctrl+C into a cancellation of this task and
            # raises KeyboardInterrupt only after the loop is gone; from here
            # the two look different and mean the same.
            self.closed_by = "you (Ctrl+C)"
        finally:
            self.closed_at = self.now()
            if microphone is not None:
                microphone.close()
            self.flush_playback()
            if self.play_task is not None:
                await asyncio.gather(self.play_task, return_exceptions=True)
            self.summary()
            if self.log_file is not None:
                self.log_file.close()
        return code

    async def until_hung_up(self) -> None:
        if self.a.minutes:
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=self.a.minutes * 60.0)
            except TimeoutError:
                self.hang_up(f"the --minutes cap ({self.a.minutes:g})")
        else:
            await self.stop.wait()

    # ------------------------------------------------------------------
    # The summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Printed, and - with `--log` - also written next to the log with the
        whole transcript, so the numbers survive the console."""
        lines: list[str] = []

        def emit(line: str = "") -> None:
            print(line)
            lines.append(line)

        emit("\n" + "=" * 78)
        emit("SUMMARY")
        emit("=" * 78)
        opened = f"opened in {self.open_ms:.0f} ms" if self.open_ms is not None else "never opened"
        lasted = (
            _clock((self.closed_at or 0.0) - self.opened_at) if self.opened_at is not None else "-"
        )
        emit(f"session   : {opened}, lasted {lasted}, ended by {self.closed_by}")
        if self.session_id:
            emit(f"            id {self.session_id}")
        emit(
            f"model     : {self.a.model}, voice {self.a.voice or '(default)'}, language "
            f"{self.a.language or '(default)'}, thinking {self.a.thinking or '(default)'}, tool "
            f"{'NON_BLOCKING/' + self.a.scheduling if self.a.non_blocking else 'blocking'}, "
            f"{'half' if self.a.half_duplex else 'full'} duplex"
        )
        interrupted = sum(1 for turn in self.turns if turn.interrupted)
        with_tool = sum(1 for turn in self.turns if turn.tool_name)
        emit(f"turns     : {len(self.turns)} ({interrupted} interrupted, {with_tool} with a tool)")
        sent = self.audio_in_bytes / (INPUT_RATE * BYTES_PER_SAMPLE)
        received = self.audio_out_bytes / (self.output_rate * BYTES_PER_SAMPLE)
        emit(
            f"audio     : sent {sent:.1f} s ({_clock(sent)}), received {received:.1f} s "
            f"({_clock(received)}) as {self.output_mime or '(none)'}"
        )
        rms = self.in_rms_sum / self.in_rms_blocks if self.in_rms_blocks else 0.0
        emit(
            f"level     : sent peak {self.in_peak:.3f} ({_dbfs(self.in_peak)}), speech rms "
            f"{rms:.4f} ({_dbfs(rms)}) over {self.in_rms_blocks} non-silent blocks, "
            f"gain x{self.a.gain:g}"
        )

        def stat(pick: Callable[[Turn], float | None]) -> str:
            return _stat([pick(turn) for turn in self.turns])

        emit(f"wait      : local end-of-speech -> first sound: {stat(Turn.wait_local_ms)}")
        emit(f"            server end-of-speech -> first sound: {stat(Turn.wait_server_ms)}")
        emit(f"talk-over : you -> server said interrupted: {stat(Turn.interrupt_ms)}")
        emit(f"            you -> speaker silent: {stat(Turn.silence_ms)}")
        emit(f"tool      : call -> sound again: {stat(Turn.tool_ms)}")
        emit(f"            silence around it: {stat(Turn.tool_silence_ms)}")
        emit(f"            our side (run + delay + send): {stat(Turn.tool_ours_ms)}")
        emit(f"            our answer -> sound again: {stat(Turn.answer_to_sound_ms)}")
        for turn in self.turns:
            if turn.tool_name and turn.audio_during_tool_bytes:
                during = turn.audio_during_tool_bytes / (self.output_rate * BYTES_PER_SAMPLE)
                emit(
                    f"            turn {turn.index}: spoke {during:.1f} s while the tool ran:"
                    f" {_shown(turn.said_during_tool)!r}"
                )
        tokens = " ".join(f"{k}={v}" for k, v in sorted(self.tokens.items()))
        emit(f"tokens    : {tokens or '(no usage reported)'} over {self.usage_reports} reports")
        signals = ", ".join(f"{k} x{v}" for k, v in sorted(self.server_signals.items()))
        emit(
            f"signals   : {signals or 'none'}; resumption updates {self.resumption_updates}; "
            f"tool cancellations {self.cancellations}"
        )
        emit(f"go_away   : {'; '.join(self.go_away) if self.go_away else 'none'}")
        reasons = sorted(
            {turn.complete_reason for turn in self.turns if turn.complete_reason}
            - {"TURN_COMPLETE_REASON_UNSPECIFIED"}
        )
        if reasons:
            emit(f"reasons   : {', '.join(reasons)}")
        for note in self.notes:
            emit(f"note      : {note}")
        if self.turns:
            emit("\n  #  wait(local/server)  tool     talk-over   speech  heard / said")
            for turn in self.turns:
                wait = f"{_n(turn.wait_local_ms())}/{_n(turn.wait_server_ms())}"
                tool = _n(turn.tool_ms())
                over = _n(turn.silence_ms()) if turn.interrupted else "-"
                speech = f"{turn.audio_out_bytes / (self.output_rate * BYTES_PER_SAMPLE):.1f} s"
                emit(
                    f"{turn.index:3d}  {wait:>18}  {tool:>7}  {over:>9}  {speech:>7}  "
                    f"{_shown(turn.heard)[:34]} / {_shown(turn.said)[:34]}"
                )
        emit()

        if self.a.log is not None:
            lines.append("TRANSCRIPT")
            for turn in self.turns:
                lines.append(f"{turn.index:3d}  you : {' '.join(turn.heard.split())}")
                lines.append(f"     it  : {' '.join(turn.said.split())}")
            report = self.a.log.with_suffix(".summary.txt")
            report.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"summary written to {report}")


def _n(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def _dbfs(value: float) -> str:
    import math

    return f"{20 * math.log10(value):.0f} dBFS" if value > 0 else "-inf dBFS"


def _host_api(device: int | str | None) -> str:
    """Which way to the microphone: WASAPI/MME go through Windows' effects, WDM-KS does not."""
    import sounddevice  # type: ignore[import-untyped]

    try:
        facts = sounddevice.query_devices(device, "input")
        return str(sounddevice.query_hostapis(facts["hostapi"])["name"])
    except (ValueError, sounddevice.PortAudioError):
        return "unknown host API"


def _rate_of(mime: str, default: int) -> int:
    for piece in mime.split(";"):
        key, _, value = piece.strip().partition("=")
        if key == "rate" and value.isdigit():
            return int(value)
    return default


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--voice", default="", help="A prebuilt voice name, e.g. Kore, Puck, Aoede."
    )
    parser.add_argument("--language", default="", help="BCP-47 for the voice, e.g. tr-TR.")
    parser.add_argument("--device", default=default_device(), help="Microphone words or index.")
    parser.add_argument(
        "--non-blocking", action="store_true", help="Declare the tool NON_BLOCKING (O5)."
    )
    parser.add_argument(
        "--tool-delay", type=float, default=0.0, help="Seconds the tool pretends to take."
    )
    parser.add_argument(
        "--thinking",
        default="",
        help="thinking_level (MINIMAL/LOW/MEDIUM/HIGH) or a thinking_budget number; "
        "empty leaves the model's default.",
    )
    parser.add_argument(
        "--scheduling", default="INTERRUPT", choices=("INTERRUPT", "WHEN_IDLE", "SILENT")
    )
    parser.add_argument(
        "--half-duplex", action="store_true", help="Send no microphone while it speaks."
    )
    parser.add_argument(
        "--silence-ms", type=int, default=None, help="Server VAD silence_duration_ms."
    )
    parser.add_argument("--prefix-ms", type=int, default=None, help="Server VAD prefix_padding_ms.")
    parser.add_argument("--start-sensitivity", default=None, choices=("HIGH", "LOW"))
    parser.add_argument("--end-sensitivity", default=None, choices=("HIGH", "LOW"))
    parser.add_argument("--text", action="append", help="A typed turn sent before listening.")
    parser.add_argument("--no-mic", action="store_true", help="Typed turns only; no microphone.")
    parser.add_argument(
        "--wav",
        action="append",
        type=Path,
        default=[],
        help="A WAV sentence played into the session instead of the microphone (repeatable).",
    )
    parser.add_argument(
        "--gain", type=float, default=1.0, help="Multiply the microphone before sending."
    )
    parser.add_argument(
        "--input-language", default="", help="BCP-47 for the input transcription, e.g. tr-TR."
    )
    parser.add_argument(
        "--user-language",
        default="",
        help="A language name added to the prompt as the user's, e.g. Turkish.",
    )
    parser.add_argument("--minutes", type=float, default=0.0, help="Hang up after this long.")
    parser.add_argument(
        "--log", type=Path, default=None, help="Write every message here (JSON lines)."
    )
    arguments = parser.parse_args()

    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    spike = Spike(arguments)
    try:
        return asyncio.run(spike.run())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
