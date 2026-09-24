"""What the command line does with each command.

`setup` is wired to the wizard here rather than driven through it - the wizard
has its own suite, and a test that opened a real prompt would hang. `run` is
the same idea one level up: the state machine, the speech model and the model
that answers all have their own suites, so what is tested here is that they are
handed to each other correctly and that starting up fails in words.

**Nothing may touch the hardware.** A test that loaded Whisper would take two
seconds and a gigabyte, and one that opened the microphone would record the
room. Both are replaced; everything else is the code that ships.

**A machine that is not set up is not a crash.** No settings, or a key that has
been removed from the Credential Manager, are things the user can fix - so they
are a sentence and an exit code, and neither of them costs a model load.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar

import pytest
from loguru import logger

from allie import __main__ as cli
from allie import app, locales, logs, setup_wizard
from allie.__main__ import BUILTIN_TOOLS, TEXT, build_parser, main, use_utf8
from allie.agent import core
from allie.agent.limits import Limits
from allie.agent.policy import NO_SUCH_TOOL
from allie.agent.prompts import SEARCH_RULE, SYSTEM_PROMPT
from allie.app import State, Turn, confirm_prompt
from allie.audio import capture
from allie.audio import wake as wake_module
from allie.config import (
    KEYRING_SERVICE,
    AudioSettings,
    LimitSettings,
    LiveSettings,
    LocaleSettings,
    MessagingSettings,
    RetentionSettings,
    Settings,
    STTSettings,
    TTSSettings,
    WakeSettings,
    WebSettings,
    config_path,
    load_settings,
    save_settings,
    store_api_key,
)
from allie.live import probe
from allie.live.base import ProviderError, SessionConfig, ToolCall, Usage
from allie.live.probe import ProbeResult, remember, remembered
from allie.messaging.contacts import CONTACTS_FILE_NAME
from allie.store import db
from allie.store.memory import MEMORY_FILE_NAME
from allie.store.repos import AuditRepo, SettingsRepo, UsageRepo
from allie.store.retention import SECONDS_PER_DAY
from allie.stt import gemini_stt, local_whisper
from allie.tools import system
from allie.tools.system import AppCatalog, AppEntry
from allie.ui import status
from allie.usage.tracker import UsageTracker
from allie.web import search as search_module
from allie.web.search import Searched
from tests.conftest import MemoryKeyring

MODEL = "gemini-3.8-live"
TURN = Turn(heard="saat kaç", said="Üç buçuk.", usage=Usage(300, 10))
# What setup wrote down about the model, unless a test says otherwise.
PASSED = ProbeResult(ok=True, first_token_ms=12.0)


@pytest.fixture(autouse=True)
def own_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The log goes into this test's directory, not the developer's machine."""
    monkeypatch.setattr(logs, "log_path", lambda: tmp_path / "logs" / "assistant.log")
    yield
    logger.remove()


@pytest.fixture
def configured(config_home: Path, vault: MemoryKeyring) -> Path:
    """A machine `allie setup` has already been run on."""
    save_settings(
        Settings(
            live=LiveSettings(primary=f"gemini:{MODEL}"),
            locale=LocaleSettings(code="tr"),
        )
    )
    store_api_key("gemini", "AIza-not-a-real-key")
    return config_home


@pytest.fixture
def own_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The database under this test's directory, never the machine's."""
    path = tmp_path / "data" / "assistant.db"
    monkeypatch.setattr(db, "database_path", lambda: path)
    return path


def said(key: str, code: str = "tr") -> str:
    """A sentence as the pack for `code` has it - whatever it was reworded to."""
    return locales.load(code).say(key, {**TEXT, **status.TEXT, **cli.DOCTOR_TEXT}[key])


# --------------------------------------------------------------------------
# The parser
# --------------------------------------------------------------------------


def test_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "assistant" in capsys.readouterr().out


def test_no_command_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["setup", "run", "cost", "mic"])
def test_the_command_names_are_declared(command: str) -> None:
    assert build_parser().parse_args([command]).command == command


def test_run_can_be_asked_for_the_tray() -> None:
    """`run --tray` (4.3); `run --terminal` keeps the line instead of the
    window (D20). Without the flags: the window, no tray."""
    assert build_parser().parse_args(["run", "--tray"]).tray is True
    assert build_parser().parse_args(["run"]).tray is False
    assert build_parser().parse_args(["run", "--terminal"]).terminal is True
    assert build_parser().parse_args(["run"]).terminal is False


def test_telegram_login_is_a_command_of_its_own() -> None:
    """`allie telegram login` (2026-09-15); a bare `telegram` is a usage error."""
    parsed = build_parser().parse_args(["telegram", "login"])

    assert (parsed.command, parsed.telegram_command) == ("telegram", "login")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["telegram"])


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------


def test_setup_runs_the_wizard(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    async def wizard(prompter: object, **kwargs: object) -> int:
        calls.append(prompter)
        return 0

    monkeypatch.setattr(setup_wizard, "run_setup", wizard)

    assert main(["setup"]) == 0
    assert isinstance(calls[0], setup_wizard.TerminalPrompter)


def test_the_exit_code_of_the_wizard_is_the_exit_code_of_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled setup must not look like a successful one to a script."""

    async def wizard(prompter: object, **kwargs: object) -> int:
        return 1

    monkeypatch.setattr(setup_wizard, "run_setup", wizard)

    assert main(["setup"]) == 1


# --------------------------------------------------------------------------
# mic
# --------------------------------------------------------------------------


def test_mic_asks_the_microphone_question_on_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    """The day a headset comes out: one command, the list, and the file is
    updated - no text editor."""
    calls: list[object] = []

    async def wizard(prompter: object, **kwargs: object) -> int:
        calls.append(prompter)
        return 0

    monkeypatch.setattr(setup_wizard, "run_microphone_setup", wizard)

    assert main(["mic"]) == 0
    assert isinstance(calls[0], setup_wizard.TerminalPrompter)


def test_the_exit_code_of_mic_is_the_exit_code_of_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def wizard(prompter: object, **kwargs: object) -> int:
        return 1

    monkeypatch.setattr(setup_wizard, "run_microphone_setup", wizard)

    assert main(["mic"]) == 1


# --------------------------------------------------------------------------
# run: what the pieces did, and what they were given
# --------------------------------------------------------------------------


@dataclass
class Wiring:
    """What the pieces did, and what they were given, while `run` ran."""

    happened: list[str] = field(default_factory=list)
    # What the state machine was built with.
    built: list[dict[str, Any]] = field(default_factory=list)
    # The tool round: the registry, the gate and the limits it was handed.
    runners: list[FakeRunner] = field(default_factory=list)
    # The doorman and the stream: what it was built with.
    captures: list[FakeLiveCapture] = field(default_factory=list)
    microphones: list[Any] = field(default_factory=list)
    databases: list[sqlite3.Connection] = field(default_factory=list)
    # What the speech model was told to expect: the words of the confirmation
    # window, since that is all it hears (D3, D10).
    prompts_for_speech: list[str] = field(default_factory=list)
    # Google's recogniser, when the settings ask for it: what it was built
    # with (2026-09-14).
    recognisers: list[dict[str, Any]] = field(default_factory=list)
    stop: BaseException | None = None
    # The probe of 2.6 at startup: what it was asked, as (provider id,
    # model, question), and what it answers.
    probes: list[tuple[str, str, str]] = field(default_factory=list)
    verdict: ProbeResult = field(default_factory=lambda: PASSED)
    probe_refusal: Exception | None = None
    # What the state machine does while it "runs", when a test wants more
    # than one turn reported: the tray's "quit" arrives in the middle of it.
    during_run: Callable[[], Awaitable[None]] | None = None
    # The level the microphone sent, as the capture reports it at the close of
    # a session and judged against the threshold (D18, `level_judged`); `None`
    # until a test sets one, and `None` from a raw-path microphone.
    level: float | None = None


class FakeRunner:
    """Stands in for `agent/core.py`'s `ToolRunner`: what it was built with."""

    def __init__(self, tools: Any, dispatch: Any, limits: Any) -> None:
        self.registry = tools
        self.dispatch = dispatch
        self.limits = limits

    @property
    def tools(self) -> list[str]:
        return [spec.name for spec in self.registry.specs()]

    def specs(self) -> list[Any]:
        return list(self.registry.specs())


class FakeLiveCapture:
    """Stands in for `audio/capture.py`'s `LiveCapture`: what it was built
    with, the switch the tray is handed, and the level it reports."""

    def __init__(
        self,
        *,
        microphone: Any,
        endpoint: Any,
        barge_in: bool,
        on_level: Any = None,
        wake: Any = None,
        on_wake: Any = None,
    ) -> None:
        self.microphone = microphone
        self.endpoint = endpoint
        self.barge_in = barge_in
        self.on_level = on_level
        self.wake = wake
        self.on_wake = on_wake
        self.level_dbfs: float | None = None
        self.level_judged: float | None = None
        self.toggles = 0

    def toggle(self) -> None:
        self.toggles += 1


class FakeWakeWord:
    """Stands in for `audio/wake.py`'s `LiveKitWakeWord`: what it was built with."""

    built: ClassVar[list[tuple[Path, float]]] = []

    def __init__(self, model_path: Path, *, threshold: float) -> None:
        self.name = model_path.stem
        self.loaded = False
        FakeWakeWord.built.append((model_path, threshold))

    async def load(self) -> None:
        self.loaded = True

    def feed(self, chunk: Any) -> bool:
        return False

    def reset(self) -> None:
        return


@pytest.fixture
def wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Wiring:
    """Replaces the speech model, the tool round, the capture and the state
    machine, and points the database at this test's directory - the real
    `open_database` runs, so that the schema is built the way it is built
    in life."""
    seen = Wiring()
    really_open = db.open_database

    def open_here(path: Path | str | None = None) -> sqlite3.Connection:
        seen.happened.append("database")
        connection = really_open(path)
        seen.databases.append(connection)
        return connection

    class FakeWhisper:
        def __init__(self, *, prompt: str = "") -> None:
            seen.prompts_for_speech.append(prompt)

        async def load(self) -> None:
            seen.happened.append("speech model")

    class FakeGemini:
        def __init__(self, api_key: str, **rest: Any) -> None:
            self.fallback = rest.get("fallback")
            seen.recognisers.append(
                {
                    "api_key": api_key,
                    "model": rest.get("model"),
                    "fallback": self.fallback,
                }
            )

        async def load(self) -> None:
            if self.fallback is not None:
                await self.fallback.load()
            seen.happened.append("gemini")

    async def catalogue_here(**_: Any) -> AppCatalog:
        seen.happened.append("app catalogue")
        return AppCatalog(INSTALLED)

    def runner_here(tools: Any, dispatch: Any, limits: Any) -> FakeRunner:
        runner = FakeRunner(tools, dispatch, limits)
        seen.runners.append(runner)
        return runner

    class FakeMicrophone:
        def __init__(self, *, device: Any = None) -> None:
            self.device = device
            seen.microphones.append(device)

    def capture_here(**parts: Any) -> FakeLiveCapture:
        built = FakeLiveCapture(**parts)
        built.level_dbfs = seen.level
        built.level_judged = seen.level
        seen.captures.append(built)
        return built

    class FakeAssistant:
        def __init__(self, **parts: Any) -> None:
            seen.happened.append("assistant")
            seen.built.append(parts)

        async def run(self) -> None:
            if seen.stop is not None:
                raise seen.stop
            parts = seen.built[-1]
            parts["on_state"](State.USER_SPEAKING)
            parts["on_session"](True)
            parts["on_turn"](TURN)
            parts["on_session"](False)
            if seen.during_run is not None:
                await seen.during_run()

    async def probed(provider: Any, model: str, *, question: str) -> ProbeResult:
        seen.happened.append("probe")
        seen.probes.append((provider.id, model, question))
        if seen.probe_refusal is not None:
            raise seen.probe_refusal
        return seen.verdict

    monkeypatch.setattr(probe, "probe_tool_support", probed)
    monkeypatch.setattr(local_whisper, "LocalWhisper", FakeWhisper)
    monkeypatch.setattr(gemini_stt, "GeminiSTT", FakeGemini)
    monkeypatch.setattr(system.AppCatalog, "load", catalogue_here)
    monkeypatch.setattr(capture, "SystemMicrophone", FakeMicrophone)
    monkeypatch.setattr(capture, "LiveCapture", capture_here)
    FakeWakeWord.built.clear()
    monkeypatch.setattr(wake_module, "LiveKitWakeWord", FakeWakeWord)
    monkeypatch.setattr(core, "ToolRunner", runner_here)
    monkeypatch.setattr(app, "LiveAssistant", FakeAssistant)
    monkeypatch.setattr(db, "database_path", lambda: tmp_path / "data" / "assistant.db")
    monkeypatch.setattr(db, "open_database", open_here)
    return seen


class OtherProvider:
    """A live provider that is not Google's, for the two engines that need
    Google's key while the model does not."""

    id = "other"
    capabilities: ClassVar[frozenset[str]] = frozenset()


@pytest.fixture
def other_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adds a second entry to the shipped catalogue, with an adapter that
    builds without a network - the OpenAI one is L2's, and nothing here
    depends on which it is."""
    from allie.live import registry

    shipped = registry.load_catalog()
    entry = registry.ProviderEntry(id="other", adapter="other", display_name="Other")
    monkeypatch.setattr(registry, "load_catalog", lambda path=None: {**shipped, "other": entry})
    monkeypatch.setitem(registry.ADAPTERS, "other", lambda entry, key: OtherProvider())


# What the machine is pretended to have, so that no test scans the real one.
INSTALLED = [
    AppEntry("Spotify", r"shell:AppsFolder\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify"),
    AppEntry("Google Chrome", r"C:\Programs\Google Chrome.lnk"),
]


def configured_with(**tables: Any) -> None:
    """The settings of `configured`, with the given tables over them."""
    chosen: dict[str, Any] = {
        "live": LiveSettings(primary=f"gemini:{MODEL}"),
        "locale": LocaleSettings(code="tr"),
        **tables,
    }
    save_settings(Settings(**chosen))


def session_of(wiring: Wiring) -> SessionConfig:
    """What the state machine would open a session with, read now."""
    [parts] = wiring.built
    config: SessionConfig = parts["session_config"]()
    return config


# --------------------------------------------------------------------------
# run: a machine that cannot start
# --------------------------------------------------------------------------


def test_a_machine_that_was_never_set_up_is_told_to_run_setup(
    config_home: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--terminal"]) == 1
    assert said("not_set_up", locales.system_code()) in capsys.readouterr().out


def test_nothing_is_loaded_before_it_is_known_there_is_anything_to_run(
    config_home: Path, wiring: Wiring
) -> None:
    """Whisper is two seconds and a gigabyte. Neither is spent finding out that
    the user has not run setup."""
    main(["run", "--terminal"])

    assert wiring.happened == []


def test_a_key_that_is_gone_is_a_sentence_rather_than_a_traceback(
    configured: Path, vault: MemoryKeyring, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """Somebody cleaned out the Credential Manager. The provider is named,
    because that is what tells the user which key to put back."""
    vault.vault.clear()

    assert main(["run", "--terminal"]) == 1

    printed = capsys.readouterr().out
    assert "gemini" in printed
    assert wiring.happened == []


@pytest.mark.parametrize(
    "problem",
    [
        app.NoVoiceError("no speech voice is installed, for 'tr' or otherwise"),
        local_whisper.ModelUnavailableError("the speech model 'small' could not be loaded"),
        capture.MicrophoneUnavailableError("the microphone 'nope' could not be opened"),
    ],
    ids=["voice", "model", "microphone"],
)
def test_what_the_user_can_fix_is_a_sentence_rather_than_a_traceback(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str], problem: Exception
) -> None:
    """A voice that is not installed, weights that could not be fetched, a
    microphone that would not open. Each is the user's to fix, and a traceback
    tells them nothing about how."""
    wiring.stop = problem

    assert main(["run", "--terminal"]) == 1

    printed = capsys.readouterr().out
    assert str(problem) in printed
    assert "Traceback" not in printed


# --------------------------------------------------------------------------
# run: a machine that starts (plan.md 4.1)
# --------------------------------------------------------------------------


def test_the_model_that_answers_is_the_one_that_was_configured(
    configured: Path, wiring: Wiring
) -> None:
    """The provider from the catalogue entry `[live] primary` names, and
    the model in what every session is opened with."""
    assert main(["run", "--terminal"]) == 0

    [parts] = wiring.built
    assert parts["provider"].id == "gemini"
    assert session_of(wiring).model == MODEL


def test_the_language_that_was_chosen_is_the_one_it_speaks(
    configured: Path, wiring: Wiring
) -> None:
    """`config.toml` says `tr`, so the pack the state machine gets says `tr` -
    the yes and no words, the fillers and the sentences (section 3.12)."""
    main(["run", "--terminal"])

    assert wiring.built[0]["locale"].code == "tr"


def test_the_pack_s_language_code_reaches_the_session(configured: Path, wiring: Wiring) -> None:
    """ADR-001 section 6: the BCP-47 hint for the recogniser behind the
    model and for its voice - from the pack, never from the code."""
    main(["run", "--terminal"])

    assert session_of(wiring).language_code == locales.load("tr").language_code == "tr-TR"


def test_the_voice_and_the_transcripts_in_the_settings_reach_the_session(
    configured: Path, wiring: Wiring
) -> None:
    configured_with(live=LiveSettings(primary=f"gemini:{MODEL}", voice="Kore", transcripts=False))

    main(["run", "--terminal"])

    config = session_of(wiring)
    assert (config.voice, config.transcripts) == ("Kore", False)


def test_by_default_the_voice_is_the_model_s_own(configured: Path, wiring: Wiring) -> None:
    """D19: the owner's ear preferred it."""
    main(["run", "--terminal"])

    assert session_of(wiring).voice == ""


def test_the_two_turn_detection_knobs_in_the_settings_reach_the_session(
    configured: Path, wiring: Wiring
) -> None:
    """ADR-001: `[live] end_sensitivity` and `silence_ms` are the owner's to
    tune in the real run, and go to the adapter as they are."""
    configured_with(
        live=LiveSettings(primary=f"gemini:{MODEL}", end_sensitivity="HIGH", silence_ms=300)
    )

    main(["run", "--terminal"])

    config = session_of(wiring)
    assert (config.end_sensitivity, config.silence_ms) == ("HIGH", 300)


def test_web_search_reaches_the_session_and_its_rule_the_prompt(
    configured: Path, wiring: Wiring
) -> None:
    """Switched on (spec section 2): the session is offered the search tool
    and the prompt says when to use it - after the pack's rule, before the
    user's facts."""
    configured_with(live=LiveSettings(primary=f"gemini:{MODEL}", web_search=True))

    main(["run", "--terminal"])

    config = session_of(wiring)
    assert config.web_search is True
    assert f"\n\n{SEARCH_RULE}\n\n" in config.system_prompt
    assert config.system_prompt.index(SEARCH_RULE) > config.system_prompt.index(SYSTEM_PROMPT)


def test_the_two_session_switches_reach_the_session(configured: Path, wiring: Wiring) -> None:
    """The defaults (D25, 2026-09-21): compression on, affective dialog off."""
    main(["run", "--terminal"])

    config = session_of(wiring)
    assert (config.affective_dialog, config.compress_context) == (False, True)


def test_the_two_session_switches_can_be_flipped(configured: Path, wiring: Wiring) -> None:
    configured_with(
        live=LiveSettings(primary=f"gemini:{MODEL}", affective_dialog=True, compress_context=False)
    )

    main(["run", "--terminal"])

    config = session_of(wiring)
    assert (config.affective_dialog, config.compress_context) == (True, False)


def test_the_wake_word_is_built_from_the_wake_table_and_handed_to_the_capture(
    configured: Path, wiring: Wiring, tmp_path: Path
) -> None:
    """`[wake]` on: the detector is built from the table - the model file,
    the threshold - loaded, and handed to the capture; the greeting reaches
    the state machine."""
    model = tmp_path / "hey_friday.onnx"
    model.write_bytes(b"\x00")
    configured_with(wake=WakeSettings(enabled=True, model=str(model), threshold=0.61))

    main(["run", "--terminal"])

    [capture] = wiring.captures
    assert isinstance(capture.wake, FakeWakeWord)
    assert capture.wake.loaded
    assert FakeWakeWord.built == [(model, 0.61)]
    [parts] = wiring.built
    assert parts["greeting"] == "chime"


def test_a_shipped_model_without_a_threshold_of_the_user_s_wakes_at_its_own(
    configured: Path, wiring: Wiring
) -> None:
    """D31: the threshold ships with the model - Vesper's eval gave 0.61 -
    and is read at every start rather than written by setup."""
    configured_with(wake=WakeSettings(enabled=True, model="hey_vesper"))

    main(["run", "--terminal"])

    [(path, threshold)] = FakeWakeWord.built
    assert path.name == "hey_vesper.onnx"
    assert threshold == pytest.approx(0.61)


def test_with_the_wake_word_off_the_capture_has_no_detector(
    configured: Path, wiring: Wiring
) -> None:
    """The default for a file setup has not written since D31: the product
    as it was before the wake word."""
    main(["run", "--terminal"])

    [capture] = wiring.captures
    assert capture.wake is None
    assert FakeWakeWord.built == []


def test_a_wake_model_that_is_not_there_is_a_sentence(
    configured: Path, wiring: Wiring, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configured_with(wake=WakeSettings(enabled=True, model=str(tmp_path / "nope.onnx")))

    assert main(["run", "--terminal"]) == 1

    out = capsys.readouterr().out
    assert "nope.onnx" in out
    assert wiring.captures == []


def test_without_google_s_key_there_is_no_look_up_and_the_prompt_is_as_it_was(
    configured: Path, vault: MemoryKeyring, wiring: Wiring, other_provider: None
) -> None:
    """D29: `look_up` asks Google on the key the Gemini entry is filed under.
    A live model elsewhere and no such key: no tool, no sentence, the frozen
    prompt as it was - and the browser's search still on offer."""
    save_settings(
        Settings(live=LiveSettings(primary="other:some-model"), locale=LocaleSettings(code="tr"))
    )
    vault.vault.clear()
    store_api_key("other", "sk-not-a-real-key")

    main(["run", "--terminal"])

    config = session_of(wiring)
    assert config.web_search is False
    assert SEARCH_RULE not in config.system_prompt
    [runner] = wiring.runners
    assert "look_up" not in runner.tools
    assert "x_trends" not in runner.tools
    assert "search_web" in runner.tools


def test_look_up_on_offer_brings_the_search_rule_without_the_session_s_search(
    configured: Path, wiring: Wiring
) -> None:
    """The rule says "look it up yourself, the browser only when asked" -
    true of `look_up` as of the session's own search (D22, D29)."""
    main(["run", "--terminal"])

    config = session_of(wiring)
    assert config.web_search is False
    assert f"\n\n{SEARCH_RULE}\n\n" in config.system_prompt


def test_look_up_asks_the_model_the_settings_name_on_the_gemini_key(
    configured: Path, wiring: Wiring, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[dict[str, Any]] = []

    class FakeSearch:
        def __init__(self, api_key: str, **rest: Any) -> None:
            built.append({"api_key": api_key, **rest})

    monkeypatch.setattr(search_module, "GroundedSearch", FakeSearch)
    configured_with(web=WebSettings(look_up_model="gemini-3.8-flash"))

    main(["run", "--terminal"])

    [made] = built
    assert made["api_key"] == "AIza-not-a-real-key"
    assert made["model"] == "gemini-3.8-flash"
    assert isinstance(made["searched"], Searched)
    assert wiring.built[0]["searched"] is made["searched"]


def test_the_state_machine_is_handed_what_look_up_keeps(configured: Path, wiring: Wiring) -> None:
    main(["run", "--terminal"])

    [parts] = wiring.built
    assert isinstance(parts["searched"], Searched)


def test_the_tools_the_session_is_opened_with_are_the_runner_s(
    configured: Path, wiring: Wiring
) -> None:
    """One list: what the model is told it can call is what the round can
    run (plan.md 4.4 rule 3)."""
    main(["run", "--terminal"])

    [runner] = wiring.runners
    assert [spec.name for spec in session_of(wiring).tools] == runner.tools


def test_the_session_config_is_read_afresh_at_every_open(
    configured: Path, wiring: Wiring, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prompt carries the time (4.2) and the user's facts (3.7), and
    both move between one session and the next."""
    from allie.tools import reminders

    main(["run", "--terminal"])
    [parts] = wiring.built
    first = parts["session_config"]().system_prompt
    monkeypatch.setattr(reminders, "current_time_line", lambda: "The current local date is later.")

    second = parts["session_config"]().system_prompt

    assert first != second
    assert second.endswith("The current local date is later.")


def test_the_session_policy_in_the_settings_is_the_one_the_state_machine_gets(
    configured: Path, wiring: Wiring
) -> None:
    """D5: `[live] idle_close_seconds` and `resume_minutes` are the state
    machine's numbers, and `barge_in` is the capture's (D6, D18)."""
    configured_with(
        live=LiveSettings(
            primary=f"gemini:{MODEL}", idle_close_seconds=30.0, resume_minutes=5.0, barge_in=False
        )
    )

    main(["run", "--terminal"])

    [parts] = wiring.built
    assert (parts["idle_close_seconds"], parts["resume_minutes"]) == (30.0, 5.0)
    [built] = wiring.captures
    assert built.barge_in is False
    assert parts["capture"] is built


def test_by_default_barge_in_is_on(configured: Path, wiring: Wiring) -> None:
    main(["run", "--terminal"])

    assert wiring.captures[0].barge_in is True


def test_the_microphone_in_the_settings_is_the_one_opened(configured: Path, wiring: Wiring) -> None:
    """`[audio] input_device` names it in `sounddevice`'s words - words rather
    than an index, because the indices shift whenever a Bluetooth device
    connects."""
    configured_with(audio=AudioSettings(input_device="Microphone Array WASAPI"))

    main(["run", "--terminal"])

    assert wiring.microphones == ["Microphone Array WASAPI"]
    assert wiring.captures[0].microphone.device == "Microphone Array WASAPI"


def test_no_microphone_in_the_settings_means_the_system_default(
    configured: Path, wiring: Wiring
) -> None:
    main(["run", "--terminal"])

    assert wiring.microphones == [None]


def test_the_device_flag_outranks_the_settings(configured: Path, wiring: Wiring) -> None:
    """One evening with a headset should not need the settings file edited."""
    main(["run", "--terminal", "--device", "9"])

    assert wiring.microphones == [9]


def test_the_speech_model_is_ready_before_the_assistant_is(
    configured: Path, wiring: Wiring
) -> None:
    """Loading Whisper at the first question would swallow the first yes
    or no. The database comes first of all: cheap, and a disk that refuses
    is better found out about before two seconds of four cores are spent.
    The probe of 2.6 comes next, for the same reason: one session on the
    network, and worth knowing about before the load. The app catalogue
    comes before the speech model, which is told its names."""
    main(["run", "--terminal"])

    assert wiring.happened == ["database", "probe", "app catalogue", "speech model", "assistant"]


# --------------------------------------------------------------------------
# run: the tools, the gate and the database (2.1c, 2.1d, 2.2)
# --------------------------------------------------------------------------


def test_every_tool_of_phase_two_is_on_offer(configured: Path, wiring: Wiring) -> None:
    main(["run", "--terminal"])

    [runner] = wiring.runners
    assert runner.tools == [
        "get_current_time",
        "system_status",
        "get_weather",
        "open_app",
        "open_url",
        "look_up",
        "search_web",
        "read_clipboard",
        "fetch_page",
        "x_trends",
        "read_latest_emails",
        "search_emails",
        "open_settings",
        "media_control",
        "set_volume",
        "play_music",
        "play_video",
        "open_media",
        "add_note",
        "search_notes",
        "delete_note",
        "create_reminder",
        "list_reminders",
        "cancel_reminder",
        "remember",
        "forget",
        "install_app",
        "send_message",
    ]


def test_a_tool_file_beside_the_settings_is_on_offer(configured: Path, wiring: Wiring) -> None:
    """Section 3.9 (13 Sep 2026): the owner's own tools, read from
    `%APPDATA%\\allie\\tools`, join the same registry as everything else."""
    folder = configured / "tools"
    folder.mkdir()
    (folder / "mine.py").write_text(
        "from allie.tools.registry import tool\n"
        "\n"
        "\n"
        '@tool(risk="safe")\n'
        "async def start_my_project() -> str:\n"
        '    """Starts the owner\'s project."""\n'
        '    return "started"\n',
        encoding="utf-8",
    )

    main(["run", "--terminal"])

    [runner] = wiring.runners
    assert runner.tools[-1] == "start_my_project"
    assert runner.tools[:-1] == list(BUILTIN_TOOLS)


def test_the_speech_model_is_told_the_words_the_window_accepts(
    configured: Path, wiring: Wiring
) -> None:
    """D3, D10: the recogniser hears the yes or no of the gate's window and
    nothing else, so that is what it is told to expect - the pack's own words
    (`app.confirm_prompt`). Until 2026-09-22 it was given the names of the
    installed applications and the address book, 120 tokens of them, which
    cost the window most of a second of decode for words it never hears."""
    main(["run", "--terminal"])

    assert wiring.prompts_for_speech == [confirm_prompt(locales.load("tr"))]
    assert "evet" in wiring.prompts_for_speech[0].casefold()


def test_send_message_asks_its_question_in_the_language_of_the_pack(
    configured: Path, wiring: Wiring
) -> None:
    from allie.tools import messaging as messaging_tools

    main(["run", "--terminal"])

    [runner] = wiring.runners
    send_message = runner.registry.get("send_message")
    assert send_message is not None
    assert send_message.risk == "confirm"
    assert send_message.confirm_prompt == locales.load("tr").say(
        "send_message_confirm", messaging_tools.TEXT["send_message_confirm"]
    )
    assert send_message.confirm_prompt != messaging_tools.TEXT["send_message_confirm"]


def test_a_contacts_file_that_names_one_person_twice_is_a_sentence_before_anything_slow(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """The worst failure of messaging is a message to the wrong person; the
    file that would cause one is refused at startup, not at the first send."""
    (configured / CONTACTS_FILE_NAME).write_text(
        '[[contact]]\nname = "Ahmet"\n\n[[contact]]\nname = "ahmet"\n', encoding="utf-8"
    )

    assert main(["run", "--terminal"]) == 1

    assert "speech model" not in wiring.happened
    out = capsys.readouterr().out
    assert said("cannot_start", "tr").split("{")[0] in out
    assert CONTACTS_FILE_NAME in out


def test_a_default_messaging_app_that_is_not_one_is_a_sentence(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    configured_with(messaging=MessagingSettings(default_app="Signal"))

    assert main(["run", "--terminal"]) == 1
    assert "Signal" in capsys.readouterr().out


def test_by_default_the_recogniser_is_whisper_alone(configured: Path, wiring: Wiring) -> None:
    """D10: local, no key, the two words stay; `[stt]` left out means that."""
    main(["run", "--terminal"])

    assert wiring.recognisers == []
    assert "gemini" not in wiring.happened
    assert type(wiring.built[-1]["stt"]).__name__ == "FakeWhisper"


def test_with_the_setting_the_recogniser_is_gemini_with_whisper_behind_it(
    configured: Path, wiring: Wiring
) -> None:
    """Google first, the local engine loaded behind it for the free tier's
    three requests a minute; the same names, the same key entry the live
    model uses."""
    configured_with(stt=STTSettings(provider="gemini", model="gemini-3.5-transcribe-live"))

    main(["run", "--terminal"])

    (built,) = wiring.recognisers
    assert built["api_key"] == "AIza-not-a-real-key"
    assert built["model"] == "gemini-3.5-transcribe-live"
    assert type(built["fallback"]).__name__ == "FakeWhisper"
    assert wiring.happened[-4:] == ["app catalogue", "speech model", "gemini", "assistant"]
    assert type(wiring.built[-1]["stt"]).__name__ == "FakeGemini"


def test_gemini_as_recogniser_without_a_gemini_key_is_one_sentence(
    configured: Path,
    vault: MemoryKeyring,
    wiring: Wiring,
    other_provider: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The live model is elsewhere and has its key; the recogniser's is
    missing. Named, with the way out, and nothing is loaded first."""
    save_settings(
        Settings(
            live=LiveSettings(primary="other:some-model"),
            locale=LocaleSettings(code="tr"),
            stt=STTSettings(provider="gemini"),
        )
    )
    vault.vault.clear()
    store_api_key("other", "sk-not-a-real-key")

    assert main(["run", "--terminal"]) == 1

    printed = capsys.readouterr().out
    assert "gemini" in printed and "allie setup" in printed
    assert "speech model" not in wiring.happened
    assert wiring.recognisers == []


def test_with_the_setting_the_voice_is_gemini_with_windows_behind_it(
    configured: Path, wiring: Wiring
) -> None:
    """`[tts] provider = "gemini"` (17 Sep 2026): Google's synthesiser with
    the same key entry, Windows' own engine behind it for the sentence
    Google refuses, and the pack's local preference for that engine. It
    reads the gate's questions and the reminders (D3, D4); the model speaks
    for itself."""
    from allie.tts.gemini_tts import GeminiTTS
    from allie.tts.sapi import SapiTTS

    configured_with(tts=TTSSettings(provider="gemini", model="gemini-3.1-flash-tts-preview"))

    main(["run", "--terminal"])

    voice = wiring.built[-1]["tts"]
    assert isinstance(voice, GeminiTTS)
    assert voice._model == "gemini-3.1-flash-tts-preview"
    assert isinstance(voice._fallback, SapiTTS)
    assert (voice._fallback_language, voice._fallback_preference) == ("tr", "Tolga")


def test_gemini_as_voice_without_a_gemini_key_is_one_sentence(
    configured: Path,
    vault: MemoryKeyring,
    wiring: Wiring,
    other_provider: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    save_settings(
        Settings(
            live=LiveSettings(primary="other:some-model"),
            locale=LocaleSettings(code="tr"),
            tts=TTSSettings(provider="gemini"),
        )
    )
    vault.vault.clear()
    store_api_key("other", "sk-not-a-real-key")

    assert main(["run", "--terminal"]) == 1

    printed = capsys.readouterr().out
    assert "gemini" in printed and "[tts]" in printed and "allie setup" in printed
    assert "speech model" not in wiring.happened


def test_without_the_setting_the_voice_is_windows(configured: Path, wiring: Wiring) -> None:
    from allie.tts.sapi import SapiTTS

    main(["run", "--terminal"])

    assert isinstance(wiring.built[-1]["tts"], SapiTTS)


def test_the_gate_the_tool_round_is_handed_is_the_permission_gate(
    configured: Path, wiring: Wiring
) -> None:
    """Not any callable: the one that refuses what it was not offered, which
    is the one thing a fake gate would not do."""
    main(["run", "--terminal"])

    [runner] = wiring.runners
    made_up = ToolCall(id="c1", name="format_disk", arguments={})

    refused = asyncio.run(runner.dispatch(made_up, turn_id="t1", confirm=nobody_asked))
    assert refused == NO_SUCH_TOOL.format(name="format_disk")


async def nobody_asked(question: str) -> bool:
    raise AssertionError(f"nobody should have been asked {question!r}")


def test_who_answers_a_tool_s_question_comes_with_the_turn_and_reaches_the_gate(
    configured: Path, wiring: Wiring, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is built before the state machine, so who to ask cannot be
    bound into it; each call brings its own, and the gate hands it to
    `policy.dispatch` unchanged."""
    from allie.agent import policy

    handed: list[Any] = []

    async def recording(call: ToolCall, **rest: Any) -> str:
        handed.append(rest["confirm"])
        return "recorded"

    monkeypatch.setattr(policy, "dispatch", recording)
    main(["run", "--terminal"])
    [runner] = wiring.runners
    order = ToolCall(id="c1", name="get_current_time", arguments={})

    async def says_yes(question: str) -> bool:
        return True

    assert asyncio.run(runner.dispatch(order, turn_id="t1", confirm=says_yes)) == "recorded"
    assert handed == [says_yes]


def test_the_tool_round_the_state_machine_is_handed_is_the_one_behind_the_gate(
    configured: Path, wiring: Wiring
) -> None:
    """One round, one gate, one caller (plan.md 4.4 rule 3): a second round
    would be a second way to run a tool, which is the thing section 3.9
    forbids."""
    main(["run", "--terminal"])

    [runner] = wiring.runners
    assert wiring.built[-1]["tool_runner"] is runner


# --------------------------------------------------------------------------
# run: what the user asked to be kept (2.10), and the prompt (plan.md 4.5)
# --------------------------------------------------------------------------


def remembered_by_hand(configured: Path, body: str) -> Path:
    path = configured / MEMORY_FILE_NAME
    path.write_text(body, encoding="utf-8")
    return path


def test_what_the_user_asked_to_be_kept_is_in_front_of_every_session(
    configured: Path, wiring: Wiring
) -> None:
    """Section 3.7: the file is read at startup and the prompt every session
    opens with carries it - behind the frozen prompt, which stays as it is."""
    from allie.agent.prompts import SYSTEM_PROMPT

    remembered_by_hand(
        configured, '[assistant]\nname = "Ada"\n\n[user]\nfacts = ["Bana Emre de."]\n'
    )

    main(["run", "--terminal"])

    prompt = session_of(wiring).system_prompt
    assert prompt.startswith(SYSTEM_PROMPT)
    assert "Bana Emre de." in prompt
    assert "Ada" in prompt


def test_the_prompt_is_the_frozen_rules_the_pack_s_language_rule_and_the_time(
    configured: Path, wiring: Wiring
) -> None:
    """The frozen prompt first, byte for byte; then the pack's sentence
    naming the language the user speaks (D19 - measured 2026-09-18: without
    it "Saat kaç" is heard as Hindi); the time last (4.2), the one line that
    changes between sessions."""
    from allie.agent.prompts import SYSTEM_PROMPT
    from allie.tools.reminders import NOW_LINE

    main(["run", "--terminal"])

    prompt = session_of(wiring).system_prompt
    rule = locales.load("tr").user_language_rule
    assert rule
    assert prompt.startswith(f"{SYSTEM_PROMPT}\n\n{rule}\n\n")
    assert prompt.rsplit("\n", 1)[1].startswith(NOW_LINE.split("{")[0])


def test_a_pack_without_a_language_rule_adds_no_line(configured: Path, wiring: Wiring) -> None:
    """`en.toml` names no language (ADR-001): the prompt's own mirroring
    rule is all there is, and the frozen prompt is followed by the search
    rule - `configured` stores the Gemini key, so `look_up` is on offer
    (D29) - and the time."""
    from allie.agent.prompts import SYSTEM_PROMPT

    configured_with(locale=LocaleSettings(code="en"))

    main(["run", "--terminal"])

    prompt = session_of(wiring).system_prompt
    assert prompt.startswith(f"{SYSTEM_PROMPT}\n\n{SEARCH_RULE}\n\n")
    assert prompt.count("\n\n") == SYSTEM_PROMPT.count("\n\n") + 2


def test_a_memory_file_that_does_not_parse_is_a_sentence_rather_than_a_traceback(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hand edit gone wrong. The user can fix it; the program must not
    write an empty file over it, and must not load a speech model first."""
    path = remembered_by_hand(configured, "[user]\nfacts = [oops\n")

    assert main(["run", "--terminal"]) == 1

    assert "speech model" not in wiring.happened
    assert path.read_text(encoding="utf-8") == "[user]\nfacts = [oops\n"
    assert said("cannot_start", "tr").split("{")[0] in capsys.readouterr().out


def test_forget_asks_its_question_in_the_language_of_the_pack(
    configured: Path, wiring: Wiring
) -> None:
    """The first question of phase 2 a user actually hears (2.3, 2.10)."""
    from allie.tools import memory as memory_tools

    main(["run", "--terminal"])

    [runner] = wiring.runners
    forget = runner.registry.get("forget")
    assert forget is not None
    assert forget.risk == "confirm"
    assert forget.confirm_prompt == locales.load("tr").say(
        "forget_confirm", memory_tools.TEXT["forget_confirm"]
    )
    assert forget.confirm_prompt != memory_tools.TEXT["forget_confirm"]


# --------------------------------------------------------------------------
# run: the verdict on the model (2.6)
# --------------------------------------------------------------------------


def test_a_model_nobody_has_tested_is_probed_at_startup_and_the_verdict_kept(
    configured: Path, wiring: Wiring
) -> None:
    """Setup wrote nothing - an older build, or a database that was deleted.
    The question is asked once and the answer kept, so that the next start
    does not ask again."""
    assert main(["run", "--terminal"]) == 0
    assert main(["run", "--terminal"]) == 0

    assert [(p, m) for p, m, _ in wiring.probes] == [("gemini", MODEL)]
    assert stored_verdict() == PASSED


def test_the_probe_asks_in_the_language_of_the_pack(configured: Path, wiring: Wiring) -> None:
    main(["run", "--terminal"])

    [(_, _, question)] = wiring.probes
    assert question == locales.load("tr").probe_question


def test_the_model_is_checked_before_the_speech_model_is_loaded(
    configured: Path, wiring: Wiring
) -> None:
    """One session on the network, before two seconds of loading are spent
    on a model that may turn out not to call tools."""
    main(["run", "--terminal"])

    assert wiring.happened.index("probe") < wiring.happened.index("speech model")


def test_a_fresh_verdict_is_not_asked_again(configured: Path, wiring: Wiring) -> None:
    write_verdict(ProbeResult(ok=True, first_token_ms=800.0), age=3 * 86400)

    main(["run", "--terminal"])

    assert wiring.probes == []


def test_a_verdict_a_week_old_is_asked_again(configured: Path, wiring: Wiring) -> None:
    """Section 3.2: the provider may have changed what is behind the name."""
    write_verdict(ProbeResult(ok=True, first_token_ms=800.0), age=8 * 86400)

    main(["run", "--terminal"])

    assert len(wiring.probes) == 1
    assert stored_verdict() == PASSED


def test_a_model_that_fails_the_probe_is_a_warning_and_not_a_refusal_to_start(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """The user may have chosen it knowing; it still answers questions. But
    they are told, in their language, on a line that stays."""
    wiring.verdict = ProbeResult(ok=False, reason="no_tool_call_emitted", first_token_ms=5.0)

    assert main(["run", "--terminal"]) == 0

    assert "assistant" in wiring.happened
    assert said("model_no_tools") in unwrapped(capsys.readouterr().out)


def test_a_kept_verdict_that_failed_is_warned_about_on_every_start(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    write_verdict(ProbeResult(ok=False, reason="no_tool_call_emitted"), age=60)

    main(["run", "--terminal"])

    assert wiring.probes == []
    assert said("model_no_tools") in unwrapped(capsys.readouterr().out)


def test_a_provider_that_cannot_be_asked_at_startup_is_left_to_the_first_turn(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """Offline at startup is not a bad model. Nothing is written down, no
    warning is shown, and the first turn says what is wrong in its own
    words (`app.py`)."""
    wiring.probe_refusal = ProviderError("gemini could not be reached (ConnectError)")

    assert main(["run", "--terminal"]) == 0

    assert "assistant" in wiring.happened
    assert stored_verdict() is None
    assert said("model_no_tools") not in unwrapped(capsys.readouterr().out)


# --------------------------------------------------------------------------
# run: the database, the log, the screen, the end
# --------------------------------------------------------------------------


def test_the_database_is_built_where_the_data_lives(configured: Path, wiring: Wiring) -> None:
    main(["run", "--terminal"])

    path = db.database_path()
    assert path.is_file()
    connection = sqlite3.connect(path)
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
        assert "tool_audit" in tables
    finally:
        connection.close()


def test_the_database_is_closed_when_the_assistant_stops(configured: Path, wiring: Wiring) -> None:
    """However it stops - here, the way Ctrl+C stops it."""
    wiring.stop = KeyboardInterrupt()

    main(["run", "--terminal"])

    [connection] = wiring.databases
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_a_finished_turn_is_written_to_the_log(configured: Path, wiring: Wiring) -> None:
    """Item 1.11's own sentence: the token count of every turn goes to the log."""
    main(["run", "--terminal"])

    assert "300 in, 10 out" in logs.log_path().read_text(encoding="utf-8")


def test_a_finished_turn_is_shown_on_screen(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["run", "--terminal"])

    assert "saat kaç" in capsys.readouterr().out


def test_what_the_assistant_is_doing_is_shown_on_screen(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["run", "--terminal"])

    assert said("state_user_speaking") in capsys.readouterr().out


def test_the_session_and_its_minutes_are_shown_on_screen(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """Plan.md 4.2: the state machine's `on_session` reaches the line's
    meter. Off a terminal the line is drawn once, at the end, so what is
    read here is the meter after the session closed; `test_status.py`
    has the rest."""
    main(["run", "--terminal"])

    printed = capsys.readouterr().out
    assert said("session_closed") in printed
    assert said("session_minutes").format(minutes=0) in printed


def test_a_quiet_microphone_is_said_on_screen_when_the_session_closes(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """D18: the level the capture judged when a session closes goes to the
    line, and under -40 dBFS it is a sentence the user reads once. Measured
    2026-09-18: the owner's array sent -45 through the audio engine. A raw-path
    microphone judges nothing (`LiveCapture.level_judged`) and says nothing."""
    wiring.level = -45.0

    main(["run", "--terminal"])

    quiet = said("microphone_quiet").format(level=-45, quiet=-40)
    assert unwrapped(quiet) in unwrapped(capsys.readouterr().out)


def test_a_microphone_heard_well_enough_is_not_mentioned(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    wiring.level = -30.0

    main(["run", "--terminal"])

    assert said("microphone_quiet").split("{")[0] not in unwrapped(capsys.readouterr().out)


def test_ctrl_c_is_how_it_is_meant_to_end(
    configured: Path, wiring: Wiring, capsys: pytest.CaptureFixture[str]
) -> None:
    """A traceback would read as a crash."""
    wiring.stop = KeyboardInterrupt()

    assert main(["run", "--terminal"]) == 0
    assert said("stopped") in capsys.readouterr().out


# --------------------------------------------------------------------------
# run: the limits and the bill (2.4)
# --------------------------------------------------------------------------


def test_the_limits_in_the_settings_are_the_ones_the_tool_round_gets(
    configured: Path, wiring: Wiring
) -> None:
    configured_with(limits=LimitSettings(tool_calls_per_turn=3))

    main(["run", "--terminal"])

    [runner] = wiring.runners
    assert runner.limits == Limits(tool_calls_per_turn=3)


def test_the_assistant_is_handed_a_tracker_over_the_usage_rows(
    configured: Path, wiring: Wiring
) -> None:
    """The tracker is what writes `usage_log` and checks the spending limits
    before a session opens (D9)."""
    configured_with(limits=LimitSettings(daily_usd=1.0))

    main(["run", "--terminal"])

    parts = wiring.built[0]
    assert isinstance(parts["tracker"], UsageTracker)
    assert parts["announcements"] is not None


# --------------------------------------------------------------------------
# run: what the audit rows still say (4.6, 17 Sep 2026)
# --------------------------------------------------------------------------


def audited(*, days_ago: float, summary: str) -> int:
    """One finished `fetch_page` row, `days_ago` days old, in the database
    `run` will open; returns its id."""
    connection = db.open_database()
    try:
        audit = AuditRepo(connection, clock=lambda: time.time() - days_ago * SECONDS_PER_DAY)
        call = ToolCall(id="c", name="fetch_page", arguments={"url": "https://example.test"})
        row_id = audit.start(call, turn_id="turn", risk="safe")
        audit.finish(row_id, status="ok", summary=summary)
    finally:
        connection.close()
    return row_id


def summaries() -> dict[int, str | None]:
    connection = sqlite3.connect(db.database_path())
    try:
        rows = connection.execute("SELECT id, result_summary FROM tool_audit").fetchall()
    finally:
        connection.close()
    return {int(row_id): summary for row_id, summary in rows}


def test_run_blanks_the_old_audit_summaries_and_keeps_the_rows(
    configured: Path, wiring: Wiring
) -> None:
    """`store/retention.py` at startup: a summary older than
    `[retention] audit_days` is gone before the first turn, the row is not."""
    old = audited(days_ago=40, summary="a page from last month")
    fresh = audited(days_ago=1, summary="a page from yesterday")

    assert main(["run", "--terminal"]) == 0

    assert summaries() == {old: None, fresh: "a page from yesterday"}


def test_a_retention_of_zero_days_keeps_every_summary(configured: Path, wiring: Wiring) -> None:
    configured_with(retention=RetentionSettings(audit_days=0))
    old = audited(days_ago=400, summary="a page from last year")

    assert main(["run", "--terminal"]) == 0

    assert summaries() == {old: "a page from last year"}


# --------------------------------------------------------------------------
# run --tray (4.3, 17 Sep 2026)
# --------------------------------------------------------------------------


class FakeTray:
    """Stands in for `ui/tray.py`'s `Tray`: what it was built with, what it
    was told, and whether it is up."""

    built: ClassVar[list[FakeTray]] = []

    def __init__(self, locale: locales.Locale, **parts: Any) -> None:
        self.code = locale.code
        self.parts = parts
        self.states: list[State] = []
        self.modes: list[bool] = []
        self.sessions: list[bool] = []
        self.up: list[str] = []
        FakeTray.built.append(self)

    def start(self) -> None:
        self.up.append("start")

    def stop(self) -> None:
        self.up.append("stop")

    def state(self, state: State) -> None:
        self.states.append(state)

    def hands_free(self, listening: bool) -> None:
        self.modes.append(listening)

    def session(self, open: bool) -> None:
        self.sessions.append(open)


@pytest.fixture
def trays(monkeypatch: pytest.MonkeyPatch) -> type[FakeTray]:
    from allie.ui import tray

    FakeTray.built = []
    monkeypatch.setattr(tray, "Tray", FakeTray)
    return FakeTray


def test_without_the_flag_there_is_no_tray(
    configured: Path, wiring: Wiring, trays: type[FakeTray]
) -> None:
    assert main(["run", "--terminal"]) == 0

    assert trays.built == []
    assert wiring.built[0]["on_state"] is not None


def test_with_the_flag_the_icon_is_up_while_it_runs_and_hears_what_the_screen_hears(
    configured: Path, wiring: Wiring, trays: type[FakeTray]
) -> None:
    """Built with the pack and the settings folder, started before the
    state machine runs and stopped after; every state and every session
    the screen is told reaches it too, and its switch is the capture's own
    `toggle`."""
    assert main(["run", "--tray", "--terminal"]) == 0

    [icon] = trays.built
    assert icon.code == "tr"
    assert icon.parts["settings_folder"] == configured
    assert icon.up == ["start", "stop"]
    assert icon.states == [State.USER_SPEAKING]
    assert icon.sessions == [True, False]
    [built] = wiring.captures
    assert icon.parts["on_toggle"] == built.toggle

    wiring.built[0]["on_mode"](False)
    assert icon.modes == [False]


def test_quit_on_the_tray_ends_the_run_the_way_ctrl_c_does(
    configured: Path, wiring: Wiring, trays: type[FakeTray], capsys: pytest.CaptureFixture[str]
) -> None:
    """ "Quit" is posted from the tray's thread and cancels the run: the
    icon comes down, the terminal says it stopped, and the exit code is
    the one Ctrl+C gives."""

    async def quit_from_the_tray() -> None:
        [icon] = trays.built
        loop = asyncio.get_running_loop()
        await asyncio.to_thread(loop.call_soon_threadsafe, icon.parts["on_quit"])
        await asyncio.sleep(5)
        raise AssertionError("the run was not cancelled")

    wiring.during_run = quit_from_the_tray

    assert main(["run", "--tray", "--terminal"]) == 0

    [icon] = trays.built
    assert icon.up == ["start", "stop"]
    assert said("stopped") in capsys.readouterr().out


# --------------------------------------------------------------------------
# run: the window (plan.md D20)
# --------------------------------------------------------------------------


class FakeWindow:
    """Stands in for `ui/window.py`'s `Window`: what it was built with, what
    it was told, and what its buttons do."""

    built: ClassVar[list[FakeWindow]] = []

    def __init__(self, locale: locales.Locale, **parts: Any) -> None:
        self.code = locale.code
        self.parts = parts
        self.told: list[tuple[str, Any]] = []
        self.up: list[str] = []
        self.questions: list[tuple[str, str]] = []
        self.answers: list[str | None] = []
        FakeWindow.built.append(self)

    def start(self) -> None:
        self.up.append("start")

    def stop(self) -> None:
        self.up.append("stop")

    def starting(self) -> None:
        self.told.append(("phase", "loading_speech"))

    def checking_model(self) -> None:
        self.told.append(("phase", "checking_model"))

    def notice(self, message: str) -> None:
        self.told.append(("notice", message))

    def state(self, state: State) -> None:
        self.told.append(("state", state))

    def session(self, open: bool) -> None:
        self.told.append(("session", open))

    def microphone_level(self, dbfs: float | None) -> None:
        self.told.append(("microphone_level", dbfs))

    def hands_free(self, listening: bool) -> None:
        self.told.append(("mode", listening))

    def turn(self, finished: Turn) -> None:
        self.told.append(("turn", finished.heard))

    def level(self, dbfs: float) -> None:
        self.told.append(("level", dbfs))

    def loading(self) -> None:
        self.told.append(("phase", "window_loading"))

    def failed(self) -> None:
        self.told.append(("phase", "window_failed"))

    def show(self) -> None:
        self.told.append(("show", None))

    def wizard(self, opened: bool) -> None:
        self.told.append(("wizard", opened))

    def say(self, text: str) -> None:
        self.told.append(("say", text))

    def ask(self, kind: str, text: str, options: Any) -> asyncio.Future[str | None]:
        self.questions.append((kind, text))
        answer: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        answer.set_result(self.answers.pop(0) if self.answers else None)
        return answer


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> type[FakeWindow]:
    from allie.ui import window

    FakeWindow.built = []
    monkeypatch.setattr(window, "Window", FakeWindow)
    return FakeWindow


def test_without_the_terminal_flag_the_window_is_the_screen(
    configured: Path, wiring: Wiring, windows: type[FakeWindow], capsys: pytest.CaptureFixture[str]
) -> None:
    """Built with the pack, up before the pieces and down after; every
    state, session and turn the line would have shown goes to it; the
    switch it was handed is the capture's toggle; nothing is drawn on the
    terminal but the last word."""
    assert main(["run"]) == 0

    [face] = windows.built
    assert face.code == "tr"
    assert face.up == ["start", "stop"]
    assert face.parts["tray"] is False
    assert ("phase", "window_loading") in face.told
    assert ("state", State.USER_SPEAKING) in face.told
    assert ("session", True) in face.told and ("session", False) in face.told
    assert ("turn", "saat kaç") in face.told
    [built] = wiring.captures
    assert face.parts["on_toggle"].target == built.toggle
    assert said("state_user_speaking") not in capsys.readouterr().out


def test_the_window_hears_the_sound_both_ways(
    configured: Path, wiring: Wiring, windows: type[FakeWindow]
) -> None:
    """The capture's and the speaker's level hooks are the window's `level`."""
    main(["run"])

    [face] = windows.built
    [built] = wiring.captures
    assert built.on_level == face.level
    assert wiring.built[0]["speaker"].on_level == face.level


def test_with_the_tray_the_icon_can_show_the_window(
    configured: Path, wiring: Wiring, windows: type[FakeWindow], trays: type[FakeTray]
) -> None:
    main(["run", "--tray"])

    [face] = windows.built
    [icon] = trays.built
    assert face.parts["tray"] is True
    assert icon.parts["on_show"] == face.show


def test_on_the_terminal_the_tray_has_no_window_to_show(
    configured: Path, wiring: Wiring, trays: type[FakeTray]
) -> None:
    main(["run", "--tray", "--terminal"])

    [icon] = trays.built
    assert icon.parts["on_show"] is None


def test_an_unconfigured_machine_runs_the_wizard_in_the_window_then_talks(
    config_home: Path,
    vault: MemoryKeyring,
    wiring: Wiring,
    windows: type[FakeWindow],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No settings: the page comes up, the wizard runs through the window's
    prompter, and once it has written the settings the assistant starts."""
    prompters: list[Any] = []

    async def wizard(prompter: Any, **_: Any) -> int:
        prompters.append(prompter)
        configured_with()
        store_api_key("gemini", "key")
        return 0

    monkeypatch.setattr(setup_wizard, "run_setup", wizard)

    assert main(["run"]) == 0

    [face] = windows.built
    [prompter] = prompters
    assert type(prompter).__name__ == "WindowPrompter"
    assert face.told.index(("wizard", True)) < face.told.index(("wizard", False))
    assert face.told.index(("wizard", False)) < face.told.index(("state", State.USER_SPEAKING))
    assert len(wiring.built) == 1


def test_a_wizard_walked_away_from_on_an_unconfigured_machine_ends_the_run(
    config_home: Path,
    vault: MemoryKeyring,
    wiring: Wiring,
    windows: type[FakeWindow],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def wizard(prompter: Any, **_: Any) -> int:
        return 1

    monkeypatch.setattr(setup_wizard, "run_setup", wizard)

    assert main(["run"]) == 0

    assert wiring.built == []
    [face] = windows.built
    assert face.up == ["start", "stop"]


def test_the_settings_button_stops_the_assistant_runs_the_wizard_and_starts_it_again(
    configured: Path, wiring: Wiring, windows: type[FakeWindow], monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[str] = []

    async def wizard(prompter: Any, **_: Any) -> int:
        runs.append("wizard")
        return 0

    async def press_settings_then_quit() -> None:
        [face] = windows.built
        loop = asyncio.get_running_loop()
        if runs == []:
            runs.append("talk")
            await asyncio.to_thread(loop.call_soon_threadsafe, face.parts["on_settings"])
        else:
            runs.append("talk again")
            await asyncio.to_thread(loop.call_soon_threadsafe, face.parts["on_quit"])
        await asyncio.sleep(5)
        raise AssertionError("the run was not cancelled")

    monkeypatch.setattr(setup_wizard, "run_setup", wizard)
    wiring.during_run = press_settings_then_quit

    assert main(["run"]) == 0

    assert runs == ["talk", "wizard", "talk again"]
    assert len(wiring.built) == 2
    [face] = windows.built
    assert face.up == ["start", "stop"]


def test_quit_on_the_window_ends_the_run_the_way_ctrl_c_does(
    configured: Path, wiring: Wiring, windows: type[FakeWindow], capsys: pytest.CaptureFixture[str]
) -> None:
    async def quit_from_the_window() -> None:
        [face] = windows.built
        loop = asyncio.get_running_loop()
        await asyncio.to_thread(loop.call_soon_threadsafe, face.parts["on_quit"])
        await asyncio.sleep(5)
        raise AssertionError("the run was not cancelled")

    wiring.during_run = quit_from_the_window

    assert main(["run"]) == 0

    [face] = windows.built
    assert face.up == ["start", "stop"]
    assert said("stopped") in capsys.readouterr().out


def test_a_fixable_failure_is_a_notice_on_the_window_and_waits_for_a_button(
    configured: Path, wiring: Wiring, windows: type[FakeWindow]
) -> None:
    """A key that is gone is a sentence on the window, not an exit: the
    settings button is the way to fix it. Here the button pressed is quit."""
    from allie.live.registry import MissingAPIKeyError

    wiring.stop = MissingAPIKeyError("no API key stored for 'gemini'")
    pressed: list[str] = []

    original_notice = FakeWindow.notice

    def notice_then_quit(self: FakeWindow, message: str) -> None:
        original_notice(self, message)
        pressed.append(message)
        asyncio.get_running_loop().call_soon(self.parts["on_quit"])

    FakeWindow.notice = notice_then_quit  # type: ignore[method-assign]
    try:
        assert main(["run"]) == 0
    finally:
        FakeWindow.notice = original_notice  # type: ignore[method-assign]

    assert pressed == [said("cannot_start").format(problem="no API key stored for 'gemini'")]


def test_a_fixable_failure_says_it_did_not_start_where_the_state_would_be(
    configured: Path, wiring: Wiring, windows: type[FakeWindow]
) -> None:
    """2026-09-23: the owner's microphone was held by Teams and the line
    under the orb went on saying what it said before - "ready", or the
    last phase of the start. It says the start failed, and the sentence
    below says why."""
    from allie.audio.capture import MicrophoneUnavailableError

    wiring.stop = MicrophoneUnavailableError("the microphone is held by another program")

    def notice_then_quit(self: FakeWindow, message: str) -> None:
        self.told.append(("notice", message))
        asyncio.get_running_loop().call_soon(self.parts["on_quit"])

    original_notice = FakeWindow.notice
    FakeWindow.notice = notice_then_quit  # type: ignore[method-assign]
    try:
        assert main(["run"]) == 0
    finally:
        FakeWindow.notice = original_notice  # type: ignore[method-assign]

    [face] = windows.built
    told = [kind_and_what for kind_and_what in face.told if kind_and_what[0] in ("phase", "notice")]
    assert told[-2] == ("phase", "window_failed")
    assert told[-1][0] == "notice"


# --------------------------------------------------------------------------
# run: the mailbox (3.3, 17 Sep 2026)
# --------------------------------------------------------------------------


def test_without_a_mail_table_the_mail_tools_are_on_offer_but_not_set_up(
    configured: Path, wiring: Wiring
) -> None:
    """The tools are always on the list, so that the model can tell the
    user what to run; without `[mail]` and a stored password they open
    nothing."""
    from allie.tools.mail import NOT_SET_UP

    main(["run", "--terminal"])

    latest = wiring.runners[0].registry.get("read_latest_emails")
    assert latest is not None
    assert asyncio.run(latest.run()) == NOT_SET_UP


def test_with_a_mail_table_and_a_password_the_mailbox_is_the_one_named(
    configured: Path, wiring: Wiring, monkeypatch: pytest.MonkeyPatch
) -> None:
    from allie.tools import mail
    from allie.tools.mail import MAIL_ENTRY, ImapMailbox

    configured_with(mail=mail_settings("imap.example.test", "emre@example.test", mailbox="Archive"))
    store_api_key(MAIL_ENTRY, "app-password")
    built: list[tuple[Any, ...]] = []

    class Recorded(ImapMailbox):
        def __init__(self, *parts: Any, **rest: Any) -> None:
            built.append(parts)

        def latest(self, count: int) -> list[Any]:
            return []

    monkeypatch.setattr(mail, "ImapMailbox", Recorded)

    main(["run", "--terminal"])

    latest = wiring.runners[0].registry.get("read_latest_emails")
    assert latest is not None
    assert asyncio.run(latest.run()) == mail.NO_MAIL
    assert built == [("imap.example.test", 993, "emre@example.test", "app-password", "Archive")]


def test_the_tools_run_offers_are_the_ones_doctor_counts(configured: Path, wiring: Wiring) -> None:
    """`BUILTIN_TOOLS` is what `doctor` reports without building anything;
    it has to be the registry `run` builds, in order."""
    main(["run", "--terminal"])

    assert wiring.runners[0].tools == list(BUILTIN_TOOLS)


# --------------------------------------------------------------------------
# Shared by the run and doctor tests
# --------------------------------------------------------------------------


def unwrapped(printed: str) -> str:
    """What was printed, with the line breaks the eighty-column terminal
    put into a long sentence taken out again."""
    return " ".join(printed.split())


def stored_verdict() -> ProbeResult | None:
    connection = sqlite3.connect(db.database_path())
    connection.row_factory = sqlite3.Row
    try:
        return remembered(SettingsRepo(connection), "gemini", MODEL)
    finally:
        connection.close()


def write_verdict(result: ProbeResult, *, age: float) -> None:
    """A verdict `age` seconds old, as setup would have left it."""
    connection = db.open_database()
    try:
        remember(SettingsRepo(connection), "gemini", MODEL, result, now=time.time() - age)
    finally:
        connection.close()


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------


def spent(*, cost: float | None) -> None:
    """One turn of 300 in and 10 out on the configured model, on the books."""
    connection = db.open_database()
    try:
        UsageRepo(connection).insert(
            turn_id="t1", provider="gemini", model=MODEL, usage=Usage(300, 10), cost_usd=cost
        )
    finally:
        connection.close()


def test_cost_on_a_machine_that_has_spent_nothing_says_so(
    config_home: Path, own_database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["cost"]) == 0
    assert TEXT["cost_none"] in capsys.readouterr().out


def test_cost_shows_today_s_spending_by_model(
    configured: Path, own_database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spent(cost=0.0004)

    assert main(["cost"]) == 0

    printed = capsys.readouterr().out
    assert said("cost_today") in printed
    assert f"gemini:{MODEL}" in printed
    assert "$0.0004" in printed


def test_a_turn_with_no_price_is_counted_and_reported_as_unpriced(
    configured: Path, own_database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Never a zero that reads as free (section 6)."""
    spent(cost=None)

    main(["cost"])

    printed = capsys.readouterr().out
    assert "pricing.toml" in printed
    assert "$0.0000" not in printed


# --------------------------------------------------------------------------
# The terminal itself
# --------------------------------------------------------------------------


def test_the_terminal_is_told_to_speak_utf8() -> None:
    """Windows hands a redirected stream the machine's legacy code page, and
    half the Turkish alphabet has no place in it - which ends the program with
    a `UnicodeEncodeError` in the middle of an answer."""

    class Stream:
        def __init__(self) -> None:
            self.asked: dict[str, str] = {}

        def reconfigure(self, **asked: str) -> None:
            self.asked = asked

    stream = Stream()
    use_utf8(stream)

    assert stream.asked["encoding"] == "utf-8"


def test_the_terminal_is_told_before_anything_is_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both streams, and before the first command runs - the wizard prints
    Turkish, and so does everything `run` shows."""

    class Recording(StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.asked: dict[str, str] = {}

        def reconfigure(self, **asked: str) -> None:
            self.asked = asked

    out, errors = Recording(), Recording()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", errors)

    main([])

    assert out.asked["encoding"] == "utf-8"
    assert errors.asked["encoding"] == "utf-8"


def test_a_stream_that_cannot_be_told_is_left_alone() -> None:
    """Something else is holding stdout - a test harness, a pipe of somebody
    else's making. Not being able to ask is not a reason to refuse to start."""
    use_utf8(object())


# --------------------------------------------------------------------------
# telegram login (2026-09-15)
# --------------------------------------------------------------------------


class ScriptedTerminal:
    """Stands in for `TerminalPrompter`: scripted answers, recorded sayings."""

    answers: ClassVar[dict[str, str | None]] = {}
    said: ClassVar[list[tuple[str, dict[str, object]]]] = []
    built_text: ClassVar[dict[str, str]] = {}

    def __init__(self, *, text: Mapping[str, str] | None = None) -> None:
        self.text = dict(text or {})
        ScriptedTerminal.built_text = self.text

    def say(self, key: str, **fields: object) -> None:
        ScriptedTerminal.said.append((key, fields))
        assert key in self.text, key

    async def choose(self, key: str, options: object) -> str | None:
        raise AssertionError("the login never offers a menu")

    async def secret(self, key: str) -> str | None:
        return ScriptedTerminal.answers.get(key)

    async def ask(self, key: str) -> str | None:
        return ScriptedTerminal.answers.get(key)


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> type[ScriptedTerminal]:
    ScriptedTerminal.answers = {}
    ScriptedTerminal.said = []
    ScriptedTerminal.built_text = {}
    monkeypatch.setattr(setup_wizard, "TerminalPrompter", ScriptedTerminal)
    return ScriptedTerminal


class FakeTelegramLogin:
    """What `messaging/telegram.py`'s real client would do, without Telegram."""

    def __init__(self, api_id: int, api_hash: str) -> None:
        self.api_id, self.api_hash = api_id, api_hash

    async def connect(self) -> None:
        pass

    async def send_code(self, phone: str) -> str:
        return "code-hash"

    async def sign_in(self, phone: str, code: str, *, code_hash: str) -> str:
        if code != "12345":
            from allie.messaging.telegram import LoginError

            raise LoginError("The phone code entered was invalid")
        return "Emre"

    async def sign_in_with_password(self, password: str) -> str:
        return "Emre"

    def session_string(self) -> str:
        return "1BVts-the-session"

    async def disconnect(self) -> None:
        pass


@pytest.fixture
def telegram_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from allie.messaging import telegram

    monkeypatch.setattr(telegram, "_login_client", FakeTelegramLogin)


def test_telegram_login_stores_the_id_in_the_file_and_the_secrets_in_the_vault(
    configured: Path, vault: MemoryKeyring, terminal: type[ScriptedTerminal], telegram_client: None
) -> None:
    """Section 10: the `api_id` is not a secret, the hash and the session
    are. And the file is rewritten from what was loaded: the microphone the
    user chose survives (the known `setup` finding, not repeated here)."""
    save_settings(
        Settings(
            live=LiveSettings(primary=f"gemini:{MODEL}"),
            locale=LocaleSettings(code="tr"),
            audio=AudioSettings(input_device="Microphone Array 1"),
        )
    )
    terminal.answers = {
        "telegram_api_id": "123456",
        "telegram_api_hash": "abcdef",
        "telegram_phone": "+90 532 000 00 00",
        "telegram_code": "12345",
    }

    assert main(["telegram", "login"]) == 0

    loaded = load_settings()
    assert loaded.telegram.api_id == 123456
    assert loaded.audio.input_device == "Microphone Array 1"
    assert vault.vault[(KEYRING_SERVICE, "telegram")] == "abcdef"
    assert vault.vault[(KEYRING_SERVICE, "telegram-session")] == "1BVts-the-session"
    assert "abcdef" not in config_path().read_text(encoding="utf-8")
    assert terminal.said == [("telegram_logged_in", {"name": "Emre"})]


def test_telegram_login_asks_in_the_language_of_the_pack(
    configured: Path, terminal: type[ScriptedTerminal], telegram_client: None
) -> None:
    from allie.messaging import telegram

    terminal.answers = {"telegram_api_id": None}

    assert main(["telegram", "login"]) == 1

    # Walking away is said, and the prompter was built with the pack's
    # sentences for every question the login asks.
    assert terminal.said == [("cancelled", {})]
    turkish = locales.load("tr")
    for key, english in telegram.TEXT.items():
        assert terminal.built_text[key] == turkish.say(key, english)
        assert terminal.built_text[key] != english


def test_a_wrong_code_is_telegrams_reason_and_nothing_is_stored(
    configured: Path, vault: MemoryKeyring, terminal: type[ScriptedTerminal], telegram_client: None
) -> None:
    terminal.answers = {
        "telegram_api_id": "1",
        "telegram_api_hash": "h",
        "telegram_phone": "+90...",
        "telegram_code": "0",
    }

    assert main(["telegram", "login"]) == 1

    assert (KEYRING_SERVICE, "telegram-session") not in vault.vault
    assert load_settings().telegram.api_id == 0
    [(key, fields)] = terminal.said
    assert key == "telegram_login_failed"
    assert "phone code entered was invalid" in str(fields["reason"])


# --------------------------------------------------------------------------
# purge --all, and what the audit rows still say (4.6, 17 Sep 2026)
# --------------------------------------------------------------------------


def test_purge_has_to_be_told_all() -> None:
    """The one flag there is, required: deleting everything is said twice,
    once on the command line and once at the question."""
    parsed = build_parser().parse_args(["purge", "--all"])

    assert (parsed.command, parsed.all) == ("purge", True)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["purge"])


@dataclass
class Recorded:
    """What a machine that has run for a while holds, by path."""

    database: Path
    sidecars: list[Path]
    memory: Path
    logs: list[Path]
    contacts: Path

    def files(self) -> list[Path]:
        return [self.database, *self.sidecars, self.memory, *self.logs]


@pytest.fixture
def recorded(configured: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Recorded:
    """A database with the two files SQLite keeps beside it, a memory file,
    the log and one rotation left of it, a contacts file the user wrote, and
    three secrets - the key `configured` stored and Telegram's two."""
    data = tmp_path / "data"
    monkeypatch.setattr(db, "database_path", lambda: data / "assistant.db")
    db.open_database().close()
    sidecars = [data / "assistant.db-wal", data / "assistant.db-shm"]
    for sidecar in sidecars:
        sidecar.write_bytes(b"rows")
    memory = configured / MEMORY_FILE_NAME
    memory.write_text('name = "Cuma"', encoding="utf-8")
    log = logs.log_path()
    log.parent.mkdir(parents=True, exist_ok=True)
    rotated = log.with_name("assistant.2026-09-01_00-00-00_000000.log")
    for path in (log, rotated):
        path.write_text("turn: 300 in, 10 out", encoding="utf-8")
    contacts = configured / CONTACTS_FILE_NAME
    contacts.write_text("# the people I message", encoding="utf-8")
    store_api_key("telegram", "abcdef")
    store_api_key("telegram-session", "1BVts-the-session")
    return Recorded(
        database=data / "assistant.db",
        sidecars=sidecars,
        memory=memory,
        logs=[log, rotated],
        contacts=contacts,
    )


def test_purge_lists_what_goes_and_deletes_it_only_after_yes(
    recorded: Recorded, vault: MemoryKeyring, terminal: type[ScriptedTerminal]
) -> None:
    """Section 3.7, 4.6: everything the assistant recorded goes - the rows,
    the WAL that holds rows too, the memory, the logs, the secrets - and
    the two files the user wrote by hand stay. Listed by path and by entry
    name: no value of a key is ever said."""
    terminal.answers = {"purge_confirm": "yes"}

    assert main(["purge", "--all"]) == 0

    assert not any(path.exists() for path in recorded.files())
    assert config_path().is_file()
    assert recorded.contacts.is_file()
    assert vault.vault == {}

    keys = [key for key, _ in terminal.said]
    assert keys[0] == "purge_will_delete"
    assert keys[-2:] == ["purge_keeps", "purge_done"]
    listed = {fields["path"] for key, fields in terminal.said if key == "purge_file"}
    assert listed == set(recorded.files())
    named = [fields["name"] for key, fields in terminal.said if key == "purge_secret"]
    assert named == ["gemini", "telegram", "telegram-session"]
    assert terminal.said[-1] == ("purge_done", {"count": 9})
    spoken = str(terminal.said)
    assert "AIza-not-a-real-key" not in spoken
    assert "abcdef" not in spoken
    assert "1BVts" not in spoken


@pytest.mark.parametrize("answer", ["evet", "y", "", None])
def test_purge_deletes_nothing_without_the_word(
    recorded: Recorded,
    vault: MemoryKeyring,
    terminal: type[ScriptedTerminal],
    answer: str | None,
) -> None:
    """The pack's yes-word is for the voice; the typed word is `yes` and
    the question says so. Anything else, and walking away, is a no."""
    terminal.answers = {"purge_confirm": answer}

    assert main(["purge", "--all"]) == 1

    assert all(path.is_file() for path in recorded.files())
    assert len(vault.vault) == 3
    assert terminal.said[-1] == ("purge_cancelled", {})


def test_purge_takes_the_word_in_any_case(
    recorded: Recorded, vault: MemoryKeyring, terminal: type[ScriptedTerminal]
) -> None:
    terminal.answers = {"purge_confirm": "  YES "}

    assert main(["purge", "--all"]) == 0

    assert vault.vault == {}
    assert not recorded.database.exists()


def test_purge_on_a_machine_with_nothing_recorded_says_so(
    config_home: Path,
    vault: MemoryKeyring,
    terminal: type[ScriptedTerminal],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(db, "database_path", lambda: tmp_path / "data" / "assistant.db")

    assert main(["purge", "--all"]) == 0

    assert terminal.said == [("purge_nothing", {})]


def test_purge_speaks_the_language_of_the_pack(
    recorded: Recorded, terminal: type[ScriptedTerminal]
) -> None:
    terminal.answers = {"purge_confirm": None}

    main(["purge", "--all"])

    turkish = locales.load("tr")
    for key in ("purge_will_delete", "purge_confirm", "purge_cancelled"):
        assert terminal.built_text[key] == turkish.say(key, TEXT[key])
        assert terminal.built_text[key] != TEXT[key]


# --------------------------------------------------------------------------
# autostart on | off | status (4.2, 17 Sep 2026)
# --------------------------------------------------------------------------


class FakeRunKey:
    """The user's `Run` key as a dictionary, shared by every instance the
    command builds."""

    values: ClassVar[dict[str, str]] = {}

    def get(self, name: str) -> str | None:
        return FakeRunKey.values.get(name)

    def set(self, name: str, command: str) -> None:
        FakeRunKey.values[name] = command

    def delete(self, name: str) -> None:
        FakeRunKey.values.pop(name, None)


@pytest.fixture
def run_key(monkeypatch: pytest.MonkeyPatch) -> type[FakeRunKey]:
    from allie import autostart

    FakeRunKey.values = {}
    monkeypatch.setattr(autostart, "WindowsRegistry", FakeRunKey)
    monkeypatch.setattr(autostart, "command_line", lambda: '"C:\\x\\allie.exe" run --tray')
    return FakeRunKey


def test_autostart_takes_exactly_one_of_three_words() -> None:
    for word in ("on", "off", "status"):
        assert build_parser().parse_args(["autostart", word]).autostart_command == word
    with pytest.raises(SystemExit):
        build_parser().parse_args(["autostart"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["autostart", "maybe"])


def test_autostart_on_writes_the_run_key_and_says_what_it_wrote(
    configured: Path, run_key: type[FakeRunKey], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["autostart", "on"]) == 0

    command = '"C:\\x\\allie.exe" run --tray'
    assert run_key.values == {"allie": command}
    printed = unwrapped(capsys.readouterr().out)
    assert unwrapped(said("autostart_on").format(command=command)) == printed


def test_autostart_off_clears_it_and_status_says_which(
    configured: Path, run_key: type[FakeRunKey], capsys: pytest.CaptureFixture[str]
) -> None:
    main(["autostart", "on"])
    capsys.readouterr()

    assert main(["autostart", "status"]) == 0
    assert '"C:\\x\\allie.exe" run --tray' in capsys.readouterr().out

    assert main(["autostart", "off"]) == 0
    assert run_key.values == {}
    assert unwrapped(capsys.readouterr().out) == unwrapped(said("autostart_off"))

    assert main(["autostart", "status"]) == 0
    assert unwrapped(capsys.readouterr().out) == unwrapped(said("autostart_status_off"))


def test_autostart_speaks_the_machines_language_before_setup(
    config_home: Path, run_key: type[FakeRunKey], capsys: pytest.CaptureFixture[str]
) -> None:
    """No `config.toml` yet: the sentence is the system's language, as
    every other command before setup."""
    assert main(["autostart", "status"]) == 0

    code = locales.system_code()
    assert unwrapped(capsys.readouterr().out) == unwrapped(said("autostart_status_off", code))


def mail_settings(host: str, user: str, *, mailbox: str = "INBOX") -> Any:
    from allie.config import MailSettings

    return MailSettings(host=host, user=user, mailbox=mailbox)


# --------------------------------------------------------------------------
# doctor (3.5, 17 Sep 2026)
# --------------------------------------------------------------------------


def test_doctor_before_setup_says_so(
    config_home: Path, vault: MemoryKeyring, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["doctor"]) == 1

    assert said("not_set_up", locales.system_code()) in unwrapped(capsys.readouterr().out)


def test_doctor_reports_the_installation_and_never_a_key(
    configured: Path,
    vault: MemoryKeyring,
    run_key: type[FakeRunKey],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One screen: who answers, what leaves, where the files are, the
    tools, the limits, what is set up. The key is reported as stored and
    is not on the screen; the Telegram hash and the mail password are not
    either."""
    monkeypatch.setattr(db, "database_path", lambda: tmp_path / "data" / "assistant.db")
    write_verdict(PASSED, age=3 * 86_400 + 60)
    (configured / MEMORY_FILE_NAME).write_text(
        '[assistant]\nname = ""\n\n[user]\nfacts = ["a", "b", "c"]\n', encoding="utf-8"
    )
    (configured / CONTACTS_FILE_NAME).write_text(
        '[[contact]]\nname = "Ahmet"\nphone = "+90 532 000 00 00"\n', encoding="utf-8"
    )
    folder = configured / "tools"
    folder.mkdir()
    (folder / "mine.py").write_text(
        "from allie.tools.registry import tool\n\n\n"
        '@tool(risk="safe")\nasync def start_my_project() -> str:\n'
        '    """Starts the owner\'s project."""\n    return "started"\n',
        encoding="utf-8",
    )
    store_api_key("telegram", "the-hash")
    store_api_key("telegram-session", "the-session")
    store_api_key("mail", "the-app-password")
    save_settings(
        Settings(
            live=LiveSettings(primary=f"gemini:{MODEL}"),
            locale=LocaleSettings(code="tr"),
            stt=STTSettings(provider="gemini"),
            limits=LimitSettings(hard_stop=True),
            mail=mail_settings("imap.example.test", "emre@example.test"),
        )
    )
    from allie.config import TelegramSettings

    loaded = load_settings()
    loaded.telegram = TelegramSettings(api_id=123456)
    save_settings(loaded)
    command = '"C:\\x\\allie.exe" run --tray'
    run_key.values["allie"] = command

    assert main(["doctor"]) == 0

    out = unwrapped(capsys.readouterr().out)

    def line(key: str, **fields: object) -> str:
        return unwrapped(said(key).format(**fields))

    assert line("doctor_model", model=f"gemini:{MODEL}", provider="Google Gemini Live") in out
    assert line("doctor_key_stored") in out
    assert line("doctor_verdict_ok", days=3) in out
    assert line("doctor_stt_gemini", model="gemini-3.5-transcribe-live", size="small") in out
    assert line("doctor_voice_goes") in out
    assert line("doctor_answer_stays") in out
    assert line("doctor_text_goes", provider="Google Gemini Live") in out
    assert line("doctor_memory", path=configured / MEMORY_FILE_NAME, count=3) in out
    assert line("doctor_contacts", path=configured / CONTACTS_FILE_NAME, count=1) in out
    assert line("doctor_builtin", count=len(BUILTIN_TOOLS)) in out
    assert line("doctor_local", count=1, names="start_my_project") in out
    assert line("doctor_spend_stop", daily="2.00", monthly="30.00") in out
    assert line("doctor_retention", days=30) in out
    mail_line = line(
        "doctor_mail", user="emre@example.test", host="imap.example.test", mailbox="INBOX"
    )
    assert mail_line in out
    assert line("doctor_telegram", api_id=123456) in out
    assert line("doctor_autostart_on", command=command) in out
    for secret in ("AIza-not-a-real-key", "the-hash", "the-session", "the-app-password"):
        assert secret not in out


def test_doctor_on_a_bare_setup_says_what_is_not_there(
    configured: Path,
    vault: MemoryKeyring,
    run_key: type[FakeRunKey],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(db, "database_path", lambda: tmp_path / "data" / "assistant.db")

    assert main(["doctor"]) == 0

    out = unwrapped(capsys.readouterr().out)
    for key in (
        "doctor_verdict_none",
        "doctor_voice_stays",
        "doctor_answer_stays",
        "doctor_local_none",
        "doctor_mail_none",
        "doctor_telegram_none",
        "doctor_autostart_off",
    ):
        assert unwrapped(said(key)) in out, key
    assert "(henüz oluşmadı)" in out
    assert "speech model" not in out
