"""The questions of `allie setup`, without a terminal.

The wizard is the only place where a key the user typed exists in memory, so
the tests that matter are about what happens to it: it is checked before it is
kept, a key that does not work is never written anywhere, and it is never
printed back. The rest - which provider, which model, which language - is
ordinary bookkeeping.

Nothing here draws on a screen. `run_setup` talks to a `Prompter`, and this
suite hands it a scripted one, so a wizard run is a function call with a
recorded transcript rather than a session someone has to sit through.

Since 2.6 the wizard also sends the chosen model one request to see whether
it calls a tool, and refuses one that does not. The fake provider below
answers that request from a class attribute, so a test can say which of
its models call tools and which only talk.

Since 2026-09-15 the last question is which microphone to listen through,
and `allie mic` asks that one question on its own. The device list is
handed in, so no test touches PortAudio.

The live product (plan.md L1.6) added a question between the model and the
microphone: who hears the yes or no of a confirmation. D36 (2026-09-26)
took it away again - Google hears it, nothing on this machine does. Who
reads the questions out loud was a second one until D32: the assistant
does, in its own voice. And the microphone list puts
the Windows audio engine first and warns on a raw kernel-streaming choice
(D18): that path has no echo cancellation, and the assistant hears itself.

Since 2026-09-23 the first question is which assistant (D31): Jarvis,
Vesper, Allie or Friday, a name, a voice and a wake phrase each. It took
the place of L1.6's free-text voice question.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, ClassVar

import pytest
import questionary
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from allie import locales
from allie.assistants import load_assistants
from allie.audio.capture import MicrophoneInfo, Microphones
from allie.config import (
    KEYRING_SERVICE,
    AudioSettings,
    LiveSettings,
    LocaleSettings,
    Settings,
    WakeSettings,
    config_path,
    load_settings,
    save_settings,
    store_api_key,
)
from allie.live.base import (
    AuthenticationError,
    LiveEvent,
    ModelInfo,
    OutputText,
    ProviderError,
    SessionConfig,
    ToolCall,
    ToolCallEvent,
    TurnComplete,
)
from allie.live.probe import NO_TOOL_CALL, QUESTION, ProbeResult, remembered
from allie.live.registry import ADAPTERS, ProviderEntry
from allie.setup_wizard import (
    TEXT,
    Option,
    TerminalPrompter,
    run_microphone_setup,
    run_settings,
    run_setup,
    wording,
)
from allie.store import db
from allie.store.db import open_database
from allie.store.memory import UserMemory, memory_path
from allie.store.repos import SettingsRepo
from tests.conftest import MemoryKeyring
from tests.live_contract import FakeLiveSession

GOOD_KEY = "good-key"
# A key checked while the network is down: the provider cannot say whether it
# works, and the adapter reports that as a refusal of the request, not the key.
OFFLINE_KEY = "offline"
# A key that was stored once and has been revoked since: only asking for the
# models finds out.
DEAD_KEY = "revoked-since"


class FakeProvider:
    """A provider that accepts one key, offers two models, and answers the
    probe of 2.6 for each of them as the class attributes say: a model in
    `tool_callers` calls the clock, one in `refusing` makes the provider
    refuse the request, any other only talks."""

    id = "fake"
    capabilities: ClassVar[frozenset[str]] = frozenset()
    models: ClassVar[list[ModelInfo]] = [
        ModelInfo(id="fast", display_name="Fast"),
        ModelInfo(id="smart", display_name="Smart"),
    ]
    tool_callers: ClassVar[set[str]] = {"fast", "smart"}
    refusing: ClassVar[set[str]] = set()
    # A server that is not running: every request fails to reach it.
    down: ClassVar[bool] = False

    def __init__(self, api_key: str, *, keyless: bool = False, base_url: str | None = None) -> None:
        self.api_key = api_key
        # Built for an entry that needs no key: an empty key is then right.
        self.keyless = keyless
        self.base_url = base_url
        # What the probe asked, as (model, question, tool names).
        self.probed: list[tuple[str, str, list[str]]] = []

    async def validate_credentials(self) -> bool:
        if self.api_key == OFFLINE_KEY or self.down:
            raise ProviderError("fake could not be reached (ConnectError)")
        if self.keyless:
            return self.api_key == ""
        return self.api_key == GOOD_KEY

    async def list_models(self) -> list[ModelInfo]:
        if self.down:
            raise ProviderError("fake could not be reached (ConnectError)")
        if self.api_key == DEAD_KEY:
            raise AuthenticationError("fake refused the key")
        return list(self.models)

    @asynccontextmanager
    async def connect(self, config: SessionConfig) -> AsyncIterator[FakeLiveSession]:
        if config.model in self.refusing:
            raise ProviderError("fake refused the request (429): slow down", kind="rate_limit")
        answer: LiveEvent
        if config.model in self.tool_callers:
            answer = ToolCallEvent(
                ToolCall(id="c1", name="get_current_time", arguments={"city": "x"})
            )
        else:
            answer = OutputText("It is about three.")
        session = FakeLiveSession([answer, TurnComplete()])
        yield session
        # What the probe asked: the model, the one text turn, the tools offered.
        question = session.texts[0][0] if session.texts else ""
        self.probed.append((config.model, question, [tool.name for tool in config.tools]))


class ScriptedPrompter:
    """Answers the wizard from a script and records the whole exchange.

    Questions are addressed by a stable key rather than by their wording, so a
    test says what it answers instead of repeating a sentence. Every key is
    checked against `TEXT`: a wizard that asks something the text table has no
    words for fails here rather than in front of the user.
    """

    def __init__(self, **answers: str | list[str | None] | None) -> None:
        self._script: dict[str, list[str | None]] = {
            key: list(value) if isinstance(value, list) else [value]
            for key, value in answers.items()
        }
        self.asked: list[str] = []
        self.offered: dict[str, list[str]] = {}
        self.labelled: dict[str, list[str]] = {}
        self.said: list[tuple[str, dict[str, object]]] = []

    def say(self, key: str, **fields: object) -> None:
        assert key in TEXT, f"the wizard said {key!r}, which has no text"
        self.said.append((key, fields))

    async def choose(self, key: str, options: Sequence[Option]) -> str | None:
        self.offered[key] = [option.value for option in options]
        self.labelled[key] = [option.label for option in options]
        return self._answer(key)

    async def secret(self, key: str) -> str | None:
        return self._answer(key)

    async def ask(self, key: str) -> str | None:
        return self._answer(key)

    async def menu(self, key: str, options: Sequence[Option]) -> str | None:
        self.offered[key] = [option.value for option in options]
        self.labelled[key] = [option.label for option in options]
        return self._answer(key)

    def _answer(self, key: str) -> str | None:
        assert key in TEXT, f"the wizard asked {key!r}, which has no text"
        self.asked.append(key)
        queue = self._script.get(key)
        assert queue, f"the wizard asked {key!r} more often than the script answers"
        return queue.pop(0)


def fake_catalog(*provider_ids: str) -> dict[str, ProviderEntry]:
    return {
        provider_id: ProviderEntry(
            id=provider_id,
            adapter="fake",
            display_name=provider_id.title(),
            key_url=f"https://{provider_id}.example/apikey",
        )
        for provider_id in (provider_ids or ("gemini",))
    }


@pytest.fixture(autouse=True)
def fake_adapter(monkeypatch: pytest.MonkeyPatch) -> list[FakeProvider]:
    """Registers an adapter that answers without a network, and keeps every
    provider it built so a test can read what the wizard asked of it."""
    built: list[FakeProvider] = []

    def build(entry: ProviderEntry, api_key: str) -> FakeProvider:
        provider = FakeProvider(api_key, keyless=not entry.requires_key, base_url=entry.base_url)
        built.append(provider)
        return provider

    monkeypatch.setitem(ADAPTERS, "fake", build)
    return built


@pytest.fixture(autouse=True)
def own_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The wizard opens the machine's database to write the verdict on the
    model (2.6). Here that is a file under this test's directory."""
    path = tmp_path / "data" / "assistant.db"
    monkeypatch.setattr(db, "database_path", lambda: path)
    return path


@pytest.fixture
def verdicts() -> Iterator[sqlite3.Connection]:
    """A database handed to the wizard, so that what it wrote can be read."""
    connection = open_database(":memory:")
    yield connection
    connection.close()


ARRAY = MicrophoneInfo(name="Microphone Array (Intel Smart ", host_api="MME")
RAW_ARRAY = MicrophoneInfo(name="Microphone Array 1 ()", host_api="Windows WDM-KS")
HEADSET = MicrophoneInfo(name="Headset (Buds3 Hands-Free AG Audio)", host_api="Windows WASAPI")
LAPTOP = Microphones(default="Microphone Array (Intel Smart ", devices=(ARRAY, RAW_ARRAY, HEADSET))
NO_MICROPHONE = Microphones(default=None, devices=())


@pytest.fixture(autouse=True)
def laptop_microphones(monkeypatch: pytest.MonkeyPatch) -> None:
    """What the wizard finds when nobody hands it a list: the laptop above,
    never PortAudio."""
    monkeypatch.setattr("allie.setup_wizard.available_microphones", lambda: LAPTOP)


def complete_run(**overrides: str | list[str | None] | None) -> ScriptedPrompter:
    """A prompter scripted to walk the wizard from end to end."""
    answers: dict[str, str | list[str | None] | None] = {
        "assistant": "vesper",
        "locale": "tr",
        "api_key": GOOD_KEY,
        "model": "fast",
        "microphone": "",
    }
    answers.update(overrides)
    return ScriptedPrompter(**answers)


# --------------------------------------------------------------------------
# What the wizard leaves behind
# --------------------------------------------------------------------------


async def test_the_answers_end_up_in_the_settings_and_the_vault(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = complete_run(model="smart")

    exit_code = await run_setup(prompter, catalog=fake_catalog())

    settings = load_settings()
    assert exit_code == 0
    assert settings.live.primary == "gemini:smart"
    assert settings.locale.code == "tr"
    assert vault.vault == {(KEYRING_SERVICE, "gemini"): GOOD_KEY}


async def test_the_key_is_not_written_to_the_settings_file(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """The one claim section 3.3 makes about key safety, from the other end."""
    await run_setup(complete_run(), catalog=fake_catalog())

    assert GOOD_KEY not in config_path().read_text(encoding="utf-8")


async def test_the_key_is_never_printed_back(config_home: Path, vault: MemoryKeyring) -> None:
    """A key echoed to the terminal outlives the session in the scrollback."""
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    printed = [str(value) for _, fields in prompter.said for value in fields.values()]
    assert not any(GOOD_KEY in text for text in printed)


async def test_setup_run_again_changes_the_model_and_keeps_the_rest(
    config_home: Path, vault: MemoryKeyring
) -> None:
    save_settings(Settings(live=LiveSettings(primary="gemini:fast")))

    await run_setup(complete_run(model="smart"), catalog=fake_catalog())

    assert load_settings().live.primary == "gemini:smart"


# --------------------------------------------------------------------------
# The provider
# --------------------------------------------------------------------------


async def test_one_buildable_provider_is_not_worth_a_question(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    assert "provider" not in prompter.asked
    assert load_settings().live.provider == "gemini"


async def test_only_providers_this_build_can_construct_are_offered(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """An entry naming an adapter this build does not have - the `litellm`
    escape hatch of section 5.11, say - is a dead end the user would only
    discover after typing their key in."""
    catalog: Mapping[str, ProviderEntry] = {
        **fake_catalog("gemini", "openrouter"),
        "proxy": ProviderEntry(id="proxy", adapter="litellm", display_name="Proxy"),
    }
    prompter = complete_run(provider="openrouter")

    await run_setup(prompter, catalog=catalog)

    assert prompter.offered["provider"] == ["gemini", "openrouter"]
    assert load_settings().live.provider == "openrouter"


# --------------------------------------------------------------------------
# A provider that needs no key, and one that needs an address (2.7)
# --------------------------------------------------------------------------


def local_catalog() -> dict[str, ProviderEntry]:
    """Ollama, as the catalogue has it: no key, an address of its own."""
    return {
        "ollama": ProviderEntry(
            id="ollama",
            adapter="fake",
            display_name="Ollama",
            base_url="http://localhost:11434/v1",
            requires_key=False,
        )
    }


def custom_catalog() -> dict[str, ProviderEntry]:
    """`custom`, as the catalogue has it: a key, and no address."""
    return {"custom": ProviderEntry(id="custom", adapter="openai_compat", display_name="Other")}


@pytest.fixture
def addressed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lets the fake adapter stand in for the one that needs an address, so
    the wizard asks the question the registry says it must."""
    monkeypatch.setitem(ADAPTERS, "openai_compat", ADAPTERS["fake"])


async def test_a_provider_that_needs_no_key_is_not_asked_for_one(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """A local Ollama has nothing to authenticate with. The server is asked
    whether it answers, and nothing goes to the Credential Manager."""
    prompter = ScriptedPrompter(assistant="vesper", locale="tr", model="fast", microphone="")

    exit_code = await run_setup(prompter, catalog=local_catalog())

    said = [key for key, _ in prompter.said]
    assert exit_code == 0
    assert "api_key" not in prompter.asked
    assert "api_key_keep" not in prompter.asked
    assert "checking_server" in said
    assert "key_url" not in said
    assert vault.vault == {}
    assert load_settings().live.primary == "ollama:fast"


async def test_a_local_server_that_does_not_answer_ends_the_wizard(
    config_home: Path, vault: MemoryKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is no key to ask for again: the user starts the server and
    comes back. Nothing is written."""
    monkeypatch.setattr(FakeProvider, "down", True)
    prompter = ScriptedPrompter(assistant="vesper", locale="tr")

    exit_code = await run_setup(prompter, catalog=local_catalog())

    said = [key for key, _ in prompter.said]
    assert exit_code != 0
    assert "provider_unreachable" in said
    assert not config_path().exists()


async def test_a_custom_server_is_asked_for_its_address_and_the_address_is_kept(
    config_home: Path, vault: MemoryKeyring, fake_adapter: list[FakeProvider], addressed: None
) -> None:
    """The one entry the catalogue gives no address for. The answer reaches
    the adapter and `config.toml`, beside the model, where a text editor
    can reach it."""
    prompter = complete_run(base_url="http://localhost:1234/v1")

    exit_code = await run_setup(prompter, catalog=custom_catalog())

    assert exit_code == 0
    assert "base_url" in prompter.asked
    assert fake_adapter[-1].base_url == "http://localhost:1234/v1"
    assert load_settings().live.primary == "custom:fast"
    assert load_settings().live.base_url == "http://localhost:1234/v1"


async def test_an_empty_address_is_asked_again(
    config_home: Path, vault: MemoryKeyring, addressed: None
) -> None:
    prompter = complete_run(base_url=["", "   ", " http://localhost:1234/v1 "])

    exit_code = await run_setup(prompter, catalog=custom_catalog())

    assert exit_code == 0
    assert prompter.asked.count("base_url") == 3
    assert load_settings().live.base_url == "http://localhost:1234/v1"


async def test_a_provider_with_an_address_of_its_own_is_not_asked_for_one(
    config_home: Path, vault: MemoryKeyring, fake_adapter: list[FakeProvider]
) -> None:
    """Gemini speaks to one vendor, Groq's address is in the catalogue:
    neither is a question, and nothing about an address lands in the file."""
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    assert "base_url" not in prompter.asked
    assert load_settings().live.base_url == ""


async def test_the_address_is_asked_before_the_key(
    config_home: Path, vault: MemoryKeyring, addressed: None
) -> None:
    """Where the server is comes before what it is told; a key typed for a
    server the wizard cannot yet reach would be checked against nothing."""
    prompter = complete_run(base_url="http://localhost:1234/v1")

    await run_setup(prompter, catalog=custom_catalog())

    assert prompter.asked.index("base_url") < prompter.asked.index("api_key")


# --------------------------------------------------------------------------
# The key
# --------------------------------------------------------------------------


async def test_a_key_that_does_not_work_is_asked_again(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = complete_run(api_key=["typo", GOOD_KEY])

    exit_code = await run_setup(prompter, catalog=fake_catalog())

    assert exit_code == 0
    assert prompter.asked.count("api_key") == 2
    assert vault.vault == {(KEYRING_SERVICE, "gemini"): GOOD_KEY}


async def test_a_key_that_does_not_work_is_never_stored(
    config_home: Path, vault: MemoryKeyring
) -> None:
    exit_code = await run_setup(complete_run(api_key=["typo", None]), catalog=fake_catalog())

    assert exit_code != 0
    assert vault.vault == {}


async def test_a_provider_that_cannot_be_reached_is_not_called_a_bad_key(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """Offline during setup. "That key did not work" would send the user to
    the provider's console to replace a key that is fine; the wizard says
    what actually happened and asks again."""
    prompter = complete_run(api_key=[OFFLINE_KEY, GOOD_KEY])

    exit_code = await run_setup(prompter, catalog=fake_catalog())

    said = [key for key, _ in prompter.said]
    assert exit_code == 0
    assert "provider_unreachable" in said
    assert "bad_key" not in said
    assert vault.vault == {(KEYRING_SERVICE, "gemini"): GOOD_KEY}


async def test_an_empty_answer_is_not_a_key(config_home: Path, vault: MemoryKeyring) -> None:
    """Enter on an empty prompt is a slip, not a key to go and check.

    The provider is never asked about an empty string: `genai.Client` raises on
    one, so what should be "you left it blank" would be a traceback.
    """
    prompter = complete_run(api_key=["", GOOD_KEY])

    exit_code = await run_setup(prompter, catalog=fake_catalog())

    said = [key for key, _ in prompter.said]
    assert exit_code == 0
    assert prompter.asked.count("api_key") == 2
    assert said.count("key_needed") == 1
    assert "bad_key" not in said


async def test_a_stored_key_is_kept_when_the_answer_is_empty(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """Changing the model must not mean fetching the key from the browser again."""
    store_api_key("gemini", GOOD_KEY)
    prompter = complete_run(api_key_keep="")

    exit_code = await run_setup(prompter, catalog=fake_catalog())

    assert exit_code == 0
    assert "api_key" not in prompter.asked
    assert vault.vault == {(KEYRING_SERVICE, "gemini"): GOOD_KEY}


async def test_a_stored_key_that_stopped_working_is_replaced(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """A revoked key is the reason the owner reruns setup in the first place."""
    store_api_key("gemini", "revoked")
    prompter = complete_run(api_key_keep="")

    exit_code = await run_setup(prompter, catalog=fake_catalog())

    assert exit_code == 0
    assert prompter.asked[:5] == ["assistant", "locale", "api_key_keep", "api_key", "model"]
    assert vault.vault == {(KEYRING_SERVICE, "gemini"): GOOD_KEY}


# --------------------------------------------------------------------------
# The model and the language
# --------------------------------------------------------------------------


async def test_the_models_offered_are_the_ones_the_key_can_reach(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    assert prompter.offered["model"] == ["fast", "smart"]


async def test_every_locale_pack_in_the_package_is_offered(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """The menu is the contents of `locales/`, which is what makes a new
    language one TOML file and no code change (section 3.12)."""
    prompter = complete_run(locale="en")

    await run_setup(prompter, catalog=fake_catalog())

    assert prompter.offered["locale"] == [pack.code for pack in locales.available()]
    assert set(prompter.offered["locale"]) == {"en", "tr"}
    assert load_settings().locale.code == "en"


def test_the_wizard_speaks_the_language_it_was_told_to_last_time(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """Coming back to change the model should not mean reading English again."""
    save_settings(Settings(locale=LocaleSettings(code="tr")))

    assert wording()["model"] == locales.load("tr").say("model", TEXT["model"])
    assert wording()["model"] != TEXT["model"]


def test_the_first_run_speaks_whatever_language_windows_speaks(
    config_home: Path, vault: MemoryKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing has been chosen yet, so the machine's own language is the only
    thing there is to go on."""
    monkeypatch.setattr(locales, "system_code", lambda: "tr")

    assert wording()["model"] == locales.load("tr").say("model", TEXT["model"])


def test_every_question_has_words_whatever_language_the_wizard_starts_in(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """With no answer from last time it is Windows' own language, and that may
    be one nobody has translated - which is what `TEXT` is for."""
    assert set(wording()) == set(TEXT)
    assert all(sentence.strip() for sentence in wording().values())


# --------------------------------------------------------------------------
# The tool-use probe (2.6)
# --------------------------------------------------------------------------


async def test_a_model_that_does_not_call_tools_cannot_be_chosen(
    config_home: Path, vault: MemoryKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The most valuable thirty lines of section 3.2: the model is offered
    again until one that calls the tool is picked, and only that one is
    written down."""
    monkeypatch.setattr(FakeProvider, "tool_callers", {"smart"})
    prompter = complete_run(model=["fast", "smart"])

    exit_code = await run_setup(prompter, catalog=fake_catalog())

    said = [key for key, _ in prompter.said]
    assert exit_code == 0
    assert prompter.asked.count("model") == 2
    assert said.count("tools_failed") == 1
    assert said.count("tools_ok") == 1
    assert load_settings().live.primary == "gemini:smart"


async def test_a_model_that_calls_tools_is_taken_at_the_first_answer(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    said = [key for key, _ in prompter.said]
    assert prompter.asked.count("model") == 1
    assert "tools_failed" not in said
    assert said.index("probing_tools") < said.index("tools_ok") < said.index("saved")


async def test_the_first_token_time_is_shown_as_a_whole_number(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    [fields] = [fields for key, fields in prompter.said if key == "tools_ok"]
    assert str(fields["ms"]).isdigit()


async def test_the_probe_asks_in_the_language_that_was_just_chosen(
    config_home: Path, vault: MemoryKeyring, fake_adapter: list[FakeProvider]
) -> None:
    """Section 3.2 wanted the question in the user's language, so that one
    request checks tool calling and understanding together; the pack is
    where the language comes from (section 3.12)."""
    await run_setup(complete_run(locale="tr"), catalog=fake_catalog())

    [(model, question, tools)] = fake_adapter[-1].probed
    assert model == "fast"
    assert question == locales.load("tr").probe_question
    assert question != QUESTION
    assert tools == ["get_current_time"]


async def test_the_probe_falls_back_to_the_english_question(
    config_home: Path, vault: MemoryKeyring, fake_adapter: list[FakeProvider]
) -> None:
    """`en.toml` carries no question; the constant beside the code asks."""
    await run_setup(complete_run(locale="en"), catalog=fake_catalog())

    [(_, question, _)] = fake_adapter[-1].probed
    assert question == QUESTION


async def test_the_verdict_on_the_chosen_model_is_written_down(
    config_home: Path, vault: MemoryKeyring, verdicts: sqlite3.Connection
) -> None:
    """So that `allie run` need not ask the same question for a week."""
    await run_setup(complete_run(model="smart"), catalog=fake_catalog(), database=verdicts)

    found = remembered(SettingsRepo(verdicts), "gemini", "smart")
    assert found is not None
    assert found.ok is True
    assert remembered(SettingsRepo(verdicts), "gemini", "fast") is None


async def test_the_wizard_opens_the_machine_s_database_when_none_is_handed_over(
    config_home: Path, vault: MemoryKeyring, own_database: Path
) -> None:
    """On a fresh machine the wizard is the first thing to touch the
    database, and it has to build it - with its schema - to write into it."""
    await run_setup(complete_run(), catalog=fake_catalog())

    connection = open_database(own_database)
    try:
        found = remembered(SettingsRepo(connection), "gemini", "fast")
    finally:
        connection.close()
    assert found is not None
    assert found.ok is True


async def test_a_provider_that_refuses_the_probe_has_said_nothing_about_the_model(
    config_home: Path, vault: MemoryKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rate limit or a bad day is not "this model cannot call tools":
    that sentence would send the user away from a model that is fine. The
    provider's own words are shown and the list is offered again."""
    monkeypatch.setattr(FakeProvider, "refusing", {"smart"})
    prompter = complete_run(model=["smart", "fast"])

    exit_code = await run_setup(prompter, catalog=fake_catalog())

    said = [key for key, _ in prompter.said]
    refusals = [fields for key, fields in prompter.said if key == "probe_refused"]
    assert exit_code == 0
    assert "tools_failed" not in said
    assert "slow down" in str(refusals[0]["problem"])
    assert load_settings().live.primary == "gemini:fast"


async def test_walking_away_from_the_probe_s_verdict_writes_nothing(
    config_home: Path,
    vault: MemoryKeyring,
    monkeypatch: pytest.MonkeyPatch,
    verdicts: sqlite3.Connection,
) -> None:
    """Every model the key reaches only talks; the user gives up at the
    second question. Nothing is stored - not the key, not the settings,
    not the failed verdict."""
    monkeypatch.setattr(FakeProvider, "tool_callers", set())
    prompter = complete_run(model=["fast", None])

    exit_code = await run_setup(prompter, catalog=fake_catalog(), database=verdicts)

    assert exit_code != 0
    assert vault.vault == {}
    assert not config_path().exists()
    assert verdicts.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 0


def test_the_failed_verdict_has_the_reason_of_section_3_2() -> None:
    """What the probe writes down is what `allie run` will read a week
    later; the word is the one the design names."""
    assert ProbeResult(ok=False, reason=NO_TOOL_CALL).reason == "no_tool_call_emitted"


# --------------------------------------------------------------------------
# The assistant (D31): a name, a voice and a wake phrase
# --------------------------------------------------------------------------


async def test_the_four_assistants_are_offered_first_by_name_and_voice(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """In the wizard's own language, like the engines' labels: the language
    of last time, which is what the file says before the run."""
    save_settings(Settings(locale=LocaleSettings(code="tr")))
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    assert prompter.asked[0] == "assistant"
    assert prompter.offered["assistant"] == ["jarvis", "vesper", "allie", "friday"]
    male, female = turkish("assistant_male"), turkish("assistant_female")
    assert prompter.labelled["assistant"] == [
        male.format(name="Jarvis"),
        male.format(name="Vesper"),
        female.format(name="Allie"),
        female.format(name="Friday"),
    ]


async def test_the_assistant_chosen_is_its_wake_word_its_voice_and_its_name(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """Vesper: "hey Vesper" wakes it, it speaks in the voice the catalogue
    gives it on Gemini, and the model is told its name through memory.toml.
    No threshold is written - the one the model shipped with is read."""
    await run_setup(complete_run(assistant="vesper"), catalog=fake_catalog())

    settings = load_settings()
    assert (settings.wake.enabled, settings.wake.model) == (True, "hey_vesper")
    assert settings.wake.threshold is None
    assert settings.live.voice == load_assistants()["vesper"].voice_for("gemini")
    assert settings.live.voice
    assert UserMemory.load().name == "Vesper"


@pytest.mark.parametrize("chosen", ["jarvis", "vesper", "allie", "friday"])
async def test_every_assistant_is_set_up_the_same_way(
    config_home: Path, vault: MemoryKeyring, chosen: str
) -> None:
    await run_setup(complete_run(assistant=chosen), catalog=fake_catalog())

    assistant = load_assistants()[chosen]
    settings = load_settings()
    assert settings.wake.model == assistant.wake
    assert settings.live.voice == assistant.voice_for("gemini")
    assert UserMemory.load().name == assistant.name


async def test_another_assistant_renames_the_one_there_was(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """The owner's memory.toml says Friday; choosing Vesper must not leave
    a model that wakes to "hey Vesper" and calls itself Friday. The facts
    stay where they were."""
    UserMemory(name="Friday", facts=["Emre Patron"]).save()

    await run_setup(complete_run(assistant="vesper"), catalog=fake_catalog())

    memory = UserMemory.load()
    assert (memory.name, memory.facts) == ("Vesper", ["Emre Patron"])


async def test_a_memory_file_that_does_not_parse_is_not_written_over(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """`run` will say what is wrong with it; setup still writes its settings."""
    broken = memory_path()
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("[assistant\nname = ", encoding="utf-8")

    exit_code = await run_setup(complete_run(), catalog=fake_catalog())

    assert exit_code == 0
    assert broken.read_text(encoding="utf-8") == "[assistant\nname = "
    assert load_settings().wake.model == "hey_vesper"


async def test_a_provider_with_no_voice_for_the_assistant_speaks_in_its_own(
    config_home: Path, vault: MemoryKeyring
) -> None:
    await run_setup(
        complete_run(provider="openrouter"), catalog=fake_catalog("gemini", "openrouter")
    )

    assert load_settings().live.voice == ""


async def test_the_same_assistant_again_keeps_the_threshold_measured_for_it(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """A threshold from the owner's own recordings belongs to the model it
    was measured on; the greeting is theirs whatever they choose."""
    save_settings(
        Settings(
            wake=WakeSettings(enabled=True, model="hey_vesper", threshold=0.7, greeting="none")
        )
    )

    await run_setup(complete_run(assistant="vesper"), catalog=fake_catalog())

    wake = load_settings().wake
    assert (wake.threshold, wake.greeting) == (0.7, "none")


async def test_another_assistant_starts_from_its_own_model_s_threshold(
    config_home: Path, vault: MemoryKeyring
) -> None:
    save_settings(
        Settings(
            wake=WakeSettings(enabled=True, model="hey_vesper", threshold=0.7, greeting="none")
        )
    )

    await run_setup(complete_run(assistant="allie"), catalog=fake_catalog())

    wake = load_settings().wake
    assert (wake.model, wake.threshold, wake.greeting) == ("hey_allie", None, "none")


async def test_walking_away_from_the_assistant_writes_nothing(
    config_home: Path, vault: MemoryKeyring
) -> None:
    exit_code = await run_setup(complete_run(assistant=None), catalog=fake_catalog())

    assert exit_code != 0
    assert not config_path().exists()
    assert not memory_path().exists()


# --------------------------------------------------------------------------
# Who hears the yes or no (L1.6, D36)
# --------------------------------------------------------------------------


async def test_who_hears_the_yes_or_no_is_no_question_any_more(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """D36: Google hears it, on the Gemini entry - there is nothing to
    choose, nothing is written about it, and no sentence of the wizard
    names a recogniser any more."""
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    assert "hears" not in prompter.asked
    assert "[stt]" not in config_path().read_text(encoding="utf-8")
    assert not any(key.startswith(("hears", "setting_hears")) for key in TEXT)


async def test_who_reads_the_questions_is_no_question_any_more(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """D32: the assistant reads them, in its own voice - there is nothing
    to choose, and nothing about a voice engine is written."""
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    assert "reads" not in prompter.asked
    assert "[tts]" not in config_path().read_text(encoding="utf-8")


def turkish(key: str) -> str:
    """A sentence of the wizard's as the Turkish pack has it."""
    return locales.load("tr").say(key, TEXT[key])


async def test_the_questions_come_in_the_plan_s_order(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """Plan.md L1.6 and D31: the assistant first, then provider, key,
    model, microphone - the microphone last, so that
    walking away there still leaves nothing written."""
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    assert prompter.asked == [
        "assistant",
        "locale",
        "api_key",
        "model",
        "microphone",
    ]


async def test_setup_run_again_keeps_the_session_tuning(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """The owner tunes `[live]` by hand in the real run (ADR-001) and may
    come back to setup for the voice: what setup does not ask about, it
    keeps - the session numbers."""
    save_settings(
        Settings(
            live=LiveSettings(
                primary="gemini:fast", idle_close_seconds=30.0, end_sensitivity="HIGH"
            ),
        )
    )

    await run_setup(complete_run(model="smart", assistant="friday"), catalog=fake_catalog())

    settings = load_settings()
    assert settings.live.primary == "gemini:smart"
    assert settings.live.voice == load_assistants()["friday"].voice_for("gemini")
    assert (settings.live.idle_close_seconds, settings.live.end_sensitivity) == (30.0, "HIGH")


# --------------------------------------------------------------------------
# The microphone
# --------------------------------------------------------------------------


async def test_the_microphone_chosen_lands_in_the_settings(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = complete_run(microphone=RAW_ARRAY.setting)

    await run_setup(prompter, catalog=fake_catalog())

    assert load_settings().audio.input_device == "Microphone Array 1 (), Windows WDM-KS"


async def test_the_microphone_is_the_last_question(config_home: Path, vault: MemoryKeyring) -> None:
    """After the model, so that the probe has already said its piece; before
    the file is written, so that walking away here still leaves nothing."""
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    assert prompter.asked[-1] == "microphone"


async def test_whatever_windows_has_chosen_is_offered_first_and_stored_as_nothing(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """The one answer that needs no upkeep: an empty setting has always meant
    the system default, and the label says which device that is right now."""
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    assert prompter.offered["microphone"][0] == ""
    assert "Microphone Array (Intel Smart" in prompter.labelled["microphone"][0]
    assert load_settings().audio.input_device == ""


async def test_every_device_is_offered_by_its_line_and_labelled_by_its_name(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    assert prompter.offered["microphone"][1:] == [
        "Headset (Buds3 Hands-Free AG Audio), Windows WASAPI",
        "Microphone Array (Intel Smart , MME",
        "Microphone Array 1 (), Windows WDM-KS",
    ]
    assert prompter.labelled["microphone"][3] == "Microphone Array 1 () - Windows WDM-KS"


async def test_the_windows_audio_engine_is_offered_before_raw_kernel_streaming(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """D18: through WASAPI or MME the microphone has echo cancellation and
    the assistant can be talked over; through WDM-KS it hears itself. The
    safe paths lead the list, in the order `LiveCapture` trusts them, and
    PortAudio's order is kept within each."""
    prompter = complete_run()

    await run_setup(prompter, catalog=fake_catalog())

    assert prompter.offered["microphone"] == ["", HEADSET.setting, ARRAY.setting, RAW_ARRAY.setting]


async def test_a_raw_kernel_streaming_choice_is_warned_about_and_still_taken(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """The owner's array through WDM-KS was the old product's choice; here
    it is a loop that answers itself (ADR-001 section 4). The choice is the
    user's, the sentence says what it costs."""
    prompter = complete_run(microphone=RAW_ARRAY.setting)

    await run_setup(prompter, catalog=fake_catalog())

    assert ("microphone_raw", {}) in prompter.said
    assert load_settings().audio.input_device == RAW_ARRAY.setting


async def test_a_windows_audio_engine_choice_is_not_warned_about(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = complete_run(microphone=HEADSET.setting)

    await run_setup(prompter, catalog=fake_catalog())

    assert "microphone_raw" not in [key for key, _ in prompter.said]


async def test_windows_own_choice_is_not_warned_about_either(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """Whatever Windows has chosen comes through the audio engine."""
    prompter = complete_run(microphone="")

    await run_setup(prompter, catalog=fake_catalog())

    assert "microphone_raw" not in [key for key, _ in prompter.said]


async def test_mic_warns_on_raw_kernel_streaming_too(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = ScriptedPrompter(microphone=RAW_ARRAY.setting)

    await run_microphone_setup(prompter, microphones=LAPTOP)

    assert [key for key, _ in prompter.said] == ["microphone_raw", "microphone_saved"]


async def test_a_machine_without_a_microphone_is_told_so_and_setup_goes_on(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """A desktop with nothing plugged in yet can still be set up; the default
    is what it listens through once something is."""
    prompter = complete_run()

    exit_code = await run_setup(prompter, catalog=fake_catalog(), microphones=NO_MICROPHONE)

    assert exit_code == 0
    assert "microphone" not in prompter.asked
    assert ("no_microphones", {}) in prompter.said
    assert load_settings().audio.input_device == ""


async def test_walking_away_at_the_microphone_writes_nothing(
    config_home: Path, vault: MemoryKeyring
) -> None:
    exit_code = await run_setup(complete_run(microphone=None), catalog=fake_catalog())

    assert exit_code != 0
    assert vault.vault == {}
    assert not config_path().exists()


async def test_mic_changes_the_microphone_and_keeps_the_rest(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """`allie mic` is the one question again, for the day the headset
    comes out: it rewrites `[audio]` and nothing else in the file."""
    save_settings(
        Settings(
            live=LiveSettings(primary="gemini:fast"),
            locale=LocaleSettings(code="tr"),
            audio=AudioSettings(input_device=RAW_ARRAY.setting),
        )
    )
    prompter = ScriptedPrompter(microphone=HEADSET.setting)

    exit_code = await run_microphone_setup(prompter, microphones=LAPTOP)

    settings = load_settings()
    assert exit_code == 0
    assert settings.audio.input_device == "Headset (Buds3 Hands-Free AG Audio), Windows WASAPI"
    assert settings.live.primary == "gemini:fast"
    assert settings.locale.code == "tr"
    assert prompter.asked == ["microphone"]
    assert prompter.said[-1][0] == "microphone_saved"
    assert (
        prompter.said[-1][1]["microphone"] == "Headset (Buds3 Hands-Free AG Audio) - Windows WASAPI"
    )


async def test_mic_finds_the_devices_itself_when_handed_none(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = ScriptedPrompter(microphone="")

    await run_microphone_setup(prompter)

    assert prompter.offered["microphone"] == ["", HEADSET.setting, ARRAY.setting, RAW_ARRAY.setting]


async def test_mic_cancelled_leaves_the_file_as_it_was(
    config_home: Path, vault: MemoryKeyring
) -> None:
    save_settings(Settings(audio=AudioSettings(input_device=RAW_ARRAY.setting)))
    prompter = ScriptedPrompter(microphone=None)

    exit_code = await run_microphone_setup(prompter, microphones=LAPTOP)

    assert exit_code != 0
    assert load_settings().audio.input_device == RAW_ARRAY.setting
    assert prompter.said[-1] == ("cancelled", {})


async def test_mic_with_nothing_to_choose_from_gives_up(
    config_home: Path, vault: MemoryKeyring
) -> None:
    prompter = ScriptedPrompter()

    exit_code = await run_microphone_setup(prompter, microphones=NO_MICROPHONE)

    assert exit_code != 0
    assert prompter.asked == []
    assert prompter.said == [("no_microphones", {})]
    assert not config_path().exists()


# --------------------------------------------------------------------------
# The settings list (plan.md D32): one row at a time, after the first setup
# --------------------------------------------------------------------------


def set_up(*, key: str | None = GOOD_KEY) -> None:
    """A machine the first setup has run on: Vesper, Turkish, `fast`."""
    save_settings(
        Settings(
            live=LiveSettings(primary="gemini:fast", voice="Orus", idle_close_seconds=30.0),
            wake=WakeSettings(enabled=True, model="hey_vesper", threshold=0.5),
            locale=LocaleSettings(code="tr"),
        )
    )
    if key is not None:
        store_api_key("gemini", key)


def settings_list(**answers: str | list[str | None] | None) -> ScriptedPrompter:
    """A prompter that opens the rows `settings` names, in turn, and closes
    the list after them."""
    rows = answers.pop("settings", [])
    assert isinstance(rows, list)
    return ScriptedPrompter(settings=[*rows, None], **answers)


async def settings_run(
    prompter: ScriptedPrompter,
    *providers: str,
    database: sqlite3.Connection | None = None,
) -> int:
    return await run_settings(
        prompter, catalog=fake_catalog(*providers), database=database, microphones=LAPTOP
    )


def row(key: str, value: str, language: str = "tr") -> str:
    """A row of the list as the pack words it."""
    return locales.load(language).say(f"setting_{key}", TEXT[f"setting_{key}"]).format(value=value)


async def test_the_list_shows_every_setting_as_it_stands(
    config_home: Path, vault: MemoryKeyring
) -> None:
    set_up()
    prompter = settings_list()

    exit_code = await settings_run(prompter)

    assert exit_code == 0
    assert prompter.offered["settings"] == [
        "assistant",
        "locale",
        "api_key",
        "model",
        "microphone",
    ]
    stored = locales.load("tr").say("key_stored", TEXT["key_stored"])
    assert prompter.labelled["settings"][:4] == [
        row("assistant", "Vesper"),
        row("locale", "Türkçe"),
        row("api_key", stored),
        row("model", "fast"),
    ]
    assert prompter.asked == ["settings"]


async def test_changing_the_assistant_asks_that_one_question_and_keeps_the_rest(
    config_home: Path, vault: MemoryKeyring, fake_adapter: list[FakeProvider]
) -> None:
    """The owner's case: Friday to Jarvis is one question - not the
    language, the model and the key again."""
    set_up()
    prompter = settings_list(settings=["assistant"], assistant="jarvis")

    await settings_run(prompter)

    settings = load_settings()
    jarvis = load_assistants()["jarvis"]
    assert prompter.asked == ["settings", "assistant", "settings"]
    assert settings.live.voice == jarvis.voice_for("gemini")
    assert (settings.wake.enabled, settings.wake.model) == (True, jarvis.wake)
    # Vesper's tuned threshold means nothing for Jarvis's model.
    assert settings.wake.threshold is None
    assert UserMemory.load().name == "Jarvis"
    # Everything else as it was, and no provider built to change a voice.
    assert settings.live.primary == "gemini:fast"
    assert settings.live.idle_close_seconds == 30.0
    assert settings.locale.code == "tr"
    assert vault.vault == {(KEYRING_SERVICE, "gemini"): GOOD_KEY}
    assert fake_adapter == []


async def test_a_saved_row_is_said_with_what_it_says_now(
    config_home: Path, vault: MemoryKeyring
) -> None:
    set_up()
    prompter = settings_list(settings=["assistant"], assistant="friday")

    await settings_run(prompter)

    assert ("setting_saved", {"setting": row("assistant", "Friday")}) in prompter.said


async def test_leaving_a_row_s_question_is_back_to_the_list_with_nothing_changed(
    config_home: Path, vault: MemoryKeyring
) -> None:
    set_up()
    before = config_path().read_text(encoding="utf-8")
    prompter = settings_list(settings=["assistant"], assistant=None)

    exit_code = await settings_run(prompter)

    assert exit_code == 0
    assert prompter.asked == ["settings", "assistant", "settings"]
    assert config_path().read_text(encoding="utf-8") == before


async def test_changing_the_language_writes_the_language_alone(
    config_home: Path, vault: MemoryKeyring
) -> None:
    set_up()
    prompter = settings_list(settings=["locale"], locale="en")

    await settings_run(prompter)

    settings = load_settings()
    assert settings.locale.code == "en"
    assert (settings.live.primary, settings.live.voice) == ("gemini:fast", "Orus")


async def test_the_list_is_worded_in_the_language_just_chosen(
    config_home: Path, vault: MemoryKeyring
) -> None:
    set_up()
    prompter = settings_list(settings=["locale"], locale="en")

    await settings_run(prompter)

    assert prompter.labelled["settings"][0] == TEXT["setting_assistant"].format(value="Vesper")


async def test_changing_the_model_uses_the_key_already_stored(
    config_home: Path, vault: MemoryKeyring, verdicts: sqlite3.Connection
) -> None:
    """No key question to change a model: the stored one is used."""
    set_up()
    prompter = settings_list(settings=["model"], model="smart")

    await settings_run(prompter, database=verdicts)

    assert load_settings().live.primary == "gemini:smart"
    assert "api_key" not in prompter.asked and "api_key_keep" not in prompter.asked
    verdict = remembered(SettingsRepo(verdicts), "gemini", "smart", ttl=60.0)
    assert verdict is not None and verdict.ok


async def test_changing_the_model_with_no_key_stored_asks_for_one_first(
    config_home: Path, vault: MemoryKeyring
) -> None:
    set_up(key=None)
    prompter = settings_list(settings=["model"], api_key=GOOD_KEY, model="smart")

    await settings_run(prompter)

    assert prompter.asked == ["settings", "api_key", "model", "settings"]
    assert vault.vault == {(KEYRING_SERVICE, "gemini"): GOOD_KEY}
    assert load_settings().live.primary == "gemini:smart"


async def test_a_stored_key_that_died_is_said_and_the_model_stays(
    config_home: Path, vault: MemoryKeyring
) -> None:
    set_up(key=DEAD_KEY)
    prompter = settings_list(settings=["model"])

    await settings_run(prompter)

    assert ("bad_key", {}) in prompter.said
    assert load_settings().live.primary == "gemini:fast"


async def test_a_provider_that_cannot_be_reached_is_a_sentence_and_the_list_again(
    config_home: Path, vault: MemoryKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_up()
    monkeypatch.setattr(FakeProvider, "down", True)
    prompter = settings_list(settings=["model"])

    exit_code = await settings_run(prompter)

    assert exit_code == 0
    assert ("provider_unreachable", {}) in prompter.said
    assert prompter.asked == ["settings", "settings"]
    assert load_settings().live.primary == "gemini:fast"


async def test_changing_the_key_checks_it_before_keeping_it(
    config_home: Path, vault: MemoryKeyring
) -> None:
    set_up(key="an-old-key")
    prompter = settings_list(settings=["api_key"], api_key_keep="mistyped", api_key=GOOD_KEY)

    await settings_run(prompter)

    assert ("bad_key", {}) in prompter.said
    assert vault.vault == {(KEYRING_SERVICE, "gemini"): GOOD_KEY}
    assert load_settings().live.primary == "gemini:fast"


async def test_changing_the_microphone_warns_about_a_raw_path(
    config_home: Path, vault: MemoryKeyring
) -> None:
    set_up()
    prompter = settings_list(settings=["microphone"], microphone=RAW_ARRAY.setting)

    await settings_run(prompter)

    assert load_settings().audio.input_device == RAW_ARRAY.setting
    assert ("microphone_raw", {}) in prompter.said


async def test_the_provider_is_a_row_only_when_there_is_a_choice(
    config_home: Path, vault: MemoryKeyring
) -> None:
    set_up()
    one, two = settings_list(), settings_list()

    await settings_run(one)
    await settings_run(two, "gemini", "openrouter")

    assert "provider" not in one.offered["settings"]
    assert "provider" in two.offered["settings"]


async def test_changing_the_provider_asks_its_key_and_its_model_together(
    config_home: Path, vault: MemoryKeyring
) -> None:
    set_up()
    prompter = settings_list(
        settings=["provider"], provider="openrouter", api_key=GOOD_KEY, model="smart"
    )

    await settings_run(prompter, "gemini", "openrouter")

    settings = load_settings()
    assert settings.live.primary == "openrouter:smart"
    # Vesper has no voice by OpenRouter's name: the model's own.
    assert settings.live.voice == ""
    assert vault.vault[(KEYRING_SERVICE, "openrouter")] == GOOD_KEY


# --------------------------------------------------------------------------
# Giving up
# --------------------------------------------------------------------------


async def test_walking_away_at_the_last_question_writes_nothing(
    config_home: Path, vault: MemoryKeyring
) -> None:
    """Everything is written at the end, so a wizard that was not finished
    leaves the machine exactly as it found it."""
    exit_code = await run_setup(complete_run(model=None), catalog=fake_catalog())

    assert exit_code != 0
    assert vault.vault == {}
    assert not config_path().exists()


async def test_a_settings_file_that_exists_survives_a_cancelled_run(
    config_home: Path, vault: MemoryKeyring
) -> None:
    save_settings(Settings(live=LiveSettings(primary="gemini:fast")))

    await run_setup(complete_run(model=None), catalog=fake_catalog())

    assert load_settings().live.primary == "gemini:fast"


async def test_a_key_that_reaches_no_model_stops_the_wizard(
    config_home: Path, vault: MemoryKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The key works but the account has no model; offering an empty list
    would put the user in a menu with nothing in it."""
    monkeypatch.setattr(FakeProvider, "models", [])

    exit_code = await run_setup(
        ScriptedPrompter(assistant="vesper", locale="tr", api_key=GOOD_KEY), catalog=fake_catalog()
    )

    assert exit_code != 0
    assert not config_path().exists()


# --------------------------------------------------------------------------
# The real terminal
# --------------------------------------------------------------------------


class FakeQuestion:
    """What `questionary` hands back: something with an `ask_async()`."""

    def __init__(self, answer: object) -> None:
        self._answer = answer

    async def ask_async(self) -> object:
        return self._answer


def test_the_terminal_says_the_sentence_with_its_fields_filled_in(
    config_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    TerminalPrompter().say("key_url", url="https://example.test/apikey")

    assert "https://example.test/apikey" in capsys.readouterr().out


async def test_the_terminal_stores_the_value_and_shows_the_label(
    config_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swap these two and the wizard writes "Fast" into config.toml."""
    seen: dict[str, object] = {}

    def select(message: str, choices: list[questionary.Choice], **kwargs: object) -> FakeQuestion:
        seen["message"] = message
        seen["titles"] = [choice.title for choice in choices]
        seen["values"] = [choice.value for choice in choices]
        return FakeQuestion(choices[0].value)

    monkeypatch.setattr(questionary, "select", select)

    answer = await TerminalPrompter().choose(
        "model", [Option("fast", "Fast"), Option("smart", "Smart")]
    )

    assert answer == "fast"
    assert seen["values"] == ["fast", "smart"]
    assert seen["titles"] == ["Fast", "Smart"]
    assert seen["message"] == wording()["model"]


async def test_the_terminal_reads_a_key_without_echoing_it(
    config_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`password` is what hides the typing; `text` would put the key on screen."""
    monkeypatch.setattr(questionary, "password", lambda message: FakeQuestion("typed-key"))
    monkeypatch.setattr(
        questionary, "text", lambda *a, **k: pytest.fail("the key must not be echoed")
    )

    assert await TerminalPrompter().secret("api_key") == "typed-key"


async def test_an_interrupted_question_is_not_an_answer(
    config_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl+C makes `questionary` return None, which the wizard reads as walking away."""
    monkeypatch.setattr(questionary, "password", lambda message: FakeQuestion(None))

    assert await TerminalPrompter().secret("api_key") is None


# --------------------------------------------------------------------------
# The terminal the wizard really runs in
#
# Everything above hands the wizard a scripted prompter, which is the right
# way to test what it asks and what it does with the answers - and is exactly
# why the terminal itself went untested until somebody ran `allie setup`.
# These two drive the real one, through a pipe instead of a keyboard.
# --------------------------------------------------------------------------


@contextmanager
def typed(keys: str) -> Iterator[None]:
    """A terminal that answers with `keys` and draws nowhere."""
    with create_pipe_input() as keyboard:
        keyboard.send_text(keys)
        with create_app_session(input=keyboard, output=DummyOutput()):
            yield


async def test_a_question_can_be_asked_from_inside_the_event_loop() -> None:
    """`run_setup` is a coroutine, so every prompt is drawn while an event loop
    is already running. `questionary`'s synchronous `ask` starts a second one
    and Python refuses outright - which no scripted prompter can find out."""
    prompter = TerminalPrompter(text={"locale": "Which language?"})

    with typed("\r"):
        chosen = await prompter.choose("locale", [Option("tr", "Türkçe"), Option("en", "English")])

    assert chosen == "tr"


async def test_a_key_can_be_typed_from_inside_the_event_loop() -> None:
    """The same for the one question whose answer is a secret."""
    prompter = TerminalPrompter(text={"api_key": "Paste your API key"})

    with typed("a-key\r"):
        assert await prompter.secret("api_key") == "a-key"


async def test_the_terminal_s_settings_list_ends_in_close() -> None:
    """A terminal has no Close button: the list's last choice is one, and
    choosing it is done with the list."""
    prompter = TerminalPrompter(text={"settings": "Settings", "settings_close": "Close"})
    rows = [Option("assistant", "Assistant: Vesper"), Option("model", "Model: fast")]

    with typed("\r"):
        first = await prompter.menu("settings", rows)
    # Down twice from the top: past both rows, onto Close.
    with typed("\x1b[B\x1b[B\r"):
        closed = await prompter.menu("settings", rows)

    assert first == "assistant"
    assert closed is None


# --------------------------------------------------------------------------
# The wizard on the window (plan.md D20): the same questions, a page each
# --------------------------------------------------------------------------


class PageAnswers:
    """A Tk half without Tk: answers each question from a script by key,
    the way a person would from the page, and records the pages."""

    def __init__(self, view: Any, window: Any, answers: dict[str, str | None]) -> None:
        self.view = view
        self.window = window
        self.answers = answers
        self.pages: list[tuple[str, str]] = []
        self.quit = False

    def run(self) -> None:
        import time

        while not self.quit:
            for message in self.window.drain():
                if message == ("quit",):
                    self.quit = True
                    return
                self.view.apply(message)
            page = self.view.wizard
            if page is not None and page.open:
                self.pages.append((page.kind, page.question))
                key = next(key for key, text in TEXT.items() if text == page.question)
                self.window.answer(self.view.take_answer(), self.answers[key])
            time.sleep(0.005)


async def test_the_wizard_runs_through_the_window_and_leaves_the_same_settings(
    config_home: Path, vault: MemoryKeyring
) -> None:
    from allie.ui.window import Window, WindowPrompter

    answers: dict[str, str | None] = {
        "assistant": "vesper",
        "locale": "tr",
        "api_key": GOOD_KEY,
        "model": "smart",
        "microphone": "",
    }
    pages: list[PageAnswers] = []

    def panel(view: Any, window: Any) -> PageAnswers:
        pages.append(PageAnswers(view, window, answers))
        return pages[-1]

    window = Window(
        locales.load("en"),
        loop=asyncio.get_running_loop(),
        on_toggle=lambda: None,
        on_quit=lambda: None,
        on_settings=lambda: None,
        tray=False,
        panel=panel,
    )
    window.start()
    try:
        window.wizard(True)
        exit_code = await run_setup(WindowPrompter(window, text=TEXT), catalog=fake_catalog())
    finally:
        window.stop()

    [page] = pages
    assert exit_code == 0
    assert load_settings().live.primary == "gemini:smart"
    assert vault.vault == {(KEYRING_SERVICE, "gemini"): GOOD_KEY}
    # Assistant, language, key, model, microphone: "who hears" left with D36.
    assert [kind for kind, _ in page.pages] == [
        "choose",
        "choose",
        "secret",
        "choose",
        "choose",
    ]
    assert page.view.wizard is not None
    assert TEXT["welcome"] in page.view.wizard.lines


async def test_walking_away_from_the_window_s_page_is_the_wizard_s_cancel(
    config_home: Path, vault: MemoryKeyring
) -> None:
    from allie.ui.window import Window, WindowPrompter

    def panel(view: Any, window: Any) -> PageAnswers:
        return PageAnswers(view, window, {"assistant": None})

    window = Window(
        locales.load("en"),
        loop=asyncio.get_running_loop(),
        on_toggle=lambda: None,
        on_quit=lambda: None,
        on_settings=lambda: None,
        tray=False,
        panel=panel,
    )
    window.start()
    try:
        window.wizard(True)
        exit_code = await run_setup(WindowPrompter(window, text=TEXT), catalog=fake_catalog())
    finally:
        window.stop()

    assert exit_code == 1
    assert not config_path().is_file()
    assert vault.vault == {}
