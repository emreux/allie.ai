"""The questions of `live-assistant setup`, without a terminal.

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
and `live-assistant mic` asks that one question on its own. The device list is
handed in, so no test touches PortAudio.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

import pytest
import questionary
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from assistant import locales
from assistant.audio.capture import MicrophoneInfo, Microphones
from assistant.config import (
    KEYRING_SERVICE,
    AudioSettings,
    LiveSettings,
    LocaleSettings,
    Settings,
    config_path,
    load_settings,
    save_settings,
    store_api_key,
)
from assistant.live.base import Delta, Message, ModelInfo, ProviderError, ToolCall, ToolSpec
from assistant.live.probe import NO_TOOL_CALL, QUESTION, ProbeResult, remembered
from assistant.live.registry import ADAPTERS, ProviderEntry
from assistant.setup_wizard import (
    TEXT,
    Option,
    TerminalPrompter,
    run_microphone_setup,
    run_setup,
    wording,
)
from assistant.store import db
from assistant.store.db import open_database
from assistant.store.repos import SettingsRepo
from tests.conftest import MemoryKeyring

GOOD_KEY = "good-key"
# A key checked while the network is down: the provider cannot say whether it
# works, and the adapter reports that as a refusal of the request, not the key.
OFFLINE_KEY = "offline"


class FakeProvider:
    """A provider that accepts one key, offers two models, and answers the
    probe of 2.6 for each of them as the class attributes say: a model in
    `tool_callers` calls the clock, one in `refusing` makes the provider
    refuse the request, any other only talks."""

    id = "fake"
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
        return list(self.models)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[Delta]:
        self.probed.append((model, messages[-1].content, [tool.name for tool in tools]))
        if model in self.refusing:
            raise ProviderError("fake refused the request (429): slow down")
        if model in self.tool_callers:
            yield Delta(
                tool_call=ToolCall(id="c1", name="get_current_time", arguments={"city": "x"})
            )
        else:
            yield Delta(text="It is about three.")
        yield Delta(finish_reason="stop")


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
    monkeypatch.setattr("assistant.setup_wizard.available_microphones", lambda: LAPTOP)


def complete_run(**overrides: str | list[str | None] | None) -> ScriptedPrompter:
    """A prompter scripted to walk the wizard from end to end."""
    answers: dict[str, str | list[str | None] | None] = {
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
    prompter = ScriptedPrompter(locale="tr", model="fast", microphone="")

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
    prompter = ScriptedPrompter(locale="tr")

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
    assert prompter.asked == ["locale", "api_key_keep", "api_key", "model", "microphone"]
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
    """So that `live-assistant run` need not ask the same question for a week."""
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
    """What the probe writes down is what `live-assistant run` will read a week
    later; the word is the one the design names."""
    assert ProbeResult(ok=False, reason=NO_TOOL_CALL).reason == "no_tool_call_emitted"


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
        "Microphone Array (Intel Smart , MME",
        "Microphone Array 1 (), Windows WDM-KS",
        "Headset (Buds3 Hands-Free AG Audio), Windows WASAPI",
    ]
    assert prompter.labelled["microphone"][2] == "Microphone Array 1 () - Windows WDM-KS"


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
    """`live-assistant mic` is the one question again, for the day the headset
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

    assert prompter.offered["microphone"] == ["", ARRAY.setting, RAW_ARRAY.setting, HEADSET.setting]


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
        ScriptedPrompter(locale="tr", api_key=GOOD_KEY), catalog=fake_catalog()
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
# why the terminal itself went untested until somebody ran `live-assistant setup`.
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
