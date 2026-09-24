"""`allie setup` - the few questions phase 1 asks (design.md section 3.3).

Which assistant, provider, key, model, the language the assistant speaks,
who hears the yes or no of a confirmation, and the microphone it listens
through. The fourteen step wizard of section 3.3 - device tests, tool probe,
latency measurement, a fallback model - is phase 4.5; what is here is the
smallest thing that can produce a working `config.toml`.

**In a row once, one at a time after that** (plan.md D32, 2026-09-24).
`run_setup` walks the questions on a machine that has never been set up.
Every time after, the settings button and `allie setup` open
`run_settings`: a list of the settings as they stand, each row changed on
its own and saved at once - changing Friday to Jarvis is one question, not
the language, the model and the key again. A row writes only what it owns.

Two rules shape the code more than the questions do.

**Nothing is written until the wizard finishes.** The key is checked against
the provider before it is kept, and both the key and the settings are stored in
the last step. A run that was abandoned, or a key that turned out to be
revoked, leaves the machine exactly as it was found.

**No user-facing sentence lives in this module.** The wizard asks for a
question by key; the `Prompter` turns that key into words, taking them from the
locale pack and falling back to `TEXT` below - English, which section 3.12
names as the end of the chain. That is also why the tests can script a wizard
run without repeating a single sentence, and why adding a language changes no
line of this file.

**A model is not taken at its word (2.6).** Every model can be asked a
question; not every one can be asked to do something, and the one that
cannot fails silently, in prose, at two in the morning. So the model the
user picks is sent one request with one tool before it is accepted
(`live/probe.py`, section 3.2), and a model that does not call the tool is
not taken - the list is offered again. The verdict on the model that was
taken goes to the `settings` table, which makes this the first thing to
touch the database on a fresh machine.

**Two providers are questions about a catalogue field (2.7).** One that
needs no key - a local Ollama - is not asked for one: the server is asked
whether it answers, and that is all. One that has no address of its own -
`custom`, any OpenAI-compatible server - is asked where it is, and the
address goes to `config.toml` beside the model, where a text editor can
reach it. Neither question names an adapter here: the registry says which
entries need an address, the catalogue says which need a key.

**The microphone is a list, not a name to type.** PortAudio's table, every
input once per host API, with "whatever Windows has chosen" on top and stored
as the empty string that has meant the default since phase 1. `allie mic`
asks this one question again on its own - the day a headset comes out - and
rewrites `[audio]` alone, so that nobody edits `config.toml` by hand for it.
The Windows audio engine's entries lead the list and a raw kernel-streaming
choice is warned about (plan.md D18): that path has no echo cancellation,
and the assistant hears itself through the speakers.

**The assistant is asked first** (plan.md D31): Jarvis, Vesper, Allie or
Friday, each a name, a voice and a wake phrase of its own
(`wake/assistants.toml`). The answer is three settings - `[wake] model`,
`[live] voice` by the provider's own name, and the name in `memory.toml`,
the one file besides `config.toml` setup writes, so that the model knows
what it is called. It replaced the free-text voice question: a voice now
comes with a name. A threshold tuned by hand stays only when the same
assistant is chosen again - it was measured for that model.

**The local recogniser serves the gate window** (plan.md D3, D10). The
model speaks for itself; Whisper or Google's recogniser hears the yes or no
after a tool asks first - Google's offered when its key is at hand, the one
just checked or one stored earlier, since it uses the same entry. Who reads
the question is no longer a question: the assistant does, in its own voice
(D32). What setup does not ask about in `[live]` and `[stt]` it keeps from
the file: the session numbers are the owner's to tune.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import questionary
from loguru import logger
from rich.console import Console

from allie import locales
from allie.assistants import Assistant, load_assistants
from allie.audio.capture import (
    FULL_DUPLEX_HOST_APIS,
    MicrophoneInfo,
    Microphones,
    available_microphones,
    duplex_for,
)
from allie.config import (
    AudioSettings,
    LocaleSettings,
    Settings,
    config_path,
    load_api_key,
    load_settings,
    save_settings,
    store_api_key,
)
from allie.live import probe
from allie.live.base import AuthenticationError, LiveProvider, ModelInfo, ProviderError
from allie.live.registry import (
    ADAPTERS,
    ProviderEntry,
    create_provider,
    load_catalog,
    needs_base_url,
)
from allie.store.db import open_database
from allie.store.memory import MemoryFileError, UserMemory
from allie.store.repos import SettingsRepo

__all__ = [
    "TEXT",
    "Option",
    "Prompter",
    "TerminalPrompter",
    "run_microphone_setup",
    "run_settings",
    "run_setup",
    "wording",
]

_OK = 0
_GAVE_UP = 1


@dataclass(frozen=True, slots=True)
class Option:
    """One answer to a question: what gets stored, and what the user reads.

    The label is data - a provider's display name, a model's name, a language's
    name in that language - so it is not part of the text table.
    """

    value: str
    label: str


class Prompter(Protocol):
    """Everything the wizard needs from a terminal, and nothing more.

    A question is named, not worded. Keeping the wording on this side of the
    line is what lets the wizard be tested without a screen and translated
    without being rewritten.
    """

    def say(self, key: str, **fields: object) -> None: ...

    async def choose(self, key: str, options: Sequence[Option]) -> str | None:
        """Returns the chosen value, or `None` if the user walked away."""
        ...

    async def secret(self, key: str) -> str | None:
        """Reads a line without echoing it, or `None` if the user walked away."""
        ...

    async def ask(self, key: str) -> str | None:
        """Reads a line the user can see - an address, not a secret - or
        `None` if the user walked away."""
        ...

    async def menu(self, key: str, options: Sequence[Option]) -> str | None:
        """The settings list (D32): the row to change, or `None` when the
        user is done with the list."""
        ...


# The last link of the fallback chain of section 3.12: what the wizard says
# when no locale pack offers a translation. English lives here, beside the code
# that says it, rather than in `locales/en.toml` - one copy cannot drift from
# the other.
TEXT: dict[str, str] = {
    "welcome": "The assistant answers through an AI provider, using your own API key.",
    "assistant": (
        "Which assistant do you want? Each has a name, a voice and a wake phrase of its own."
    ),
    "assistant_male": '{name} - a man\'s voice, wakes to "hey {name}"',
    "assistant_female": '{name} - a woman\'s voice, wakes to "hey {name}"',
    "provider": "Which provider do you want to use?",
    "only_provider": "Provider: {name} - the only one this version can talk to.",
    "no_provider": "This version cannot build any provider in the catalogue.",
    "locale": "Which language should the assistant speak?",
    "key_url": "You can create a key at {url}",
    "api_key": "Paste your API key (nothing is shown as you type)",
    "api_key_keep": "Paste your API key, or press Enter to keep the one already stored",
    "key_needed": "This provider needs a key before it will answer.",
    "checking_key": "Checking the key...",
    "base_url": "The server's address - the OpenAI-compatible base URL, ending in /v1",
    "checking_server": "Checking that the server answers...",
    "bad_key": "That key did not work - mistyped, revoked, or out of credit.",
    "provider_unreachable": (
        "The provider could not be reached. Check the connection and try again."
    ),
    "loading_models": "Asking which models the key can reach...",
    "no_models": "The key works, but it reaches no model. Check the provider's console.",
    "model": "Which model should answer?",
    "probing_tools": "Checking whether the model calls tools...",
    "tools_ok": "The model calls tools - first token in {ms} ms.",
    "tools_failed": (
        "This model does not call tools, and most of what the assistant does depends on that. "
        "Choose another model."
    ),
    "probe_refused": (
        "The model could not be tested: {problem}. Choose another model, or try again."
    ),
    "hears": "Who hears your yes or no when a tool asks first?",
    "hears_local": "Whisper, on this machine - nothing leaves it",
    "hears_gemini": "Google's recogniser, with the same key - those two words go to Google",
    "microphone": "Which microphone should the assistant listen through?",
    "windows_microphone": "Whatever Windows has chosen (right now: {name})",
    "no_microphones": "No microphone was found.",
    "microphone_raw": (
        "That path has no echo cancellation: on speakers the assistant will hear itself "
        "and cut itself off. The same microphone's Windows WASAPI or MME entry is safer."
    ),
    "microphone_saved": "Microphone: {microphone}. Settings: {path}",
    "saved": "Ready. Settings: {path} - the key itself is in the Windows Credential Manager.",
    "cancelled": "Setup cancelled. Nothing was changed.",
    # The settings list (D32): one row per setting, "title: {value}".
    "settings": "Settings - choose one to change it",
    "settings_close": "Close",
    "setting_assistant": "Assistant: {value}",
    "setting_provider": "Provider: {value}",
    "setting_locale": "Language: {value}",
    "setting_api_key": "API key: {value}",
    "key_stored": "stored",
    "key_missing": "not set",
    "setting_model": "Model: {value}",
    "setting_hears": "Who hears your yes or no: {value}",
    "hears_short_local": "Whisper, on this machine",
    "hears_short_gemini": "Google's recogniser",
    "setting_microphone": "Microphone: {value}",
    "setting_saved": "Saved: {setting}",
}


def wording() -> dict[str, str]:
    """What the wizard says, in the language the user is likeliest to read.

    The language question decides what the *assistant* speaks from then on,
    which is no help in wording the question: the wizard has to ask before it
    has an answer. So it uses the answer from last time when setup has run
    before - changing the model should not mean reading English again - and
    Windows' own language on the very first run. When neither names a language
    anybody has translated, this is `TEXT` unchanged.
    """
    pack = locales.load(_chosen_before() or locales.system_code())
    return {key: pack.say(key, default) for key, default in TEXT.items()}


def _chosen_before() -> str | None:
    """The language chosen the last time setup ran, if it ever ran."""
    return load_settings().locale.code if config_path().is_file() else None


def _languages() -> list[Option]:
    """Every locale pack there is, each named in its own language."""
    return [Option(pack.code, pack.name) for pack in locales.available()]


class _WalkedAwayError(Exception):
    """The user pressed Ctrl+C, or answered nothing to a question with no default."""


async def run_setup(
    prompter: Prompter,
    *,
    catalog: Mapping[str, ProviderEntry] | None = None,
    database: sqlite3.Connection | None = None,
    microphones: Microphones | None = None,
    assistants: Mapping[str, Assistant] | None = None,
) -> int:
    """Asks the questions, then writes the answers. Returns a process exit code.

    `database` is where the verdict on the model goes; left out, the
    machine's own is opened for it at the end, and closed again.
    `microphones` is what the last question offers; left out, PortAudio is
    asked. `assistants` is what the first one offers; left out, the
    packaged catalogue.
    """
    try:
        return await _ask(
            prompter,
            catalog,
            database,
            microphones,
            load_assistants() if assistants is None else assistants,
        )
    except _WalkedAwayError:
        prompter.say("cancelled")
        return _GAVE_UP


async def run_microphone_setup(
    prompter: Prompter, *, microphones: Microphones | None = None
) -> int:
    """`allie mic`: the microphone question on its own.

    Rewrites `[audio]` and leaves every other table as the file has it, so
    that switching to a headset is one command rather than a text editor.
    """
    found = _found(microphones)
    if not found.devices:
        prompter.say("no_microphones")
        return _GAVE_UP
    options = _microphone_options(found)
    try:
        chosen = _answered(await prompter.choose("microphone", options))
    except _WalkedAwayError:
        prompter.say("cancelled")
        return _GAVE_UP
    _warn_if_raw(prompter, chosen, found)

    path = save_settings(Settings(audio=AudioSettings(input_device=chosen)))
    label = next(option.label for option in options if option.value == chosen)
    prompter.say("microphone_saved", microphone=label, path=path)
    return _OK


async def _ask(
    prompter: Prompter,
    catalog: Mapping[str, ProviderEntry] | None,
    database: sqlite3.Connection | None,
    microphones: Microphones | None,
    assistants: Mapping[str, Assistant],
) -> int:
    entries = load_catalog() if catalog is None else catalog
    buildable = {
        provider_id: entry for provider_id, entry in entries.items() if entry.adapter in ADAPTERS
    }
    if not buildable:
        # A catalogue listing only adapters from later phases. Worth its own
        # sentence: the user has done nothing wrong and retrying will not help.
        prompter.say("no_provider")
        return _GAVE_UP

    prompter.say("welcome")
    assistant = await _pick_assistant(prompter, assistants)
    provider_id, entry = await _pick_provider(prompter, buildable)
    locale = _answered(await prompter.choose("locale", _languages()))
    base_url = await _address(prompter, entry)

    connected = await _connection(prompter, provider_id, entry, entries, base_url, locale)
    if connected is None:
        return _GAVE_UP
    api_key, model, verdict = connected
    hears = await _pick_engine(prompter, "hears", ("local", "gemini"), google=_google(provider_id))
    input_device = await _pick_microphone(prompter, _found(microphones))

    # Everything above could still be abandoned; from here it is written down.
    # The tables setup asks part of keep the rest from the file: the session
    # numbers of `[live]`, the engines' models and the greeting are the
    # owner's to tune.
    if api_key:
        store_api_key(provider_id, api_key)
    kept = load_settings()
    path = save_settings(
        Settings(
            live=kept.live.model_copy(
                update={
                    "primary": f"{provider_id}:{model}",
                    "base_url": base_url or "",
                    "voice": assistant.voice_for(provider_id),
                }
            ),
            wake=kept.wake.model_copy(
                update={
                    "enabled": True,
                    "model": assistant.wake,
                    # Measured for one model, meaningless for another.
                    "threshold": (
                        kept.wake.threshold if kept.wake.model == assistant.wake else None
                    ),
                }
            ),
            locale=LocaleSettings(code=locale),
            audio=AudioSettings(input_device=input_device),
            stt=kept.stt.model_copy(update={"provider": hears}),
        )
    )
    _name(assistant.name)
    _remember(database, provider_id, model, verdict)
    prompter.say("saved", path=path)
    return _OK


async def _connection(
    prompter: Prompter,
    provider_id: str,
    entry: ProviderEntry,
    entries: Mapping[str, ProviderEntry],
    base_url: str | None,
    locale: str,
) -> tuple[str, str, probe.ProbeResult] | None:
    """A provider that answers and a model of it that calls tools: the key
    (checked) or the server's answer, then the model list and the probe.
    The key, the model and the verdict - or `None` when there is nothing
    to go on with: a keyless server that does not answer, a key that reaches
    no model."""
    if entry.requires_key:
        if entry.key_url is not None:
            prompter.say("key_url", url=entry.key_url)
        provider, api_key = await _working_key(prompter, provider_id, entries, base_url=base_url)
    else:
        api_key = ""
        answering = await _answering_server(prompter, provider_id, entries, base_url=base_url)
        if answering is None:
            return None
        provider = answering

    prompter.say("loading_models")
    models = await provider.list_models()
    if not models:
        prompter.say("no_models")
        return None
    model, verdict = await _model_that_calls_tools(
        prompter, provider, models, question=_probe_question(locale)
    )
    return api_key, model, verdict


def _google(provider_id: str) -> bool:
    """Whether Google's recogniser can be offered: its key is the one just
    checked, or one stored earlier under the same entry."""
    return provider_id == "gemini" or load_api_key("gemini") is not None


# --------------------------------------------------------------------------
# The settings list (plan.md D32)
# --------------------------------------------------------------------------


async def run_settings(
    prompter: Prompter,
    *,
    catalog: Mapping[str, ProviderEntry] | None = None,
    database: sqlite3.Connection | None = None,
    microphones: Microphones | None = None,
    assistants: Mapping[str, Assistant] | None = None,
) -> int:
    """The settings one at a time: the list of them as they stand, one row
    chosen, that one question asked, the answer saved at once, and the list
    again - until the user closes it. Leaving a row's question halfway is
    back to the list with nothing of that row changed. Returns a process
    exit code; closing the list is success.
    """
    entries = load_catalog() if catalog is None else catalog
    buildable = {
        provider_id: entry for provider_id, entry in entries.items() if entry.adapter in ADAPTERS
    }
    offered = load_assistants() if assistants is None else assistants
    # Asked once: PortAudio's table does not change while the list is open.
    found = _found(microphones)
    setting = _Setting(prompter, entries, buildable, database, found, offered)

    while True:
        rows = setting.rows()
        chosen = await prompter.menu("settings", rows)
        if chosen is None:
            return _OK
        try:
            changed = await setting.change(chosen)
        except _WalkedAwayError:
            continue
        except ProviderError:
            # The model list, asked of a provider that went away meanwhile:
            # a sentence and the list again, never the end of the program.
            prompter.say("provider_unreachable")
            continue
        after = next((row.label for row in setting.rows() if row.value == chosen), None)
        if changed and after is not None:
            prompter.say("setting_saved", setting=after)


class _Setting:
    """What each row of the settings list says, and what changing it asks
    and writes. Every read of the settings is fresh: the row changed a
    moment ago is the one shown now."""

    def __init__(
        self,
        prompter: Prompter,
        entries: Mapping[str, ProviderEntry],
        buildable: Mapping[str, ProviderEntry],
        database: sqlite3.Connection | None,
        found: Microphones,
        assistants: Mapping[str, Assistant],
    ) -> None:
        self._prompter = prompter
        self._entries = entries
        self._buildable = buildable
        self._database = database
        self._found = found
        self._assistants = assistants

    def rows(self) -> list[Option]:
        """One row per setting, "title: value", in the words of the language
        chosen - read again every time, so that a new language shows."""
        said = wording()
        settings = load_settings()
        provider_id = settings.live.provider
        entry = self._entries.get(provider_id)

        def row(key: str, value: str) -> Option:
            return Option(key, said[f"setting_{key}"].format(value=value))

        assistant = self._current(settings)
        rows = [row("assistant", assistant.name if assistant else settings.wake.model)]
        if len(self._buildable) > 1:
            rows.append(row("provider", entry.display_name if entry else provider_id))
        rows.append(row("locale", locales.load(settings.locale.code).name))
        if entry is None or entry.requires_key:
            stored = load_api_key(provider_id) is not None
            rows.append(row("api_key", said["key_stored" if stored else "key_missing"]))
        rows.append(row("model", settings.live.model))
        rows.append(row("hears", said[f"hears_short_{settings.stt.provider}"]))
        rows.append(row("microphone", self._microphone(settings.audio.input_device)))
        return rows

    async def change(self, key: str) -> bool:
        """Asks the one question of `key` and writes the answer; whether
        anything was written."""
        prompter = self._prompter
        settings = load_settings()
        provider_id = settings.live.provider

        if key == "assistant":
            assistant = await _pick_assistant(prompter, self._assistants)
            _save(
                live={"voice": assistant.voice_for(provider_id)},
                wake={
                    "enabled": True,
                    "model": assistant.wake,
                    "threshold": (
                        settings.wake.threshold if settings.wake.model == assistant.wake else None
                    ),
                },
            )
            _name(assistant.name)
            return True

        if key == "locale":
            code = _answered(await prompter.choose("locale", _languages()))
            _save(locale={"code": code})
            return True

        if key == "api_key":
            entry = self._entries[provider_id]
            if entry.key_url is not None:
                prompter.say("key_url", url=entry.key_url)
            _, api_key = await _working_key(
                prompter, provider_id, self._entries, base_url=settings.live.base_url or None
            )
            store_api_key(provider_id, api_key)
            return True

        if key == "model":
            return await self._model(settings)

        if key == "hears":
            options = [Option("local", wording()["hears_local"])]
            if _google(provider_id):
                options.append(Option("gemini", wording()["hears_gemini"]))
            hears = _answered(await prompter.choose("hears", options))
            _save(stt={"provider": hears})
            return True

        if key == "microphone":
            _save(audio={"input_device": await _pick_microphone(prompter, self._found)})
            return True

        if key == "provider":
            return await self._provider(settings)

        raise ValueError(f"no setting is called {key!r}")

    async def _model(self, settings: Settings) -> bool:
        """Another model of the same provider, on the key already stored -
        asked for a key only when there is none. The list, the probe, and
        the verdict written beside the choice as setup writes it."""
        prompter = self._prompter
        provider_id = settings.live.provider
        entry = self._entries[provider_id]
        base_url = settings.live.base_url or None
        provider: LiveProvider | None
        if not entry.requires_key:
            provider = await _answering_server(
                prompter, provider_id, self._entries, base_url=base_url
            )
            if provider is None:
                return False
        elif (stored := load_api_key(provider_id)) is not None:
            provider = create_provider(
                provider_id, api_key=stored, catalog=self._entries, base_url=base_url
            )
        else:
            prompter.say("key_needed")
            provider, api_key = await _working_key(
                prompter, provider_id, self._entries, base_url=base_url
            )
            store_api_key(provider_id, api_key)

        prompter.say("loading_models")
        try:
            models = await provider.list_models()
        except AuthenticationError:
            # The stored key has died since; the key row is where it is fixed.
            prompter.say("bad_key")
            return False
        if not models:
            prompter.say("no_models")
            return False
        model, verdict = await _model_that_calls_tools(
            prompter, provider, models, question=_probe_question(settings.locale.code)
        )
        _save(live={"primary": f"{provider_id}:{model}"})
        _remember(self._database, provider_id, model, verdict)
        return True

    async def _provider(self, settings: Settings) -> bool:
        """Another provider: its address, its key and a model of it belong
        together, so they are asked together - and the assistant's voice by
        the new provider's own name."""
        prompter = self._prompter
        provider_id, entry = await _pick_provider(prompter, self._buildable)
        base_url = await _address(prompter, entry)
        connected = await _connection(
            prompter, provider_id, entry, self._entries, base_url, settings.locale.code
        )
        if connected is None:
            return False
        api_key, model, verdict = connected
        if api_key:
            store_api_key(provider_id, api_key)
        assistant = self._current(settings)
        _save(
            live={
                "primary": f"{provider_id}:{model}",
                "base_url": base_url or "",
                "voice": assistant.voice_for(provider_id) if assistant else "",
            }
        )
        _remember(self._database, provider_id, model, verdict)
        return True

    def _current(self, settings: Settings) -> Assistant | None:
        """The assistant whose wake phrase the settings name."""
        return next(
            (one for one in self._assistants.values() if one.wake == settings.wake.model), None
        )

    def _microphone(self, setting: str) -> str:
        """The row's words for `[audio] input_device`: the list's own label."""
        for option in _microphone_options(self._found):
            if option.value == setting:
                return option.label
        return setting


def _save(**changes: Mapping[str, object]) -> Path:
    """Writes the tables named, each the file's own with only `changes`
    replaced - so that a row never takes the owner's tuning with it."""
    kept = load_settings()
    tables = {
        table: getattr(kept, table).model_copy(update=dict(values))
        for table, values in changes.items()
    }
    return save_settings(Settings(**tables))


async def _pick_assistant(prompter: Prompter, assistants: Mapping[str, Assistant]) -> Assistant:
    """Which of them (plan.md D31): each offered by its name and its voice,
    in the words of the wizard's own language, like the engines' labels."""
    said = wording()
    options = [
        Option(assistant.id, said[f"assistant_{assistant.gender}"].format(name=assistant.name))
        for assistant in assistants.values()
    ]
    return assistants[_answered(await prompter.choose("assistant", options))]


def _name(name: str) -> None:
    """Tells the model what it is called: `memory.toml`'s `[assistant]
    name`, which it reads at every session (D23). A file edited into
    something that does not parse is never written over - `run` says so in
    a sentence - and the name waits for the next setup."""
    try:
        memory = UserMemory.load()
    except MemoryFileError as problem:
        logger.warning("setup: the assistant's name was not written: {problem}", problem=problem)
        return
    if memory.name != name:
        memory.rename(name)


async def _pick_provider(
    prompter: Prompter, buildable: Mapping[str, ProviderEntry]
) -> tuple[str, ProviderEntry]:
    """Asks which provider - unless there is only one, which is no question."""
    if len(buildable) == 1:
        provider_id, entry = next(iter(buildable.items()))
        prompter.say("only_provider", name=entry.display_name)
        return provider_id, entry

    options = [Option(provider_id, e.display_name) for provider_id, e in buildable.items()]
    chosen = _answered(await prompter.choose("provider", options))
    return chosen, buildable[chosen]


async def _address(prompter: Prompter, entry: ProviderEntry) -> str | None:
    """Where the server is, for an entry the catalogue gives no address for
    (`custom`); `None` for every other, whose address is in the file. An
    empty answer is asked again - there is no default to fall back on."""
    if not needs_base_url(entry):
        return None

    while True:
        typed = _answered(await prompter.ask("base_url")).strip()
        if typed:
            return typed


async def _answering_server(
    prompter: Prompter,
    provider_id: str,
    entries: Mapping[str, ProviderEntry],
    *,
    base_url: str | None,
) -> LiveProvider | None:
    """A provider that needs no key, checked once: does the server answer?

    There is no key to ask for again, so a server that cannot be reached
    ends the wizard with the sentence that says so - the user starts the
    server and comes back. A server that turns out to want a key after all
    is told about in those words; the catalogue said it would not.
    """
    prompter.say("checking_server")
    provider = create_provider(provider_id, api_key="", catalog=entries, base_url=base_url)
    try:
        accepted = await provider.validate_credentials()
    except ProviderError:
        prompter.say("provider_unreachable")
        return None
    if not accepted:
        prompter.say("key_needed")
        return None
    return provider


async def _working_key(
    prompter: Prompter,
    provider_id: str,
    entries: Mapping[str, ProviderEntry],
    *,
    base_url: str | None = None,
) -> tuple[LiveProvider, str]:
    """Asks for a key until the provider accepts one (section 3.3).

    A key that fails is not stored, not retried and not silently swapped for a
    fallback: mistyped, revoked and out of credit all mean the same thing to
    the user, and all three are fixed by pasting a different key.

    A provider that could not be reached is a different sentence. The key has
    not been proved dead - nothing has been proved - so a stored one is still
    offered on the next round, and the user is told to look at the connection
    rather than at the provider's console.
    """
    stored = load_api_key(provider_id)

    while True:
        typed = _answered(await prompter.secret("api_key_keep" if stored else "api_key"))
        api_key = typed or stored or ""
        if not api_key:
            prompter.say("key_needed")
            continue

        prompter.say("checking_key")
        provider = create_provider(provider_id, api_key=api_key, catalog=entries, base_url=base_url)
        try:
            accepted = await provider.validate_credentials()
        except ProviderError:
            prompter.say("provider_unreachable")
            continue
        if accepted:
            return provider, api_key

        prompter.say("bad_key")
        # Whatever was in the vault has just been proved useless, so it stops
        # being offered - otherwise Enter would retry the same dead key.
        stored = None


async def _model_that_calls_tools(
    prompter: Prompter, provider: LiveProvider, models: Sequence[ModelInfo], *, question: str
) -> tuple[str, probe.ProbeResult]:
    """Offers the models until one is chosen that passes the probe (section 3.2).

    The probe is one request: the question, and the one canonical tool. A
    model that answers in prose is said to have failed and the list is
    offered again - it is not stored, not marked "chat only", not taken
    with a warning, because everything the assistant does from 2.1 on
    depends on the answer being a call. A provider that refuses the
    request has said nothing about the model, so that is a different
    sentence with the provider's own words in it, and the list again.
    """
    options = [_offer(m) for m in models]

    while True:
        model = _answered(await prompter.choose("model", options))
        prompter.say("probing_tools")
        try:
            verdict = await probe.probe_tool_support(provider, model, question=question)
        except ProviderError as refusal:
            prompter.say("probe_refused", problem=refusal)
            continue

        if verdict.ok:
            prompter.say("tools_ok", ms=_whole(verdict.first_token_ms))
            return model, verdict
        prompter.say("tools_failed")


async def _pick_engine(
    prompter: Prompter, key: str, engines: tuple[str, str], *, google: bool
) -> str:
    """Which recogniser hears the yes or no (`hears`): the one on this
    machine, or Google's when its key is at hand. With one choice there is
    no question."""
    local, remote = engines
    if not google:
        return local
    said = wording()
    options = [Option(local, said[f"{key}_{local}"]), Option(remote, said[f"{key}_{remote}"])]
    return _answered(await prompter.choose(key, options))


async def _pick_microphone(prompter: Prompter, found: Microphones) -> str:
    """The `[audio] input_device` line - or nothing, for a machine with no
    microphone yet: setup goes on, and the default is what it listens through
    once something is plugged in."""
    if not found.devices:
        prompter.say("no_microphones")
        return ""
    chosen = _answered(await prompter.choose("microphone", _microphone_options(found)))
    _warn_if_raw(prompter, chosen, found)
    return chosen


def _found(microphones: Microphones | None) -> Microphones:
    return available_microphones() if microphones is None else microphones


def _microphone_options(found: Microphones) -> list[Option]:
    """Windows' own choice first, then PortAudio's table, the audio engine's
    entries before the raw ones (plan.md D18).

    The first label is the one option worded rather than named, so it is
    the one place the wizard reads its own wording: the device it stands
    for changes with every headset, and the label says which it is today.
    The rest are sorted by the trust `LiveCapture` puts in their path -
    WASAPI, MME, DirectSound, then the rest - and PortAudio's order within.
    """
    current = " ".join((found.default or "?").split())
    options = [Option("", wording()["windows_microphone"].format(name=current))]
    devices = sorted(found.devices, key=_trust)
    options.extend(Option(device.setting, device.label) for device in devices)
    return options


def _trust(device: MicrophoneInfo) -> int:
    """Where a device's path stands in `FULL_DUPLEX_HOST_APIS`; past the end
    for a path that is not there."""
    for rank, name in enumerate(FULL_DUPLEX_HOST_APIS):
        if name in device.host_api:
            return rank
    return len(FULL_DUPLEX_HOST_APIS)


def _warn_if_raw(prompter: Prompter, chosen: str, found: Microphones) -> None:
    """Says what a raw kernel-streaming path costs (plan.md D18): no echo
    cancellation, so the assistant hears itself on speakers. The choice is
    still the user's. Windows' own choice comes through the engine."""
    device = next((device for device in found.devices if device.setting == chosen), None)
    if device is not None and duplex_for(device.host_api, barge_in=True) == "half":
        prompter.say("microphone_raw")


def _probe_question(locale: str) -> str:
    """The question in the language just chosen, or the English beside the
    code: the chain of section 3.12, for a sentence the model reads."""
    return locales.load(locale).probe_question or probe.QUESTION


def _whole(milliseconds: float | None) -> str:
    """`812`, for a number that is shown once and not calculated with."""
    return "?" if milliseconds is None else f"{milliseconds:.0f}"


def _remember(
    database: sqlite3.Connection | None, provider_id: str, model: str, verdict: probe.ProbeResult
) -> None:
    """Writes the verdict to `settings`, so that `allie run` need not ask
    again for a week. On a fresh machine this is the first thing to touch
    the database, which is why the wizard opens it - and closes it - here."""
    connection = open_database() if database is None else database
    try:
        probe.remember(SettingsRepo(connection), provider_id, model, verdict)
    finally:
        if database is None:
            connection.close()


def _offer(model: ModelInfo) -> Option:
    # The id is what lands in `config.toml`, so it is shown next to the name
    # rather than hidden behind it.
    label = model.display_name
    if model.id != model.display_name:
        label = f"{model.display_name}  ({model.id})"
    return Option(model.id, label)


def _answered(value: str | None) -> str:
    if value is None:
        raise _WalkedAwayError
    return value


class TerminalPrompter:
    """The real terminal: `rich` for what is said, `questionary` for answers.

    Asked with `ask_async`, never with `ask`. The wizard checks the key against
    the provider and asks it for a model list, so `run_setup` is a coroutine
    and every question is drawn while an event loop is already running -
    `questionary`'s synchronous `ask` starts a second one underneath, and
    Python refuses. A scripted prompter cannot find that out, which is why
    `test_setup_wizard.py` drives this class through a pipe as well.
    """

    def __init__(self, *, text: Mapping[str, str] | None = None) -> None:
        self._text = wording() if text is None else text
        self._console = Console()

    def say(self, key: str, **fields: object) -> None:
        self._console.print(self._text[key].format(**fields))

    async def choose(self, key: str, options: Sequence[Option]) -> str | None:
        answer = await questionary.select(
            self._text[key],
            choices=[
                questionary.Choice(title=option.label, value=option.value) for option in options
            ],
            # A key can reach forty models; typing part of a name beats forty
            # arrow presses. With the filter on, j and k are letters again.
            use_search_filter=True,
            use_jk_keys=False,
        ).ask_async()
        return _as_answer(answer)

    async def secret(self, key: str) -> str | None:
        return _as_answer(await questionary.password(self._text[key]).ask_async())

    async def ask(self, key: str) -> str | None:
        return _as_answer(await questionary.text(self._text[key]).ask_async())

    async def menu(self, key: str, options: Sequence[Option]) -> str | None:
        # A terminal has no Close button, so the list ends in one: its value
        # is the one no row has.
        answer = await questionary.select(
            self._text[key],
            choices=[
                *(questionary.Choice(title=option.label, value=option.value) for option in options),
                questionary.Choice(title=self._text["settings_close"], value=_CLOSE),
            ],
            use_jk_keys=False,
        ).ask_async()
        chosen = _as_answer(answer)
        return None if chosen == _CLOSE else chosen


# What the terminal's settings list answers for its Close choice.
_CLOSE = ""


def _as_answer(value: object) -> str | None:
    """Narrows what `questionary` returns, which is typed `Any`.

    An interrupted question comes back as `None`, and so does anything else
    unexpected - which is what the wizard reads as walking away.
    """
    return value if isinstance(value, str) else None
