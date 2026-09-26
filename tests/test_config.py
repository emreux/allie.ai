"""Where settings live, what shape they have, and where the key does not go.

Two claims are worth a test each. The first is that a directory this code
writes to is the one design.md section 3.3 names - `platformdirs` is called
with arguments, and the wrong ones silently produce a different, plausible
looking path. The second is that the API key never reaches the settings file;
that is the whole reason `keyring` is a dependency.

The `vault` and `config_home` fixtures come from `conftest.py`, which keeps
this suite off the developer's own Credential Manager and settings directory.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest

from allie import config as config_module
from allie.agent.limits import Limits
from allie.config import (
    KEYRING_SERVICE,
    AudioSettings,
    DocumentSettings,
    LimitSettings,
    LiveSettings,
    LocaleSettings,
    MessagingSettings,
    Settings,
    STTSettings,
    TelegramSettings,
    ToolSettings,
    WakeSettings,
    WebSettings,
    config_dir,
    config_path,
    data_dir,
    delete_api_key,
    is_configured,
    load_api_key,
    load_settings,
    log_dir,
    save_settings,
    store_api_key,
)
from tests.conftest import MemoryKeyring

WINDOWS_ONLY = pytest.mark.skipif(os.name != "nt", reason="the product ships on Windows")


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def test_the_environment_variable_moves_the_whole_config_directory(config_home: Path) -> None:
    """Tests need somewhere to write that is not the developer's own settings."""
    assert config_dir() == config_home
    assert config_path() == config_home / "config.toml"


@WINDOWS_ONLY
def test_settings_land_in_appdata_under_a_single_assistant_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ALLIE_CONFIG_DIR", raising=False)

    assert config_dir() == Path(os.environ["APPDATA"]) / "allie"


@WINDOWS_ONLY
def test_data_and_logs_are_local_not_roaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """A database and a log file must not follow the user to another machine."""
    monkeypatch.delenv("ALLIE_CONFIG_DIR", raising=False)
    local = Path(os.environ["LOCALAPPDATA"]) / "allie"

    assert data_dir() == local
    assert log_dir() == local / "Logs"
    assert data_dir() != config_dir()


def test_reading_settings_creates_nothing(config_home: Path) -> None:
    """Merely asking what the settings are must not litter the disk."""
    load_settings()

    assert list(config_home.iterdir()) == []


# --------------------------------------------------------------------------
# The settings themselves
# --------------------------------------------------------------------------


def test_before_setup_the_settings_load_with_defaults(config_home: Path) -> None:
    """The first run has no file; it must not be an error."""
    settings = load_settings()

    assert settings.live.primary == ""
    assert is_configured() is False


def test_after_setup_the_settings_come_back(config_home: Path) -> None:
    save_settings(Settings(live=LiveSettings(primary="gemini:gemini-3.5-flash-lite")))

    settings = load_settings()

    assert settings.live.primary == "gemini:gemini-3.5-flash-lite"
    assert is_configured() is True


def test_the_input_device_round_trips_through_the_file(config_home: Path) -> None:
    save_settings(Settings(audio=AudioSettings(input_device="Microphone Array WASAPI")))

    assert load_settings().audio.input_device == "Microphone Array WASAPI"


def test_a_kernel_streaming_device_line_round_trips_through_the_file(config_home: Path) -> None:
    """What `allie mic` stores is the device line as PortAudio prints it,
    and a Bluetooth headset's kernel-streaming line carries a driver path with
    backslashes and a line break. It must come back byte for byte, since
    `sounddevice` is matched against it exactly."""
    line = (
        "Headset (@System32\\drivers\\bthhfenum.sys,#2;%1 Hands-Free%0\r\n;(Buds3)), Windows WDM-KS"
    )
    save_settings(Settings(audio=AudioSettings(input_device=line)))

    assert load_settings().audio.input_device == line


def test_no_input_device_means_the_system_default(config_home: Path) -> None:
    assert load_settings().audio.input_device == ""


def test_the_unblocked_tools_round_trip_through_the_file(config_home: Path) -> None:
    save_settings(Settings(tools=ToolSettings(unblocked=["delete_file", "run_command"])))

    assert load_settings().tools.unblocked == ["delete_file", "run_command"]


def test_no_tool_is_unblocked_by_default(config_home: Path) -> None:
    assert load_settings().tools.unblocked == []


def test_a_file_written_before_there_was_an_audio_table_still_loads(config_home: Path) -> None:
    """Every `config.toml` `allie setup` wrote before 2026-09-05 has no
    `[audio]` table. It means what it always meant: the system default."""
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text('[live]\nprimary = "gemini:x"\n', encoding="utf-8")

    assert load_settings().audio.input_device == ""
    assert load_settings().live.primary == "gemini:x"


def test_the_saved_file_is_toml(config_home: Path) -> None:
    save_settings(Settings(live=LiveSettings(primary="gemini:x"), locale=LocaleSettings(code="tr")))

    written = tomllib.loads(config_path().read_text(encoding="utf-8"))

    assert written["live"]["primary"] == "gemini:x"
    assert written["locale"]["code"] == "tr"


def test_a_value_with_quotes_survives_the_round_trip(config_home: Path) -> None:
    """There is no TOML writer in the standard library, so ours must escape."""
    awkward = 'custom:he said "hi" \\ back'
    save_settings(Settings(live=LiveSettings(primary=awkward)))

    assert load_settings().live.primary == awkward


def test_the_provider_and_the_model_are_read_off_one_line() -> None:
    settings = LiveSettings(primary="gemini:gemini-3.5-flash-lite")

    assert settings.provider == "gemini"
    assert settings.model == "gemini-3.5-flash-lite"


def test_a_model_name_may_contain_colons_and_slashes() -> None:
    """`ollama:llama3:8b` and `openrouter:google/gemini-2.5-flash` are real names."""
    assert LiveSettings(primary="ollama:llama3:8b").model == "llama3:8b"
    assert LiveSettings(primary="openrouter:google/gemini-2.5-flash").provider == "openrouter"


def test_the_server_address_round_trips_through_the_file(config_home: Path) -> None:
    """`custom` has no address in the catalogue; the one the wizard asked
    for lives here, beside the model (2.7)."""
    save_settings(
        Settings(live=LiveSettings(primary="custom:llama3", base_url="http://localhost:1234/v1"))
    )

    assert load_settings().live.base_url == "http://localhost:1234/v1"


def test_no_address_by_default(config_home: Path) -> None:
    """Every other provider's address is in `providers.toml`, and a file
    written before there was an address still loads."""
    save_settings(Settings(live=LiveSettings(primary="gemini:x")))

    assert load_settings().live.base_url == ""


def test_the_live_table_defaults_to_the_session_policy_of_the_plan(config_home: Path) -> None:
    """Plan.md 4.6: barge-in on, a minute of silence closes the session,
    transcripts on, the adapter's own voice - and four minutes of resumption,
    not the ten D5 wrote: Gemini refuses the handle at five (measured
    2026-09-21), so a longer window only buys a failed open."""
    live = load_settings().live

    assert (
        live.voice,
        live.barge_in,
        live.idle_close_seconds,
        live.resume_minutes,
        live.transcripts,
    ) == ("", True, 60.0, 4.0, True)


def test_the_server_s_turn_detection_is_left_to_the_server_by_default(config_home: Path) -> None:
    """ADR-001: the fast setting clips a paused sentence, so the two knobs
    ship empty - the owner tunes them in the real run."""
    live = load_settings().live

    assert (live.end_sensitivity, live.silence_ms) == ("", 0)


def test_the_two_turn_detection_knobs_round_trip_through_the_file(config_home: Path) -> None:
    save_settings(Settings(live=LiveSettings(end_sensitivity="HIGH", silence_ms=300)))

    live = load_settings().live

    assert (live.end_sensitivity, live.silence_ms) == ("HIGH", 300)


def test_compression_is_on_and_affective_dialog_off_unless_the_owner_says_otherwise(
    config_home: Path,
) -> None:
    """D25: the model refuses `enable_affective_dialog` (2026-09-21), so that
    one ships off; the sliding window is accepted and ships on."""
    live = load_settings().live
    assert (live.affective_dialog, live.compress_context) == (False, True)

    save_settings(Settings(live=LiveSettings(affective_dialog=True, compress_context=False)))

    live = load_settings().live
    assert (live.affective_dialog, live.compress_context) == (True, False)


def test_web_search_is_off_until_the_owner_switches_it_on(config_home: Path) -> None:
    """D22: Google refuses the tool on the free-tier key (2026-09-21), so
    the default cannot be on; a key with billing gets `web_search = true`."""
    assert load_settings().live.web_search is False
    save_settings(Settings(live=LiveSettings(web_search=True)))
    assert load_settings().live.web_search is True


def test_the_sensitivity_is_read_in_any_letter_case_and_kept_in_capitals() -> None:
    """The adapter builds the SDK's constant name from it; `high` typed by
    hand is the same setting."""
    assert LiveSettings(end_sensitivity="low").end_sensitivity == "LOW"
    assert LiveSettings(end_sensitivity=" High ").end_sensitivity == "HIGH"
    assert LiveSettings(end_sensitivity="").end_sensitivity == ""


def test_a_sensitivity_the_server_does_not_have_is_refused() -> None:
    with pytest.raises(ValueError, match="HIGH, LOW"):
        LiveSettings(end_sensitivity="medium")


def test_a_negative_silence_is_refused() -> None:
    with pytest.raises(ValueError, match="0 or more"):
        LiveSettings(silence_ms=-1)


def test_the_live_table_round_trips_through_the_file(config_home: Path) -> None:
    chosen = LiveSettings(
        primary="gemini:gemini-3.8-live",
        voice="Kore",
        barge_in=False,
        idle_close_seconds=30.0,
        resume_minutes=5.0,
        transcripts=False,
        end_sensitivity="LOW",
        silence_ms=300,
    )
    save_settings(Settings(live=chosen))

    assert load_settings().live == chosen


def test_a_negative_session_number_is_refused() -> None:
    """Zero is a policy (close at once, never resume); less than zero is a typo."""
    with pytest.raises(ValueError, match="0 or more"):
        LiveSettings(idle_close_seconds=-1.0)
    with pytest.raises(ValueError, match="0 or more"):
        LiveSettings(resume_minutes=-0.5)


def test_the_search_engine_round_trips_through_the_file(config_home: Path) -> None:
    """`[web] search_url` (15 Sep 2026): which engine `search_web` opens is
    the user's, not the code's."""
    save_settings(Settings(web=WebSettings(search_url="https://duckduckgo.com/?q={query}")))

    assert load_settings().web.search_url == "https://duckduckgo.com/?q={query}"


def test_a_file_written_before_there_was_a_web_table_searches_google(config_home: Path) -> None:
    save_settings(Settings(live=LiveSettings(primary="gemini:x")))

    assert load_settings().web.search_url == "https://www.google.com/search?q={query}"


def test_the_look_up_model_is_the_one_the_free_key_may_search_with() -> None:
    """D29: the 2.5 family is the free tier's only one with Google's search."""
    assert WebSettings().look_up_model == "gemini-2.5-flash"


def test_the_messaging_default_and_the_telegram_id_round_trip(config_home: Path) -> None:
    """`[messaging] default_app` and `[telegram] api_id` (2026-09-15). The
    hash and the session are secrets and never reach this file."""
    save_settings(
        Settings(
            messaging=MessagingSettings(default_app="WhatsApp"),
            telegram=TelegramSettings(api_id=123456),
        )
    )

    loaded = load_settings()

    assert loaded.messaging.default_app == "WhatsApp"
    assert loaded.telegram.api_id == 123456
    assert "hash" not in config_path().read_text(encoding="utf-8")


def test_without_the_tables_no_app_is_the_default_and_telegram_is_not_set_up(
    config_home: Path,
) -> None:
    save_settings(Settings(live=LiveSettings(primary="gemini:x")))

    assert load_settings().messaging.default_app == ""
    assert load_settings().telegram.api_id == 0


def test_a_primary_without_a_provider_is_refused() -> None:
    """Left unchecked this reads as a provider named after the model, with no model."""
    with pytest.raises(ValueError, match="provider:model"):
        LiveSettings(primary="gemini-3.5-flash-lite")


def test_the_limits_round_trip_through_the_file(config_home: Path) -> None:
    save_settings(
        Settings(limits=LimitSettings(daily_usd=5.0, hard_stop=True, tool_calls_per_turn=3))
    )

    limits = load_settings().limits

    assert (limits.daily_usd, limits.hard_stop, limits.tool_calls_per_turn) == (5.0, True, 3)


def test_the_limits_default_to_the_table_of_section_3_11(config_home: Path) -> None:
    """Written once, in `agent/limits.py`; the file's defaults are read off it."""
    assert Limits.from_settings(load_settings().limits) == Limits()


def test_a_file_written_before_there_was_a_limits_table_still_loads(config_home: Path) -> None:
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text('[live]\nprimary = "gemini:x"\n', encoding="utf-8")

    assert load_settings().limits == LimitSettings()


def test_one_limit_in_the_file_leaves_the_others_at_their_defaults(config_home: Path) -> None:
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text("[limits]\nhard_stop = true\n", encoding="utf-8")

    limits = load_settings().limits

    assert limits.hard_stop is True
    assert limits.daily_usd == LimitSettings().daily_usd


def test_the_locale_defaults_to_english(config_home: Path) -> None:
    """The fallback chain of section 3.12 ends at `en`, so it is the safe default."""
    assert load_settings().locale.code == "en"


def test_an_unknown_table_in_the_file_does_not_stop_the_assistant(config_home: Path) -> None:
    """A newer version's key, or a typo, is not worth refusing to start over."""
    config_path().write_text('[live]\nprimary = "gemini:x"\n[watchers]\nkap = true\n', "utf-8")

    assert load_settings().live.primary == "gemini:x"


def test_the_environment_can_override_a_setting(
    config_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """So a second model can be tried without editing the file."""
    save_settings(Settings(live=LiveSettings(primary="gemini:one")))
    monkeypatch.setenv("ALLIE_LIVE__PRIMARY", "gemini:two")

    assert load_settings().live.primary == "gemini:two"


# --------------------------------------------------------------------------
# The key
# --------------------------------------------------------------------------


def test_the_key_is_stored_and_read_back(vault: MemoryKeyring) -> None:
    store_api_key("gemini", "AQ.secret")

    assert load_api_key("gemini") == "AQ.secret"


def test_the_key_is_filed_under_the_provider_id(vault: MemoryKeyring) -> None:
    """Setup writes it and the registry reads it; they must agree on the name."""
    store_api_key("gemini", "AQ.secret")

    assert vault.vault == {(KEYRING_SERVICE, "gemini"): "AQ.secret"}


def test_no_key_stored_reads_as_none(vault: MemoryKeyring) -> None:
    assert load_api_key("gemini") is None


def test_deleting_a_key_that_was_never_there_is_not_an_error(vault: MemoryKeyring) -> None:
    """`allie purge` must not fail on a provider the user never configured."""
    delete_api_key("gemini")

    assert load_api_key("gemini") is None


def test_deleting_a_key_removes_it(vault: MemoryKeyring) -> None:
    store_api_key("gemini", "AQ.secret")

    delete_api_key("gemini")

    assert load_api_key("gemini") is None


def test_the_key_never_reaches_the_settings_file(config_home: Path, vault: MemoryKeyring) -> None:
    """The one claim section 3.3 makes about key safety."""
    store_api_key("gemini", "AQ.secret")
    save_settings(Settings(live=LiveSettings(primary="gemini:gemini-3.5-flash-lite")))

    assert "AQ.secret" not in config_path().read_text(encoding="utf-8")


def test_the_settings_model_has_nowhere_to_put_a_key() -> None:
    """Not an oversight to be fixed later: the field is absent on purpose."""
    assert "api_key" not in LiveSettings.model_fields
    assert not any("key" in name for name in Settings.model_fields)


def test_the_stt_settings_round_trip_through_the_file(config_home: Path) -> None:
    save_settings(Settings(stt=STTSettings(provider="gemini", model="x")))

    assert load_settings().stt == STTSettings(provider="gemini", model="x")


def test_the_recogniser_defaults_to_local(config_home: Path) -> None:
    """ADR-001: local Whisper is the default and never leaves; a file written
    before there was an `[stt]` table means exactly that."""
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text('[live]\nprimary = "gemini:x"\n', encoding="utf-8")

    stt = load_settings().stt

    assert stt == STTSettings()
    assert (stt.provider, stt.model) == ("local", "gemini-3.5-transcribe-live")


def test_an_unknown_recogniser_is_refused() -> None:
    with pytest.raises(ValueError, match="local, gemini"):
        STTSettings(provider="azure")


def test_a_voice_table_from_before_is_read_past_and_not_written_back(config_home: Path) -> None:
    """D32: the program speaks in `[live] voice` and in nothing else. A file
    that still names Windows' voice loads, and the next save drops it."""
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text(
        '[live]\nprimary = "gemini:x"\nvoice = "Orus"\n\n[tts]\nprovider = "sapi"\n',
        encoding="utf-8",
    )

    save_settings(load_settings())

    assert "[tts]" not in config_path().read_text(encoding="utf-8")
    assert load_settings().live.voice == "Orus"


# --------------------------------------------------------------------------
# [wake] (plan.md D21)
# --------------------------------------------------------------------------


def test_the_wake_table_has_the_owner_s_defaults(config_home: Path) -> None:
    """Off for a file setup has not written since the assistants arrived
    (D31): setup turns it on with the model, the voice and the name of the
    one chosen. No threshold of its own: the model's is read. The rest is
    the owner's choice: the chime."""
    wake = load_settings().wake

    assert (wake.enabled, wake.model, wake.greeting) == (False, "hey_friday", "chime")
    assert wake.threshold is None


def test_a_threshold_left_out_stays_out_of_the_file(config_home: Path) -> None:
    """So that the one the model shipped with is read at every start, and
    a better number reaches the user with the next version."""
    save_settings(Settings(wake=WakeSettings(enabled=True, model="hey_vesper")))

    assert "threshold" not in config_path().read_text(encoding="utf-8")
    assert load_settings().wake.threshold is None


def test_the_wake_table_round_trips_through_the_file(config_home: Path) -> None:
    chosen = WakeSettings(
        enabled=True, model=r"C:\models\mine.onnx", threshold=0.62, greeting="sentence"
    )
    save_settings(Settings(wake=chosen))

    assert load_settings().wake == chosen


def test_a_threshold_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        WakeSettings(threshold=1.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        WakeSettings(threshold=0)


def test_a_greeting_the_program_does_not_have_is_refused() -> None:
    with pytest.raises(ValueError):
        WakeSettings(greeting="bell")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# [documents] (plan.md D35)
# --------------------------------------------------------------------------


def test_the_documents_live_under_local_app_data_unless_the_user_says(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Local, not Roaming and not Documents: OneDrive's folder backup never
    covers AppData, and a roaming profile never copies AppData\\Local (the
    owner, 2026-09-25: nothing may go to OneDrive)."""
    monkeypatch.setattr(config_module, "data_dir", lambda: tmp_path / "local")

    assert DocumentSettings().root() == tmp_path / "local" / "documents"


def test_a_written_folder_is_taken_with_its_variables_expanded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ALLIE_TEST_DOCUMENTS", str(tmp_path))

    written = DocumentSettings(folder="%ALLIE_TEST_DOCUMENTS%\\Belgeler")

    assert written.root() == tmp_path / "Belgeler"


def test_the_document_model_is_given_a_positive_time() -> None:
    with pytest.raises(ValueError, match="more than 0"):
        DocumentSettings(timeout_seconds=0)


def test_the_documents_table_round_trips(config_home: Path) -> None:
    save_settings(
        Settings(documents=DocumentSettings(folder="D:\\Belgeler", model="gemini-3.5-flash-lite"))
    )

    loaded = load_settings().documents

    assert (loaded.folder, loaded.model, loaded.timeout_seconds) == (
        "D:\\Belgeler",
        "gemini-3.5-flash-lite",
        20.0,
    )
