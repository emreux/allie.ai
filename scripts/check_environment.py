"""Reports whether this machine can run the assistant (design.md section 8, phase 0).

Checks the four things phase 0 has to prove, and prints what it found instead
of asserting: on a fresh clone the point is to see which piece is missing.

    uv run python scripts/check_environment.py

Covered: the Python version, the audio devices, the installed speech voices
(including the ones Windows hides from SAPI), the credential store, and the
local speech-to-text model. It never touches the network except through the
model cache, and it never records or plays anything - `scripts/smoke_audio.py`
does that, because it needs a person to listen.
"""

from __future__ import annotations

import sys
import winreg
from typing import Any

OK = "  ok   "
WARN = " warn  "
FAIL = " FAIL  "

TURKISH_LANGUAGE_ID = "41f"


def _line(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def check_python() -> bool:
    """Phase 1 targets 3.13; 3.12 works but is not what uv.lock was solved for."""
    version = sys.version_info
    text = f"Python {version.major}.{version.minor}.{version.micro}"
    if (version.major, version.minor) == (3, 13):
        _line(OK, text)
        return True
    _line(WARN, f"{text} - the project pins 3.13")
    return False


def check_audio_devices() -> bool:
    """Lists the default input and output; without an input there is nothing to hear."""
    try:
        import sounddevice
    except Exception as error:
        _line(FAIL, f"sounddevice could not be imported: {error}")
        return False

    try:
        source: Any = sounddevice.query_devices(kind="input")
        sink: Any = sounddevice.query_devices(kind="output")
    except Exception as error:
        _line(FAIL, f"no usable audio device: {error}")
        return False

    _line(OK, f"microphone: {source['name']}")
    _line(OK, f"speaker   : {sink['name']}")
    return True


def _registry_voices(path: str) -> list[str]:
    """Reads voice tokens from one registry hive path."""
    names: list[str] = []
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
    except OSError:
        return names

    index = 0
    while True:
        try:
            token = winreg.EnumKey(key, index)
        except OSError:
            break
        try:
            with winreg.OpenKey(key, token) as sub:
                names.append(str(winreg.QueryValueEx(sub, "")[0]))
        except OSError:
            names.append(token)
        index += 1
    return names


def check_speech_voices() -> bool:
    """Finds the voices SAPI exposes, and the modern ones it usually hides.

    Windows installs newer voices under Speech_OneCore. `SAPI.SpVoice` does not
    list those, so a language pack can be installed and still be invisible to
    `GetVoices()`. The assistant enumerates both hives itself (`tts/sapi.py`),
    so these are listed for information rather than as a problem.
    """
    try:
        import win32com.client
    except Exception as error:
        _line(FAIL, f"pywin32 could not be imported: {error}")
        return False

    try:
        engine = win32com.client.Dispatch("SAPI.SpVoice")
        sapi_voices = [voice.GetDescription() for voice in engine.GetVoices()]
    except Exception as error:
        _line(FAIL, f"SAPI is not available: {error}")
        return False

    for name in sapi_voices:
        _line(OK, f"SAPI voice: {name}")

    one_core = _registry_voices(r"SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens")
    hidden = [name for name in one_core if name not in sapi_voices]
    for name in hidden:
        _line(OK, f"Speech_OneCore voice, read directly by the assistant: {name}")

    turkish = [
        name
        for name in sapi_voices
        if "turk" in name.lower() or TURKISH_LANGUAGE_ID in name.lower()
    ]
    if turkish:
        _line(OK, f"Turkish voice available: {turkish[0]}")
        return True

    turkish_hidden = [name for name in hidden if "turk" in name.lower()]
    if turkish_hidden:
        _line(
            OK,
            f"Turkish voice {turkish_hidden[0]!r} is installed under Speech_OneCore; the "
            f"assistant reads that hive itself (tts/sapi.py), so nothing needs copying",
        )
        return True

    _line(
        WARN,
        "no Turkish voice - install one from Settings > Time & Language > Speech, "
        "or let phase 1 speak Turkish text with an English voice",
    )
    return False


def check_credential_store() -> bool:
    """Writes, reads and deletes a probe secret in the Windows Credential Manager."""
    try:
        import keyring
    except Exception as error:
        _line(FAIL, f"keyring could not be imported: {error}")
        return False

    backend = type(keyring.get_keyring()).__name__
    try:
        keyring.set_password("assistant", "probe", "value")
        stored = keyring.get_password("assistant", "probe")
        keyring.delete_password("assistant", "probe")
    except Exception as error:
        _line(FAIL, f"credential store ({backend}) failed: {error}")
        return False

    if stored != "value":
        _line(FAIL, f"credential store ({backend}) returned {stored!r}")
        return False

    _line(OK, f"credential store: {backend}")
    return True


def check_speech_to_text_model(size: str = "small") -> bool:
    """Loads the local model from the cache; the first run downloads about 500 MB."""
    try:
        from faster_whisper import WhisperModel
    except Exception as error:
        _line(FAIL, f"faster-whisper could not be imported: {error}")
        return False

    try:
        WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=4)
    except Exception as error:
        _line(FAIL, f"the {size} model could not be loaded: {error}")
        return False

    _line(OK, f"speech-to-text model '{size}' (int8, cpu) loads")
    return True


def main() -> int:
    """Runs every check and returns 1 if a required one failed."""
    print("\nPhase 0 environment report\n" + "-" * 42)
    required = [
        check_python(),
        check_audio_devices(),
        check_credential_store(),
        check_speech_to_text_model(),
    ]
    check_speech_voices()  # a missing Turkish voice is a warning, not a failure
    print("-" * 42)

    if all(required):
        print("All required checks passed.\n")
        return 0
    print("At least one required check failed - see the lines marked FAIL above.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
