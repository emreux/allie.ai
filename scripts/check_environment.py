"""Reports whether this machine can run the assistant (design.md section 8, phase 0).

Checks what a machine has to have, and prints what it found instead of
asserting: on a fresh clone the point is to see which piece is missing.

    uv run python scripts/check_environment.py

Covered: the Python version, the audio devices and the credential store. No
speech model is checked: nothing on this machine turns speech into text since
plan.md D36, and the assistant speaks in the live model's own voice (D32). It
never touches the network, and it never records or plays anything -
`scripts/smoke_audio.py` does that, because it needs a person to listen.
"""

from __future__ import annotations

import sys
from typing import Any

OK = "  ok   "
WARN = " warn  "
FAIL = " FAIL  "


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


def main() -> int:
    """Runs every check and returns 1 if a required one failed."""
    print("\nPhase 0 environment report\n" + "-" * 42)
    required = [
        check_python(),
        check_audio_devices(),
        check_credential_store(),
    ]
    print("-" * 42)

    if all(required):
        print("All required checks passed.\n")
        return 0
    print("At least one required check failed - see the lines marked FAIL above.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
