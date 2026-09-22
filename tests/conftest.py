"""Fixtures shared by every test that touches the user's machine.

Two things in this project are global by nature: the Credential Manager and
the settings directory. A test that used the real ones would read the
developer's own key and write into their own `%APPDATA%`, so both are
redirected here - the keyring to a backend held in memory, the settings to a
temporary directory through `ALLIE_CONFIG_DIR`.

The keyring backend is a real `KeyringBackend`, not a stand-in for the library.
The calls under test are the calls production makes; only the vault is fake.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import keyring
import pytest
from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError


class MemoryKeyring(KeyringBackend):
    """A real keyring backend that forgets everything when the test ends."""

    priority = 1.0

    def __init__(self) -> None:
        # `keyring` is typed but this constructor is not; calling it anyway
        # keeps the fake on the same code path as a real backend.
        super().__init__()  # type: ignore[no-untyped-call]
        self.vault: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self.vault[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self.vault.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) not in self.vault:
            raise PasswordDeleteError(f"nothing stored for {username}")
        del self.vault[(service, username)]


@pytest.fixture
def vault() -> Iterator[MemoryKeyring]:
    """Swaps the machine's Credential Manager for one that lives in memory."""
    previous = keyring.get_keyring()
    fake = MemoryKeyring()
    keyring.set_keyring(fake)
    try:
        yield fake
    finally:
        keyring.set_keyring(previous)


@pytest.fixture
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points the settings directory at a temporary one for this test only."""
    monkeypatch.setenv("ALLIE_CONFIG_DIR", str(tmp_path))
    return tmp_path
