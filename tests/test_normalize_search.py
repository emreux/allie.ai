"""Search folding (design.md section 3.7): one key for every spelling.

The fixtures are the ones measured on this machine when `anyascii` was chosen
over NFKD and over a hand-written map: the Turkish spellings that broke both,
and the other scripts that would break a Turkish-only map.
"""

from __future__ import annotations

import pytest

from assistant.store.normalize import normalize_search


@pytest.mark.parametrize("spelling", ["IŞIK", "Işık", "ışık", "isik", "ISIK", "İSİK"])
def test_every_spelling_of_isik_is_one_key(spelling: str) -> None:
    """`I`/`ı`/`İ`/`i` and the cedilla all fold away, whichever way they came."""
    assert normalize_search(spelling) == "isik"


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("Łódź", "lodz"),
        ("STRAßE", "strasse"),
        ("København", "kobenhavn"),
        ("Ελλάδα", "ellada"),
        ("Москва", "moskva"),
        ("ödeme", "odeme"),
    ],
)
def test_other_scripts_fold_by_the_same_rule(text: str, key: str) -> None:
    """No language is named in the code; one rule covers every script."""
    assert normalize_search(text) == key


def test_plain_ascii_is_only_lowercased() -> None:
    assert normalize_search("Google Chrome") == "google chrome"


def test_the_key_removes_information_and_cannot_be_read_back() -> None:
    """Two spellings, one key: which one it was is gone. That is what makes
    this the wrong function for speech, where `25.08.2026` has to become
    *more* text, not less (section 3.5)."""
    assert normalize_search("Işık") == normalize_search("isik")
