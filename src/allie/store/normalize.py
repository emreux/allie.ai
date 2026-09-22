"""Folding text into a search key (design.md section 3.7).

Two people say the same app name and the recogniser writes it two ways:
`Işık` for one, `isik` for the other. To match them the letters have to be
folded - accents off, case off - and the obvious ways of doing that are
wrong. SQLite's own folding follows the Unicode default, in which `I` becomes
`i` and never `ı`. NFKD with the combining marks stripped is inconsistent:
`Ş` decomposes to `S` plus a cedilla, but `ı` is a base letter and stays, so
`IŞIK` and `Işık` come out as two different keys. A hand-written map is
consistent - and Turkish only. Measured on this machine, all three, before
`anyascii` was chosen: it transliterates every script to ASCII by one rule,
and every spelling of `ışık` lands on `isik`, as do `Łódź` on `Lodz` and
`Москва` on `Moskva`.

Phase 2.2 uses the key to find an app by name; phase 4.1 will index notes
with the same function. It *removes* information on purpose. The TTS
normalisation of phase 3.4 *adds* it - `25.08.2026` read out as words - and
the two are never one function, however alike they look.
"""

from __future__ import annotations

from anyascii import anyascii

__all__ = ["normalize_search"]


def normalize_search(text: str) -> str:
    """The script-neutral, case-neutral key under which `text` is looked up.

    `anyascii` first, so that the case folding is done on plain letters and
    the Turkish `İ` does not turn into `i` with a stray combining dot.
    """
    return anyascii(text).casefold()
