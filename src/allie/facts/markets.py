"""Exchange rates and coin prices (D39): TCMB's daily bulletin, the ECB's
reference rates through frankfurter, CoinGecko. Parsing and saying only;
`service.py` asks."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

__all__ = [
    "COINGECKO_PRICE",
    "COINGECKO_SEARCH",
    "DEFAULT_COINS",
    "DEFAULT_CURRENCIES",
    "FRANKFURTER",
    "TCMB_TODAY",
    "Bulletin",
    "Rate",
    "amount",
    "currency_codes",
    "describe_coins",
    "describe_frankfurter",
    "describe_tcmb",
    "first_coin",
    "parse_tcmb",
]

TCMB_TODAY = "https://www.tcmb.gov.tr/kurlar/today.xml"
FRANKFURTER = "https://api.frankfurter.dev/v1/latest"
COINGECKO_PRICE = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_SEARCH = "https://api.coingecko.com/api/v3/search"

# What "how much is the dollar" means when the model named nothing: the two
# currencies the user most likely holds, as codes and not as words - the
# model turns "dolar" into USD, the code carries no language.
DEFAULT_CURRENCIES = ("USD", "EUR")
DEFAULT_COINS = ("bitcoin", "ethereum")

_CODE = re.compile(r"\b[A-Za-z]{3}\b")


@dataclass(frozen=True, slots=True)
class Rate:
    code: str
    unit: int
    buying: float | None
    selling: float | None


@dataclass(frozen=True, slots=True)
class Bulletin:
    day: str
    rates: dict[str, Rate]


def amount(value: float) -> str:
    """A price as a model reads it: thousands grouped, no trailing zeros."""
    text = f"{value:,.4f}" if abs(value) < 100 else f"{value:,.2f}"
    return text.rstrip("0").rstrip(".")


def currency_codes(about: str) -> tuple[str, ...]:
    """The ISO 4217 codes in `about`, upper case, in order, once each; the
    defaults when it names none."""
    found = tuple(dict.fromkeys(code.upper() for code in _CODE.findall(about)))
    return found or DEFAULT_CURRENCIES


def parse_tcmb(xml_text: str) -> Bulletin:
    """TCMB's `today.xml`; `ValueError` for anything that is not it."""
    try:
        root = ET.fromstring(xml_text)  # noqa: S314  # expat 2.6 refuses entity bombs; no external entities
    except ET.ParseError as failure:
        raise ValueError(f"not XML: {failure}") from failure
    written = root.get("Date") or ""
    day = datetime.strptime(written, "%m/%d/%Y").date().isoformat() if written else ""
    rates: dict[str, Rate] = {}
    for node in root.iter("Currency"):
        code = (node.get("CurrencyCode") or node.get("Kod") or "").strip().upper()
        if not code:
            continue
        rates[code] = Rate(
            code=code,
            unit=int(_number(node.findtext("Unit")) or 1),
            buying=_number(node.findtext("ForexBuying")),
            selling=_number(node.findtext("ForexSelling")),
        )
    return Bulletin(day=day, rates=rates)


def describe_tcmb(bulletin: Bulletin, codes: tuple[str, ...], currency: str) -> str:
    parts: list[str] = []
    missing: list[str] = []
    for code in codes:
        rate = bulletin.rates.get(code)
        if rate is None or (rate.buying is None and rate.selling is None):
            missing.append(code)
            continue
        sides = [
            f"{amount(value)} {side}"
            for value, side in ((rate.buying, "buying"), (rate.selling, "selling"))
            if value is not None
        ]
        parts.append(f"{rate.unit} {code} = {' / '.join(sides)} {currency}")
    said = (
        f"TCMB indicative rates of {bulletin.day} (the day's bulletin, not the live market): "
        + ("; ".join(parts) + "." if parts else "none of those.")
    )
    if missing:
        said += f" TCMB lists no rate for {', '.join(missing)}."
    return said


def describe_frankfurter(body: Mapping[str, Any], codes: tuple[str, ...], currency: str) -> str:
    rates = body.get("rates") or {}
    parts: list[str] = []
    missing: list[str] = []
    for code in codes:
        value = rates.get(code)
        if not isinstance(value, int | float) or value == 0:
            missing.append(code)
            continue
        parts.append(f"1 {code} = {amount(1 / value)} {currency}")
    said = (
        f"European Central Bank reference rates of {body.get('date', '')} (daily, not the live "
        "market): " + ("; ".join(parts) + "." if parts else "none of those.")
    )
    if missing:
        said += f" No reference rate for {', '.join(missing)}."
    return said


def first_coin(body: Mapping[str, Any]) -> str | None:
    coins = body.get("coins") or []
    first = coins[0] if coins and isinstance(coins[0], dict) else None
    return str(first["id"]) if first and first.get("id") else None


def describe_coins(
    body: Mapping[str, Any], ids: tuple[str, ...], currencies: tuple[str, ...]
) -> str:
    lines: list[str] = []
    missing: list[str] = []
    for coin in ids:
        prices = body.get(coin)
        if not isinstance(prices, dict):
            missing.append(coin)
            continue
        said = ", ".join(
            f"{amount(float(prices[unit]))} {unit.upper()}" for unit in currencies if unit in prices
        )
        change = prices.get(f"{currencies[0]}_24h_change")
        if isinstance(change, int | float):
            said += f" ({change:+.1f} % in 24 hours)"
        lines.append(f"{coin}: {said}")
    text = "CoinGecko prices now: " + ("; ".join(lines) + "." if lines else "none of those.")
    if missing:
        text += f" No price for {', '.join(missing)}."
    return text


def _number(value: str | None) -> float | None:
    try:
        return float(value) if value and value.strip() else None
    except ValueError:
        return None
