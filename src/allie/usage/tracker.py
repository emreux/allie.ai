"""What a turn cost in dollars, and whether the day can afford another one
(design.md section 6 and section 3.11).

Every turn that reached the model is one row of `usage_log`: the tokens the
provider counted, and what they cost at the price of the model that
answered. `allie cost` adds the rows up; the two spending limits of
section 3.11 are checked against the same sums, once per turn.

**The price is data, not code** (section 6). `defaults/pricing.toml` ships
the prices this build knows, and `%APPDATA%\\allie\\pricing.toml`, if the
user writes one, is read on top of it: a price that changed is an edit, and
a model this file never heard of is a table the user adds. A model with no
price is priced at `None`, not at zero - a zero would be a made-up number
that reads as free, and `allie cost` says the price is unknown instead.

**Cached tokens are a subset of the input, not an addition to it.** A turn
of 1000 input tokens with 400 cached is 600 fresh tokens at the input price
and 400 at the cached price. Adding the 400 to the 1000 would bill the cache
hit as a cost, which is the mistake the formula of section 6 made in version
2 of the design. A provider that reports no cached tokens reports zero, and
zero reads correctly as "no cache hit".

**The rows are UTC; the day and the month are the user's.** A row's `ts` is
an epoch second like every other time in the database (section 3.10), and
"today" is the local calendar day, which starts at a different epoch second
every day and at a different one again on the two days the clocks change.
So the boundary is computed from the local calendar and turned into an epoch
second, and the query stays a comparison of integers.

**The limits warn; they stop only if told to.** Past the day's or the
month's limit every answer starts with a warning and the assistant goes on
answering: a limit that silenced the assistant at two in the morning would
make it useless exactly when the user cannot see why. `hard_stop` in
`[limits]` is the user's decision to be silenced anyway (section 12,
decision 10).
"""

from __future__ import annotations

import time
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

from allie.agent.limits import Limits
from allie.config import config_dir
from allie.live.base import Usage
from allie.store.repos import UsageRepo

__all__ = [
    "PRICING_FILE_NAME",
    "Price",
    "Pricing",
    "UsageTracker",
    "start_of_day",
    "start_of_month",
]

PRICING_FILE_NAME = "pricing.toml"

# Vendors quote per million tokens; the counts arrive per token.
MILLION = 1_000_000


@dataclass(frozen=True, slots=True)
class Price:
    """One model's prices, in dollars per million tokens, as vendors quote them."""

    input_per_mtok: float
    output_per_mtok: float
    cached_per_mtok: float

    def cost(self, usage: Usage) -> float:
        """What `usage` cost at this price.

        The cached tokens are taken out of the input before it is charged,
        never added to it (section 6). A provider that reported more cached
        than input tokens has miscounted, and is billed as if it had not.
        """
        cached = min(usage.cached_tokens, usage.input_tokens)
        fresh = usage.input_tokens - cached
        return (
            fresh * self.input_per_mtok
            + cached * self.cached_per_mtok
            + usage.output_tokens * self.output_per_mtok
        ) / MILLION


class Pricing:
    """The prices this build knows, by provider and model."""

    def __init__(self, prices: Mapping[tuple[str, str], Price]) -> None:
        self._prices = dict(prices)

    @classmethod
    def load(cls, *, directory: Path | None = None) -> Pricing:
        """The packaged table, with the user's own `pricing.toml` read over it.

        `directory` is where the user's file is looked for: `config_dir()`
        in life, somewhere under `tmp_path` in a test.
        """
        prices = _parse(_packaged())
        own = (config_dir() if directory is None else directory) / PRICING_FILE_NAME
        if own.is_file():
            prices.update(_parse(own.read_text(encoding="utf-8")))
        return cls(prices)

    @classmethod
    def from_toml(cls, text: str) -> Pricing:
        """A table read from text, which is what a test hands over."""
        return cls(_parse(text))

    def price(self, provider: str, model: str) -> Price | None:
        return self._prices.get((provider, model))

    def cost(self, provider: str, model: str, usage: Usage) -> float | None:
        """What `usage` cost on `model`, or `None` if its price is not known.

        `None` and not zero: a zero would be a number nothing backs, and it
        would read as free (section 6).
        """
        price = self.price(provider, model)
        return None if price is None else price.cost(usage)


def _packaged() -> str:
    # Through `importlib.resources` rather than from `__file__`: once
    # installed, the package need not be a directory on disk.
    return (resources.files("allie") / "defaults" / PRICING_FILE_NAME).read_text(encoding="utf-8")


def _parse(text: str) -> dict[tuple[str, str], Price]:
    """`[provider."model"]` tables into prices.

    A table missing the input or the output price is skipped: half a price
    is no price, and a guessed other half would be a made-up number. A
    missing cached price is the input price - no discount known, none
    applied.
    """
    prices: dict[tuple[str, str], Price] = {}
    for provider, models in tomllib.loads(text).items():
        if not isinstance(models, Mapping):
            continue
        for model, fields in models.items():
            if isinstance(fields, Mapping) and (price := _price(fields)) is not None:
                prices[(provider, model)] = price
    return prices


def _price(fields: Mapping[str, Any]) -> Price | None:
    input_price = _dollars(fields.get("input_per_mtok"))
    output_price = _dollars(fields.get("output_per_mtok"))
    if input_price is None or output_price is None:
        return None
    cached_price = _dollars(fields.get("cached_per_mtok"))
    return Price(
        input_per_mtok=input_price,
        output_per_mtok=output_price,
        cached_per_mtok=input_price if cached_price is None else cached_price,
    )


def _dollars(value: object) -> float | None:
    """A number, or nothing. A bool is an int in Python, and is not a price."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


class UsageTracker:
    """Every turn's tokens to `usage_log` with their price, and whether the
    day can afford another turn.

    Bound to one provider and one model, because that is what `config.toml`
    says answers and every turn of a run goes there. When the fallback chain
    of phase 4.5 lets a turn go elsewhere, the turn will have to say where.
    `clock` is `time.time` unless a test holds one still.
    """

    def __init__(
        self,
        repo: UsageRepo,
        pricing: Pricing,
        *,
        provider: str,
        model: str,
        limits: Limits | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._repo = repo
        self._pricing = pricing
        self._provider = provider
        self._model = model
        self._limits = limits if limits is not None else Limits()
        self._clock = clock

    def record(self, turn_id: str, usage: Usage) -> float | None:
        """Writes the turn down and returns what it cost - `None` for a
        model whose price is not known, and the row says the same."""
        cost = self._pricing.cost(self._provider, self._model, usage)
        self._repo.insert(
            turn_id=turn_id,
            provider=self._provider,
            model=self._model,
            usage=usage,
            cost_usd=cost,
        )
        return cost

    def spent_today(self) -> float:
        """Dollars since local midnight, over the turns that have a price."""
        return self._repo.sum_since(start_of_day(self._clock()))

    def spent_this_month(self) -> float:
        """Dollars since the first of the local month, likewise."""
        return self._repo.sum_since(start_of_month(self._clock()))

    def warning(self) -> str | None:
        """The key of the sentence to say before the answer, if a limit is passed.

        The month's is checked first: it is the larger of the two, and when
        it is passed the day's usually is as well.
        """
        if self.spent_this_month() > self._limits.monthly_usd:
            return "monthly_over"
        if self.spent_today() > self._limits.daily_usd:
            return "daily_over"
        return None

    def stopped(self) -> bool:
        """Whether the model may not be asked at all: `hard_stop`, and a limit passed."""
        return self._limits.hard_stop and self.warning() is not None


def start_of_day(now: float) -> int:
    """The epoch second at which the local day holding `now` began."""
    local = datetime.fromtimestamp(now).astimezone()
    return _epoch(datetime.combine(local.date(), datetime.min.time()))


def start_of_month(now: float) -> int:
    """The epoch second at which the local month holding `now` began."""
    local = datetime.fromtimestamp(now).astimezone()
    return _epoch(datetime.combine(local.date().replace(day=1), datetime.min.time()))


def _epoch(local_midnight: datetime) -> int:
    # A naive datetime is read as local time by `astimezone`, which is the
    # one thing wanted here: the wall clock's midnight, at whatever offset
    # the clock happens to have that day.
    return int(local_midnight.astimezone().timestamp())
