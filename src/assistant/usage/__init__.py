"""What the assistant costs, turn by turn (design.md section 6, section 3.11).

`tracker.py` turns a turn's token counts into dollars with the price table
of `defaults/pricing.toml`, writes them to `usage_log`, and answers whether
the day or the month has gone over its limit. `live-assistant cost` reads the
same table back. The limits themselves are `agent/limits.py`'s; this package
only applies the two rows of that table that are about money.
"""

__all__: list[str] = []
