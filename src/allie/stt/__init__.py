"""Speech to text: one protocol, one engine per vendor (design.md section 3.4).

Nothing outside this package imports a speech SDK. `app.py` sees only what
`base.py` declares; since D36 (2026-09-26) the one engine is Google's, and
nothing on this machine turns speech into text.
"""

__all__: list[str] = []
