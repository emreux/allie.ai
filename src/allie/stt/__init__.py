"""Speech to text: one protocol, one engine per vendor (design.md section 3.4).

Nothing outside this package imports a speech SDK. `app.py` sees only what
`base.py` declares, which is what lets the local model be swapped for a hosted
one in phase 3 without touching the state machine.
"""

__all__: list[str] = []
