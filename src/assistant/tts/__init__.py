"""Text to speech: one protocol, one engine per vendor (design.md section 3.5).

Nothing outside this package imports a speech SDK. `app.py` sees only what
`base.py` declares, which is what lets Windows be swapped for Azure in phase
3.4 without touching the state machine.
"""

__all__: list[str] = []
