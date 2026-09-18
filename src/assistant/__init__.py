"""A provider-agnostic, language-agnostic Windows desktop voice assistant,
on live speech-to-speech models.

The successor of `windows-voice-assistant` (shelved at v0.4.0): the same
product - tools behind one permission gate, notes, reminders, mail,
messaging, media, the tray, `doctor` and `purge`, all on the user's own API
key - with the three-stage pipeline (local recogniser, text model, local
voice) replaced by one live session: the microphone streams to the model,
the model's voice streams back, the model decides when the user has
finished speaking, and the user can interrupt it by talking. `v0.1.0` is the
live loop on Gemini Live; the plan in `docs/plan.md` says what follows.
"""

__all__ = ["__version__"]

# Kept in step with `[project] version` in pyproject.toml by hand: two places,
# both read by a person, and neither worth a build plugin to reconcile.
__version__ = "0.1.0"
