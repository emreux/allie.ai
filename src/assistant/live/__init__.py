"""The provider layer: one narrow protocol, one adapter per vendor.

Nothing outside this package may import a provider SDK. `agent/`, `tools/`
and `policy.py` see only what `base.py` declares, which is what lets the same
state machine run against Gemini Live and OpenAI's live API (plan.md 4.3).

Until task L1.1, `base.py` is the old repository's text contract - the
value types the whole tree shares (`ToolSpec`, `ToolCall`, `Usage`,
`ModelInfo`, the two errors) and, beside them, `LLMProvider`, `Message` and
`Delta`, which `probe.py`, the wizard and their tests still speak. L1.1
replaces those three with `LiveProvider`, `SessionConfig` and `LiveSession`;
the value types stay. There is no adapter in this build yet, so the
catalogue's entries cannot be built and the wizard says so.
"""

__all__: list[str] = []
