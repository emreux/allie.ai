"""The provider layer: one narrow protocol, one adapter per vendor.

Nothing outside this package may import a provider SDK. `agent/`, `tools/`
and `policy.py` see only what `base.py` declares, which is what lets the same
state machine run against Gemini Live and OpenAI's live API (plan.md 4.3).

`base.py` is the session contract: the value types the whole tree shares
(`ToolSpec`, `ToolCall`, `Usage`, `ModelInfo`, the two errors), the two
protocols (`LiveProvider`, `LiveSession`), the `SessionConfig` a session is
opened with and the events it produces. `gemini_live.py` is the first
adapter; `registry.py` builds one from a name in `providers.toml` and a key
from the Credential Manager; `probe.py` asks a model once whether it calls
a tool. The OpenAI adapter is L2.
"""

__all__: list[str] = []
