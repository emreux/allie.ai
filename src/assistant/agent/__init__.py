"""The agent: the loop, the prompt, and later the gate every tool goes through.

Nothing in this package knows which provider is answering. It sees the protocol
of `live/base.py` and nothing else, which is what lets the same loop run against
Gemini, Anthropic and any OpenAI-compatible endpoint (design.md section 3.2).

Phase 1 was `core.py` and `prompts.py` - a loop with no tools in it. Phase 2.1
added `policy.py`, the one permission gate of section 3.9, and gave the loop
its `while`; 2.4 added `limits.py`, the limits of section 3.11 the loop asks
before every call; 2.5 added `intents.py`, the fast path of section 4 that
answers a short command without the model and without skipping the gate.
`memory.py` arrives with 2.10; do not assume a module exists because section
7 lists it.
"""

__all__: list[str] = []
