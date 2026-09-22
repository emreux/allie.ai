"""The agent: the loop, the prompt, and later the gate every tool goes through.

Nothing in this package knows which provider is answering. It sees the protocol
of `live/base.py` and nothing else, which is what lets the same code run against
Gemini and, from L2.1, against an OpenAI session (design.md section 3.2).

Four modules. `core.py` is `ToolRunner`: the round that takes a tool call off
the session, asks the guard, runs it through the gate and sends the result
back. `policy.py` is that gate, the only one (section 3.9). `limits.py` holds
the limits of section 3.11 the runner asks before every call. `prompts.py` is
the frozen system prompt. The old product's fast path - `intents.py`, a short
command answered without the model - went with the pipeline (plan.md D7), and
so did the `while` loop that drove a text model: a live session drives itself,
and the state machine is `app.py`.
"""

__all__: list[str] = []
