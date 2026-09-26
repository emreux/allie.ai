# Allie

A Windows desktop voice assistant on live speech-to-speech models, running on
your own API key and in your own language.

You talk, it answers in its own voice, and you can cut it off by talking over
it. There is no recogniser and no text model in between: the microphone streams
to the model and the model's voice streams back over one session.

**Status: it runs.** This repository is the successor of
[windows-voice-assistant](https://github.com/emreux/windows-voice-assistant)
(complete for its scope, shelved at v0.4.0). Everything that does not depend on
how speech reaches the model was carried over from it - the tools behind one
permission gate, notes, reminders, mail, messaging, media playback, the tray
icon, `doctor`, `purge`, `autostart`, the locale packs - and the three-stage
pipeline was replaced by one live session against Google's Gemini Live API.
This README is still short: it describes what the program does today rather
than everything it does.

## Getting started

```
uv sync
uv run allie setup     # assistant, key, model, language, microphone; later, one setting at a time
uv run allie run       # the window; --terminal for a status line, --tray for the icon
uv run allie doctor    # who answers, what leaves this machine, where the files are
```

Ctrl+Alt+H stops and starts listening, and cuts an answer short. Ctrl+C quits.

## What it does

A conversation, and twenty-six tools the model may call: the time, the weather,
notes with search, reminders that are read out loud between turns, mail
reading, WhatsApp and Telegram messages, music and video, the system volume,
web pages, the clipboard, opening applications and settings pages, installing
from the Store, and what it remembers about you. Anything that could matter is
read back to you and waits for a spoken yes.

The session opens when you start speaking and closes after a minute of silence,
because a live model bills by the minute while it is open. `allie cost`
reports what the turns used.

## What leaves this machine

While a session is open the microphone streams to Google, and so does the
model's answer. That is what a live model is. When a tool asks for
permission, your yes or no goes to Google's recogniser on the same Gemini
key; nothing on this machine turns speech into text. Your API key lives in
the Windows Credential Manager and is never written to a file or a log.

## Not there yet

- **OpenAI.** The second adapter waits for a funded key; Gemini Live is the
  only provider this build can open.
- **The wake word.** The detector is written and off (`[wake] enabled`): the
  model that recognises the phrase is not in the repository yet.
- **Web search.** The model can be given Google's own search
  (`[live] web_search`), which a free AI Studio key refuses, so it ships off.
- **A price for the live model.** It bills by audio as well as by tokens, so
  `pricing.toml` names no price for it and `cost` says so rather than claiming
  a number that is only part of the bill.

Mail has only ever been run against a fake IMAP server.

## For developers

`docs/` holds the plan, the decisions and the measurements; it is not in git.
The tests are the specification of the parts: `uv run pytest`. Lint, format and
types are `uv run ruff check .`, `uv run ruff format .`, `uv run mypy src
--strict`, and all four are green before every commit.
