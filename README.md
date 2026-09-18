# live-assistant

A Windows desktop voice assistant on live speech-to-speech models, running on
your own API key and in your own language.

**Status: not usable yet.** This repository is the successor of
[windows-voice-assistant](https://github.com/emreux/windows-voice-assistant)
(complete for its scope, shelved at v0.4.0). Everything that does not depend on
how speech reaches the model was carried over from it - the tools behind one
permission gate, notes, reminders, mail, messaging, media playback, the tray
icon, `doctor`, `purge`, `autostart`, the locale packs - and the three-stage
pipeline (local recogniser, text model, local voice) is being replaced by one
live session: the microphone streams to the model, the model's voice streams
back, and you can interrupt it by talking.

What exists today is that baseline: `live-assistant setup`, `doctor`, `cost`,
`purge --all`, `autostart` and the logins work; `live-assistant run` says that
the live loop is not wired yet and exits. The first live adapter is Google's
Gemini Live API on a free AI Studio key; an OpenAI adapter follows.

This README is a placeholder and is rewritten when the live loop runs.
