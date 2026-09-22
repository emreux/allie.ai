"""Sending a message from the user's own account to a person they know
(design.md section 3.6; spec of 2026-09-15).

Two channels, two files, one address book. `contacts.py` reads the people
the user wrote down; `whatsapp.py` drives the official WhatsApp application
through its own click-to-chat link and one Enter key; `telegram.py` speaks
to Telegram's API as the user, through Telethon. What the model sees is one
tool, `send_message`, in `tools/messaging.py`, which asks the user first.

Nothing in this package speaks WhatsApp's protocol, and nothing ever will
by default: Meta bans unofficial clients, and the only thing here that
talks to WhatsApp's servers is WhatsApp.
"""
