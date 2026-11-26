-- Hold Super+Alt+V to talk to Omi; let go to send.
o.bind_hold("SUPER + ALT + V", "Omi: talk (hold)", "@OMI@ ptt start", "Omi: send", "@OMI@ ptt stop")
-- Stop Omi speaking and cancel what it is doing. Escape alone stays with apps.
o.bind("SUPER + ALT + ESCAPE", "Omi: stop", "@OMI@ stop")
-- Open the Omi panel: the conversation, a field to type a request, Yes and No.
o.bind("SUPER + ALT + O", "Omi panel", "omarchy-shell omi toggle")
