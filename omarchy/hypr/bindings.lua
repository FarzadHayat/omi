-- Hold Super+Alt+V to talk to Hark; let go to send.
o.bind_hold("SUPER + ALT + V", "Hark: talk (hold)", "@HARK@ ptt start", "Hark: send", "@HARK@ ptt stop")
-- Stop Hark speaking and cancel what it is doing. Escape alone stays with apps.
o.bind("SUPER + ALT + ESCAPE", "Hark: stop", "@HARK@ stop")
-- Open the Hark panel: the conversation, a field to type a request, Yes and No.
o.bind("SUPER + ALT + H", "Hark panel", "omarchy-shell hark toggle")
