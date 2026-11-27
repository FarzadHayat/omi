import QtQuick
import Quickshell
import Quickshell.Io
import "Model.js" as Model

// The Omi daemon as seen from the bar and companion overlay: one subscribed socket
// that streams every event (no polling), plus a short-lived socket per typed request.
Item {
  id: root

  readonly property string socketPath: Quickshell.env("OMI_SOCKET")
    || (Quickshell.env("XDG_RUNTIME_DIR") || "/tmp") + "/omi/omi.sock"

  property QtObject sock: null  // the connected event socket
  readonly property bool connected: sock !== null
  // offline | idle | listening | thinking | speaking | acting | confirming
  property string phase: "offline"
  property var lines: []        // [{ role: "user" | "assistant" | "progress", text }]
  property var confirm: null    // { id, question, detail } while a question waits
  property string notice: ""    // the last refusal from the daemon, shown briefly
  property string lastUserText: ""
  property string liveSpeech: ""     // spoken reply updated clause by clause
  property string lastProgress: ""   // current background action label
  property var mascot: null          // { action, x, y, w, h, label, duration } while pointing/moving
  property real cursorX: 0
  property real cursorY: 0
  property bool hasCursor: false
  property bool wakeEnabled: false
  property string wakeCallMode: "auto"  // auto (smart) | continuous | single
  property bool wakeConversational: true
  property string wakeSensitivity: "medium"
  property bool soundsEnabled: true
  property string talker: ""
  property string sttEngine: "parakeet"
  property string ttsEngine: "kokoro"
  property real ttsSpeed: 1.0
  property string voice: "af_heart"
  property var providers: []
  // { providers, models, model, talker, fallbacks, active, fallback_reason, local_model, local_endpoint }
  property var brain: ({})
  property var sttEngines: []
  property var ttsEngines: []
  property var voices: ["af_heart", "af_bella", "am_adam", "bf_emma", "bm_george"]
  property string healthError: ""
  property string _pendingSay: ""

  function receive(line) {
    var msg = Model.parseLine(line)
    if (!msg) return
    var ev = msg.event
    if (ev === "cursor") {
      var cx = Number(msg.x)
      var cy = Number(msg.y)
      if (isFinite(cx) && isFinite(cy)) {
        cursorX = cx
        cursorY = cy
        hasCursor = true
      }
      return
    }
    if (ev === "state") {
      var next = Model.knownState(msg.state)
      if (next === "listening") {
        liveSpeech = ""
        lastProgress = ""
      }
      phase = next
      if (phase !== "confirming") confirm = null
    } else if (ev === "transcript") {
      lines = Model.appendLine(lines, msg.role, msg.text)
      if (msg.role === "user") {
        lastUserText = Model.clip(msg.text, 240)
        liveSpeech = ""
      } else {
        liveSpeech = Model.clip(msg.text, 600)
      }
    } else if (ev === "speech") {
      var clause = Model.clip(msg.text, 300).trim()
      if (clause !== "") {
        liveSpeech = liveSpeech === "" ? clause : Model.clip(liveSpeech + " " + clause, 600)
      }
    } else if (ev === "progress") {
      lines = Model.appendLine(lines, "progress", msg.text)
      lastProgress = Model.clip(msg.text, 140)
    } else if (ev === "mascot") {
      var m = Model.mascotOf(msg)
      mascot = m
      if (m) {
        mascotTimer.interval = m.duration * 1000
        mascotTimer.restart()
      } else {
        mascotTimer.stop()
      }
    } else if (ev === "wake") {
      if (typeof msg.enabled === "boolean") wakeEnabled = msg.enabled
    } else if (ev === "brain") {
      var next = Object.assign({}, brain)
      if (typeof msg.active === "string") next.active = msg.active
      if (typeof msg.reason === "string") next.fallback_reason = msg.reason
      brain = next
    } else if (ev === "settings") {
      if (typeof msg.talker === "string") talker = msg.talker
      if (msg.brain && typeof msg.brain === "object") brain = msg.brain
      if (typeof msg.voice === "string" && msg.voice !== "") voice = msg.voice
      if (Array.isArray(msg.voices)) voices = msg.voices
      if (Array.isArray(msg.stt_engines)) sttEngines = msg.stt_engines
      if (Array.isArray(msg.tts_engines)) ttsEngines = msg.tts_engines
      if (msg.engines) {
        if (typeof msg.engines.stt === "string" && msg.engines.stt !== "") sttEngine = msg.engines.stt
        if (typeof msg.engines.tts === "string" && msg.engines.tts !== "") ttsEngine = msg.engines.tts
        if (typeof msg.engines.voice === "string" && msg.engines.voice !== "") voice = msg.engines.voice
        if (typeof msg.engines.speed === "number" && isFinite(msg.engines.speed)) ttsSpeed = msg.engines.speed
      }
      if (msg.wake) {
        if (typeof msg.wake.enabled === "boolean") wakeEnabled = msg.wake.enabled
        if (typeof msg.wake.call_mode === "string") wakeCallMode = msg.wake.call_mode
        if (typeof msg.wake.conversational === "boolean") wakeConversational = msg.wake.conversational
        if (typeof msg.wake.sensitivity === "string") wakeSensitivity = msg.wake.sensitivity
      }
      if (msg.audio && typeof msg.audio.sounds === "boolean") soundsEnabled = msg.audio.sounds
    } else if (ev === "confirm_request") {
      confirm = Model.confirmOf(msg)
    } else if (ev === undefined && msg.ok === true && typeof msg.state === "string") {
      phase = Model.knownState(msg.state)
      confirm = Model.confirmOf(msg.confirm)
      if (typeof msg.talker === "string") talker = msg.talker
      if (msg.engines) {
        if (typeof msg.engines.stt === "string" && msg.engines.stt !== "") sttEngine = msg.engines.stt
        if (typeof msg.engines.tts === "string" && msg.engines.tts !== "") ttsEngine = msg.engines.tts
        if (typeof msg.engines.voice === "string" && msg.engines.voice !== "") voice = msg.engines.voice
        if (typeof msg.engines.speed === "number" && isFinite(msg.engines.speed)) ttsSpeed = msg.engines.speed
      }
      if (Array.isArray(msg.providers)) providers = msg.providers
      if (msg.brain && typeof msg.brain === "object") brain = msg.brain
      if (Array.isArray(msg.stt_engines)) sttEngines = msg.stt_engines
      if (Array.isArray(msg.tts_engines)) ttsEngines = msg.tts_engines
      if (Array.isArray(msg.voices)) voices = msg.voices
      if (msg.wake) {
        if (typeof msg.wake.enabled === "boolean") wakeEnabled = msg.wake.enabled
        if (typeof msg.wake.call_mode === "string") wakeCallMode = msg.wake.call_mode
        if (typeof msg.wake.conversational === "boolean") wakeConversational = msg.wake.conversational
        if (typeof msg.wake.sensitivity === "string") wakeSensitivity = msg.wake.sensitivity
      }
      if (msg.audio && typeof msg.audio.sounds === "boolean") soundsEnabled = msg.audio.sounds
      var errText = ""
      if (msg.errors && typeof msg.errors === "object") {
        var keys = Object.keys(msg.errors)
        if (keys.length > 0) errText = keys[0] + ": " + Model.clip(msg.errors[keys[0]], 160)
      }
      healthError = errText
    } else if (ev === undefined && msg.ok === false) {
      flash(Model.clip(msg.error, 300))
    }
  }

  function flash(message) {
    notice = message
    noticeTimer.restart()
  }

  function clearMascot() {
    mascot = null
    mascotTimer.stop()
  }

  function send(line) {
    if (!sock) return false
    sock.write(line)
    sock.flush()
    return true
  }

  function stop() {
    clearMascot()
    send(Model.request("stop"))
  }

  function setSetting(key, value) {
    return send(Model.request("set", { key: key, value: value, save: true }))
  }

  function toggleWake() {
    return setSetting("wake.enabled", wakeEnabled ? "off" : "on")
  }

  function cycleCallMode() {
    var order = ["auto", "continuous", "single"]
    var idx = order.indexOf(wakeCallMode)
    return setSetting("wake.call_mode", order[(idx + 1) % order.length])
  }

  function toggleCallMode() {
    return cycleCallMode()
  }

  function toggleSounds() {
    return setSetting("audio.sounds", soundsEnabled ? "off" : "on")
  }

  function cycleSensitivity() {
    var order = ["low", "medium", "high"]
    var idx = order.indexOf(wakeSensitivity)
    return setSetting("wake.sensitivity", order[(idx + 1) % order.length])
  }

  function cycleVoice() {
    if (!voices || voices.length === 0) return false
    var idx = voices.indexOf(voice)
    return setSetting("tts.voice", voices[(idx + 1) % voices.length])
  }

  function _readyNames(list, fallback) {
    var out = []
    var arr = list || []
    for (var i = 0; i < arr.length; i++) {
      if (arr[i].installed) out.push(arr[i].name)
    }
    return out.length > 0 ? out : fallback
  }

  function cycleStt() {
    var order = _readyNames(sttEngines, ["parakeet", "nemotron", "moonshine"])
    var idx = order.indexOf(sttEngine)
    return setSetting("stt.engine", order[(idx + 1) % order.length])
  }

  function cycleTts() {
    var order = _readyNames(ttsEngines, ["kokoro", "piper"])
    var idx = order.indexOf(ttsEngine)
    return setSetting("tts.engine", order[(idx + 1) % order.length])
  }

  function cycleSpeed() {
    var steps = [0.9, 1.0, 1.15, 1.3]
    var next = steps[0]
    for (var i = 0; i < steps.length; i++) {
      if (ttsSpeed < steps[i] - 0.02) { next = steps[i]; break }
    }
    return setSetting("tts.speed", String(next))
  }

  function setProvider(name) {
    if (!name || name === talker) return false
    return setSetting("brain.talker", name)
  }

  // Local models without a custom endpoint are files; everything else is a model name.
  readonly property bool localFiles: talker === "local" && !(brain && brain.local_endpoint)

  function currentModel() {
    if (!brain) return ""
    return localFiles ? (brain.local_model || "") : (brain.model || "")
  }

  function setModel(value) {
    if (value === currentModel()) return false
    return setSetting(localFiles ? "brain.local_model" : "brain.model", value)
  }

  function setFallbacks(list) {
    return setSetting("brain.fallbacks", (list || []).join(","))
  }

  // Answers exactly the question on screen: the daemon refuses an id that is no
  // longer the one waiting, so a late click never approves a newer question.
  function answer(yes) {
    if (!confirm) return
    if (send(Model.request("confirm", { answer: yes === true, id: confirm.id }))) confirm = null
  }

  function say(text) {
    var line = Model.sayRequest(text)
    if (line === "") return false
    if (!connected) return false
    _pendingSay = line
    turnLink.active = false
    turnLink.active = true
    return true
  }

  Timer {
    id: noticeTimer
    interval: 5000
    onTriggered: root.notice = ""
  }

  Timer {
    id: mascotTimer
    interval: 8000
    onTriggered: root.mascot = null
  }

  // Quickshell's Socket cannot retry after a failed connect, so each attempt is a
  // fresh one, every two seconds while the daemon is away.
  Loader {
    id: link
    sourceComponent: Socket {
      id: socket
      path: root.socketPath
      connected: true
      parser: SplitParser { onRead: function(line) { root.receive(line) } }
      onConnectionStateChanged: {
        if (connected) {
          root.sock = socket
          socket.write(Model.request("status") + Model.request("subscribe"))
          socket.flush()
        } else {
          if (root.sock === socket) root.sock = null
          root.phase = "offline"
          root.confirm = null
          root.mascot = null
        }
      }
    }
  }

  Timer {
    running: !root.connected
    interval: 2000
    repeat: true
    onTriggered: { link.active = false; link.active = true }
  }

  Loader {
    id: turnLink
    active: false
    sourceComponent: Socket {
      id: turnSocket
      path: root.socketPath
      connected: true
      parser: SplitParser {
        onRead: function(line) {
          var msg = Model.parseLine(line)
          if (msg && msg.event === undefined && msg.ok === false) root.flash(Model.clip(msg.error, 300))
        }
      }
      onConnectionStateChanged: {
        if (connected && root._pendingSay !== "") {
          turnSocket.write(root._pendingSay)
          turnSocket.flush()
          root._pendingSay = ""
        }
      }
      onError: root._pendingSay = ""
    }
  }
}
