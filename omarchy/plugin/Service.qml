import QtQuick
import Quickshell
import Quickshell.Io
import "Model.js" as Model

// The hark daemon as seen from the bar: one subscribed socket that streams every
// event (no polling), plus a short-lived socket per typed request, because the
// daemon answers a turn on the connection that asked for it and reads nothing
// else from that connection until the turn ends.
Item {
  id: root

  readonly property string socketPath: Quickshell.env("HARK_SOCKET")
    || (Quickshell.env("XDG_RUNTIME_DIR") || "/tmp") + "/hark/hark.sock"

  property QtObject sock: null  // the connected event socket
  readonly property bool connected: sock !== null
  // offline | idle | listening | thinking | speaking | acting | confirming
  property string phase: "offline"
  property var lines: []        // [{ role: "user" | "assistant" | "progress", text }]
  property var confirm: null    // { id, question, detail } while a question waits
  property string notice: ""    // the last refusal from the daemon, shown briefly
  property string _pendingSay: ""

  function receive(line) {
    var msg = Model.parseLine(line)
    if (!msg) return
    var ev = msg.event
    if (ev === "state") {
      phase = Model.knownState(msg.state)
      if (phase !== "confirming") confirm = null
    } else if (ev === "transcript") {
      lines = Model.appendLine(lines, msg.role, msg.text)
    } else if (ev === "progress") {
      lines = Model.appendLine(lines, "progress", msg.text)
    } else if (ev === "confirm_request") {
      confirm = Model.confirmOf(msg)
    } else if (ev === undefined && msg.ok === true && typeof msg.state === "string") {
      // The status reply sent right after connecting.
      phase = Model.knownState(msg.state)
      confirm = Model.confirmOf(msg.confirm)
    } else if (ev === undefined && msg.ok === false) {
      flash(Model.clip(msg.error, 300))
    }
  }

  function flash(message) {
    notice = message
    noticeTimer.restart()
  }

  function send(line) {
    if (!sock) return false  // the panel already says the daemon is not running
    sock.write(line)
    sock.flush()
    return true
  }

  function stop() { send(Model.request("stop")) }

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
      // The turn's events also reach the event socket; only a refusal matters here.
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
