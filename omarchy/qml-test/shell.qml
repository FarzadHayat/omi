import QtQuick
import Quickshell
import "plugin" as Hark

// Harness for the hark bar widget: loads the real plugin against a fake bar and a
// fake daemon (fake_daemon.py), drives it, and prints one CHECK line per result.
// Run by tests/test_install.py in a separate quickshell on the offscreen platform,
// so it never touches the running Omarchy shell or the screen.
ShellRoot {
  id: shell

  QtObject {
    id: fakeBar
    property color foreground: "#cdd6f4"
    property color barForeground: "#cdd6f4"
    property color background: "#1e1e2e"
    property color urgent: "#f38ba8"
    property string fontFamily: "monospace"
    property string position: "top"
    property bool vertical: false
    property int barSize: 26
    property bool foregroundAnimationEnabled: false
    function showTooltip(target, text) {}
    function hideTooltip(target) {}
    function setCenterHoverRevealSuppressed(value) {}
    function switchPanelFrom(owner, direction) { return false }
  }

  FloatingWindow {
    id: window
    visible: true
    implicitWidth: 120
    implicitHeight: 40

    Hark.Panel {
      id: widget
      bar: fakeBar
      width: 40
      height: 30
    }
  }

  function check(name, ok, detail) {
    console.log("CHECK " + (ok ? "pass " : "FAIL ") + name + (detail !== undefined ? " " + JSON.stringify(detail) : ""))
  }

  function find(item, name) {
    if (!item) return null
    if (item.objectName === name) return item
    var kids = item.children || []
    for (var i = 0; i < kids.length; i++) {
      var hit = find(kids[i], name)
      if (hit) return hit
    }
    return null
  }

  property var hark: widget.service
  property int step: 0
  property var phasesSeen: []

  Connections {
    target: shell.hark
    function onPhaseChanged() { shell.phasesSeen = shell.phasesSeen.concat([shell.hark.phase]) }
  }

  // The fake daemon plays its script as soon as the widget subscribes; each step
  // waits for what the previous one should have produced.
  Timer {
    interval: 100
    repeat: true
    running: true
    onTriggered: {
      var h = shell.hark
      if (shell.step === 0 && h.confirm !== null) {
        shell.check("connected", h.connected)
        shell.check("phases", ["listening", "thinking", "speaking", "acting", "confirming"].every(function(p) {
          return shell.phasesSeen.indexOf(p) >= 0
        }), shell.phasesSeen)
        shell.check("transcript", h.lines.length === 2 && h.lines[0].role === "user"
          && h.lines[1].text === "<b>not bold</b>", h.lines)
        shell.check("confirm", h.confirm.id === "q-1" && h.confirm.question === "Delete the file notes.txt?", h.confirm)
        shell.check("mascot point", h.mascot !== null && h.mascot.x === 420 && h.mascot.label === "Save button", h.mascot)
        shell.check("cursor tracked", h.hasCursor === true && h.cursorX === 640 && h.cursorY === 420, [h.hasCursor, h.cursorX, h.cursorY])
        shell.check("companion shown", widget.companion.shown === true && widget.companion.interactive === true, widget.companion.shown)
        shell.check("accessible label", widget.accessibleLabel === "Omi: Waiting for your answer", widget.accessibleLabel)
        widget.open()
        shell.step = 1
      } else if (shell.step === 1 && widget.opened) {
        shell.check("cursor on yes", widget.currentRow === "yes", widget.currentRow)
        widget.moveCursor(1)
        shell.check("cursor moves to no", widget.currentRow === "no", widget.currentRow)
        widget.moveCursor(-1)
        widget.activate(widget.currentRow)
        shell.check("answered", h.confirm === null)
        shell.step = 2
      } else if (shell.step === 2 && h.phase === "idle") {
        shell.check("rows without question", JSON.stringify(widget.rows) === JSON.stringify(["input", "stop"]), widget.rows)
        shell.check("say", h.say("  open the browser  "))
        h.stop()
        shell.step = 3
      } else if (shell.step === 3 && h.notice !== "") {
        shell.check("refusal shown", h.notice === "a question is waiting for the user's answer", h.notice)
        shell.step = 4
      } else if (shell.step === 4 && !h.connected) {
        shell.check("offline when daemon goes", h.phase === "offline" && widget.accessibleLabel === "Omi: Not running", h.phase)
        console.log("HARNESS DONE")
        Qt.quit()
      }
    }
  }

  Timer {
    interval: 15000
    running: true
    onTriggered: { console.log("HARNESS TIMEOUT at step " + shell.step); Qt.quit() }
  }
}
