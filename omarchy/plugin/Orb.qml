import QtQuick
import qs.Commons

// Hark's state as a small mark. Each state has its own shape as well as its own
// colour, so it never relies on colour alone:
//   offline     faint ring with a slash     idle        faint ring
//   listening   solid accent dot, breathing thinking    turning arc
//   speaking    dot inside a ring           acting      accent diamond
//   confirming  urgent dot with a "?"
Item {
  id: root

  property string phase: "idle"
  property real size: Style.font.icon
  property color foreground: "white"
  property color accent: "deepskyblue"
  property color urgent: "orangered"
  property color background: "black"

  implicitWidth: size
  implicitHeight: size
  width: size
  height: size

  readonly property real stroke: Math.max(1.5, size / 9)
  readonly property bool animate: !Style.reduceMotion

  // offline and idle: a ring, faint enough to sit quietly in the bar.
  Rectangle {
    anchors.centerIn: parent
    visible: root.phase === "idle" || root.phase === "offline"
    width: root.size * 0.62
    height: width
    radius: width / 2
    color: "transparent"
    border.width: root.stroke
    border.color: root.foreground
    opacity: root.phase === "offline" ? 0.35 : 0.6
  }

  Rectangle {
    anchors.centerIn: parent
    visible: root.phase === "offline"
    width: root.size * 0.8
    height: root.stroke
    rotation: -45
    color: root.foreground
    opacity: 0.35
  }

  // listening: the microphone is open.
  Rectangle {
    id: listeningDot
    anchors.centerIn: parent
    visible: root.phase === "listening"
    width: root.size * 0.7
    height: width
    radius: width / 2
    color: root.accent

    SequentialAnimation on scale {
      running: listeningDot.visible && root.animate
      loops: Animation.Infinite
      alwaysRunToEnd: false
      NumberAnimation { from: 0.82; to: 1.0; duration: Style.duration(650); easing.type: Easing.InOutSine }
      NumberAnimation { from: 1.0; to: 0.82; duration: Style.duration(650); easing.type: Easing.InOutSine }
    }
  }

  // thinking: a three-quarter arc that turns.
  Canvas {
    id: arc
    anchors.centerIn: parent
    visible: root.phase === "thinking"
    width: root.size * 0.72
    height: width
    onPaint: {
      var ctx = getContext("2d")
      ctx.reset()
      ctx.strokeStyle = root.foreground
      ctx.lineWidth = root.stroke
      ctx.lineCap = "round"
      ctx.beginPath()
      ctx.arc(width / 2, height / 2, width / 2 - root.stroke, 0, Math.PI * 1.5)
      ctx.stroke()
    }
    onVisibleChanged: requestPaint()
    Connections {
      target: root
      function onForegroundChanged() { arc.requestPaint() }
      function onSizeChanged() { arc.requestPaint() }
    }

    RotationAnimator on rotation {
      running: arc.visible && root.animate
      from: 0
      to: 360
      duration: 900
      loops: Animation.Infinite
    }
  }

  // speaking: a dot inside a ring, like sound leaving.
  Item {
    anchors.fill: parent
    visible: root.phase === "speaking"

    Rectangle {
      anchors.centerIn: parent
      width: root.size * 0.86
      height: width
      radius: width / 2
      color: "transparent"
      border.width: root.stroke
      border.color: root.foreground
      opacity: 0.55
    }

    Rectangle {
      anchors.centerIn: parent
      width: root.size * 0.42
      height: width
      radius: width / 2
      color: root.foreground
    }
  }

  // acting: running a tool on the computer.
  Rectangle {
    anchors.centerIn: parent
    visible: root.phase === "acting"
    width: root.size * 0.52
    height: width
    rotation: 45
    radius: Math.max(1, width / 6)
    color: root.accent
  }

  // confirming: waiting for a yes or no.
  Rectangle {
    anchors.centerIn: parent
    visible: root.phase === "confirming"
    width: root.size * 0.86
    height: width
    radius: width / 2
    color: root.urgent

    Text {
      anchors.centerIn: parent
      textFormat: Text.PlainText
      text: "?"
      color: root.background
      font.pixelSize: Math.round(root.size * 0.62)
      font.bold: true
    }
  }
}
