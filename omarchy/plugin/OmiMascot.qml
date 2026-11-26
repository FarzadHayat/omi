import QtQuick
import qs.Commons

// Omi: the Omarchy mascot with expressive animated eyes.
// Drawn cleanly on a pixel-inspired Omarchy "O" silhouette so it scales from the
// status bar (18 px) to the floating on-screen companion (60 px) and follows the
// active Omarchy theme live.
//
// Each state has a distinct expression (never colour alone):
//   offline     dimmed body, sleepy closed eyes
//   idle        calm open eyes with natural periodic blink
//   listening   wide bright accent eyes + gentle breathing halo
//   thinking    eyes glancing around curiously + gentle bob
//   speaking    happy expressive eyes + animated voice bars below
//   acting      focused slightly squinted eyes + accent frame glow
//   confirming  curious wide eyes + urgent "?" badge
Item {
  id: root

  property string phase: "idle"
  property real size: Style.font.icon
  property color foreground: "white"
  property color accent: "deepskyblue"
  property color urgent: "orangered"
  property color background: "black"
  // Normalized gaze vector [-1, 1] for looking toward a target on screen.
  property real lookX: 0
  property real lookY: 0
  property bool showHalo: false

  implicitWidth: size
  implicitHeight: size
  width: size
  height: size

  readonly property bool animate: !Style.reduceMotion
  readonly property bool compact: size < 28
  property real blinkFactor: 1.0      // 1 = open, 0.08 = closed
  property real thinkGlanceX: 0.0     // smooth gaze while thinking
  property real thinkGlanceY: 0.0
  property real speakPulse: 0.35      // voice bar animation while speaking

  readonly property real effectiveLookX: phase === "thinking" ? thinkGlanceX : Math.max(-1, Math.min(1, lookX))
  readonly property real effectiveLookY: phase === "thinking" ? thinkGlanceY : Math.max(-1, Math.min(1, lookY))

  readonly property color bodyColor: {
    if (phase === "offline") return Qt.rgba(foreground.r, foreground.g, foreground.b, 0.38)
    if (phase === "confirming") return urgent
    if (phase === "listening" || phase === "acting") return accent
    return foreground
  }

  readonly property color eyeColor: {
    if (phase === "offline") return Qt.rgba(foreground.r, foreground.g, foreground.b, 0.45)
    if (phase === "listening") return accent
    if (phase === "confirming") return urgent
    return foreground
  }

  // Natural blink timer (idle, speaking, acting).
  Timer {
    id: blinkTimer
    interval: 3400
    running: root.animate && root.phase !== "offline" && root.phase !== "listening"
    repeat: true
    onTriggered: {
      blinkAnim.restart()
      interval = 2800 + Math.round(Math.random() * 2600)
    }
  }

  SequentialAnimation {
    id: blinkAnim
    NumberAnimation { target: root; property: "blinkFactor"; to: 0.08; duration: Style.duration(75); easing.type: Easing.OutCubic }
    NumberAnimation { target: root; property: "blinkFactor"; to: 1.0; duration: Style.duration(110); easing.type: Easing.OutCubic }
  }

  // Curious eye glance while thinking.
  SequentialAnimation {
    running: root.animate && root.phase === "thinking"
    loops: Animation.Infinite
    ParallelAnimation {
      NumberAnimation { target: root; property: "thinkGlanceX"; to: -0.7; duration: Style.duration(280); easing.type: Easing.OutCubic }
      NumberAnimation { target: root; property: "thinkGlanceY"; to: -0.35; duration: Style.duration(280); easing.type: Easing.OutCubic }
    }
    PauseAnimation { duration: Style.duration(320) }
    ParallelAnimation {
      NumberAnimation { target: root; property: "thinkGlanceX"; to: 0.7; duration: Style.duration(280); easing.type: Easing.OutCubic }
      NumberAnimation { target: root; property: "thinkGlanceY"; to: -0.25; duration: Style.duration(280); easing.type: Easing.OutCubic }
    }
    PauseAnimation { duration: Style.duration(320) }
    ParallelAnimation {
      NumberAnimation { target: root; property: "thinkGlanceX"; to: 0.0; duration: Style.duration(240); easing.type: Easing.OutCubic }
      NumberAnimation { target: root; property: "thinkGlanceY"; to: 0.0; duration: Style.duration(240); easing.type: Easing.OutCubic }
    }
    PauseAnimation { duration: Style.duration(220) }
  }

  // Gentle voice mouth pulse while speaking.
  SequentialAnimation {
    running: root.animate && root.phase === "speaking"
    loops: Animation.Infinite
    NumberAnimation { target: root; property: "speakPulse"; to: 1.0; duration: Style.duration(160); easing.type: Easing.OutCubic }
    NumberAnimation { target: root; property: "speakPulse"; to: 0.35; duration: Style.duration(140); easing.type: Easing.OutCubic }
    NumberAnimation { target: root; property: "speakPulse"; to: 0.8; duration: Style.duration(130); easing.type: Easing.OutCubic }
    NumberAnimation { target: root; property: "speakPulse"; to: 0.25; duration: Style.duration(150); easing.type: Easing.OutCubic }
  }

  // Soft listening / acting halo behind Omi (companion mode).
  Rectangle {
    id: halo
    anchors.centerIn: parent
    visible: root.showHalo && (root.phase === "listening" || root.phase === "acting" || root.phase === "speaking")
    width: root.size * 1.24
    height: width
    radius: width * 0.34
    color: "transparent"
    border.width: Math.max(1.5, root.size * 0.04)
    border.color: root.phase === "speaking" ? root.foreground : root.accent
    opacity: 0.36

    SequentialAnimation on scale {
      running: halo.visible && root.animate
      loops: Animation.Infinite
      NumberAnimation { from: 0.96; to: 1.07; duration: Style.duration(720); easing.type: Easing.InOutSine }
      NumberAnimation { from: 1.07; to: 0.96; duration: Style.duration(720); easing.type: Easing.InOutSine }
    }
  }

  // Main Omi body container (subtle breathing/bob motion).
  Item {
    id: body
    anchors.centerIn: parent
    width: root.size
    height: root.size

    SequentialAnimation on y {
      running: root.animate && !root.compact && (root.phase === "thinking" || root.phase === "listening" || root.phase === "speaking")
      loops: Animation.Infinite
      NumberAnimation { from: 0; to: -3; duration: Style.duration(650); easing.type: Easing.InOutSine }
      NumberAnimation { from: -3; to: 0; duration: Style.duration(650); easing.type: Easing.InOutSine }
    }

    // Outer chunky Omarchy "O" shell.
    Rectangle {
      id: shell
      anchors.centerIn: parent
      width: root.size * 0.92
      height: root.size * 0.84
      radius: root.compact ? Math.max(3, width * 0.28) : width * 0.26
      color: root.bodyColor

      Behavior on color {
        ColorAnimation { duration: Style.duration(160); easing.type: Easing.OutCubic }
      }

      // Inner dark face visor.
      Rectangle {
        id: visor
        anchors.centerIn: parent
        width: shell.width * 0.74
        height: shell.height * 0.72
        radius: root.compact ? Math.max(2, width * 0.22) : width * 0.20
        color: root.background

        // Eyes container, shifted smoothly by gaze vector.
        Item {
          id: eyes
          width: visor.width * 0.74
          height: visor.height * 0.56
          x: (visor.width - width) / 2 + root.effectiveLookX * (visor.width * 0.10)
          y: (visor.height - height) / 2 + (root.phase === "speaking" && !root.compact ? -visor.height * 0.07 : 0)
             + root.effectiveLookY * (visor.height * 0.10)

          Behavior on x { NumberAnimation { duration: Style.duration(140); easing.type: Easing.OutCubic } }
          Behavior on y { NumberAnimation { duration: Style.duration(140); easing.type: Easing.OutCubic } }

          readonly property real eyeW: Math.max(2, eyes.width * 0.29)
          readonly property real baseEyeH: {
            if (root.phase === "offline") return Math.max(1.5, eyes.height * 0.16)
            if (root.phase === "listening") return eyes.height * 0.94
            if (root.phase === "acting") return eyes.height * 0.62
            if (root.phase === "speaking") return eyes.height * 0.76
            return eyes.height * 0.82
          }
          readonly property real eyeH: Math.max(1.5, baseEyeH * (root.phase === "offline" ? 1.0 : root.blinkFactor))

          // Left eye
          Rectangle {
            id: leftEye
            width: eyes.eyeW
            height: eyes.eyeH
            radius: width / 2
            x: eyes.width * 0.10
            anchors.verticalCenter: parent.verticalCenter
            color: root.eyeColor
            rotation: root.phase === "acting" ? 6 : 0

            // Cute specular catchlight (companion/panel size)
            Rectangle {
              visible: !root.compact && root.phase !== "offline" && eyes.eyeH > 6
              width: Math.max(2, leftEye.width * 0.34)
              height: width
              radius: width / 2
              x: leftEye.width * 0.48
              y: leftEye.height * 0.16
              color: root.background
              opacity: 0.75
            }
          }

          // Right eye
          Rectangle {
            id: rightEye
            width: eyes.eyeW
            height: root.phase === "confirming" ? eyes.eyeH * 0.82 : eyes.eyeH
            radius: width / 2
            x: eyes.width - width - eyes.width * 0.10
            anchors.verticalCenter: parent.verticalCenter
            anchors.verticalCenterOffset: root.phase === "confirming" ? -Math.max(1, eyes.height * 0.08) : 0
            color: root.eyeColor
            rotation: root.phase === "acting" ? -6 : 0

            Rectangle {
              visible: !root.compact && root.phase !== "offline" && eyes.eyeH > 6
              width: Math.max(2, rightEye.width * 0.34)
              height: width
              radius: width / 2
              x: rightEye.width * 0.48
              y: rightEye.height * 0.16
              color: root.background
              opacity: 0.75
            }
          }
        }

        // Cute mini voice wave / mouth when speaking (non-compact).
        Row {
          visible: !root.compact && root.phase === "speaking"
          anchors.horizontalCenter: parent.horizontalCenter
          anchors.bottom: parent.bottom
          anchors.bottomMargin: visor.height * 0.10
          spacing: Math.max(2, visor.width * 0.06)

          Repeater {
            model: 3
            Rectangle {
              required property int index
              width: Math.max(2, visor.width * 0.09)
              height: Math.max(2, visor.height * (0.08 + (index === 1 ? 0.14 : 0.08) * root.speakPulse))
              radius: width / 2
              anchors.verticalCenter: parent.verticalCenter
              color: root.accent
            }
          }
        }
      }
    }

    // Confirming badge "?" at top-right.
    Rectangle {
      visible: root.phase === "confirming"
      width: Math.max(10, root.size * 0.38)
      height: width
      radius: width / 2
      anchors.right: parent.right
      anchors.top: parent.top
      color: root.urgent
      border.width: 1
      border.color: root.background

      Text {
        anchors.centerIn: parent
        textFormat: Text.PlainText
        text: "?"
        color: root.background
        font.pixelSize: Math.max(8, Math.round(parent.width * 0.68))
        font.bold: true
      }
    }
  }
}
