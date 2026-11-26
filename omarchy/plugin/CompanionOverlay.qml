import QtQuick
import QtQuick.Layouts
import Quickshell
import Quickshell.Wayland
import qs.Commons
import qs.Commons as Commons
import qs.Ui

// Floating on-screen Omi companion: pops up when you interact with Omi, shows a live
// speech bubble of what Omi is saying, glides across the screen while acting, and
// spotlights / points at targets on screen when asked about something.
// Click-through everywhere except interactive confirm buttons.
Item {
  id: root

  required property var service
  required property var strings

  readonly property string phase: service ? service.phase : "offline"
  readonly property var confirm: service ? service.confirm : null
  readonly property var mascot: service ? service.mascot : null
  readonly property string liveSpeech: service ? service.liveSpeech : ""
  readonly property string lastUserText: service ? service.lastUserText : ""
  readonly property string lastProgress: service ? service.lastProgress : ""

  readonly property bool activeNow: phase === "listening" || phase === "thinking"
    || phase === "speaking" || phase === "acting" || phase === "confirming" || mascot !== null

  property bool lingering: false
  readonly property bool shown: activeNow || lingering
  readonly property bool interactive: confirm !== null

  onActiveNowChanged: {
    if (activeNow) {
      lingerTimer.stop()
      lingering = false
    } else if (phase === "idle" && (liveSpeech !== "" || lastProgress !== "")) {
      lingering = true
      lingerTimer.restart()
    }
  }

  Timer {
    id: lingerTimer
    interval: 4500
    onTriggered: root.lingering = false
  }

  readonly property string bubbleHeadline: {
    if (confirm) return confirm.question
    if (mascot && mascot.label !== "") return mascot.label
    if (liveSpeech !== "") return liveSpeech
    if (phase === "listening") return strings.t("state.listening") + "…"
    if (phase === "thinking") return strings.t("state.thinking") + "…"
    if (phase === "acting") return lastProgress !== "" ? lastProgress : (strings.t("state.acting") + "…")
    return ""
  }

  OverlayWindow {
    id: overlay
    shown: root.shown
    shownLayer: WlrLayer.Overlay
    shownKeyboardFocus: WlrKeyboardFocus.None
    // Visual-only unless a Yes/No confirmation is waiting (then only the card catches clicks).
    mask: Region {
      item: root.interactive ? companionGroup : null
    }

    readonly property real screenW: overlay.width > 0 ? overlay.width : 1920
    readonly property real screenH: overlay.height > 0 ? overlay.height : 1080
    readonly property real margin: Style.space(26)
    readonly property real mascotSize: Style.space(58)

    // Target point when Omi is pointing/moving on screen.
    readonly property bool hasTarget: root.mascot !== null
    readonly property real targetX: hasTarget ? Math.max(12, Math.min(screenW - 12, root.mascot.x)) : screenW - margin - mascotSize / 2
    readonly property real targetY: hasTarget ? Math.max(12, Math.min(screenH - 12, root.mascot.y)) : screenH - margin - mascotSize / 2
    readonly property real targetW: hasTarget && root.mascot.w > 0 ? Math.min(root.mascot.w, screenW - 24) : Style.space(44)
    readonly property real targetH: hasTarget && root.mascot.h > 0 ? Math.min(root.mascot.h, screenH - 24) : Style.space(44)

    // Position the companion neatly next to the target (or bottom-right by default),
    // automatically flipping sides so it never clips offscreen.
    readonly property bool placeRightOfTarget: hasTarget && (targetX + targetW / 2 + companionGroup.width + Style.space(36) < screenW)
    readonly property real desiredX: {
      if (!hasTarget) return screenW - companionGroup.width - margin
      if (placeRightOfTarget) return targetX + Math.min(targetW / 2, 160) + Style.space(22)
      return Math.max(margin, targetX - Math.min(targetW / 2, 160) - companionGroup.width - Style.space(22))
    }
    readonly property real desiredY: {
      if (!hasTarget) return screenH - companionGroup.height - margin
      return Math.max(Style.space(48), Math.min(screenH - companionGroup.height - margin, targetY - companionGroup.height / 2))
    }

    // Gaze vector: look directly at pointed target when pointing, otherwise follow the live mouse cursor!
    readonly property real omiCenterX: companionGroup.x + mascotAvatar.x + mascotAvatar.width / 2
    readonly property real omiCenterY: companionGroup.y + mascotAvatar.y + mascotAvatar.height / 2
    readonly property bool followCursor: !hasTarget && root.service && root.service.hasCursor
    readonly property real gazeTargetX: hasTarget ? targetX : (followCursor ? root.service.cursorX : omiCenterX)
    readonly property real gazeTargetY: hasTarget ? targetY : (followCursor ? root.service.cursorY : omiCenterY)
    readonly property real dx: (hasTarget || followCursor) ? (gazeTargetX - omiCenterX) : 0
    readonly property real dy: (hasTarget || followCursor) ? (gazeTargetY - omiCenterY) : 0
    readonly property real dist: Math.max(1, Math.sqrt(dx * dx + dy * dy))
    // Scale gaze softly with distance so nearby cursor movements feel playful and natural.
    readonly property real gazeScale: hasTarget ? 1.0 : Math.min(1.0, dist / 240.0)

    // ---- Spotlight ring / region highlight at pointed target ----
    Rectangle {
      id: spotlight
      visible: overlay.hasTarget && root.mascot.action === "point"
      x: Math.max(4, Math.min(overlay.screenW - width - 4, overlay.targetX - width / 2))
      y: Math.max(4, Math.min(overlay.screenH - height - 4, overlay.targetY - height / 2))
      width: Math.max(Style.space(38), overlay.targetW)
      height: Math.max(Style.space(38), overlay.targetH)
      radius: root.mascot && root.mascot.w > 60 ? Style.cornerRadius : width / 2
      color: Util.alpha(Commons.Color.accent, 0.10)
      border.width: Math.max(2, Style.space(2))
      border.color: Commons.Color.accent

      Behavior on x { NumberAnimation { duration: Style.duration(220); easing.type: Easing.OutCubic } }
      Behavior on y { NumberAnimation { duration: Style.duration(220); easing.type: Easing.OutCubic } }
      Behavior on width { NumberAnimation { duration: Style.duration(220); easing.type: Easing.OutCubic } }
      Behavior on height { NumberAnimation { duration: Style.duration(220); easing.type: Easing.OutCubic } }

      SequentialAnimation on opacity {
        running: spotlight.visible && !Style.reduceMotion
        loops: Animation.Infinite
        NumberAnimation { from: 0.95; to: 0.45; duration: Style.duration(750); easing.type: Easing.InOutSine }
        NumberAnimation { from: 0.45; to: 0.95; duration: Style.duration(750); easing.type: Easing.InOutSine }
      }
    }

    // ---- Subtle directional pointer line from Omi to target ----
    Canvas {
      id: pointerCanvas
      anchors.fill: parent
      visible: overlay.hasTarget && root.mascot.action === "point"
      onPaint: {
        var ctx = getContext("2d")
        ctx.reset()
        if (!visible) return
        var sx = overlay.omiCenterX
        var sy = overlay.omiCenterY
        var tx = overlay.targetX
        var ty = overlay.targetY
        var vx = tx - sx
        var vy = ty - sy
        var len = Math.sqrt(vx * vx + vy * vy)
        if (len < 36) return
        var ux = vx / len
        var uy = vy / len
        var x1 = sx + ux * (overlay.mascotSize * 0.62)
        var y1 = sy + uy * (overlay.mascotSize * 0.62)
        var x2 = tx - ux * 16
        var y2 = ty - uy * 16
        ctx.strokeStyle = Commons.Color.accent
        ctx.lineWidth = 2.5
        ctx.lineCap = "round"
        ctx.setLineDash([6, 6])
        ctx.beginPath()
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
        ctx.stroke()
        ctx.setLineDash([])
        // Small arrowhead
        var head = 9
        var ang = Math.atan2(y2 - y1, x2 - x1)
        ctx.fillStyle = Commons.Color.accent
        ctx.beginPath()
        ctx.moveTo(x2, y2)
        ctx.lineTo(x2 - head * Math.cos(ang - 0.45), y2 - head * Math.sin(ang - 0.45))
        ctx.lineTo(x2 - head * Math.cos(ang + 0.45), y2 - head * Math.sin(ang + 0.45))
        ctx.closePath()
        ctx.fill()
      }
      Connections {
        target: overlay
        function onOmiCenterXChanged() { pointerCanvas.requestPaint() }
        function onOmiCenterYChanged() { pointerCanvas.requestPaint() }
        function onTargetXChanged() { pointerCanvas.requestPaint() }
        function onTargetYChanged() { pointerCanvas.requestPaint() }
        function onHasTargetChanged() { pointerCanvas.requestPaint() }
      }
    }

    // ---- Omi + Speech Bubble group ----
    Item {
      id: companionGroup
      objectName: "omiCompanion"
      x: Math.max(overlay.margin, Math.min(overlay.screenW - width - overlay.margin, overlay.desiredX))
      y: Math.max(Style.space(44), Math.min(overlay.screenH - height - overlay.margin, overlay.desiredY))
      width: row.implicitWidth
      height: Math.max(overlay.mascotSize, row.implicitHeight)
      opacity: root.shown ? 1 : 0
      scale: root.shown ? 1 : 0.96

      Behavior on x { NumberAnimation { duration: Style.duration(260); easing.type: Easing.Bezier; easing.bezierCurve: Style.easeOut } }
      Behavior on y { NumberAnimation { duration: Style.duration(260); easing.type: Easing.Bezier; easing.bezierCurve: Style.easeOut } }
      Behavior on opacity { NumberAnimation { duration: Style.duration(180); easing.type: Easing.OutCubic } }
      Behavior on scale { NumberAnimation { duration: Style.duration(180); easing.type: Easing.Bezier; easing.bezierCurve: Style.easeOut } }

      RowLayout {
        id: row
        spacing: Style.space(12)
        layoutDirection: overlay.placeRightOfTarget ? Qt.RightToLeft : Qt.LeftToRight

        // Speech bubble card
        BorderSurface {
          id: bubbleCard
          objectName: "omiBubble"
          visible: root.bubbleHeadline !== ""
          Layout.maximumWidth: Style.space(340)
          implicitWidth: Math.min(Style.space(340), Math.max(Style.space(140), bubbleCol.implicitWidth + Style.space(26)))
          implicitHeight: bubbleCol.implicitHeight + Style.space(20)
          color: Util.alpha(Commons.Color.popups.background, 0.96)
          borderSpec: Border.surfaceSpec(
            "popups",
            "border",
            root.confirm ? Commons.Color.urgent : Commons.Color.popups.border,
            Math.max(1, Style.space(2))
          )
          radius: Style.cornerRadius

          ColumnLayout {
            id: bubbleCol
            anchors.fill: parent
            anchors.margins: Style.space(11)
            spacing: Style.space(5)

            // Header row: Omi name + state pill
            RowLayout {
              Layout.fillWidth: true
              spacing: Style.space(8)

              Text {
                textFormat: Text.PlainText
                text: strings.t("title")
                color: Commons.Color.accent
                font.family: Style.font.family
                font.pixelSize: Style.font.caption
                font.bold: true
              }

              Text {
                textFormat: Text.PlainText
                text: strings.t("state." + root.phase).toUpperCase()
                color: root.confirm ? Commons.Color.urgent : Util.alpha(Commons.Color.popups.text, 0.6)
                font.family: Style.font.family
                font.pixelSize: Math.max(9, Style.font.caption - 1)
                font.bold: true
                font.letterSpacing: 0.9
              }

              Item { Layout.fillWidth: true }
            }

            // User utterance preview (subtle)
            Text {
              visible: root.lastUserText !== "" && !root.confirm && root.phase !== "listening"
              Layout.fillWidth: true
              textFormat: Text.PlainText
              text: "“" + root.lastUserText + "”"
              color: Util.alpha(Commons.Color.popups.text, 0.58)
              font.family: Style.font.family
              font.pixelSize: Style.font.caption
              wrapMode: Text.WordWrap
              maximumLineCount: 2
              elide: Text.ElideRight
            }

            // Main speech / action / confirm text
            Text {
              objectName: "omiBubbleText"
              Layout.fillWidth: true
              textFormat: Text.PlainText
              text: root.bubbleHeadline
              color: Commons.Color.popups.text
              font.family: Style.font.family
              font.pixelSize: Style.font.body
              wrapMode: Text.WordWrap
              maximumLineCount: 6
              elide: Text.ElideRight
            }

            // Detail text on confirmation
            Text {
              visible: root.confirm !== null && root.confirm.detail !== ""
              Layout.fillWidth: true
              textFormat: Text.PlainText
              text: root.confirm ? root.confirm.detail : ""
              color: Util.alpha(Commons.Color.popups.text, 0.65)
              font.family: Style.font.family
              font.pixelSize: Style.font.caption
              wrapMode: Text.WrapAnywhere
              maximumLineCount: 4
              elide: Text.ElideRight
            }

            // Interactive Yes / No confirmation buttons right on the speech bubble
            RowLayout {
              visible: root.confirm !== null
              Layout.fillWidth: true
              spacing: Style.space(8)

              Button {
                objectName: "bubbleYesButton"
                text: strings.t("confirm.yes")
                bordered: true
                foreground: Commons.Color.popups.text
                fontFamily: Style.font.family
                onClicked: root.service.answer(true)
              }

              Button {
                objectName: "bubbleNoButton"
                text: strings.t("confirm.no")
                bordered: true
                foreground: Commons.Color.popups.text
                fontFamily: Style.font.family
                onClicked: root.service.answer(false)
              }

              Item { Layout.fillWidth: true }
            }
          }
        }

        // Animated Omi mascot character (adapts live to active Omarchy theme)
        OmiMascot {
          id: mascotAvatar
          objectName: "companionMascot"
          size: overlay.mascotSize
          phase: root.phase
          showHalo: true
          foreground: Commons.Color.popups.text
          accent: Commons.Color.accent
          urgent: Commons.Color.urgent
          background: Commons.Color.background
          lookX: (overlay.hasTarget || overlay.followCursor) ? (overlay.dx / overlay.dist) * overlay.gazeScale : 0
          lookY: (overlay.hasTarget || overlay.followCursor) ? (overlay.dy / overlay.dist) * overlay.gazeScale : 0
          Layout.alignment: Qt.AlignBottom
        }
      }
    }
  }
}
