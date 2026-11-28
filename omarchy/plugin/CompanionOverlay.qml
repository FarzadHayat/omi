import QtQuick
import QtQuick.Layouts
import Quickshell
import Quickshell.Wayland
import qs.Commons
import qs.Commons as Commons
import qs.Ui

// Floating on-screen Omi companion: pops up when you interact with Omi, shows a live
// speech bubble of what Omi is saying, and, when asked about something on screen, stays
// anchored in the bottom-right corner and shows a large arrow pointing at the target
// (plus a spotlight ring around it).
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
  readonly property bool tuning: service ? service.tuning : false

  // Look to talk tuning: the bubble moves to the middle of the screen, large, and stays
  // up for the whole setup (plus long enough to read the last line after it ends).
  property bool tuneLingering: false
  readonly property bool tuneShown: tuning || tuneLingering

  onTuningChanged: {
    if (tuning) {
      tuneLingerTimer.stop()
      tuneLingering = false
    } else if (service && service.tuneText !== "") {
      tuneLingerTimer.interval = Math.max(6000, service.tuneText.length * 70)
      tuneLingering = true
      tuneLingerTimer.restart()
    }
  }

  Timer {
    id: tuneLingerTimer
    onTriggered: root.tuneLingering = false
  }

  readonly property bool activeNow: phase === "listening" || phase === "thinking"
    || phase === "speaking" || phase === "acting" || phase === "confirming" || mascot !== null

  property bool lingering: false
  readonly property bool shown: activeNow || lingering || tuneShown
  // The bubble catches clicks for Yes / No, and for Cancel while tuning.
  readonly property bool interactive: confirm !== null || tuning
  readonly property bool revealed: overlay.contentRevealed

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
    if (tuneShown) return service.tuneText
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
    // Visual-only unless a Yes/No confirmation or tuning is on (then only the bubble catches clicks).
    mask: Region {
      item: root.interactive ? bubbleCard : null
    }

    readonly property real screenW: overlay.width > 0 ? overlay.width : 1920
    readonly property real screenH: overlay.height > 0 ? overlay.height : 1080
    readonly property real margin: Style.space(26)
    readonly property real mascotSize: Style.space(58)
    readonly property real bubbleGap: Style.space(12)

    // Only animate the spotlight ring once the fullscreen window is revealed and settled,
    // so nothing flies in from the unconfigured window size on first appearance.
    property bool animateMotion: false

    onContentRevealedChanged: {
      if (!contentRevealed) {
        animateMotion = false
        if (spotlightXAnim.running) spotlightXAnim.complete()
        if (spotlightYAnim.running) spotlightYAnim.complete()
        if (spotlightWAnim.running) spotlightWAnim.complete()
        if (spotlightHAnim.running) spotlightHAnim.complete()
      } else {
        Qt.callLater(function() {
          if (overlay.contentRevealed) overlay.animateMotion = true
        })
      }
    }

    // Target point when Omi is pointing/moving on screen. Omi itself never moves.
    readonly property bool hasTarget: root.mascot !== null
    readonly property real targetX: hasTarget ? Math.max(12, Math.min(screenW - 12, root.mascot.x)) : 0
    readonly property real targetY: hasTarget ? Math.max(12, Math.min(screenH - 12, root.mascot.y)) : 0
    readonly property real targetW: hasTarget && root.mascot.w > 0 ? Math.min(root.mascot.w, screenW - 24) : Style.space(44)
    readonly property real targetH: hasTarget && root.mascot.h > 0 ? Math.min(root.mascot.h, screenH - 24) : Style.space(44)

    // Omi stays anchored in the bottom-right corner; the bubble sits to its left.
    readonly property real mascotX: screenW - margin - mascotSize
    readonly property real mascotY: screenH - margin - mascotSize

    // Gaze vector: look directly at pointed target when pointing, otherwise follow the live mouse cursor!
    readonly property real omiCenterX: mascotAvatar.x + mascotAvatar.width / 2
    readonly property real omiCenterY: mascotAvatar.y + mascotAvatar.height / 2
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

      property bool animateMove: false
      onVisibleChanged: {
        if (!visible) {
          animateMove = false
          if (spotlightXAnim.running) spotlightXAnim.complete()
          if (spotlightYAnim.running) spotlightYAnim.complete()
          if (spotlightWAnim.running) spotlightWAnim.complete()
          if (spotlightHAnim.running) spotlightHAnim.complete()
        } else {
          Qt.callLater(function() {
            if (spotlight.visible && overlay.contentRevealed) spotlight.animateMove = true
          })
        }
      }

      Behavior on x { enabled: overlay.animateMotion && spotlight.animateMove; NumberAnimation { id: spotlightXAnim; duration: Style.duration(220); easing.type: Easing.OutCubic } }
      Behavior on y { enabled: overlay.animateMotion && spotlight.animateMove; NumberAnimation { id: spotlightYAnim; duration: Style.duration(220); easing.type: Easing.OutCubic } }
      Behavior on width { enabled: overlay.animateMotion && spotlight.animateMove; NumberAnimation { id: spotlightWAnim; duration: Style.duration(220); easing.type: Easing.OutCubic } }
      Behavior on height { enabled: overlay.animateMotion && spotlight.animateMove; NumberAnimation { id: spotlightHAnim; duration: Style.duration(220); easing.type: Easing.OutCubic } }

      SequentialAnimation on opacity {
        running: spotlight.visible && !Style.reduceMotion
        loops: Animation.Infinite
        NumberAnimation { from: 0.95; to: 0.45; duration: Style.duration(750); easing.type: Easing.InOutSine }
        NumberAnimation { from: 0.45; to: 0.95; duration: Style.duration(750); easing.type: Easing.InOutSine }
      }
    }

    // ---- Large arrow from Omi to the target ----
    Canvas {
      id: pointerCanvas
      objectName: "omiPointerArrow"
      anchors.fill: parent
      visible: overlay.hasTarget
      readonly property real shaftWidth: Style.space(14)
      readonly property real headLength: Style.space(52)
      readonly property real headWidth: Style.space(52)
      // True when the target is far enough from Omi for an arrow to be drawn.
      readonly property bool drawn: overlay.hasTarget && overlay.dist > overlay.mascotSize * 0.62 + headLength + Style.space(24)

      onPaint: {
        var ctx = getContext("2d")
        ctx.reset()
        if (!visible || !drawn) return
        var sx = overlay.omiCenterX
        var sy = overlay.omiCenterY
        var ux = (overlay.targetX - sx) / overlay.dist
        var uy = (overlay.targetY - sy) / overlay.dist
        // Stop at the edge of the spotlight ring so the tip touches the target, not its centre.
        var ringRadius = Math.min(Math.max(Style.space(38), Math.max(overlay.targetW, overlay.targetH)) / 2, overlay.dist * 0.4)
        var x1 = sx + ux * (overlay.mascotSize * 0.62)
        var y1 = sy + uy * (overlay.mascotSize * 0.62)
        var tipX = overlay.targetX - ux * (ringRadius + Style.space(6))
        var tipY = overlay.targetY - uy * (ringRadius + Style.space(6))
        var baseX = tipX - ux * headLength
        var baseY = tipY - uy * headLength
        // Perpendicular unit vector.
        var px = -uy
        var py = ux
        var sw = shaftWidth / 2
        var hw = headWidth / 2
        ctx.beginPath()
        ctx.moveTo(x1 + px * sw, y1 + py * sw)
        ctx.lineTo(baseX + px * sw, baseY + py * sw)
        ctx.lineTo(baseX + px * hw, baseY + py * hw)
        ctx.lineTo(tipX, tipY)
        ctx.lineTo(baseX - px * hw, baseY - py * hw)
        ctx.lineTo(baseX - px * sw, baseY - py * sw)
        ctx.lineTo(x1 - px * sw, y1 - py * sw)
        ctx.closePath()
        ctx.fillStyle = Commons.Color.accent
        ctx.fill()
        ctx.lineJoin = "round"
        ctx.lineWidth = Math.max(2, Style.space(3))
        ctx.strokeStyle = Commons.Color.background
        ctx.stroke()
      }

      onDrawnChanged: requestPaint()
      onVisibleChanged: requestPaint()
      onWidthChanged: requestPaint()
      onHeightChanged: requestPaint()

      Connections {
        target: overlay
        function onTargetXChanged() { pointerCanvas.requestPaint() }
        function onTargetYChanged() { pointerCanvas.requestPaint() }
        function onTargetWChanged() { pointerCanvas.requestPaint() }
        function onTargetHChanged() { pointerCanvas.requestPaint() }
        function onHasTargetChanged() { pointerCanvas.requestPaint() }
        function onMascotXChanged() { pointerCanvas.requestPaint() }
        function onMascotYChanged() { pointerCanvas.requestPaint() }
      }

      SequentialAnimation on opacity {
        running: pointerCanvas.visible && !Style.reduceMotion
        loops: Animation.Infinite
        NumberAnimation { from: 1.0; to: 0.7; duration: Style.duration(650); easing.type: Easing.InOutSine }
        NumberAnimation { from: 0.7; to: 1.0; duration: Style.duration(650); easing.type: Easing.InOutSine }
      }
    }

    // ---- Speech Bubble Card (anchored beside Omi; grows upward/leftward without moving Omi) ----
    // While tuning look to talk it is a large card centred on screen instead.
    BorderSurface {
      id: bubbleCard
      objectName: "omiBubble"
      readonly property bool big: root.tuneShown
      visible: root.bubbleHeadline !== ""
      width: big ? Math.min(overlay.screenW - 2 * overlay.margin, Style.space(760))
                 : Math.min(Style.space(340), Math.max(Style.space(140), bubbleCol.implicitWidth + Style.space(26)))
      height: bubbleCol.implicitHeight + (big ? Style.space(48) : Style.space(20))
      x: big ? (overlay.screenW - width) / 2 : Math.max(overlay.margin, mascotAvatar.x - width - overlay.bubbleGap)
      y: big ? Math.max(overlay.margin, (overlay.screenH - height) / 2)
             : Math.max(Style.space(44), Math.min(overlay.screenH - height - overlay.margin, mascotAvatar.y + mascotAvatar.height - height))
      opacity: overlay.contentRevealed ? 1 : 0
      scale: overlay.contentRevealed ? 1 : 0.96
      color: Util.alpha(Commons.Color.popups.background, 0.96)
      borderSpec: Border.surfaceSpec(
        "popups",
        "border",
        root.confirm ? Commons.Color.urgent : Commons.Color.popups.border,
        Math.max(1, Style.space(2))
      )
      radius: Style.cornerRadius

      Behavior on opacity { NumberAnimation { duration: Style.duration(180); easing.type: Easing.OutCubic } }
      Behavior on scale { NumberAnimation { duration: Style.duration(180); easing.type: Easing.Bezier; easing.bezierCurve: Style.easeOut } }

      ColumnLayout {
        id: bubbleCol
        anchors.fill: parent
        anchors.margins: bubbleCard.big ? Style.space(24) : Style.space(11)
        spacing: bubbleCard.big ? Style.space(14) : Style.space(5)

        // Header row: Omi name + state pill
        RowLayout {
          Layout.fillWidth: true
          spacing: Style.space(8)

          Text {
            objectName: "omiBubbleTitle"
            Layout.fillWidth: bubbleCard.big
            textFormat: Text.PlainText
            text: bubbleCard.big && root.service.tuneTitle !== "" ? root.service.tuneTitle : strings.t("title")
            color: Commons.Color.accent
            font.family: Style.font.family
            font.pixelSize: bubbleCard.big ? Style.font.subtitle : Style.font.caption
            font.bold: true
            wrapMode: Text.WordWrap
          }

          Text {
            visible: !bubbleCard.big
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
          visible: root.lastUserText !== "" && !root.confirm && root.phase !== "listening" && !bubbleCard.big
          Layout.fillWidth: true
          textFormat: Text.PlainText
          text: "“" + root.lastUserText + "”"
          color: Util.alpha(Commons.Color.popups.text, 0.58)
          font.family: Style.font.family
          font.pixelSize: Style.font.caption
          wrapMode: Text.WordWrap
          maximumLineCount: 4
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
          font.pixelSize: bubbleCard.big ? Math.round(Style.font.body * 1.7) : Style.font.body
          lineHeight: bubbleCard.big ? 1.15 : 1.0
          wrapMode: Text.WordWrap
          maximumLineCount: bubbleCard.big ? 16 : 6
          elide: Text.ElideRight
        }

        // Tuning: a warning such as "I can't see your face"
        Text {
          objectName: "tuneNote"
          visible: bubbleCard.big && root.tuning && root.service.tuneNote !== ""
          Layout.fillWidth: true
          textFormat: Text.PlainText
          text: root.service ? root.service.tuneNote : ""
          color: Commons.Color.urgent
          font.family: Style.font.family
          font.pixelSize: Style.font.body
          font.bold: true
          wrapMode: Text.WordWrap
        }

        // Tuning: how much of the current step is left
        Rectangle {
          id: tuneTrack
          objectName: "tuneTimer"
          visible: bubbleCard.big && root.tuning && root.service.tuneSeconds > 0 && !root.confirm
          Layout.fillWidth: true
          Layout.preferredHeight: Math.max(4, Style.space(6))
          radius: height / 2
          color: Util.alpha(Commons.Color.popups.text, 0.15)

          property real done: 0  // share of the step already gone, 0..1

          Rectangle {
            height: parent.height
            width: parent.width * tuneTrack.done
            radius: parent.radius
            color: Commons.Color.accent
          }

          NumberAnimation {
            id: tuneFillAnim
            target: tuneTrack
            property: "done"
            to: 1
          }

          function restart() {
            tuneFillAnim.stop()
            var left = root.service ? root.service.tuneSeconds : 0
            var total = root.service ? Math.max(left, root.service.tuneTotal) : 0
            if (left <= 0 || total <= 0) { done = 0; return }
            done = 1 - left / total
            tuneFillAnim.from = done
            tuneFillAnim.duration = left * 1000
            tuneFillAnim.start()
          }

          Connections {
            target: root.service
            function onTuneSerialChanged() { tuneTrack.restart() }
          }
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
            Accessible.role: Accessible.Button
            Accessible.name: text
          }

          Button {
            objectName: "bubbleNoButton"
            text: strings.t("confirm.no")
            bordered: true
            foreground: Commons.Color.popups.text
            fontFamily: Style.font.family
            onClicked: root.service.answer(false)
            Accessible.role: Accessible.Button
            Accessible.name: text
          }

          Item { Layout.fillWidth: true }
        }

        // Tuning: stop part way (the shortcut does the same without the mouse)
        RowLayout {
          visible: bubbleCard.big && root.tuning
          Layout.fillWidth: true
          spacing: Style.space(10)

          Button {
            objectName: "tuneCancelButton"
            text: strings.t("tune.cancel")
            bordered: true
            foreground: Commons.Color.urgent
            fontFamily: Style.font.family
            onClicked: root.service.cancelLookTune()
            Accessible.role: Accessible.Button
            Accessible.name: text
          }

          Text {
            textFormat: Text.PlainText
            text: strings.t("tune.cancel_hint")
            color: Util.alpha(Commons.Color.popups.text, 0.6)
            font.family: Style.font.family
            font.pixelSize: Style.font.caption
          }

          Item { Layout.fillWidth: true }
        }
      }
    }

    // ---- Animated Omi Mascot (always anchored in the bottom-right corner) ----
    OmiMascot {
      id: mascotAvatar
      objectName: "companionMascot"
      x: overlay.mascotX
      y: overlay.mascotY
      opacity: overlay.contentRevealed ? 1 : 0
      scale: overlay.contentRevealed ? 1 : 0.96
      size: overlay.mascotSize
      phase: root.phase
      showHalo: true
      foreground: Commons.Color.popups.text
      accent: Commons.Color.accent
      urgent: Commons.Color.urgent
      background: Commons.Color.background
      lookX: (overlay.hasTarget || overlay.followCursor) ? (overlay.dx / overlay.dist) * overlay.gazeScale : 0
      lookY: (overlay.hasTarget || overlay.followCursor) ? (overlay.dy / overlay.dist) * overlay.gazeScale : 0

      Behavior on opacity { NumberAnimation { duration: Style.duration(180); easing.type: Easing.OutCubic } }
      Behavior on scale { NumberAnimation { duration: Style.duration(180); easing.type: Easing.Bezier; easing.bezierCurve: Style.easeOut } }
    }
  }
}
