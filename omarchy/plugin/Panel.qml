import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Quickshell
import qs.Commons
import qs.Commons as Commons
import qs.Ui
import "Model.js" as Model

// Omi in the bar: the animated Omi mascot showing live state, a keyboard-navigable
// conversation panel, and the floating on-screen companion overlay.
// omarchy-shell omi toggle opens the panel from a key.
Panel {
  id: root
  moduleName: "omi"
  ipcTarget: "omi"

  readonly property color foreground: bar ? bar.foreground : Commons.Color.foreground
  readonly property color urgent: bar ? bar.urgent : Commons.Color.urgent
  readonly property color dim: Qt.darker(foreground, 1.55)
  readonly property string fontFamily: bar ? bar.fontFamily : Style.font.family

  Commons.Translations { id: strings; directory: Qt.resolvedUrl("i18n") }

  Service { id: omi }
  // For the QML harness.
  property alias service: omi
  property alias companion: companionOverlay

  readonly property string phase: omi.phase
  readonly property string stateLabel: strings.t("state." + phase)
  readonly property string accessibleLabel: strings.t("orb.label").arg(stateLabel)

  // Floating on-screen Omi companion with live speech bubble and screen pointer.
  CompanionOverlay {
    id: companionOverlay
    service: omi
    strings: strings
  }

  // The keyboard cursor walks these rows; Yes and No exist only while a question waits.
  property int rowIndex: 0
  property bool cursorActive: false
  property bool settingsOpen: false
  readonly property var rows: (omi.confirm ? ["yes", "no"] : []).concat(["input", "stop"])
  readonly property string currentRow: cursorActive ? rows[Math.max(0, Math.min(rowIndex, rows.length - 1))] : ""

  readonly property var installedProviders: {
    var out = []
    var list = omi.providers || []
    for (var i = 0; i < list.length; i++) {
      if (list[i].installed) out.push(list[i].name)
    }
    return out.length > 0 ? out : ["claude", "opencode", "local"]
  }

  function voiceLabel(id) {
    var map = {
      "af_heart": "Heart",
      "af_bella": "Bella",
      "am_adam": "Adam",
      "bf_emma": "Emma",
      "bm_george": "George"
    }
    if (map[id]) return map[id]
    if (!id) return ""
    return id.charAt(0).toUpperCase() + id.slice(1)
  }

  function sensitivityLabel(level) {
    if (level === "high") return strings.t("settings.high")
    if (level === "low") return strings.t("settings.low")
    return strings.t("settings.medium")
  }

  function callModeLabel(mode) {
    if (mode === "continuous") return strings.t("settings.continuous")
    if (mode === "single") return strings.t("settings.single_turn")
    return strings.t("settings.smart")
  }

  function dictationModeLabel(mode) {
    var map = {
      "auto": strings.t("settings.dictate_auto"),
      "raw": strings.t("settings.dictate_raw"),
      "clean": strings.t("settings.dictate_clean"),
      "message": strings.t("settings.dictate_message"),
      "mail": strings.t("settings.dictate_mail"),
      "note": strings.t("settings.dictate_note"),
      "code": strings.t("settings.dictate_code")
    }
    return map[mode] || strings.t("settings.dictate_auto")
  }

  function engineLabel(id) {
    var map = {
      "parakeet": "Parakeet v3",
      "nemotron": "Nemotron",
      "moonshine": "Moonshine",
      "groq": "Groq (Cloud)",
      "deepgram": "Deepgram (Cloud)",
      "openai": "OpenAI (Cloud)",
      "kokoro": "Kokoro",
      "piper": "Piper",
      "elevenlabs": "ElevenLabs (Cloud)"
    }
    if (map[id]) return map[id]
    if (!id) return ""
    return id.charAt(0).toUpperCase() + id.slice(1)
  }

  function speedLabel(speed) {
    return (Math.round(speed * 100) / 100) + "x"
  }

  function setRowCursor(id) {
    cursorActive = true
    rowIndex = Math.max(0, rows.indexOf(id))
  }

  function moveCursor(dy) {
    if (!cursorActive) { cursorActive = true; return }
    rowIndex = Math.max(0, Math.min(rows.length - 1, rowIndex + dy))
  }

  function activate(id) {
    if (id === "yes") omi.answer(true)
    else if (id === "no") omi.answer(false)
    else if (id === "input") focusInput()
    else if (id === "stop") omi.stop()
  }

  function focusInput() {
    cursorActive = false
    input.forceActiveFocus()
  }

  function leaveInput(dy) {
    keyCatcher.forceActiveFocus()
    setRowCursor("input")
    moveCursor(dy)
  }

  function submit() {
    if (omi.say(input.text)) input.text = ""
  }

  onOpenedChanged: if (opened) {
    cursorActive = omi.confirm !== null
    rowIndex = 0
    Qt.callLater(function() { transcriptView.positionViewAtEnd() })
  } else {
    settingsOpen = false
  }

  Connections {
    target: omi
    function onConfirmChanged() {
      if (omi.confirm && root.opened && !input.activeFocus) {
        keyCatcher.forceActiveFocus()
        root.setRowCursor("yes")
      }
    }
  }

  implicitWidth: button.implicitWidth
  implicitHeight: button.implicitHeight
  Accessible.role: Accessible.Button
  Accessible.name: root.accessibleLabel

  BarIconButton {
    id: button
    objectName: "omiButton"
    anchors.fill: parent
    bar: root.bar
    iconComponent: Component { Item {} }
    tooltipText: root.accessibleLabel
    Accessible.role: Accessible.Button
    Accessible.name: root.accessibleLabel
    Accessible.description: strings.t("hint.talk")

    OmiMascot {
      objectName: "barMascot"
      x: (button.width - width) / 2
      y: (button.height - height) / 2
      size: Math.max(18, Style.bar.iconSize > 0 ? Style.bar.iconSize : Style.font.icon)
      phase: root.phase
      foreground: root.barForeground
      accent: Commons.Color.accent
      urgent: root.urgent
      background: Commons.Color.background
      // Follow the user's cursor gently from the top-right bar area.
      lookX: omi.hasCursor ? Math.max(-1, Math.min(1, (omi.cursorX - 1650) / 520)) : 0
      lookY: omi.hasCursor ? Math.max(-1, Math.min(1, (omi.cursorY - 18) / 360)) : 0
    }

    onPressed: function(buttonCode) { root.toggle() }
  }

  KeyboardPanel {
    id: panel
    anchorItem: button
    owner: root
    bar: root.bar
    open: root.opened
    focusTarget: omi.confirm || !omi.connected ? keyCatcher : input
    contentWidth: panel.fittedContentWidth(Style.space(380))
    contentHeight: panel.fittedContentHeight(column.implicitHeight, Style.space(620))

    PanelKeyCatcher {
      id: keyCatcher
      anchors.fill: parent
      clip: true
      blocked: input.activeFocus
      onMoveRequested: function(dx, dy) { if (dy !== 0) root.moveCursor(dy) }
      onActivateRequested: if (root.cursorActive) root.activate(root.currentRow)
      onCloseRequested: root.close()
      onTabRequested: function(direction) { root.switchPanel(direction) }
      onTextKey: function(t) {
        var key = t.toLowerCase()
        if (key === "y" && omi.confirm) omi.answer(true)
        else if (key === "n" && omi.confirm) omi.answer(false)
        else if (key === "i") root.focusInput()
      }

      ColumnLayout {
        id: column
        anchors.fill: parent
        anchors.leftMargin: Style.space(2)
        anchors.rightMargin: Style.space(2)
        spacing: Style.space(10)

        // ---- Hero: the animated Omi mascot, name and state ----------------
        RowLayout {
          Layout.fillWidth: true
          spacing: Style.space(14)

          OmiMascot {
            objectName: "panelMascot"
            size: Math.max(36, Style.font.display)
            phase: root.phase
            foreground: Commons.Color.popups.text
            accent: Commons.Color.accent
            urgent: root.urgent
            background: Commons.Color.background
            lookX: omi.hasCursor ? Math.max(-1, Math.min(1, (omi.cursorX - 1600) / 480)) : 0
            lookY: omi.hasCursor ? Math.max(-1, Math.min(1, (omi.cursorY - 120) / 380)) : 0
            Layout.alignment: Qt.AlignVCenter
          }

          ColumnLayout {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            spacing: Style.space(2)

            Text {
              textFormat: Text.PlainText
              text: strings.t("title")
              color: root.foreground
              font.family: root.fontFamily
              font.pixelSize: Style.font.subtitle
              font.bold: true
            }

            Text {
              objectName: "stateText"
              Layout.fillWidth: true
              textFormat: Text.PlainText
              text: (omi.notice !== "" ? omi.notice : root.stateLabel).toUpperCase()
              color: omi.notice !== "" ? root.urgent : root.dim
              font.family: root.fontFamily
              font.pixelSize: Style.font.caption
              font.bold: true
              font.letterSpacing: 1.2
              elide: Text.ElideRight
              Accessible.role: Accessible.StaticText
              Accessible.name: text
            }

            Text {
              Layout.fillWidth: true
              textFormat: Text.PlainText
              text: root.phase === "offline" ? strings.t("hint.offline") : (omi.wakeEnabled ? strings.t("hint.wake") : strings.t("hint.talk"))
              color: root.dim
              font.family: root.fontFamily
              font.pixelSize: Style.font.caption
              wrapMode: Text.WordWrap
            }
          }
        }

        PanelSeparator { Layout.fillWidth: true; foreground: root.foreground }

        // ---- Inline health / setup notice (missing CLI login, mic, model) --
        BorderSurface {
          objectName: "healthCard"
          visible: omi.healthError !== ""
          Layout.fillWidth: true
          implicitHeight: healthText.implicitHeight + Style.space(12)
          color: "transparent"
          borderSpec: Border.controlSpec("normal", root.urgent, root.urgent)
          radius: Style.cornerRadius

          Text {
            id: healthText
            anchors.fill: parent
            anchors.margins: Style.space(6)
            textFormat: Text.PlainText
            text: omi.healthError
            color: root.urgent
            font.family: root.fontFamily
            font.pixelSize: Style.font.caption
            wrapMode: Text.WordWrap
            Accessible.role: Accessible.StaticText
            Accessible.name: text
          }
        }

        // ---- The conversation ---------------------------------------------
        Text {
          visible: omi.lines.length === 0
          Layout.fillWidth: true
          textFormat: Text.PlainText
          text: strings.t("transcript.empty")
          color: root.dim
          font.family: root.fontFamily
          font.pixelSize: Style.font.body
        }

        ListView {
          id: transcriptView
          objectName: "transcript"
          visible: omi.lines.length > 0
          Layout.fillWidth: true
          Layout.fillHeight: true
          Layout.minimumHeight: Math.min(contentHeight, Style.space(72))
          Layout.preferredHeight: Math.min(contentHeight, root.settingsOpen ? Style.space(150) : Style.space(260))
          Layout.maximumHeight: Math.max(Style.space(72), contentHeight)
          clip: true
          spacing: Style.space(8)
          boundsBehavior: Flickable.StopAtBounds
          model: omi.lines
          onCountChanged: Qt.callLater(positionViewAtEnd)
          ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }
          Accessible.role: Accessible.List
          Accessible.name: strings.t("title")

          delegate: Column {
            required property var modelData
            width: transcriptView.width - Style.space(8)
            spacing: Style.space(1)
            Accessible.role: Accessible.StaticText
            Accessible.name: strings.t("role." + modelData.role) + ": " + modelData.text

            Text {
              textFormat: Text.PlainText
              text: strings.t("role." + modelData.role)
              color: modelData.role === "user" ? root.dim : Commons.Color.accent
              font.family: root.fontFamily
              font.pixelSize: Style.font.caption
              font.bold: true
            }

            Text {
              width: parent.width
              textFormat: Text.PlainText
              text: modelData.text
              color: modelData.role === "progress" ? root.dim : root.foreground
              font.family: root.fontFamily
              font.pixelSize: Style.font.body
              wrapMode: Text.Wrap
            }
          }
        }

        // ---- A question waiting for yes or no -----------------------------
        BorderSurface {
          objectName: "confirmCard"
          visible: omi.confirm !== null
          Layout.fillWidth: true
          implicitHeight: confirmColumn.implicitHeight + Style.space(20)
          color: "transparent"
          borderSpec: Border.controlSpec("normal", root.urgent, root.urgent)
          radius: Style.cornerRadius

          ColumnLayout {
            id: confirmColumn
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            anchors.margins: Style.space(10)
            spacing: Style.space(6)

            Text {
              textFormat: Text.PlainText
              text: strings.t("confirm.title").toUpperCase()
              color: root.urgent
              font.family: root.fontFamily
              font.pixelSize: Style.font.caption
              font.bold: true
            }

            Text {
              Layout.fillWidth: true
              textFormat: Text.PlainText
              text: omi.confirm ? omi.confirm.question : ""
              color: root.foreground
              font.family: root.fontFamily
              font.pixelSize: Style.font.body
              wrapMode: Text.Wrap
              Accessible.role: Accessible.StaticText
              Accessible.name: text
            }

            Text {
              visible: text !== ""
              Layout.fillWidth: true
              textFormat: Text.PlainText
              text: omi.confirm ? omi.confirm.detail : ""
              color: root.dim
              font.family: root.fontFamily
              font.pixelSize: Style.font.bodySmall
              wrapMode: Text.WrapAnywhere
              maximumLineCount: 8
              elide: Text.ElideRight
            }

            RowLayout {
              Layout.fillWidth: true
              spacing: Style.space(8)

              Button {
                objectName: "yesButton"
                text: strings.t("confirm.yes")
                bordered: true
                hasCursor: root.currentRow === "yes"
                foreground: root.foreground
                fontFamily: root.fontFamily
                onHovered: function(on) { if (on) root.setRowCursor("yes") }
                onClicked: omi.answer(true)
                Accessible.role: Accessible.Button
                Accessible.name: text
              }

              Button {
                objectName: "noButton"
                text: strings.t("confirm.no")
                bordered: true
                hasCursor: root.currentRow === "no"
                foreground: root.foreground
                fontFamily: root.fontFamily
                onHovered: function(on) { if (on) root.setRowCursor("no") }
                onClicked: omi.answer(false)
                Accessible.role: Accessible.Button
                Accessible.name: text
              }

              Item { Layout.fillWidth: true }

              Text {
                textFormat: Text.PlainText
                text: strings.t("confirm.keys")
                color: root.dim
                font.family: root.fontFamily
                font.pixelSize: Style.font.caption
              }
            }
          }
        }

        // ---- Type a request -----------------------------------------------
        TextField {
          id: input
          objectName: "requestField"
          Layout.fillWidth: true
          enabled: omi.connected
          placeholderText: strings.t("input.placeholder")
          hasCursor: root.currentRow === "input"
          foreground: root.foreground
          font.family: root.fontFamily
          maximumLength: Model.MAX_REQUEST
          Accessible.name: strings.t("input.label")

          Keys.onPressed: function(event) {
            if (event.key === Qt.Key_Return || event.key === Qt.Key_Enter) {
              root.submit()
              event.accepted = true
            } else if (event.key === Qt.Key_Escape) {
              root.close()
              event.accepted = true
            } else if (event.key === Qt.Key_Up || event.key === Qt.Key_Down) {
              root.leaveInput(event.key === Qt.Key_Up ? -1 : 1)
              event.accepted = true
            }
          }
        }

        // ---- Stop ---------------------------------------------------------
        RowLayout {
          Layout.fillWidth: true
          spacing: Style.space(8)

          Button {
            objectName: "stopButton"
            text: strings.t("stop")
            iconText: "\u{f04db}"
            bordered: true
            enabled: omi.connected
            hasCursor: root.currentRow === "stop"
            foreground: root.foreground
            fontFamily: root.fontFamily
            onHovered: function(on) { if (on) root.setRowCursor("stop") }
            onClicked: omi.stop()
            Accessible.role: Accessible.Button
            Accessible.name: text
          }

          Button {
            objectName: "wakeButton"
            text: omi.wakeEnabled ? strings.t("wake.on") : strings.t("wake.off")
            bordered: true
            enabled: omi.connected
            foreground: omi.wakeEnabled ? Commons.Color.accent : root.dim
            fontFamily: root.fontFamily
            onClicked: omi.toggleWake()
            Accessible.role: Accessible.Button
            Accessible.name: text
          }

          Item { Layout.fillWidth: true }

          Button {
            objectName: "settingsButton"
            text: strings.t("settings.button")
            bordered: true
            enabled: omi.connected
            foreground: root.settingsOpen ? Commons.Color.accent : root.dim
            fontFamily: root.fontFamily
            onClicked: root.settingsOpen = !root.settingsOpen
            Accessible.role: Accessible.Button
            Accessible.name: strings.t("settings.title")
          }
        }

        // ---- Collapsible Settings drawer (scrollable, persists to config.toml) -
        BorderSurface {
          id: settingsCard
          objectName: "settingsCard"
          visible: root.settingsOpen && omi.connected
          Layout.fillWidth: true
          Layout.preferredHeight: Math.min(settingsCol.implicitHeight + Style.space(18), Style.space(270))
          implicitHeight: Layout.preferredHeight
          color: "transparent"
          borderSpec: Border.controlSpec("normal", root.dim, Commons.Color.popups.border)
          radius: Style.cornerRadius

          Flickable {
            anchors.fill: parent
            anchors.margins: Style.space(9)
            clip: true
            contentWidth: width
            contentHeight: settingsCol.implicitHeight
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }

            ColumnLayout {
              id: settingsCol
              width: parent.width
              spacing: Style.space(8)

              // Provider picker
              ColumnLayout {
                Layout.fillWidth: true
                spacing: Style.space(4)

                Text {
                  textFormat: Text.PlainText
                  text: strings.t("settings.provider").toUpperCase()
                  color: root.dim
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.caption
                  font.bold: true
                  font.letterSpacing: 0.8
                }

                Flow {
                  Layout.fillWidth: true
                  spacing: Style.space(6)

                  Repeater {
                    model: root.installedProviders
                    delegate: Button {
                      required property string modelData
                      text: modelData
                      bordered: true
                      foreground: omi.talker === modelData ? Commons.Color.accent : root.foreground
                      fontFamily: root.fontFamily
                      onClicked: omi.setProvider(modelData)
                      Accessible.role: Accessible.Button
                      Accessible.name: strings.t("settings.provider") + ": " + modelData
                    }
                  }
                }
              }

              // STT engine selector
              RowLayout {
                Layout.fillWidth: true
                spacing: Style.space(8)

                Text {
                  textFormat: Text.PlainText
                  text: strings.t("settings.stt")
                  color: root.dim
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.caption
                }

                Item { Layout.fillWidth: true }

                Button {
                  objectName: "sttButton"
                  text: root.engineLabel(omi.sttEngine)
                  bordered: true
                  foreground: root.foreground
                  fontFamily: root.fontFamily
                  onClicked: omi.cycleStt()
                  Accessible.role: Accessible.Button
                  Accessible.name: strings.t("settings.stt") + ": " + text
                }
              }

              // TTS engine selector
              RowLayout {
                Layout.fillWidth: true
                spacing: Style.space(8)

                Text {
                  textFormat: Text.PlainText
                  text: strings.t("settings.tts")
                  color: root.dim
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.caption
                }

                Item { Layout.fillWidth: true }

                Button {
                  objectName: "ttsButton"
                  text: root.engineLabel(omi.ttsEngine)
                  bordered: true
                  foreground: root.foreground
                  fontFamily: root.fontFamily
                  onClicked: omi.cycleTts()
                  Accessible.role: Accessible.Button
                  Accessible.name: strings.t("settings.tts") + ": " + text
                }
              }

              // Voice selector (shown whenever the active TTS engine has voice presets)
              RowLayout {
                visible: omi.voices && omi.voices.length > 0
                Layout.fillWidth: true
                spacing: Style.space(8)

                Text {
                  textFormat: Text.PlainText
                  text: strings.t("settings.voice")
                  color: root.dim
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.caption
                }

                Item { Layout.fillWidth: true }

                Button {
                  objectName: "voiceButton"
                  text: root.voiceLabel(omi.voice)
                  bordered: true
                  foreground: root.foreground
                  fontFamily: root.fontFamily
                  onClicked: omi.cycleVoice()
                  Accessible.role: Accessible.Button
                  Accessible.name: strings.t("settings.voice") + ": " + text
                }
              }

              // Speech speed selector
              RowLayout {
                Layout.fillWidth: true
                spacing: Style.space(8)

                Text {
                  textFormat: Text.PlainText
                  text: strings.t("settings.speed")
                  color: root.dim
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.caption
                }

                Item { Layout.fillWidth: true }

                Button {
                  objectName: "speedButton"
                  text: root.speedLabel(omi.ttsSpeed)
                  bordered: true
                  foreground: root.foreground
                  fontFamily: root.fontFamily
                  onClicked: omi.cycleSpeed()
                  Accessible.role: Accessible.Button
                  Accessible.name: strings.t("settings.speed") + ": " + text
                }
              }

              // Dictation mode (Hex / Superwhisper style: Super+Alt+D)
              RowLayout {
                Layout.fillWidth: true
                spacing: Style.space(8)

                Text {
                  textFormat: Text.PlainText
                  text: strings.t("settings.dictation")
                  color: root.dim
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.caption
                }

                Item { Layout.fillWidth: true }

                Button {
                  objectName: "dictationModeButton"
                  text: root.dictationModeLabel(omi.dictationMode)
                  bordered: true
                  foreground: Commons.Color.accent
                  fontFamily: root.fontFamily
                  onClicked: omi.cycleDictationMode()
                  Accessible.role: Accessible.Button
                  Accessible.name: strings.t("settings.dictation") + ": " + text
                }
              }

              // Pause media while dictating
              RowLayout {
                Layout.fillWidth: true
                spacing: Style.space(8)

                Text {
                  textFormat: Text.PlainText
                  text: strings.t("settings.pause_media")
                  color: root.dim
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.caption
                }

                Item { Layout.fillWidth: true }

                Button {
                  objectName: "pauseMediaButton"
                  text: omi.dictationPauseMedia ? strings.t("settings.on") : strings.t("settings.off")
                  bordered: true
                  foreground: omi.dictationPauseMedia ? Commons.Color.accent : root.dim
                  fontFamily: root.fontFamily
                  onClicked: omi.togglePauseMedia()
                  Accessible.role: Accessible.Button
                  Accessible.name: strings.t("settings.pause_media") + ": " + text
                }
              }

              // Call mode toggle
              RowLayout {
                Layout.fillWidth: true
                spacing: Style.space(8)

                Text {
                  textFormat: Text.PlainText
                  text: strings.t("settings.call_mode")
                  color: root.dim
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.caption
                }

                Item { Layout.fillWidth: true }

                Button {
                  objectName: "callModeButton"
                  text: root.callModeLabel(omi.wakeCallMode)
                  bordered: true
                  foreground: omi.wakeCallMode !== "single" ? Commons.Color.accent : root.dim
                  fontFamily: root.fontFamily
                  onClicked: omi.cycleCallMode()
                  Accessible.role: Accessible.Button
                  Accessible.name: strings.t("settings.call_mode") + ": " + text
                }
              }

              // Wake word sensitivity selector
              RowLayout {
                Layout.fillWidth: true
                spacing: Style.space(8)

                Text {
                  textFormat: Text.PlainText
                  text: strings.t("settings.sensitivity")
                  color: root.dim
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.caption
                }

                Item { Layout.fillWidth: true }

                Button {
                  objectName: "sensitivityButton"
                  text: root.sensitivityLabel(omi.wakeSensitivity)
                  bordered: true
                  foreground: root.foreground
                  fontFamily: root.fontFamily
                  onClicked: omi.cycleSensitivity()
                  Accessible.role: Accessible.Button
                  Accessible.name: strings.t("settings.sensitivity") + ": " + text
                }
              }

              // Cute sound effects toggle
              RowLayout {
                Layout.fillWidth: true
                spacing: Style.space(8)

                Text {
                  textFormat: Text.PlainText
                  text: strings.t("settings.sounds")
                  color: root.dim
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.caption
                }

                Item { Layout.fillWidth: true }

                Button {
                  objectName: "soundsButton"
                  text: omi.soundsEnabled ? strings.t("settings.on") : strings.t("settings.off")
                  bordered: true
                  foreground: omi.soundsEnabled ? Commons.Color.accent : root.dim
                  fontFamily: root.fontFamily
                  onClicked: omi.toggleSounds()
                  Accessible.role: Accessible.Button
                  Accessible.name: strings.t("settings.sounds") + ": " + text
                }
              }
            }
          }
        }
      }
    }
  }
}
