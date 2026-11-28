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
  // Dropdown lists draw inside the panel, where the compositor cannot blur what
  // is under them, so a theme's see-through popup color is made solid here or
  // the settings below show through the options.
  readonly property color dropdownBackground: Qt.rgba(Commons.Color.popups.background.r,
    Commons.Color.popups.background.g, Commons.Color.popups.background.b, 1)

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
  // Stop only shows while there is something to stop.
  readonly property bool busy: ["listening", "thinking", "speaking", "acting", "confirming"].indexOf(phase) >= 0
  readonly property var rows: (omi.confirm ? ["yes", "no"] : []).concat(busy ? ["input", "stop", "wake", "settings"] : ["input", "wake", "settings"])
  readonly property string currentRow: cursorActive ? rows[Math.max(0, Math.min(rowIndex, rows.length - 1))] : ""

  readonly property var installedProviders: {
    var out = []
    var list = omi.providers || []
    for (var i = 0; i < list.length; i++) {
      if (list[i].installed) out.push(list[i].name)
    }
    return out.length > 0 ? out : ["claude", "opencode", "local"]
  }

  property bool customModel: false

  // Brain choices from the daemon, each labelled by where it runs.
  readonly property var providerOptions: {
    var list = (omi.brain && omi.brain.providers) || []
    if (list.length > 0) return list
    return installedProviders.map(function(n) { return { value: n, label: n } })
  }

  readonly property var modelOptions: {
    var models = (omi.brain && omi.brain.models && omi.brain.models[omi.talker]) || []
    return models.concat([{ value: "__custom__", label: strings.t("settings.custom") }])
  }

  readonly property var workerOptions: (omi.brain && omi.brain.worker_providers) || []
  readonly property string worker: (omi.brain && omi.brain.worker) || ""
  readonly property string workerModel: (omi.brain && omi.brain.worker_model) || ""
  readonly property var workerModelOptions: (omi.brain && omi.brain.worker_models && omi.brain.worker_models[worker]) || []
  readonly property var workerFallbackList: (omi.brain && omi.brain.worker_fallbacks) || []
  property bool customWorkerBackup: false

  readonly property var workerBackupChoices: {
    var out = []
    var list = workerOptions
    for (var i = 0; i < list.length; i++) {
      var v = list[i].value
      if (v !== "" && v !== "off" && v !== worker && workerFallbackList.indexOf(v) < 0)
        out.push({ value: v, label: list[i].label })
    }
    if (worker !== "" && worker !== "off") {
      var wLabel = worker
      for (var j = 0; j < list.length; j++) if (list[j].value === worker) wLabel = list[j].label
      var models = workerModelOptions
      for (var k = 0; k < models.length; k++) {
        var m = models[k].value
        if (!m || m === workerModel) continue
        var entry = worker + ":" + m
        if (workerFallbackList.indexOf(entry) < 0)
          out.push({ value: entry, label: wLabel + " · " + models[k].label })
      }
    }
    out.push({ value: "__custom__", label: strings.t("settings.custom") })
    return out
  }

  readonly property string workerFallbackNotice: {
    var b = omi.brain || {}
    if (!b.worker_active || b.worker_active === worker || !b.worker_fallback_reason) return ""
    return strings.t("settings.using_backup").arg(workerEntryLabel(b.worker_active))
  }

  readonly property var fallbackList: (omi.brain && omi.brain.fallbacks) || []

  readonly property var backupChoices: {
    var out = []
    var list = providerOptions
    for (var i = 0; i < list.length; i++) {
      var v = list[i].value
      if (v !== omi.talker && fallbackList.indexOf(v) < 0) out.push({ value: v, label: list[i].label })
    }
    return out
  }

  readonly property string fallbackNotice: {
    var b = omi.brain || {}
    if (!b.active || b.active === omi.talker || !b.fallback_reason) return ""
    return strings.t("settings.using_backup").arg(entryLabel(b.active))
  }

  function entryLabel(entry) {
    var name = String(entry).split(":")[0]
    var model = String(entry).indexOf(":") >= 0 ? String(entry).slice(name.length + 1) : ""
    var list = providerOptions
    var label = name
    for (var i = 0; i < list.length; i++) if (list[i].value === name) label = list[i].label
    return model !== "" ? label + " · " + model : label
  }

  function removeFallback(index) {
    var next = fallbackList.slice()
    next.splice(index, 1)
    omi.setFallbacks(next)
  }

  function workerEntryLabel(entry) {
    var name = String(entry).split(":")[0]
    var model = String(entry).indexOf(":") >= 0 ? String(entry).slice(name.length + 1) : ""
    var list = workerOptions
    var label = name
    for (var i = 0; i < list.length; i++) if (list[i].value === name) label = list[i].label
    if (model === "") return label
    var models = (omi.brain && omi.brain.worker_models && omi.brain.worker_models[name]) || []
    var mLabel = model
    for (var j = 0; j < models.length; j++) if (models[j].value === model) mLabel = models[j].label
    return label + " · " + mLabel
  }

  function removeWorkerFallback(index) {
    var next = workerFallbackList.slice()
    next.splice(index, 1)
    omi.setWorkerFallbacks(next)
  }

  // Speech engines: "Parakeet v3 · on this PC", "Groq · cloud · needs GROQ_API_KEY".
  function engineOptions(list, current) {
    var out = []
    var arr = list || []
    for (var i = 0; i < arr.length; i++) {
      var e = arr[i]
      var where = e.kind === "local" ? strings.t("settings.on_this_pc") : strings.t("settings.cloud")
      var label = (e.label || engineLabel(e.name)) + " · " + where
      if (!e.installed) {
        var key = (String(e.detail || "").match(/\(([A-Z0-9_]+)\)/) || [])[1]
        label += " · " + (e.kind === "local" ? strings.t("settings.not_downloaded") : strings.t("settings.needs_key").arg(key || "API key"))
      }
      out.push({ value: e.name, label: label })
    }
    if (out.length === 0 && current) out.push({ value: current, label: engineLabel(current) })
    return out
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
    else if (id === "wake") omi.toggleWake()
    else if (id === "settings") root.settingsOpen = !root.settingsOpen
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
    // Settings grow into the screen's spare height so there is less to scroll.
    contentHeight: panel.fittedContentHeight(column.implicitHeight,
      root.settingsOpen ? Math.max(Style.space(620), Math.round(panel.availableCardHeight * 0.9)) : Style.space(620))

    PanelKeyCatcher {
      id: keyCatcher
      anchors.fill: parent
      clip: true
      blocked: input.activeFocus || customModelField.activeFocus || customWorkerBackupField.activeFocus
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
            visible: root.busy
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
            hasCursor: root.currentRow === "wake"
            foreground: omi.wakeEnabled ? Commons.Color.accent : root.dim
            fontFamily: root.fontFamily
            onHovered: function(on) { if (on) root.setRowCursor("wake") }
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
            hasCursor: root.currentRow === "settings"
            foreground: root.settingsOpen ? Commons.Color.accent : root.dim
            fontFamily: root.fontFamily
            onHovered: function(on) { if (on) root.setRowCursor("settings") }
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
          Layout.preferredHeight: Math.min(settingsCol.implicitHeight + Style.space(18),
            Math.max(Style.space(340), Math.round(panel.availableCardHeight * 0.55)))
          implicitHeight: Layout.preferredHeight
          color: "transparent"
          borderSpec: Border.controlSpec("normal", root.dim, Commons.Color.popups.border)
          radius: Style.cornerRadius

          Flickable {
            id: settingsFlick
            anchors.fill: parent
            anchors.margins: Style.space(9)
            clip: true
            contentWidth: width
            contentHeight: settingsCol.implicitHeight
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar { id: settingsScroll; policy: ScrollBar.AsNeeded }

            ColumnLayout {
              id: settingsCol
              // Leave a lane for the scrollbar so it never sits on top of the controls.
              width: settingsFlick.width
                - (settingsFlick.contentHeight > settingsFlick.height ? settingsScroll.width + Style.space(6) : 0)
              spacing: Style.space(8)

              // ---- Brain: who answers, which model, and backups ------------
              Text {
                textFormat: Text.PlainText
                text: strings.t("settings.brain").toUpperCase()
                color: root.dim
                font.family: root.fontFamily
                font.pixelSize: Style.font.caption
                font.bold: true
                font.letterSpacing: 0.8
              }

              // A refused switch leaves the daemon on the old provider and model, so
              // the dropdowns go back to what is really in use.
              Connections {
                target: omi
                function onNoticeChanged() {
                  if (omi.notice === "") return
                  providerDropdown.value = providerDropdown.bound
                  modelDropdown.value = modelDropdown.bound
                  workerDropdown.value = workerDropdown.bound
                  workerModelDropdown.value = workerModelDropdown.bound
                }
              }

              Dropdown {
                id: providerDropdown
                objectName: "providerDropdown"
                Layout.fillWidth: true
                label: strings.t("settings.provider")
                fontFamily: root.fontFamily
                background: root.dropdownBackground
                options: root.providerOptions
                readonly property string bound: omi.talker
                value: bound
                onBoundChanged: value = bound
                onChanged: function(v) {
                  root.customModel = false
                  omi.setProvider(v)
                }
                Accessible.name: strings.t("settings.provider")
              }

              Dropdown {
                id: modelDropdown
                objectName: "modelDropdown"
                Layout.fillWidth: true
                label: strings.t("settings.model")
                fontFamily: root.fontFamily
                background: root.dropdownBackground
                options: root.modelOptions
                readonly property string bound: root.customModel ? "__custom__" : omi.currentModel()
                value: bound
                onBoundChanged: value = bound
                onChanged: function(v) {
                  if (v === "__custom__") {
                    root.customModel = true
                    Qt.callLater(function() { customModelField.forceActiveFocus() })
                  } else {
                    root.customModel = false
                    omi.setModel(v)
                  }
                }
                Accessible.name: strings.t("settings.model")
              }

              TextField {
                id: customModelField
                objectName: "customModelField"
                visible: root.customModel
                Layout.fillWidth: true
                placeholderText: omi.localFiles ? strings.t("settings.custom_local") : strings.t("settings.custom_model")
                foreground: root.foreground
                font.family: root.fontFamily
                Accessible.name: strings.t("settings.custom_model")
                Keys.onPressed: function(event) {
                  if (event.key === Qt.Key_Return || event.key === Qt.Key_Enter) {
                    if (text.trim() !== "") omi.setModel(text.trim())
                    root.customModel = false
                    text = ""
                    event.accepted = true
                  } else if (event.key === Qt.Key_Escape) {
                    root.customModel = false
                    text = ""
                    event.accepted = true
                  }
                }
              }

              Text {
                objectName: "fallbackNotice"
                visible: text !== ""
                Layout.fillWidth: true
                textFormat: Text.PlainText
                text: root.fallbackNotice
                color: root.urgent
                font.family: root.fontFamily
                font.pixelSize: Style.font.caption
                wrapMode: Text.WordWrap
              }

              Text {
                textFormat: Text.PlainText
                text: strings.t("settings.backups")
                color: Qt.darker(root.foreground, 1.4)
                font.family: root.fontFamily
                font.pixelSize: Style.font.caption
                font.bold: true
              }

              Repeater {
                model: root.fallbackList
                delegate: RowLayout {
                  required property string modelData
                  required property int index
                  Layout.fillWidth: true
                  spacing: Style.space(6)

                  Text {
                    Layout.fillWidth: true
                    textFormat: Text.PlainText
                    text: (index + 1) + ". " + root.entryLabel(modelData)
                    color: root.foreground
                    font.family: root.fontFamily
                    font.pixelSize: Style.font.body
                    elide: Text.ElideRight
                  }

                  Button {
                    text: "\u{f0156}"
                    bordered: true
                    foreground: root.dim
                    fontFamily: root.fontFamily
                    onClicked: root.removeFallback(index)
                    Accessible.role: Accessible.Button
                    Accessible.name: strings.t("settings.remove_backup").arg(root.entryLabel(modelData))
                  }
                }
              }

              Dropdown {
                objectName: "addBackupDropdown"
                visible: root.backupChoices.length > 0
                Layout.fillWidth: true
                showLabel: false
                fontFamily: root.fontFamily
                background: root.dropdownBackground
                options: [{ value: "", label: strings.t("settings.add_backup") }].concat(root.backupChoices)
                value: ""
                onChanged: function(v) {
                  if (v !== "") omi.setFallbacks(root.fallbackList.concat([v]))
                  value = ""
                }
                Accessible.name: strings.t("settings.add_backup")
              }

              PanelSeparator { Layout.fillWidth: true; foreground: root.foreground }

              // ---- Worker: who does long jobs in the background ------------
              Text {
                textFormat: Text.PlainText
                text: strings.t("settings.worker_section").toUpperCase()
                color: root.dim
                font.family: root.fontFamily
                font.pixelSize: Style.font.caption
                font.bold: true
                font.letterSpacing: 0.8
              }

              Dropdown {
                id: workerDropdown
                objectName: "workerDropdown"
                visible: root.workerOptions.length > 0
                Layout.fillWidth: true
                label: strings.t("settings.worker")
                fontFamily: root.fontFamily
                background: root.dropdownBackground
                options: root.workerOptions
                readonly property string bound: root.worker
                value: bound
                onBoundChanged: value = bound
                onChanged: function(v) {
                  root.customWorkerBackup = false
                  if (v !== root.worker) omi.setSetting("brain.worker", v)
                }
                Accessible.name: strings.t("settings.worker")
              }

              Dropdown {
                id: workerModelDropdown
                objectName: "workerModelDropdown"
                visible: root.worker !== "" && root.worker !== "off" && root.workerModelOptions.length > 0
                Layout.fillWidth: true
                label: strings.t("settings.worker_model")
                fontFamily: root.fontFamily
                background: root.dropdownBackground
                options: root.workerModelOptions
                readonly property string bound: root.workerModel
                value: bound
                onBoundChanged: value = bound
                onChanged: function(v) { if (v !== bound) omi.setSetting("brain.worker_model", v) }
                Accessible.name: strings.t("settings.worker_model")
              }

              Dropdown {
                objectName: "workerModeDropdown"
                visible: root.worker !== "" && root.worker !== "off"
                Layout.fillWidth: true
                label: strings.t("settings.worker_mode")
                fontFamily: root.fontFamily
                background: root.dropdownBackground
                options: [{ value: "auto", label: strings.t("settings.worker_mode_auto") },
                          { value: "unrestricted", label: strings.t("settings.worker_mode_unrestricted") }]
                readonly property string bound: (omi.brain && omi.brain.worker_mode) || "auto"
                value: bound
                onBoundChanged: value = bound
                onChanged: function(v) { if (v !== bound) omi.setSetting("brain.worker_mode", v) }
                Accessible.name: strings.t("settings.worker_mode")
              }

              Text {
                objectName: "workerFallbackNotice"
                visible: text !== ""
                Layout.fillWidth: true
                textFormat: Text.PlainText
                text: root.workerFallbackNotice
                color: root.urgent
                font.family: root.fontFamily
                font.pixelSize: Style.font.caption
                wrapMode: Text.WordWrap
              }

              Text {
                visible: root.worker !== "" && root.worker !== "off"
                textFormat: Text.PlainText
                text: strings.t("settings.backups")
                color: Qt.darker(root.foreground, 1.4)
                font.family: root.fontFamily
                font.pixelSize: Style.font.caption
                font.bold: true
              }

              Repeater {
                model: root.worker !== "" && root.worker !== "off" ? root.workerFallbackList : []
                delegate: RowLayout {
                  required property string modelData
                  required property int index
                  Layout.fillWidth: true
                  spacing: Style.space(6)

                  Text {
                    Layout.fillWidth: true
                    textFormat: Text.PlainText
                    text: (index + 1) + ". " + root.workerEntryLabel(modelData)
                    color: root.foreground
                    font.family: root.fontFamily
                    font.pixelSize: Style.font.body
                    elide: Text.ElideRight
                  }

                  Button {
                    text: "\u{f0156}"
                    bordered: true
                    foreground: root.dim
                    fontFamily: root.fontFamily
                    onClicked: root.removeWorkerFallback(index)
                    Accessible.role: Accessible.Button
                    Accessible.name: strings.t("settings.remove_backup").arg(root.workerEntryLabel(modelData))
                  }
                }
              }

              Dropdown {
                objectName: "addWorkerBackupDropdown"
                visible: root.worker !== "" && root.worker !== "off" && root.workerBackupChoices.length > 0
                Layout.fillWidth: true
                showLabel: false
                fontFamily: root.fontFamily
                background: root.dropdownBackground
                options: [{ value: "", label: strings.t("settings.add_backup") }].concat(root.workerBackupChoices)
                value: ""
                onChanged: function(v) {
                  if (v === "__custom__") {
                    root.customWorkerBackup = true
                    Qt.callLater(function() { customWorkerBackupField.forceActiveFocus() })
                  } else if (v !== "") {
                    root.customWorkerBackup = false
                    omi.setWorkerFallbacks(root.workerFallbackList.concat([v]))
                  }
                  value = ""
                }
                Accessible.name: strings.t("settings.add_backup")
              }

              TextField {
                id: customWorkerBackupField
                objectName: "customWorkerBackupField"
                visible: root.customWorkerBackup && root.worker !== "" && root.worker !== "off"
                Layout.fillWidth: true
                placeholderText: strings.t("settings.custom_backup")
                foreground: root.foreground
                font.family: root.fontFamily
                Accessible.name: strings.t("settings.custom_backup")
                Keys.onPressed: function(event) {
                  if (event.key === Qt.Key_Return || event.key === Qt.Key_Enter) {
                    if (text.trim() !== "") omi.setWorkerFallbacks(root.workerFallbackList.concat([text.trim()]))
                    root.customWorkerBackup = false
                    text = ""
                    event.accepted = true
                  } else if (event.key === Qt.Key_Escape) {
                    root.customWorkerBackup = false
                    text = ""
                    event.accepted = true
                  }
                }
              }

              PanelSeparator { Layout.fillWidth: true; foreground: root.foreground }

              // ---- Voice: hearing and speaking ------------------------------
              Text {
                textFormat: Text.PlainText
                text: strings.t("settings.voice_section").toUpperCase()
                color: root.dim
                font.family: root.fontFamily
                font.pixelSize: Style.font.caption
                font.bold: true
                font.letterSpacing: 0.8
              }

              Dropdown {
                objectName: "sttDropdown"
                Layout.fillWidth: true
                label: strings.t("settings.stt")
                fontFamily: root.fontFamily
                background: root.dropdownBackground
                options: root.engineOptions(omi.sttEngines, omi.sttEngine)
                readonly property string bound: omi.sttEngine
                value: bound
                onBoundChanged: value = bound
                onChanged: function(v) { if (v !== omi.sttEngine) omi.setSetting("stt.engine", v) }
                Accessible.name: strings.t("settings.stt")
              }

              Dropdown {
                objectName: "ttsDropdown"
                Layout.fillWidth: true
                label: strings.t("settings.tts")
                fontFamily: root.fontFamily
                background: root.dropdownBackground
                options: root.engineOptions(omi.ttsEngines, omi.ttsEngine)
                readonly property string bound: omi.ttsEngine
                value: bound
                onBoundChanged: value = bound
                onChanged: function(v) { if (v !== omi.ttsEngine) omi.setSetting("tts.engine", v) }
                Accessible.name: strings.t("settings.tts")
              }

              Dropdown {
                objectName: "voiceDropdown"
                visible: omi.voices && omi.voices.length > 0
                Layout.fillWidth: true
                label: strings.t("settings.voice")
                fontFamily: root.fontFamily
                background: root.dropdownBackground
                options: (omi.voices || []).map(function(v) { return { value: v, label: root.voiceLabel(v) } })
                readonly property string bound: omi.voice
                value: bound
                onBoundChanged: value = bound
                onChanged: function(v) { if (v !== omi.voice) omi.setSetting("tts.voice", v) }
                Accessible.name: strings.t("settings.voice")
              }

              Dropdown {
                objectName: "speedDropdown"
                Layout.fillWidth: true
                label: strings.t("settings.speed")
                fontFamily: root.fontFamily
                background: root.dropdownBackground
                options: [0.9, 1.0, 1.15, 1.3].map(function(v) { return { value: String(v), label: root.speedLabel(v) } })
                readonly property string bound: String(Math.round(omi.ttsSpeed * 100) / 100)
                value: bound
                onBoundChanged: value = bound
                onChanged: function(v) { omi.setSetting("tts.speed", v) }
                Accessible.name: strings.t("settings.speed")
              }

              PanelSeparator { Layout.fillWidth: true; foreground: root.foreground }

              // ---- Listening ---------------------------------------------------
              Text {
                textFormat: Text.PlainText
                text: strings.t("settings.listening").toUpperCase()
                color: root.dim
                font.family: root.fontFamily
                font.pixelSize: Style.font.caption
                font.bold: true
                font.letterSpacing: 0.8
              }

              Dropdown {
                objectName: "callModeDropdown"
                Layout.fillWidth: true
                label: strings.t("settings.call_mode")
                fontFamily: root.fontFamily
                background: root.dropdownBackground
                options: ["auto", "continuous", "single"].map(function(v) { return { value: v, label: root.callModeLabel(v) } })
                readonly property string bound: omi.wakeCallMode
                value: bound
                onBoundChanged: value = bound
                onChanged: function(v) { if (v !== omi.wakeCallMode) omi.setSetting("wake.call_mode", v) }
                Accessible.name: strings.t("settings.call_mode")
              }

              Dropdown {
                objectName: "sensitivityDropdown"
                Layout.fillWidth: true
                label: strings.t("settings.sensitivity")
                fontFamily: root.fontFamily
                background: root.dropdownBackground
                options: ["low", "medium", "high"].map(function(v) { return { value: v, label: root.sensitivityLabel(v) } })
                readonly property string bound: omi.wakeSensitivity
                value: bound
                onBoundChanged: value = bound
                onChanged: function(v) { if (v !== omi.wakeSensitivity) omi.setSetting("wake.sensitivity", v) }
                Accessible.name: strings.t("settings.sensitivity")
              }

              // Look to talk: the webcam tells when you are talking to Omi
              RowLayout {
                Layout.fillWidth: true
                spacing: Style.space(8)

                Text {
                  textFormat: Text.PlainText
                  text: strings.t("settings.look")
                  color: root.dim
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.caption
                }

                Item { Layout.fillWidth: true }

                Button {
                  objectName: "lookTuneButton"
                  text: omi.tuning ? strings.t("settings.tune_cancel") : strings.t("settings.tune")
                  bordered: true
                  foreground: omi.tuning ? Commons.Color.urgent : root.dim
                  fontFamily: root.fontFamily
                  onClicked: omi.tuning ? omi.cancelLookTune() : omi.openLookTune()
                  Accessible.role: Accessible.Button
                  Accessible.name: omi.tuning ? strings.t("settings.tune_cancel_name") : strings.t("settings.tune_name")
                }

                Button {
                  objectName: "lookButton"
                  text: omi.lookToTalk ? strings.t("settings.on") : strings.t("settings.off")
                  bordered: true
                  foreground: omi.lookToTalk ? Commons.Color.accent : root.dim
                  fontFamily: root.fontFamily
                  onClicked: omi.toggleLook()
                  Accessible.role: Accessible.Button
                  Accessible.name: strings.t("settings.look") + ": " + text
                }
              }

              // Only my voice: ignore people who did not save their voice
              RowLayout {
                Layout.fillWidth: true
                spacing: Style.space(8)

                Text {
                  textFormat: Text.PlainText
                  text: strings.t("settings.my_voice")
                  color: root.dim
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.caption
                }

                Item { Layout.fillWidth: true }

                Button {
                  objectName: "myVoiceButton"
                  text: omi.onlyMyVoice ? strings.t("settings.on") : strings.t("settings.off")
                  bordered: true
                  foreground: omi.onlyMyVoice ? Commons.Color.accent : root.dim
                  fontFamily: root.fontFamily
                  onClicked: omi.toggleMyVoice()
                  Accessible.role: Accessible.Button
                  Accessible.name: strings.t("settings.my_voice") + ": " + text
                }
              }

              // Which webcam watches, only worth asking with more than one
              Dropdown {
                objectName: "cameraDropdown"
                visible: omi.cameras.length > 1
                Layout.fillWidth: true
                label: strings.t("settings.camera")
                fontFamily: root.fontFamily
                background: root.dropdownBackground
                options: [{ value: "", label: strings.t("settings.camera_auto") }].concat(
                  omi.cameras.map(function(c) { return { value: c.id, label: c.name } }))
                readonly property string bound: omi.lookCamera
                value: bound
                onBoundChanged: value = bound
                onChanged: function(v) { if (v !== omi.lookCamera) omi.setSetting("wake.camera", v === "" ? "auto" : v) }
                Accessible.name: strings.t("settings.camera")
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
