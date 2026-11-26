import QtQuick

// Offscreen stand-in for qs.Ui.OverlayWindow used by omarchy/qml-test/shell.qml.
// Real layer-shell surfaces have no Wayland backend under QT_QPA_PLATFORM=offscreen.
Item {
  id: window
  property bool shown: false
  property int shownLayer: 0
  property int shownKeyboardFocus: 0
  property var mask: null
  property var targetScreen: null
  property var targetMonitor: null
  visible: shown
  width: 1920
  height: 1080
}
