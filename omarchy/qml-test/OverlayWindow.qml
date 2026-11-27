import QtQuick

// Offscreen stand-in for qs.Ui.OverlayWindow used by omarchy/qml-test/shell.qml.
// Real layer-shell surfaces have no Wayland backend under QT_QPA_PLATFORM=offscreen.
Item {
  id: window
  property bool shown: false
  property bool contentReady: false
  property bool contentRevealed: false
  property int shownLayer: 0
  property int shownKeyboardFocus: 0
  property var mask: null
  property var targetScreen: null
  property var targetMonitor: null
  visible: shown
  // Mirror real Wayland OverlayWindow: unconfigured before open, then configured
  // to fullscreen before contentRevealed turns true on the next turn.
  width: shown ? 1920 : 100
  height: shown ? 1080 : 100

  onShownChanged: {
    contentReady = false
    contentRevealed = false
    if (shown) {
      contentReady = true
      Qt.callLater(function() {
        if (window.shown) window.contentRevealed = true
      })
    }
  }
}
