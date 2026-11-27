import QtQuick

// Stand-in for qs.Ui KeyboardPanel in the harness. The real one is a layer-shell
// PanelWindow, which the offscreen platform cannot create; this keeps its API
// (the properties and helpers the omi panel uses) and hosts the same content.
Item {
  id: root

  property Item anchorItem: null
  property QtObject bar: null
  property var owner: null
  property bool open: false
  property bool centerOnBar: false
  property Item focusTarget: null
  property int contentWidth: 280
  property int contentHeight: 200
  default property alias contentItem: holder.children

  function fittedContentWidth(width, cap) { return Math.round(cap > 0 ? Math.min(width, cap) : width) }
  function fittedContentHeight(implicitHeight, cap) { return Math.round(cap > 0 ? Math.min(implicitHeight, cap) : implicitHeight) }

  visible: open
  width: contentWidth
  height: contentHeight
  onOpenChanged: if (open && focusTarget) Qt.callLater(function() { root.focusTarget.forceActiveFocus() })

  Item {
    id: holder
    anchors.fill: parent
  }
}
