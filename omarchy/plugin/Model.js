.pragma library

// Pure helpers for the hark bar widget: no QML, so the harness can check them.

var STATES = ["offline", "idle", "listening", "thinking", "speaking", "acting", "confirming"]
var MAX_LINES = 40      // transcript entries kept in the panel
var MAX_TEXT = 2000     // characters kept from any one daemon string
var MAX_REQUEST = 4000  // characters sent in one typed request

function clip(value, limit) {
  var text = value === undefined || value === null ? "" : String(value)
  var max = limit || MAX_TEXT
  return text.length > max ? text.slice(0, max - 1) + "…" : text
}

// One line from the daemon socket, or null when it is not a JSON object.
function parseLine(line) {
  if (typeof line !== "string" || line.length === 0 || line.length > 1 << 20) return null
  try {
    var msg = JSON.parse(line)
    return msg !== null && typeof msg === "object" && !Array.isArray(msg) ? msg : null
  } catch (e) {
    return null
  }
}

function knownState(value) {
  return STATES.indexOf(String(value)) >= 0 ? String(value) : "idle"
}

// A confirm_request event or a status "confirm" object, checked and bounded.
function confirmOf(msg) {
  if (!msg || typeof msg.id !== "string" || msg.id === "" || msg.id.length > 64) return null
  return { id: msg.id, question: clip(msg.question, 600), detail: clip(msg.detail || "", 4000) }
}

// Returns a new list so QML sees the change.
function appendLine(lines, role, text) {
  var clean = clip(text).trim()
  if (clean === "") return lines
  var next = lines.concat([{ role: role === "user" ? "user" : (role === "progress" ? "progress" : "assistant"), text: clean }])
  return next.length > MAX_LINES ? next.slice(next.length - MAX_LINES) : next
}

function request(cmd, fields) {
  var req = { cmd: cmd }
  for (var key in fields || {}) req[key] = fields[key]
  return JSON.stringify(req) + "\n"
}

function sayRequest(text) {
  var clean = String(text || "").trim()
  if (clean === "") return ""
  return request("say", { text: clean.slice(0, MAX_REQUEST) })
}
