// Hark phone and tablet client. One WebSocket: mic audio up, reply audio and events down.
// Wire format: see hark/protocol.py.

const $ = (id) => document.getElementById(id);
const app = document.querySelector(".app");
const els = {
  state: $("state-text"), name: $("name"), log: $("log"), list: $("transcript"),
  confirm: $("confirm"), question: $("confirm-q"), detail: $("confirm-detail"),
  yes: $("yes"), no: $("no"), talk: $("talk"), stop: $("stop"), form: $("compose"), text: $("text"),
};
const LABELS = {
  idle: "Ready", listening: "Listening", thinking: "Thinking", speaking: "Speaking",
  acting: "Working", confirming: "Yes or no?", offline: "Offline", connecting: "Connecting",
};
const TAP_MS = 300; // a press shorter than this toggles listening on until the next tap
const MIC_IDLE_MS = 120000; // close the microphone after this long unused
const HEADER = 8; // reply audio header: uint32 sample rate, uint32 turn number

let ws = null;
let online = false;
let retry = 0;
let confirmId = null;

// --- state and transcript ---------------------------------------------------------

function setState(state) {
  app.dataset.state = state;
  els.state.textContent = LABELS[state] || state;
  if (state !== "confirming") hideConfirm();
}

function addLine(kind, text) {
  if (!text) return;
  const li = document.createElement("li");
  li.className = kind;
  const who = { user: "You: ", assistant: `${els.name.textContent}: ` }[kind];
  if (who) { // alignment and colour show the speaker; screen readers hear it
    const span = document.createElement("span");
    span.className = "sr-only";
    span.textContent = who;
    li.append(span);
  }
  li.append(text);
  els.list.append(li);
  while (els.list.children.length > 200) els.list.firstElementChild.remove();
  els.log.scrollTop = els.log.scrollHeight;
}

function showConfirm(c) {
  confirmId = c.id;
  els.question.textContent = c.question;
  els.detail.textContent = c.detail || "";
  els.detail.hidden = !c.detail;
  els.confirm.hidden = false;
}

function hideConfirm() {
  confirmId = null;
  els.confirm.hidden = true;
}

// --- connection -------------------------------------------------------------------

function connect() {
  setState("connecting");
  const sock = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  sock.binaryType = "arraybuffer";
  sock.onopen = () => { ws = sock; online = true; retry = 0; els.talk.disabled = false; };
  sock.onmessage = (e) => (typeof e.data === "string" ? onEvent(JSON.parse(e.data)) : onAudio(e.data));
  sock.onclose = () => {
    if (ws === sock) ws = null;
    online = false;
    els.talk.disabled = true;
    if (capturing) cancelTalk();
    stopPlayback();
    setState("offline");
    setTimeout(connect, Math.min(10000, 500 * 2 ** retry++));
  };
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function onEvent(ev) {
  switch (ev.event) {
    case "hello":
      if (ev.name) { els.name.textContent = ev.name; document.title = ev.name; }
      setState(ev.state);
      if (ev.confirm) showConfirm(ev.confirm);
      break;
    case "state":
      setState(ev.state);
      break;
    case "transcript":
      addLine(ev.role === "user" ? "user" : "assistant", ev.text);
      break;
    case "tool":
      if (ev.phase === "start") addLine("note", ev.name.replaceAll("_", " "));
      else if (ev.ok === false) addLine("note error", ev.text);
      break;
    case "confirm_request":
      showConfirm(ev);
      addLine("note", ev.question);
      break;
    case "progress":
      addLine("note", ev.text);
      break;
    case "audio":
      if (ev.phase === "stop") { dropSeq = Math.max(dropSeq, ev.seq); if (ev.seq === playSeq) stopPlayback(); }
      else if (ev.phase === "end") { endSeq = ev.seq; maybePlayed(); }
      break;
    case "done":
      if (ev.error) addLine("note error", ev.error);
      break;
    case "error":
      addLine("note error", ev.text);
      break;
  }
}

// --- audio out: reply chunks scheduled back to back -----------------------------------

let ctx = null;
let playHead = 0;
let playSeq = 0; // turn whose audio is playing
let dropSeq = 0; // audio of this turn and older is dropped (barge-in)
let endSeq = 0; // turn whose last chunk has arrived
const sources = new Set();

function audioContext() {
  if (!ctx) ctx = new AudioContext({ latencyHint: "interactive" });
  if (ctx.state === "suspended") ctx.resume();
  return ctx;
}

function onAudio(buf) {
  if (buf.byteLength <= HEADER) return;
  const view = new DataView(buf);
  const rate = view.getUint32(0, true);
  const seq = view.getUint32(4, true);
  if (seq <= dropSeq || !ctx) return;
  const pcm = new Int16Array(buf, HEADER, (buf.byteLength - HEADER) >> 1);
  const samples = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) samples[i] = pcm[i] / 32768;
  const audio = ctx.createBuffer(1, samples.length, rate);
  audio.copyToChannel(samples, 0);
  const src = ctx.createBufferSource();
  src.buffer = audio;
  src.connect(ctx.destination);
  const at = Math.max(ctx.currentTime + 0.02, playHead);
  src.start(at);
  playHead = at + audio.duration;
  playSeq = seq;
  sources.add(src);
  src.onended = () => { sources.delete(src); maybePlayed(); };
}

function maybePlayed() {
  if (endSeq && sources.size === 0) {
    send({ cmd: "played", seq: endSeq });
    endSeq = 0;
  }
}

function stopPlayback() {
  for (const src of sources) { try { src.stop(); } catch { /* already stopped */ } }
  sources.clear();
  playHead = 0;
}

function bargeIn() {
  if (playSeq) dropSeq = Math.max(dropSeq, playSeq);
  endSeq = 0;
  stopPlayback();
}

// --- audio in: AudioWorklet at 16 kHz Int16 --------------------------------------------

let mic = null; // {stream, node}
let micReady = null;
let micTimer = 0;
let capturing = false;

async function openMic() {
  const c = audioContext();
  if (!openMic.loaded) { await c.audioWorklet.addModule("/mic-worklet.js"); openMic.loaded = true; }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const node = new AudioWorkletNode(c, "mic-16k", { numberOfOutputs: 1 });
  node.port.onmessage = (e) => {
    if (e.data === "done") send({ cmd: "ptt_stop" });
    else if (ws && ws.readyState === WebSocket.OPEN) ws.send(e.data);
  };
  c.createMediaStreamSource(stream).connect(node);
  node.connect(c.destination); // silent; keeps the node pulled by the graph
  mic = { stream, node };
  return mic;
}

function ensureMic() {
  clearTimeout(micTimer);
  if (!micReady) micReady = openMic().catch((err) => { micReady = null; throw err; });
  return micReady;
}

function closeMic() {
  if (!mic) return;
  mic.stream.getTracks().forEach((t) => t.stop());
  mic.node.disconnect();
  mic = null;
  micReady = null;
}

function idleMic() {
  clearTimeout(micTimer);
  micTimer = setTimeout(() => { if (!capturing) closeMic(); }, MIC_IDLE_MS);
}

async function startTalk() {
  if (!online || capturing) return;
  audioContext();
  bargeIn();
  capturing = true;
  els.talk.setAttribute("aria-pressed", "true");
  send({ cmd: "ptt_start" });
  try {
    const m = await ensureMic();
    if (capturing) m.node.port.postMessage("start");
  } catch (err) {
    capturing = false;
    held = toggled = false;
    els.talk.setAttribute("aria-pressed", "false");
    send({ cmd: "stop" });
    addLine("note error", "Microphone unavailable");
  }
}

async function finishTalk() {
  if (!capturing) return;
  capturing = false;
  els.talk.setAttribute("aria-pressed", "false");
  try {
    const m = await ensureMic();
    m.node.port.postMessage("stop"); // the worklet sends its last samples, then "done"
  } catch {
    send({ cmd: "stop" });
  }
  idleMic();
}

function cancelTalk() {
  capturing = false;
  held = toggled = false;
  els.talk.setAttribute("aria-pressed", "false");
  if (mic) mic.node.port.postMessage("cancel");
  idleMic();
}

// --- talk button: hold to talk, tap to toggle -------------------------------------------

let held = false;
let toggled = false;
let pressedAt = 0;
let keyDown = false;

function press() {
  if (toggled) { toggled = false; finishTalk(); return; }
  held = true;
  pressedAt = performance.now();
  startTalk();
}

function release() {
  if (!held) return;
  held = false;
  if (!capturing) return;
  if (performance.now() - pressedAt < TAP_MS) toggled = true;
  else finishTalk();
}

els.talk.addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;
  e.preventDefault();
  els.talk.setPointerCapture(e.pointerId);
  press();
});
els.talk.addEventListener("pointerup", release);
els.talk.addEventListener("pointercancel", release);
els.talk.addEventListener("contextmenu", (e) => e.preventDefault());
els.talk.addEventListener("keydown", (e) => {
  if (e.key !== " " && e.key !== "Enter") return;
  e.preventDefault();
  if (e.repeat || keyDown) return;
  keyDown = true;
  press();
});
els.talk.addEventListener("keyup", (e) => {
  if (e.key !== " " && e.key !== "Enter") return;
  e.preventDefault();
  keyDown = false;
  release();
});
// Assistive tech activates with a click that has no pointer or key behind it: toggle.
els.talk.addEventListener("click", (e) => {
  if (e.detail !== 0 || keyDown) return;
  if (capturing) { toggled = false; finishTalk(); } else { toggled = true; startTalk(); }
});

els.stop.addEventListener("click", () => {
  bargeIn();
  if (capturing) cancelTalk();
  send({ cmd: "stop" });
});

els.yes.addEventListener("click", () => answer(true));
els.no.addEventListener("click", () => answer(false));
function answer(yes) {
  if (confirmId) send({ cmd: "confirm", answer: yes, id: confirmId });
  hideConfirm();
}

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = els.text.value.trim();
  if (!text || !online) return;
  audioContext();
  bargeIn();
  send({ cmd: "say", text });
  els.text.value = "";
});

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") {
    if (capturing) { cancelTalk(); send({ cmd: "stop" }); }
    closeMic();
  }
});

// Rotating the device keeps the newest line in view.
addEventListener("resize", () => { els.log.scrollTop = els.log.scrollHeight; });

els.talk.disabled = true;
connect();

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch(() => {});
}
