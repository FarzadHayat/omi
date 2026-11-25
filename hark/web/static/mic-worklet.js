// Microphone to 16 kHz mono Int16, in 20 ms frames (320 samples) posted to the page.
// Each output sample is the mean of the input samples it covers, which also low-passes.

const OUT_RATE = 16000;
const FRAME = 320;

class Mic16k extends AudioWorkletProcessor {
  constructor() {
    super();
    this.on = false;
    this.ratio = sampleRate / OUT_RATE;
    this.reset();
    this.port.onmessage = (e) => {
      if (e.data === "start") { this.reset(); this.on = true; }
      else if (e.data === "stop") { this.flush(); this.on = false; this.port.postMessage("done"); }
      else if (e.data === "cancel") { this.on = false; this.reset(); }
    };
  }

  reset() {
    this.pos = 0;
    this.sum = 0;
    this.count = 0;
    this.out = new Int16Array(FRAME);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!this.on || !ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.sum += ch[i];
      this.count++;
      if (++this.pos >= this.ratio) {
        this.pos -= this.ratio;
        this.emit(this.sum / this.count);
        this.sum = 0;
        this.count = 0;
      }
    }
    return true;
  }

  emit(v) {
    const s = Math.max(-1, Math.min(1, v));
    this.out[this.n++] = s < 0 ? s * 32768 : s * 32767;
    if (this.n === FRAME) {
      this.port.postMessage(this.out.buffer, [this.out.buffer]);
      this.out = new Int16Array(FRAME);
      this.n = 0;
    }
  }

  flush() {
    if (this.n) {
      const part = this.out.slice(0, this.n);
      this.port.postMessage(part.buffer, [part.buffer]);
    }
    this.reset();
  }
}

registerProcessor("mic-16k", Mic16k);
