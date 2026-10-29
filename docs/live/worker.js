// The live page's analysis, off the page's main thread (docs/design/live-page.md).
// Loads Pyodide and sonore as the landing page's worker does (docs/try/worker.js),
// gathers the microphone's 128-sample buffers into 50 ms blocks and steps
// docs/live/live.py on each one. A block is analyzed between messages, so the
// buffers that arrive meanwhile wait in `pending`, which is how far behind the
// analysis is; past one second it starts again (L5).
//
// For development, ?wheel=<url of a wheel> installs that wheel instead of
// sonore from PyPI, and ?keep=1 keeps the samples for the page's check.

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
const PACKAGES = ["numpy", "scipy", "matplotlib", "soundfile", "micropip"];
const BLOCK_SECONDS = 0.05;
const MAX_BEHIND_SECONDS = 1.0;

const options = new URL(self.location.href).searchParams;
const say = (text) => self.postMessage({ type: "status", text });

const ready = (async () => {
  say("Downloading Python and its scientific packages (about 35 MB, once)…");
  const { loadPyodide } = await import(PYODIDE + "pyodide.mjs");
  const pyodide = await loadPyodide({ indexURL: PYODIDE });
  await pyodide.loadPackage(PACKAGES, { messageCallback: () => {} });
  say("Installing sonore…");
  try {
    await pyodide.pyimport("micropip").install(options.get("wheel") || "sonore>=0.10");
  } catch (error) {
    throw new Error(`this page needs sonore 0.10 or later from PyPI, where the streams first appear (${error.message.trim().split("\n").pop()})`);
  }
  const source = await (await fetch(new URL("live.py", import.meta.url))).text();
  pyodide.FS.writeFile("/home/pyodide/live.py", source);
  const version = pyodide.runPython("import sonore; sonore.__version__");
  return { pyodide, live: pyodide.pyimport("live"), version };
})();
ready.then(
  ({ version }) => self.postMessage({ type: "installed", version }),
  (error) => self.postMessage({ type: "failed", text: String(error) }),
);

let analysis = null;  // a live.Live, once the page has said its sampling rate
let audioPort = null;
let fs = 0;
let blockLength = 0;
let pending = [];
let pendingLength = 0;
let busy = false;
let restarts = 0;
let received = 0;
let nextFrame = 0;
let gaps = 0;
let firstAudio = 0;
const restartTimes = [];

function take(length) {
  const block = new Float32Array(length);
  let filled = 0;
  while (filled < length) {
    const head = pending[0];
    const used = Math.min(head.length, length - filled);
    block.set(head.subarray(0, used), filled);
    filled += used;
    if (used === head.length) pending.shift();
    else pending[0] = head.subarray(used);
  }
  pendingLength -= length;
  return block;
}

async function drain() {
  busy = true;
  while (pendingLength >= blockLength) {
    const result = analysis.step(take(blockLength));
    const part = result.toJs({ dict_converter: Object.fromEntries });
    result.destroy();
    self.postMessage({ type: "part", part, behind: pendingLength / fs });
    await new Promise((resolve) => setTimeout(resolve, 0));  // let waiting buffers in
  }
  busy = false;
}

function onAudio(samples, frame) {
  if (!analysis) return;  // before the analysis is ready
  if (!received) firstAudio = performance.now();
  const gap = received > 0 && frame !== nextFrame;
  nextFrame = frame + samples.length;
  received += samples.length;
  if (gap) gaps += 1;
  pending.push(samples);
  pendingLength += samples.length;
  if (gap || pendingLength > MAX_BEHIND_SECONDS * fs) {
    pending = gap ? [samples] : [];
    pendingLength = gap ? samples.length : 0;
    analysis.restart();
    restarts += 1;
    restartTimes.push((performance.now() - firstAudio) / 1000);
    self.postMessage({ type: "restarted", restarts, gap });
  }
  if (!busy) drain();
}

self.onmessage = async ({ data }) => {
  if (data.type === "start") {
    pending = [];
    pendingLength = 0;
    received = 0;
    audioPort = data.audio;
    data.audio.onmessage = ({ data: message }) => onAudio(message.samples, message.frame);
    try {
      const { live, version } = await ready;
      fs = data.fs;
      blockLength = Math.round(BLOCK_SECONDS * fs);
      if (analysis) analysis.destroy();
      analysis = live.Live(fs, options.get("keep") === "1");
      const centers = analysis.band_centers().toJs();
      self.postMessage({ type: "ready", version, centers, fs });
    } catch (error) {
      self.postMessage({ type: "failed", text: String(error) });
    }
  } else if (data.type === "stop") {
    // The microphone is off: no more buffers, and what waits is dropped. The
    // analysis stays, so the page's check still covers what was heard.
    if (audioPort) audioPort.close();
    audioPort = null;
    pending = [];
    pendingLength = 0;
  } else if (data.type === "check" && analysis) {
    const result = analysis.check();
    self.postMessage({ type: "check", result: { ...result.toJs({ dict_converter: Object.fromEntries }), restarts, restartTimes, gaps, received, seconds: (performance.now() - firstAudio) / 1000 } });
    result.destroy();
  }
};
