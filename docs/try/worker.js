// Runs Python for the landing page's code cell, off the page's main thread.
// Pyodide (CPython compiled to WebAssembly) comes from its CDN; its own builds of
// numpy, scipy, matplotlib and soundfile load from there too, and sonore, a pure
// Python wheel, installs from PyPI. Pyodide 314 runs only in a module worker.

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
const PACKAGES = ["numpy", "scipy", "matplotlib", "soundfile", "micropip"];

const say = (text) => self.postMessage({ type: "status", text });

const ready = (async () => {
  const started = performance.now();
  say("Downloading Python and its scientific packages (about 35 MB, once)…");
  const { loadPyodide } = await import(PYODIDE + "pyodide.mjs");
  const pyodide = await loadPyodide({ indexURL: PYODIDE });
  await pyodide.loadPackage(PACKAGES, { messageCallback: () => {} });
  say("Installing sonore from PyPI…");
  await pyodide.pyimport("micropip").install("sonore");
  say("Importing sonore…");
  const runner = await (await fetch(new URL("runner.py", import.meta.url))).text();
  pyodide.FS.writeFile("/home/pyodide/runner.py", runner);
  const run = pyodide.pyimport("runner").run;
  const version = pyodide.runPython("import sonore; sonore.__version__");
  self.postMessage({ type: "ready", version, seconds: (performance.now() - started) / 1000 });
  return run;
})();
ready.catch((error) => self.postMessage({ type: "failed", text: String(error) }));

self.onmessage = async ({ data }) => {
  const run = await ready;
  const started = performance.now();
  const result = run(data.code);
  const items = result.toJs({ create_proxies: false });
  result.destroy();
  self.postMessage({ type: "result", items, seconds: (performance.now() - started) / 1000 });
};
