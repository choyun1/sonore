// Checks docs/live/ in headless Chromium with a fake microphone (L7 of
// docs/design/live-page.md): the page draws, and the F0s it streamed equal
// f0_track(..., delay=0.05) on the samples its worker received. Local only,
// like tools/pyodide_timing/ (it downloads Pyodide). From the repository root:
//     node tools/live_page/check.js [seconds]
// It builds a wheel of the working tree, plays docs/speech/bdl_arctic_a0131
// as the microphone (Chromium loops the file), and saves a screenshot to
// tools/live_page/screenshot.png.
const { execSync, spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

(async () => {
  const seconds = Number(process.argv[2] || 8);
  const here = path.join("tools", "live_page");
  for (const old of fs.readdirSync(here).filter((name) => name.endsWith(".whl"))) fs.unlinkSync(path.join(here, old));
  execSync(`pip wheel -q --no-deps -w ${here} .`, { stdio: "inherit" });
  const wheel = fs.readdirSync(here).find((name) => name.endsWith(".whl"));
  const wav = path.resolve(here, "speech.wav");
  execSync(`python -c "import soundfile as sf; d, fs = sf.read('docs/speech/bdl_arctic_a0131.flac'); sf.write(r'${wav}', d, fs, subtype='PCM_16')"`);
  const server = spawn("python", ["-m", "http.server", "8766"], { stdio: "ignore" });
  await new Promise((resolve) => setTimeout(resolve, 1000));
  const browser = await chromium.launch({
    args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-audio-capture=${wav}`,
      "--autoplay-policy=no-user-gesture-required"],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1000, height: 900 } });
    page.on("pageerror", (error) => console.error("page error:", error.message));
    page.on("console", (message) => { if (message.type() === "error") console.error("console:", message.text()); });
    await page.goto(`http://localhost:8766/docs/live/?wheel=/tools/live_page/${wheel}&keep=1`);
    await page.click("#start");
    await page.waitForFunction(() => document.getElementById("status").textContent.startsWith("Listening"), null, { timeout: 600000 });
    await page.waitForTimeout(seconds * 1000);
    await page.evaluate(() => window.checkStreams());
    await page.waitForFunction(() => window.CHECK, null, { timeout: 120000 });
    const check = await page.evaluate(() => window.CHECK);
    const readout = await page.evaluate(() => [document.getElementById("f0").textContent, document.getElementById("cost").textContent, document.getElementById("status").textContent]);
    await page.screenshot({ path: path.join(here, "screenshot.png"), fullPage: true });
    console.log(JSON.stringify({ check, readout }, null, 1));
    const ok = check.windows > 100 && check.same_times && check.max_difference === 0 && check.voiced > 0;
    if (check.restarts) console.log(`note: the page restarted ${check.restarts} time(s) after falling behind`);
    console.log(ok ? "OK: the page's F0s equal f0_track on the samples it received" : "FAILED");
    process.exitCode = ok ? 0 : 1;
  } finally {
    await browser.close();
    server.kill();
  }
})();
