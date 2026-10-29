// Opens index.html in headless Chromium (Playwright) and prints its table.
// Builds the wheel and serves the repository root itself; run from the root:
//     node tools/pyodide_timing/run.js
const { execSync, spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

(async () => {
  const here = path.join("tools", "pyodide_timing");
  for (const old of fs.readdirSync(here).filter((name) => name.endsWith(".whl"))) fs.unlinkSync(path.join(here, old));
  execSync(`pip wheel -q --no-deps -w ${here} .`, { stdio: "inherit" });
  const wheel = fs.readdirSync(here).find((name) => name.endsWith(".whl"));
  fs.writeFileSync(path.join(here, "wheel.txt"), wheel);
  const server = spawn("python", ["-m", "http.server", "8765"], { stdio: "ignore" });
  await new Promise((resolve) => setTimeout(resolve, 1000));
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    page.on("pageerror", (error) => console.error(error.message));
    await page.goto("http://localhost:8765/tools/pyodide_timing/index.html");
    await page.waitForFunction(() => window.TABLE, null, { timeout: 900000 });
    console.log(await page.evaluate(() => window.TABLE));
  } finally {
    await browser.close();
    server.kill();
  }
})();
