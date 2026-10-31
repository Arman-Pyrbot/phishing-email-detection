// Extension smoke test: load the unpacked extension into headless Chrome and
// exercise what the loopback benchmark cannot reach - the chrome-extension://
// origin and its CSP, chrome.runtime.getURL asset loading, and the manifest
// content script. Reads results/browser_benchmark.json, so run bench.mjs first.
import { createServer } from "node:http";
import { readFileSync, writeFileSync, mkdirSync, rmSync, mkdtempSync, existsSync } from "node:fs";
import { spawn, spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const EXT_DIR = resolve(ROOT, "extension");
const PORT = 8766;
const PROB_TOLERANCE = 0.01;
const TEST_PAGE =
  '<!DOCTYPE html><html><head><meta charset="utf-8"><title>Test Page Title</title></head>' +
  '<body><h1>Sign in</h1><p>Your account will be suspended within 24 hours. ' +
  "Verify your password immediately.</p></body></html>";

function findChrome() {
  const candidates = process.env.CHROME_PATH
    ? [process.env.CHROME_PATH]
    : process.platform === "win32"
      ? [
          join(process.env.PROGRAMFILES || "C:\\Program Files", "Google", "Chrome", "Application", "chrome.exe"),
          join(process.env["PROGRAMFILES(X86)"] || "C:\\Program Files (x86)", "Google", "Chrome", "Application", "chrome.exe"),
          join(process.env.LOCALAPPDATA || "", "Google", "Chrome", "Application", "chrome.exe"),
        ]
      : ["/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"];
  const found = candidates.find((p) => p && existsSync(p));
  if (!found) throw new Error("Chrome not found; set CHROME_PATH");
  return found;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

class Cdp {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.seq = 0;
    this.pending = new Map();
    this.logs = [];
    this.contexts = [];
  }

  connect() {
    return new Promise((ok, fail) => {
      this.ws = new WebSocket(this.wsUrl);
      this.ws.onopen = () => ok();
      this.ws.onerror = () => fail(new Error("CDP websocket failed"));
      this.ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.id && this.pending.has(msg.id)) {
          const { ok: done, fail: no } = this.pending.get(msg.id);
          this.pending.delete(msg.id);
          msg.error ? no(new Error(JSON.stringify(msg.error))) : done(msg.result);
        } else if (msg.method === "Log.entryAdded" && msg.params.entry.level === "error") {
          this.logs.push(msg.params.entry.text);
        } else if (msg.method === "Runtime.executionContextCreated") {
          this.contexts.push(msg.params.context);
        }
      };
    });
  }

  send(method, params = {}, sessionId) {
    const id = ++this.seq;
    return new Promise((ok, no) => {
      this.pending.set(id, { ok, no });
      this.ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    });
  }

  /**
   * Open a tab at `url` and attach a flat session to it. The URL goes straight
   * to Target.createTarget: navigating an existing tab to a chrome-extension://
   * URL over CDP is refused with ERR_BLOCKED_BY_CLIENT. Attaching only after
   * `settleMs` means Runtime.enable replays the contexts the page created,
   * including the content script's isolated world.
   */
  async open(url, settleMs = 0) {
    const { targetId } = await this.send("Target.createTarget", { url });
    if (settleMs) await sleep(settleMs);
    const { sessionId } = await this.send("Target.attachToTarget", { targetId, flatten: true });
    await this.send("Runtime.enable", {}, sessionId);
    await this.send("Log.enable", {}, sessionId);
    return sessionId;
  }

  async eval(sessionId, expression, contextId) {
    const res = await this.send(
      "Runtime.evaluate",
      { expression, awaitPromise: true, returnByValue: true, ...(contextId ? { contextId } : {}) },
      sessionId
    );
    if (res.exceptionDetails) {
      const d = res.exceptionDetails;
      throw new Error(d.exception?.description || d.text || "evaluate failed");
    }
    return res.result.value;
  }
}

async function cdpReady(profileDir) {
  // --remote-debugging-port=0 makes Chrome pick a free port and record it, so a
  // leftover browser from an earlier run can never be mistaken for this one.
  const portFile = join(profileDir, "DevToolsActivePort");
  for (let i = 0; i < 60; i++) {
    if (existsSync(portFile)) {
      const port = Number(readFileSync(portFile, "utf8").split("\n")[0]);
      if (port > 0) {
        try {
          const res = await fetch(`http://127.0.0.1:${port}/json/version`);
          if (res.ok) return { port, wsUrl: (await res.json()).webSocketDebuggerUrl };
        } catch {
          /* not accepting connections yet */
        }
      }
    }
    await sleep(500);
  }
  throw new Error("Chrome DevTools endpoint never came up");
}

const server = createServer((_req, res) => {
  res.writeHead(200, { "Content-Type": "text/html" }).end(TEST_PAGE);
});

let chrome = null;
let cdp = null;
const profileDir = mkdtempSync(join(tmpdir(), "phish-smoke-"));

function cleanup() {
  try { server.close(); } catch { /* already closed */ }
  try { cdp?.ws?.close(); } catch { /* already closed */ }
  if (chrome && chrome.exitCode === null) {
    if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(chrome.pid), "/T", "/F"]);
    else chrome.kill("SIGKILL");
  }
}

function finish(result, failures) {
  cleanup();
  mkdirSync(join(ROOT, "results"), { recursive: true });
  writeFileSync(join(ROOT, "results", "extension_smoke.json"), JSON.stringify(result, null, 2));
  for (const [name, detail] of Object.entries(result.checks)) {
    console.log(`  ${detail.passed ? "PASS" : "FAIL"} ${name}: ${detail.detail}`);
  }
  if (failures.length) {
    console.error(`FAIL extension smoke test: ${failures.join("; ")}`);
    process.exit(1);
  }
  console.log("PASS extension smoke test (unpacked extension in headless Chrome)");
  process.exit(0);
}

/** Inference exactly as popup.js performs it, on the chrome-extension:// origin. */
const INFER_EXPR = (text) => `(async () => {
  const base = chrome.runtime.getURL("model/");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  ort.env.wasm.wasmPaths = chrome.runtime.getURL("vendor/");
  ort.env.logLevel = "error";
  const [vocab, meta, bytes] = await Promise.all([
    fetch(base + "vocab.txt").then((r) => r.text()),
    fetch(base + "meta.json").then((r) => r.json()),
    fetch(base + "model.onnx").then((r) => r.arrayBuffer()),
  ]);
  const session = await ort.InferenceSession.create(new Uint8Array(bytes), {
    executionProviders: ["wasm"], graphOptimizationLevel: "all",
  });
  const enc = new PhishTokenizer(vocab).encode(${JSON.stringify(text)}, meta.max_length);
  const shape = [1, meta.max_length];
  const out = await session.run({
    input_ids: new ort.Tensor("int64", enc.inputIds, shape),
    attention_mask: new ort.Tensor("int64", enc.attentionMask, shape),
    token_type_ids: new ort.Tensor("int64", enc.tokenTypeIds, shape),
  });
  const [a, b] = out.logits.data;
  const peak = Math.max(a, b);
  const e0 = Math.exp(a - peak), e1 = Math.exp(b - peak);
  return { model_bytes: bytes.byteLength, p_phishing: e1 / (e0 + e1) };
})()`;

(async () => {
  const checks = {};
  const failures = [];
  const record = (name, passed, detail) => {
    checks[name] = { passed, detail };
    if (!passed) failures.push(name);
    return passed;
  };

  try {
    await new Promise((r) => server.listen(PORT, "127.0.0.1", r));
    chrome = spawn(
      findChrome(),
      [
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        // Chrome 137+ ignores --load-extension, and the Extensions domain that
        // replaces it needs this flag. Test-only: never a runtime dependency.
        "--enable-unsafe-extension-debugging",
        `--user-data-dir=${profileDir}`,
        "--remote-debugging-port=0",
        "about:blank",
      ],
      { stdio: ["ignore", "ignore", "pipe"] }
    );

    const { port: cdpPort, wsUrl } = await cdpReady(profileDir);
    cdp = new Cdp(wsUrl);
    await cdp.connect();
    const { id: extId } = await cdp.send("Extensions.loadUnpacked", { path: EXT_DIR });

    // loadUnpacked returns as soon as the manifest parses; the extension only
    // becomes routable a little later, so poll until Chrome reports it enabled.
    let installed = null;
    for (let i = 0; i < 20 && !installed?.enabled; i++) {
      await sleep(1000);
      const { extensions } = await cdp
        .send("Extensions.getExtensions", {})
        .catch(() => ({ extensions: [] }));
      installed = extensions.find((e) => e.id === extId) || installed;
    }
    if (!record("extension_loads", Boolean(installed?.enabled),
      `loaded "${installed?.name}" v${installed?.version} id=${extId} enabled=${installed?.enabled}`)) {
      throw new Error("the unpacked extension never became enabled");
    }

    // ---- the manifest content script on a real page -----------------------
    cdp.contexts = [];
    const page = await cdp.open(`http://127.0.0.1:${PORT}/`, 2500);
    await sleep(1000); // let the replayed executionContextCreated events land
    const world = cdp.contexts.find(
      (c) => c.origin === `chrome-extension://${extId}` && c.auxData?.isDefault === false
    );
    const hasListener = world
      ? await cdp.eval(page, "chrome.runtime.onMessage.hasListeners()", world.id)
      : null;
    record("content_script_injected", Boolean(world) && hasListener === true,
      world
        ? `isolated world "${world.name}" on the test page, message listener registered = ${hasListener}`
        : `no extension isolated world (contexts seen: ${cdp.contexts.length})`);

    // ---- the popup page, on the extension origin -------------------------
    let popup = null;
    let runtimeId = null;
    for (let attempt = 1; attempt <= 4 && runtimeId !== extId; attempt++) {
      popup = await cdp.open(`chrome-extension://${extId}/popup.html`, 1500);
      runtimeId = await cdp.eval(popup, "chrome.runtime && chrome.runtime.id").catch(() => null);
      if (runtimeId !== extId) await sleep(2500);
    }
    record("popup_origin", runtimeId === extId, `chrome.runtime.id on popup.html = ${runtimeId}`);

    // popup.js runs scan() on load; on a chrome-extension:// tab its own guard
    // fires, which still proves the whole script chain parsed under the CSP.
    const status = await cdp.eval(popup, "document.getElementById('status').textContent");
    record("popup_scripts_run", /cannot be scanned/i.test(status || ""),
      `#status after load = ${JSON.stringify(status)}`);

    // ---- bundled assets + real inference on the extension origin ---------
    const bench = JSON.parse(readFileSync(join(ROOT, "results", "browser_benchmark.json"), "utf8"));
    const fixtures = JSON.parse(readFileSync(join(ROOT, "verification", "fixtures.json"), "utf8"));
    const expected = bench.model_parity.rows[0].browser_int8;
    const infer = await cdp.eval(popup, INFER_EXPR(fixtures[0].text));

    const shippedBytes = readFileSync(join(EXT_DIR, "model", "model.onnx")).length;
    record("bundled_assets_load", infer.model_bytes === shippedBytes,
      `model.onnx ${infer.model_bytes} bytes fetched via chrome.runtime.getURL`);
    const drift = Math.abs(infer.p_phishing - expected);
    record("csp_wasm_inference", drift <= PROB_TOLERANCE,
      `P(phishing) = ${infer.p_phishing.toFixed(6)} vs benchmarked ${expected} (diff ${drift.toFixed(6)})`);

    const cspErrors = cdp.logs.filter((t) => /Content Security Policy|Refused to/i.test(t));
    record("no_csp_violations", cspErrors.length === 0,
      cspErrors.length ? cspErrors.join(" | ") : "no CSP violations logged");

    const version = await fetch(`http://127.0.0.1:${cdpPort}/json/version`).then((r) => r.json());
    finish({
      methodology:
        "headless Chrome, unpacked extension loaded via the DevTools Extensions " +
        "domain; all assertions run on the chrome-extension:// origin",
      chrome: version.Browser,
      extension_id: extId,
      test_page: `http://127.0.0.1:${PORT}/`,
      prob_tolerance: PROB_TOLERANCE,
      checks,
      passed: failures.length === 0,
    }, failures);
  } catch (e) {
    record("harness", false, String((e && e.message) || e));
    finish({
      methodology: "headless Chrome, unpacked extension loaded via the DevTools Extensions domain",
      error: String((e && e.message) || e),
      checks,
      passed: false,
    }, failures);
  }
})();

process.on("exit", () => {
  try {
    rmSync(profileDir, { recursive: true, force: true });
  } catch {
    // Chrome may still hold handles briefly; the temp dir is disposable.
  }
});
