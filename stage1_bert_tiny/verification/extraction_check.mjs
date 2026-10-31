// Extraction check: run the real extension/content.js against every demo page,
// build the document exactly as popup.js builds it, score it with the shipped
// model, and assert the expected verdict.
//
// This exists because a prefix of body.innerText looked fine on simple pages but
// scored a phishing email open in Gmail as "legitimate" - navigation chrome ate
// the token budget before the message was reached. Needs bench.mjs first.
import { createServer } from "node:http";
import { readFileSync, writeFileSync, existsSync, mkdirSync, rmSync, mkdtempSync, statSync } from "node:fs";
import { spawn, spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { dirname, join, resolve, normalize, extname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const EXT = join(ROOT, "extension");
const PORT = 8768;
const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".txt": "text/plain", ".png": "image/png",
  ".wasm": "application/wasm", ".onnx": "application/octet-stream",
};

// page -> the verdict the extension must show for it.
const CASES = [
  ["demo/phishing-login.html", "phishing"],
  ["demo/phishing-invoice.html", "phishing"],
  ["demo/webmail-lookalike.html", "phishing"],
  ["demo/webmail-iframe.html", "phishing"],
  ["demo/index.html", "legitimate"],
  ["demo/legit-article.html", "legitimate"],
  ["demo/legit-shop.html", "legitimate"],
];

// Read the caps and the body threshold out of popup.js so this check cannot
// drift from the extension.
const popupSrc = readFileSync(join(EXT, "popup.js"), "utf8");
const capsMatch = popupSrc.match(/const CAPS = \{([^}]*)\}/);
if (!capsMatch) throw new Error("could not read CAPS from extension/popup.js");
const CAPS = Object.fromEntries(
  capsMatch[1].split(",").map((pair) => {
    const [key, value] = pair.split(":");
    return [key.trim(), Number(value)];
  })
);
const minBodyMatch = popupSrc.match(/const MIN_BODY_CHARS = (\d+)/);
if (!minBodyMatch) throw new Error("could not read MIN_BODY_CHARS from extension/popup.js");
const MIN_BODY_CHARS = Number(minBodyMatch[1]);
const mailMatch = popupSrc.match(/const MAIL_CLIENT = \/(.+)\/i;/);
if (!mailMatch) throw new Error("could not read MAIL_CLIENT from extension/popup.js");
const MAIL_CLIENT = new RegExp(mailMatch[1], "i");
const contentSrc = readFileSync(join(EXT, "content.js"), "utf8");

// Documents that must classify correctly no matter which page they came from.
// These reproduce the reported bug: four phishing emails open in Gmail were all
// reported legitimate because the extension fed the model the *page's* URL and
// title - "mail.google.com" and "... - Gmail" - which describe the mail client,
// not the message. A mail host's own URL and title must not override the body.
const GMAIL_URL = "https://mail.google.com/mail/u/1/";
const T = (s) => s.replace(/\s+/g, " ").trim();
const DOCUMENTS = [
  ["webmail: polite BEC invoice", "phishing",
    "Action required: Updated payment details for invoice #INV-48217 - armanpytbot@gmail.com - Gmail",
    T(`Hi Arman, I am following up on invoice #INV-48217 for the September software renewal. Our
      finance team recently updated the payment portal, and the previous payment link is no longer
      active. Please review the invoice and confirm the billing details using the secure portal
      below. https://billing.review.example.net/invoice/INV-48217 The outstanding amount is 16,450
      and the payment is currently scheduled for processing on 24 September. Regards, Rohan Mehta
      Accounts & Vendor Operations Clearview Technologies`)],
  ["webmail: shared document", "phishing",
    "Re: Shared document - Arman, action required - armanpytbot@gmail.com - Gmail",
    T(`Hi Arman, A document has been shared with you in Microsoft 365. Q3 Internship Evaluation -
      Confidential The document is currently pending your review. Due to a recent security update,
      you will need to re-authenticate your Microsoft account before it can be opened. Open
      document: https://microsoft365-document-review.example.net/share/GF42KQ1 Please complete the
      verification by 5:00 PM today. Microsoft 365 Support Document Sharing Service`)],
  ["webmail: invoice payment required", "phishing",
    "Phishing test email - invoice - armanpytbot@gmail.com - Gmail",
    T(`From: Billing Department billing.notice@example.net To: Arman Subject: Invoice INV-2291 -
      Payment Required Hi Arman, We were unable to confirm the payment details associated with your
      account. Invoice INV-2291 is currently overdue, with a total outstanding balance of 1,380.00
      including a late payment fee. To avoid interruption of your service, please review your
      billing information and confirm your payment method. Review invoice and payment details:
      https://billing.acct-check.example.net/invoice/INV-2291 Please complete this process within
      24 hours to prevent your account from being temporarily suspended. Billing Department`)],
  ["webmail: account security alert", "phishing",
    "Phishing test email - account security - armanpytbot@gmail.com - Gmail",
    T(`From: PayPal Security alert@example.net To: Arman Subject: Unusual sign-in activity detected
      Hi Arman, We detected a sign-in to your PayPal account that we were unable to verify. For your
      protection, access to some account features has been temporarily restricted. Date: 22
      September 2026 Location: New Delhi, India Device: Windows PC If this activity was not
      performed by you, please verify your account immediately. Review account activity:
      https://paypal-security-check.example.net/verify Please complete verification within 24 hours.
      If we cannot confirm your account information, access to your account may be suspended.`)],
  ["webmail: ordinary reply (control)", "legitimate",
    "Re: lunch on Thursday - armanpytbot@gmail.com - Gmail",
    T(`Hi Arman, Thursday works for me. Shall we say 12:30 at the place near the station? I will
      book a table if that suits. Let me know if you would rather do Friday instead, my afternoon is
      free either way. Best, Rohan`)],
];

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

  async open(url, settleMs = 0) {
    const { targetId } = await this.send("Target.createTarget", { url });
    if (settleMs) await sleep(settleMs);
    const { sessionId } = await this.send("Target.attachToTarget", { targetId, flatten: true });
    await this.send("Runtime.enable", {}, sessionId);
    return sessionId;
  }

  async eval(sessionId, expression) {
    const res = await this.send(
      "Runtime.evaluate",
      { expression, awaitPromise: true, returnByValue: true },
      sessionId
    );
    if (res.exceptionDetails) {
      const d = res.exceptionDetails;
      throw new Error(d.exception?.description || d.text || "evaluate failed");
    }
    return res.result.value;
  }
}

const server = createServer((req, res) => {
  const rel = normalize(req.url.split("?")[0]).replace(/^([/\\])+/, "");
  const file = join(ROOT, rel);
  if (!file.startsWith(ROOT) || !existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404).end("not found");
    return;
  }
  res.writeHead(200, { "Content-Type": MIME[extname(file)] || "application/octet-stream" });
  res.end(readFileSync(file));
});

let chrome = null;
let cdp = null;
const profileDir = mkdtempSync(join(tmpdir(), "phish-extract-"));

function cleanup() {
  try { server.close(); } catch { /* already closed */ }
  try { cdp?.ws?.close(); } catch { /* already closed */ }
  if (chrome && chrome.exitCode === null) {
    if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(chrome.pid), "/T", "/F"]);
    else chrome.kill("SIGKILL");
  }
}

function finish(result) {
  cleanup();
  mkdirSync(join(ROOT, "results"), { recursive: true });
  writeFileSync(join(ROOT, "results", "extraction_check.json"), JSON.stringify(result, null, 2));
  for (const row of result.pages) {
    console.log(
      `  ${row.passed ? "PASS" : "FAIL"} ${row.page.padEnd(28)} ` +
      `${row.expected.padEnd(11)} -> ${row.verdict.padEnd(11)} P=${row.p_phishing.toFixed(4)}`
    );
  }
  if (!result.passed) {
    console.error("FAIL extraction check: at least one case scored the wrong verdict");
    process.exit(1);
  }
  console.log(
    `PASS extraction check: ${result.pages.length} cases ` +
    `(${CASES.length} pages, ${DOCUMENTS.length} documents), real content.js + popup.js builder`
  );
  process.exit(0);
}

/** Run the real content.js in the page and return what it would send the popup. */
const EXTRACT = (src) => `(() => {
  let captured = null;
  const shim = { runtime: { onMessage: { addListener: (fn) => { captured = fn; } } } };
  new Function("chrome", ${JSON.stringify(src)})(shim);
  if (!captured) throw new Error("content.js registered no message listener");
  let out = null;
  captured({ type: "EXTRACT_PAGE_TEXT" }, null, (response) => { out = response; });
  return out;
})()`;

/** The document popup.js would build, using the values parsed from popup.js. */
function buildDoc(page) {
  const text = (page.text || "").trim();
  const url = (page.url || "").split("#")[0];
  const parts = [];
  if (text.length >= MIN_BODY_CHARS) {
    let host = "";
    try { host = new URL(url).hostname; } catch { host = ""; }
    if (url && !MAIL_CLIENT.test(host)) parts.push(`[url] ${url.slice(0, CAPS.url)}`);
    parts.push(`[body] ${text.slice(0, CAPS.body)}`);
    return parts.join("\n");
  }
  if (url) parts.push(`[url] ${url.slice(0, CAPS.url)}`);
  if (page.title) parts.push(`[title] ${page.title.slice(0, CAPS.title)}`);
  return parts.join("\n");
}

const SCORE = (doc) => `(async () => {
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
  const enc = new PhishTokenizer(vocab).encode(${JSON.stringify(doc)}, meta.max_length);
  const shape = [1, meta.max_length];
  const out = await session.run({
    input_ids: new ort.Tensor("int64", enc.inputIds, shape),
    attention_mask: new ort.Tensor("int64", enc.attentionMask, shape),
    token_type_ids: new ort.Tensor("int64", enc.tokenTypeIds, shape),
  });
  const [a, b] = out.logits.data;
  const peak = Math.max(a, b);
  const e0 = Math.exp(a - peak), e1 = Math.exp(b - peak);
  const p = e1 / (e0 + e1);
  return { p, label: p >= meta.threshold ? meta.labels[1] : meta.labels[0], tokens: enc.length };
})()`;

async function main() {
  await new Promise((r) => server.listen(PORT, "127.0.0.1", r));
  chrome = spawn(
    findChrome(),
    [
      "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
      "--enable-unsafe-extension-debugging",
      `--user-data-dir=${profileDir}`, "--remote-debugging-port=0", "about:blank",
    ],
    { stdio: ["ignore", "ignore", "pipe"] }
  );

  const portFile = join(profileDir, "DevToolsActivePort");
  let wsUrl = null;
  for (let i = 0; i < 60 && !wsUrl; i++) {
    if (existsSync(portFile)) {
      const port = Number(readFileSync(portFile, "utf8").split("\n")[0]);
      if (port > 0) {
        try {
          const res = await fetch(`http://127.0.0.1:${port}/json/version`);
          if (res.ok) wsUrl = (await res.json()).webSocketDebuggerUrl;
        } catch { /* not ready */ }
      }
    }
    if (!wsUrl) await sleep(500);
  }
  cdp = new Cdp(wsUrl);
  await cdp.connect();

  const { id: extId } = await cdp.send("Extensions.loadUnpacked", { path: EXT });
  for (let i = 0; i < 20; i++) {
    await sleep(1000);
    const { extensions } = await cdp
      .send("Extensions.getExtensions", {})
      .catch(() => ({ extensions: [] }));
    if (extensions.some((e) => e.id === extId && e.enabled)) break;
  }
  const popup = await cdp.open(`chrome-extension://${extId}/popup.html`, 2000);

  const pages = [];
  for (const [rel, expected] of CASES) {
    const session = await cdp.open(`http://127.0.0.1:${PORT}/${rel}`, 2200);
    const page = await cdp.eval(session, EXTRACT(contentSrc));
    const { p, label } = await cdp.eval(popup, SCORE(buildDoc(page)));
    pages.push({
      page: rel,
      expected,
      verdict: label,
      p_phishing: Math.round(p * 1e6) / 1e6,
      passed: label === expected,
      body_chars: (page.text || "").length,
      body_head: (page.text || "").slice(0, 120),
    });
  }

  // Documents rather than pages: the same builder, fed a webmail client's own
  // URL and title, must still classify the message.
  for (const [name, expected, title, text] of DOCUMENTS) {
    const { p, label } = await cdp.eval(popup, SCORE(buildDoc({ url: GMAIL_URL, title, text })));
    pages.push({
      page: name,
      expected,
      verdict: label,
      p_phishing: Math.round(p * 1e6) / 1e6,
      passed: label === expected,
      body_chars: text.length,
      body_head: text.slice(0, 120),
    });
  }

  finish({
    methodology:
      "headless Chrome; the shipped extension/content.js is executed in each demo " +
      "page with a message-listener shim, the document is built with the caps read " +
      "out of extension/popup.js, and scored with the shipped int8 model",
    caps: CAPS,
    pages,
    passed: pages.every((row) => row.passed),
  });
}

main().catch((e) => {
  finish({ error: String((e && e.message) || e), pages: [], passed: false });
});

process.on("exit", () => {
  try {
    rmSync(profileDir, { recursive: true, force: true });
  } catch {
    // Chrome may still hold handles briefly; the temp dir is disposable.
  }
});
