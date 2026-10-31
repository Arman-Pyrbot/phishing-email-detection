/* Popup: loads the bundled ONNX model, classifies the page, shows the verdict.
 *
 * Inference lives here (not in a service worker) so the MV3 worker lifecycle
 * can never tear down the WASM session: the model is loaded once per popup
 * open and released when it closes. Everything is fetched from the
 * extension's own bundle - zero network at inference time.
 */
(() => {
  "use strict";

  const CAPS = { url: 120, title: 200, body: 1200 };
  const MIN_BODY_CHARS = 200;
  // A webmail client's own URL and title describe the client, not the message
  // being read: on Gmail the page is "mail.google.com" titled "... - Gmail".
  // Sending those alongside the message makes the model classify the client
  // instead of the mail, which is how four phishing emails came back legitimate.
  const MAIL_CLIENT = /(^|\.)(mail|webmail|email|outlook|imap|exchange|roundcube)\.|(gmail|outlook|yahoo|proton|zoho|fastmail|tutanota|icloud|aol)\./i;
  const engine = { session: null, tokenizer: null, meta: null, loadMs: 0 };

  const el = (id) => document.getElementById(id);

  function setStatus(text, isError = false) {
    const node = el("status");
    node.textContent = text;
    node.className = isError ? "error" : "";
  }

  async function initEngine() {
    if (engine.session) return;
    const t0 = performance.now();
    ort.env.wasm.numThreads = 1; // extension pages are not cross-origin isolated
    ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths = chrome.runtime.getURL("vendor/");
    ort.env.logLevel = "error";
    const [vocabText, meta] = await Promise.all([
      fetch(chrome.runtime.getURL("model/vocab.txt")).then((r) => r.text()),
      fetch(chrome.runtime.getURL("model/meta.json")).then((r) => r.json()),
    ]);
    engine.tokenizer = new PhishTokenizer(vocabText);
    engine.meta = meta;
    const bytes = await fetch(chrome.runtime.getURL("model/model.onnx"))
      .then((r) => r.arrayBuffer());
    engine.session = await ort.InferenceSession.create(new Uint8Array(bytes), {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    engine.loadMs = performance.now() - t0;
  }

  async function classify(doc) {
    const { session, tokenizer, meta } = engine;
    const enc = tokenizer.encode(doc, meta.max_length);
    const shape = [1, meta.max_length];
    const out = await session.run({
      input_ids: new ort.Tensor("int64", enc.inputIds, shape),
      attention_mask: new ort.Tensor("int64", enc.attentionMask, shape),
      token_type_ids: new ort.Tensor("int64", enc.tokenTypeIds, shape),
    });
    const [logit0, logit1] = out.logits.data; // class 0 = legitimate, 1 = phishing
    const peak = Math.max(logit0, logit1);
    const exp0 = Math.exp(logit0 - peak);
    const exp1 = Math.exp(logit1 - peak);
    const pPhishing = exp1 / (exp0 + exp1);
    const label = pPhishing >= meta.threshold ? meta.labels[1] : meta.labels[0];
    return {
      label,
      confidence: label === meta.labels[1] ? pPhishing : 1 - pPhishing,
      pPhishing,
    };
  }

  /** Hostname of a URL, or "" if it will not parse. */
  function hostOf(url) {
    try {
      return new URL(url).hostname;
    } catch (e) {
      return "";
    }
  }

  function buildDoc(page) {
    // The model knows two document shapes and never their combination: `[url]`
    // + `[title]` for URL pages, and `[from]` + `[body]` for emails, where
    // [body] carries the subject on its first line. Sending url, title and body
    // together makes the model read the document as a URL row and judge the
    // page's own URL.
    //
    // So a page with real text is sent as a body, and the url is included only
    // when it describes the content rather than a mail client hosting it. A
    // page too thin to classify falls back to the url + title shape.
    const text = (page.text || "").trim();
    const url = (page.url || "").split("#")[0];
    const parts = [];
    if (text.length >= MIN_BODY_CHARS) {
      if (url && !MAIL_CLIENT.test(hostOf(url))) parts.push(`[url] ${url.slice(0, CAPS.url)}`);
      parts.push(`[body] ${text.slice(0, CAPS.body)}`);
      return parts.join("\n");
    }
    if (url) parts.push(`[url] ${url.slice(0, CAPS.url)}`);
    if (page.title) parts.push(`[title] ${page.title.slice(0, CAPS.title)}`);
    return parts.join("\n");
  }

  async function extractPage() {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || tab.id === undefined) throw new Error("no active tab");
    if (!/^https?:\/\//i.test(tab.url || "")) throw new Error("this page cannot be scanned");
    const page = await chrome.tabs.sendMessage(tab.id, { type: "EXTRACT_PAGE_TEXT" });
    if (!page) throw new Error("page did not respond (reload the tab)");
    return page;
  }

  function render(page, result, inferMs) {
    el("status").textContent = "";
    const verdict = el("verdict");
    verdict.className = "visible " + result.label;
    el("label").textContent = result.label;
    const pct = (x) => Math.round(x * 100);
    el("confidence").textContent =
      `${pct(result.confidence)}% confident it is ${result.label}`;
    el("barFill").style.width = `${pct(result.pPhishing)}%`;
    let host = page.url;
    try { host = new URL(page.url).hostname; } catch (e) { /* keep full url */ }
    const loadNote = engine.loadMs ? ` · model load ${Math.round(engine.loadMs)} ms` : "";
    el("detail").textContent =
      `${host} · P(phishing) = ${result.pPhishing.toFixed(3)} · ` +
      `inference ${Math.round(inferMs)} ms${loadNote} · offline`;
  }

  async function scan() {
    const verdict = el("verdict");
    verdict.className = "";
    try {
      setStatus("reading page…");
      const page = await extractPage();
      setStatus(engine.session ? "classifying…" : "loading model…");
      await initEngine();
      setStatus("classifying…");
      const t0 = performance.now();
      const result = await classify(buildDoc(page));
      render(page, result, performance.now() - t0);
    } catch (e) {
      setStatus(String((e && e.message) || e), true);
    }
  }

  el("rescan").addEventListener("click", scan);
  scan();
})();
