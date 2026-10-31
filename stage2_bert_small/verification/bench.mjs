// Browser benchmark: serve the project on loopback, open bench.html in
// headless Chrome, collect its POSTed results into results/browser_benchmark.json.
import { createServer } from "node:http";
import { readFileSync, writeFileSync, existsSync, mkdirSync, rmSync, mkdtempSync, statSync } from "node:fs";
import { spawn, spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { dirname, join, normalize, extname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const PORT = 8765;
const TIMEOUT_MS = 120000;
const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".txt": "text/plain", ".wasm": "application/wasm",
  ".onnx": "application/octet-stream", ".png": "image/png",
};

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

const server = createServer((req, res) => {
  if (req.method === "POST" && req.url === "/__bench_done") {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
      res.writeHead(200).end("ok");
      finish(JSON.parse(body));
    });
    return;
  }
  const path = normalize(req.url.split("?")[0]).replace(/^([/\\])+/, "");
  const file = join(ROOT, path);
  const safe = file.startsWith(ROOT) && existsSync(file) && statSync(file).isFile();
  if (!safe) {
    res.writeHead(404).end("not found");
    return;
  }
  res.writeHead(200, { "Content-Type": MIME[extname(file)] || "application/octet-stream" });
  res.end(readFileSync(file));
});

let chrome = null;
let done = false;
function finish(result) {
  if (done) return;
  done = true;
  cleanup();
  if (result.error) {
    console.error(`FAIL browser benchmark: ${result.error}`);
    process.exit(1);
  }
  mkdirSync(join(ROOT, "results"), { recursive: true });
  writeFileSync(join(ROOT, "results", "browser_benchmark.json"), JSON.stringify(result, null, 2));
  const parity = result.model_parity;
  console.log(
    `browser: load ${result.model_load_ms} ms, ` +
    `inference p50 ${result.inference_ms.median} ms / p95 ${result.inference_ms.p95} ms`
  );
  console.log(
    `  fp32 runtime equivalence: max ${parity.fp32_runtime_equivalence.max_abs_diff} ` +
    `(tolerance ${parity.fp32_runtime_equivalence.tolerance})`
  );
  console.log(
    `  shipped int8 vs reference: max ${parity.shipped_int8_vs_reference.max_abs_diff} / ` +
    `mean ${parity.shipped_int8_vs_reference.mean_abs_diff} / ` +
    `${parity.shipped_int8_vs_reference.verdict_mismatches} verdict mismatches`
  );
  console.log(
    `  int8 cross-runtime (reported): max ${parity.int8_cross_runtime.max_abs_diff} / ` +
    `mean ${parity.int8_cross_runtime.mean_abs_diff} / ` +
    `${parity.int8_cross_runtime.verdict_mismatches} verdict mismatches`
  );
  if (!parity.passed) {
    console.error("FAIL browser model parity");
    process.exit(1);
  }
  console.log("PASS browser benchmark + model parity");
}

function cleanup() {
  server.close();
  if (chrome && chrome.exitCode === null) {
    if (process.platform === "win32") {
      spawnSync("taskkill", ["/pid", String(chrome.pid), "/T", "/F"]);
    } else {
      chrome.kill("SIGKILL");
    }
  }
}

const profileDir = mkdtempSync(join(tmpdir(), "phish-bench-"));
server.listen(PORT, "127.0.0.1", () => {
  const chromePath = findChrome();
  chrome = spawn(
    chromePath,
    [
      "--headless=new",
      "--disable-gpu",
      "--disable-dev-shm-usage",
      "--no-first-run",
      "--no-default-browser-check",
      `--user-data-dir=${profileDir}`,
      `http://127.0.0.1:${PORT}/verification/bench.html`,
    ],
    { stdio: ["ignore", "ignore", "pipe"] },
  );
  let stderrTail = "";
  chrome.stderr.on("data", (d) => {
    stderrTail = (stderrTail + d.toString()).slice(-2000);
  });
  chrome.on("exit", (code) => {
    if (!done) {
      cleanup();
      console.error(`FAIL: Chrome exited early (code ${code}):\n${stderrTail}`);
      process.exit(1);
    }
  });
});

setTimeout(() => {
  if (!done) {
    cleanup();
    console.error(`FAIL: benchmark timed out after ${TIMEOUT_MS / 1000}s`);
    process.exit(1);
  }
}, TIMEOUT_MS).unref?.();
process.on("exit", () => {
  try {
    rmSync(profileDir, { recursive: true, force: true });
  } catch {
    // Chrome may still hold handles briefly after taskkill; temp dir is disposable.
  }
});
