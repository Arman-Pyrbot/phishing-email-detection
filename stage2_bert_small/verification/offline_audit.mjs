// Offline audit: the extension must load everything from its own bundle.
// Flags remote resource loads in authored files, required permissions, and
// manifest references that do not exist on disk. vendor/ and model/ are
// exempt from content scanning (vendored binaries/minified runtime).
import { readFileSync, writeFileSync, existsSync, readdirSync, mkdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const EXT = join(ROOT, "extension");
const AUTHORED = ["manifest.json", "popup.html", "popup.js", "content.js", "tokenizer.js"];
const violations = [];

// 1. Remote-loading patterns in authored files.
const REMOTE_PATTERNS = [
  [/<script[^>]+src\s*=\s*["']https?:/i, "remote <script src>"],
  [/<link[^>]+href\s*=\s*["']https?:/i, "remote <link href>"],
  [/\bfetch\s*\(\s*["']https?:/i, "fetch of remote URL"],
  [/\bimportScripts\s*\(\s*["']https?:/i, "importScripts of remote URL"],
  [/@import\s+url\s*\(\s*["']?https?:/i, "remote CSS @import"],
  [/src\s*=\s*["']\/\//, "protocol-relative src"],
  [/\bfrom\s+["']https?:/i, "remote ES module import"],
];
for (const file of AUTHORED) {
  const text = readFileSync(join(EXT, file), "utf8");
  for (const [pattern, label] of REMOTE_PATTERNS) {
    if (pattern.test(text)) violations.push(`${file}: ${label}`);
  }
}

// 2. Manifest permissions must stay minimal, and every referenced path must exist.
const manifest = JSON.parse(readFileSync(join(EXT, "manifest.json"), "utf8"));
const permissions = manifest.permissions || [];
const allowed = new Set(["activeTab"]);
for (const p of permissions) {
  if (!allowed.has(p)) violations.push(`manifest: unexpected permission "${p}"`);
}
if (manifest.host_permissions) violations.push("manifest: host_permissions present");

const refs = [
  manifest.action?.default_popup,
  manifest.background?.service_worker,
  ...(manifest.content_scripts || []).flatMap((cs) => cs.js || []),
  ...Object.values(manifest.icons || {}),
  ...Object.values(manifest.action?.default_icon || {}),
];
for (const ref of refs.filter(Boolean)) {
  if (!existsSync(join(EXT, ref))) violations.push(`manifest: missing file "${ref}"`);
}

// 3. Required bundled runtime + model files must exist.
for (const required of [
  "vendor/ort.wasm.min.js",
  "vendor/ort-wasm-simd.wasm",
  "model/model.onnx",
  "model/vocab.txt",
  "model/meta.json",
]) {
  if (!existsSync(join(EXT, required))) violations.push(`missing bundled file "${required}"`);
}

const result = {
  authored_files_scanned: AUTHORED,
  remote_load_violations: violations.filter((v) => !v.startsWith("manifest: missing")),
  manifest_reference_ok: !violations.some((v) => v.startsWith("manifest:")),
  permissions,
  passed: violations.length === 0,
  violations,
};
mkdirSync(join(ROOT, "results"), { recursive: true });
writeFileSync(join(ROOT, "results", "offline_audit.json"), JSON.stringify(result, null, 2));

if (violations.length) {
  console.error("FAIL offline audit:");
  for (const v of violations) console.error("  - " + v);
  process.exit(1);
}
console.log(`PASS offline audit: ${AUTHORED.length} files, permissions [${permissions.join(", ")}], all references local`);
