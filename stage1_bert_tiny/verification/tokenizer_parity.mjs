// Tokenizer parity: JS WordPiece must emit the same ids as the Python
// (Hugging Face) tokenizer recorded in fixtures.json, on every fixture text.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const fixtures = JSON.parse(readFileSync(join(ROOT, "verification", "fixtures.json"), "utf8"));
const meta = JSON.parse(readFileSync(join(ROOT, "extension", "model", "meta.json"), "utf8"));
const vocab = readFileSync(join(ROOT, "extension", "model", "vocab.txt"), "utf8");

// tokenizer.js is a classic browser script: give it `self`, then evaluate it.
globalThis.self = globalThis;
(0, eval)(readFileSync(join(ROOT, "extension", "tokenizer.js"), "utf8"));
const tokenizer = new globalThis.PhishTokenizer(vocab);

if (fixtures.length < 20) {
  console.error(`FAIL: need >= 20 fixture texts, got ${fixtures.length}`);
  process.exit(1);
}

const failed = [];
for (const fixture of fixtures) {
  const enc = tokenizer.encode(fixture.text, meta.max_length);
  // encode() returns BigInt64Array; expected_ids are plain numbers.
  const got = Array.from(enc.inputIds, Number).slice(0, enc.length); // drop padding
  const expected = fixture.expected_ids;
  const equal = got.length === expected.length && got.every((v, i) => v === expected[i]);
  if (!equal) {
    failed.push({
      shape: fixture.shape,
      text: fixture.text.slice(0, 80),
      expected,
      got,
    });
  }
}

const result = {
  n_fixtures: fixtures.length,
  n_passed: fixtures.length - failed.length,
  n_failed: failed.length,
  failed,
};
mkdirSync(join(ROOT, "results"), { recursive: true });
writeFileSync(join(ROOT, "results", "tokenizer_parity.json"), JSON.stringify(result, null, 2));

if (failed.length) {
  console.error(`FAIL tokenizer parity: ${failed.length}/${fixtures.length} texts differ`);
  process.exit(1);
}
console.log(`PASS tokenizer parity: ${fixtures.length}/${fixtures.length} texts identical`);
