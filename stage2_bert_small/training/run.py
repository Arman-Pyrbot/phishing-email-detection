"""One command: data -> train -> export -> checks -> metrics.json."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TR, RES, VER = ROOT / "training", ROOT / "results", ROOT / "verification"


def step(title: str, cmd: list[str], cwd: Path) -> None:
    print(f"\n=== {title} ===", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load(n: str) -> dict:
    return json.loads((RES / n).read_text())


def assemble() -> None:
    data, train, quant = load("data_stats.json"), load("train_report.json"), load("quantization_report.json")
    opt = lambda n, d: load(n) if (RES / n).exists() else {"status": d}
    metrics = {"data": data, "model": train["model"], "training": train["training"],
               "preprocessing": train["preprocessing"],
               "evaluation_test_set": train["evaluation_test_set"],
               "quantization_impact": quant, "browser": opt("browser_benchmark.json", "not measured"),
               "extension_smoke": opt("extension_smoke.json", "not run"),
               "extraction_check": opt("extraction_check.json", "not run"),
               "environment": train["environment"]}
    (RES / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nassembled {RES / 'metrics.json'}")


def main() -> None:
    node, py = shutil.which("node") or "node", sys.executable
    step("prepare data", [py, "data.py"], TR)
    step("fine-tune", [py, "train.py"], TR)
    step("export + quantise", [py, "export.py"], TR)
    step("tokenizer parity", [node, "tokenizer_parity.mjs"], VER)
    step("offline audit", [node, "offline_audit.mjs"], VER)
    step("browser benchmark", [node, "bench.mjs"], VER)
    step("extension smoke", [node, "extension_smoke.mjs"], VER)
    step("extraction check", [node, "extraction_check.mjs"], VER)
    assemble()
    print("\npipeline complete.")


if __name__ == "__main__":
    main()
