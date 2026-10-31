"""Run the full pipeline end-to-end and assemble results/metrics.json.

    .venv/Scripts/python training/run.py

Stages: fetch extra corpora -> prepare data -> fine-tune -> export + quantise
-> JS tokenizer parity -> offline audit -> headless-Chrome benchmark
-> extension smoke test -> extraction check -> merge every fragment into
results/metrics.json. Fails loudly at the first failing stage.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "training"
RESULTS = ROOT / "results"
VERIFICATION = ROOT / "verification"


def step(title: str, cmd: list[str], cwd: Path) -> None:
    print(f"\n=== {title} ===", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def assemble_metrics() -> None:
    data = load("data_stats.json")
    train = load("train_report.json")
    quant = load("quantization_report.json")
    browser_path = RESULTS / "browser_benchmark.json"
    browser = load("browser_benchmark.json") if browser_path.exists() else {
        "status": "not measured (verification/bench.mjs did not run)"
    }
    smoke_path = RESULTS / "extension_smoke.json"
    smoke = load("extension_smoke.json") if smoke_path.exists() else {
        "status": "not run (verification/extension_smoke.mjs did not run)"
    }
    extract_path = RESULTS / "extraction_check.json"
    extraction = load("extraction_check.json") if extract_path.exists() else {
        "status": "not run (verification/extraction_check.mjs did not run)"
    }
    metrics = {
        "data": {
            "dataset": data["dataset"],
            "deduplication": data["deduplication"],
            "split": data["split"],
        },
        "preprocessing": {
            "normalization_steps": data["preprocessing"],
            **train["preprocessing"],
        },
        "model": {
            **train["model"],
            "quantization_settings": quant["quantization"],
        },
        "training": train["training"],
        "evaluation_test_set": train["evaluation_test_set"],
        "quantization_impact": {
            "delta_int8_minus_float32": quant["delta_int8_minus_float32"],
            "test_set_float32_onnx": quant["test_set_float32_onnx"],
            "test_set_int8_shipped": quant["test_set_int8_shipped"],
            "verdict_disagreements": quant["verdict_disagreements_fp32_vs_int8"],
            "fixture_drift": quant["parity"],
            "external_holdout": quant["external_holdout_int8"],
        },
        "browser": browser,
        "extension_smoke": smoke,
        "extraction_check": extraction,
        "environment": train["environment"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nassembled {RESULTS / 'metrics.json'}")


def main() -> None:
    node = shutil.which("node") or "node"
    py = sys.executable
    step("fetch extra corpora", [py, "fetch_external.py"], TRAINING)
    step("prepare data", [py, "prepare_data.py"], TRAINING)
    step("fine-tune model", [py, "train.py"], TRAINING)
    step("export + quantise", [py, "export.py"], TRAINING)
    step("tokenizer parity (Python vs JS)", [node, "tokenizer_parity.mjs"], VERIFICATION)
    step("offline audit", [node, "offline_audit.mjs"], VERIFICATION)
    step("browser benchmark", [node, "bench.mjs"], VERIFICATION)
    step("extension smoke test", [node, "extension_smoke.mjs"], VERIFICATION)
    step("extraction check (demo pages)", [node, "extraction_check.mjs"], VERIFICATION)
    assemble_metrics()
    print("\npipeline complete.")


if __name__ == "__main__":
    main()
