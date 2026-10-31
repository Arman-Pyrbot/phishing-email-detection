"""Export the trained model to browser artifacts: fp32 ONNX -> dynamic int8.

Writes the shipped int8 model + vocab.txt + meta.json into extension/model/,
checks numerical parity (PyTorch vs fp32 ONNX vs int8 ONNX), measures the
quantisation impact on the full test set, and writes parity fixtures for the
JS tokenizer/model checks plus results/quantization_report.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

TRAINING_DIR = Path(__file__).resolve().parent
ROOT = TRAINING_DIR.parent
DATA = TRAINING_DIR / "data" / "processed"
MODEL_DIR = TRAINING_DIR / "out"
EXT_MODEL = ROOT / "extension" / "model"
RESULTS = ROOT / "results"
VERIFICATION = ROOT / "verification"

# Fixtures cover every document shape the extension can produce (the tagged
# [url]/[title]/[from]/[body] format) plus sampled real test rows.
FIXTURE_TEXTS = [
    ("[url] http://paypal-secure-login.tk/verify-account?id=1", "url only"),
    ("[url] https://example.com/", "url only"),
    ("[url] https://github.com/features/actions", "url only"),
    ("[url] http://192.168.0.1/login", "url only"),
    ("[url] https://github.com/features/actions\n[title] GitHub Actions", "url + title"),
    ("[url] https://www.southbankmosaics.com\n[title] Southbank Mosaics", "url + title"),
    ("[url] https://en.wikipedia.org/wiki/Phishing\n[body] Phishing is a form of social engineering.", "url + body"),
    ("[body] Your account will be suspended within 24 hours. Verify your password immediately.", "body only"),
    ("[from] security@paypal-secure-login.tk\n[body] Dear customer, confirm your billing information.", "from + body"),
    ("[url] http://paypal-secure-login.tk/verify\n[title] Sign in\n[from] security@paypal-secure-login.tk\n[body] Verify your password immediately.", "url + title + from + body"),
]
N_SAMPLED_FIXTURES = 14  # 10 crafted + 14 sampled = 24 texts for the parity tests
EVAL_BATCH = 256


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Threshold-0.5 metrics for the quantisation comparison."""
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "accuracy": round(float((pred == y).mean()), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
    }


def torch_probs(model, tokenizer, texts: list[str], max_length: int) -> np.ndarray:
    batch = tokenizer(texts, return_tensors="pt", truncation=True,
                      max_length=max_length, padding="max_length")
    with torch.no_grad():
        logits = model(**batch).logits.numpy()
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True))[:, 1]


def onnx_probs(session, tokenizer, texts: list[str], max_length: int) -> np.ndarray:
    batch = tokenizer(texts, return_tensors="np", truncation=True,
                      max_length=max_length, padding="max_length")
    feeds = {k: batch[k].astype(np.int64)
             for k in ("input_ids", "attention_mask", "token_type_ids")}
    logits = session.run(["logits"], feeds)[0]
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True))[:, 1]


def batch_onnx_probs(session, tokenizer, texts: list[str], max_length: int) -> np.ndarray:
    """Score a whole split in batches (an all-at-once batch would OOM)."""
    parts = [onnx_probs(session, tokenizer, texts[i:i + EVAL_BATCH], max_length)
             for i in range(0, len(texts), EVAL_BATCH)]
    return np.concatenate(parts)


def main() -> None:
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    report = json.loads((RESULTS / "train_report.json").read_text(encoding="utf-8"))
    max_length = int(report["training"]["max_sequence_length"])
    EXT_MODEL.mkdir(parents=True, exist_ok=True)
    VERIFICATION.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).eval()

    # ---- PyTorch -> fp32 ONNX ------------------------------------------------
    fp32_path = MODEL_DIR / "model.fp32.onnx"
    dummy = tokenizer(FIXTURE_TEXTS[0][0], return_tensors="pt",
                      truncation=True, max_length=max_length)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
            str(fp32_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={name: {0: "batch", 1: "sequence"}
                          for name in ("input_ids", "attention_mask", "token_type_ids")}
                         | {"logits": {0: "batch"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
    print(f"fp32 ONNX: {fp32_path.stat().st_size / 1e6:.1f} MB")

    # ---- parity: PyTorch vs fp32 ONNX vs int8 ONNX --------------------------
    crafted_texts = [t for t, _ in FIXTURE_TEXTS]
    ref = torch_probs(model, tokenizer, crafted_texts, max_length)
    fp32_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    fp32 = onnx_probs(fp32_session, tokenizer, crafted_texts, max_length)
    fp32_diff = float(np.abs(ref - fp32).max())
    if fp32_diff > 1e-4:
        raise AssertionError(f"fp32 ONNX diverges from PyTorch: {fp32_diff:.2e}")

    int8_path = EXT_MODEL / "model.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    int8_session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    int8 = onnx_probs(int8_session, tokenizer, crafted_texts, max_length)
    drift = np.abs(fp32 - int8)
    print(f"int8 ONNX: {int8_path.stat().st_size / 1e6:.1f} MB  "
          f"fixture drift max {drift.max():.4f} mean {drift.mean():.4f}")

    # ---- quantisation impact on the full test set ---------------------------
    test = pd.read_parquet(DATA / "test.parquet")
    y_test = test["label"].to_numpy(dtype=np.int64)
    texts = test["text"].tolist()
    p_fp32 = batch_onnx_probs(fp32_session, tokenizer, texts, max_length)
    p_int8 = batch_onnx_probs(int8_session, tokenizer, texts, max_length)
    m_fp32, m_int8 = metrics(y_test, p_fp32), metrics(y_test, p_int8)
    delta = {k: round(m_int8[k] - m_fp32[k], 6)
             for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}
    disagreements = int(((p_fp32 >= 0.5) != (p_int8 >= 0.5)).sum())

    # ---- external holdout: a corpus the model has never been trained on -----
    # SpamAssassin stays out of training so this measures cross-corpus
    # generalisation rather than memorisation.
    holdout = pd.read_parquet(DATA / "external_holdout.parquet")
    y_hold = holdout["label"].to_numpy(dtype=np.int64)
    p_hold = batch_onnx_probs(int8_session, tokenizer, holdout["text"].tolist(), max_length)
    m_hold = metrics(y_hold, p_hold)
    negatives = int((y_hold == 0).sum())
    holdout_report = {
        "source": "external:spamassassin",
        "never_trained_on": True,
        "rows": int(len(holdout)),
        **m_hold,
        "false_positive_rate": round(
            float(((p_hold >= 0.5) & (y_hold == 0)).sum()) / max(negatives, 1), 4
        ),
    }

    # ---- fixtures for the JS parity tests -----------------------------------
    rng = np.random.default_rng(42)
    sampled = rng.choice(len(texts), size=N_SAMPLED_FIXTURES, replace=False)
    fixtures = []
    for text, shape in FIXTURE_TEXTS:
        fixtures.append({"text": text, "shape": shape})
    for idx in sampled:
        fixtures.append({"text": texts[int(idx)], "shape": "test sample"})
    expected_fp32 = torch_probs(model, tokenizer, [f["text"] for f in fixtures], max_length)
    expected_int8 = batch_onnx_probs(int8_session, tokenizer, [f["text"] for f in fixtures], max_length)
    for i, fixture in enumerate(fixtures):
        ids = tokenizer(fixture["text"], truncation=True, max_length=max_length)["input_ids"]
        fixture["expected_ids"] = [int(x) for x in ids]
        fixture["expected_fp32"] = round(float(expected_fp32[i]), 6)
        fixture["expected_int8"] = round(float(expected_int8[i]), 6)
    (VERIFICATION / "fixtures.json").write_text(json.dumps(fixtures, indent=2), encoding="utf-8")

    # ---- browser artifacts ---------------------------------------------------
    # save_vocabulary() is broken in transformers v5 for tokenizers that were
    # not loaded from a vocab file; rebuild the line-numbered vocab by hand.
    vocab = tokenizer.get_vocab()
    ordered = [None] * (max(vocab.values()) + 1)
    for token, idx in vocab.items():
        ordered[idx] = token
    (EXT_MODEL / "vocab.txt").write_text(
        "\n".join("" if t is None else t for t in ordered) + "\n", encoding="utf-8"
    )
    meta = {
        "backbone": report["model"]["base_checkpoint"],
        "max_length": max_length,
        "threshold": 0.5,
        "labels": ["legitimate", "phishing"],
        "quantized_int8": True,
        "parameters_total": report["model"]["parameters_total"],
    }
    (EXT_MODEL / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    quant_report = {
        "quantization": {
            "method": "dynamic QInt8 (onnxruntime.quantization.quantize_dynamic)",
            "applies_to": (
                "MatMul weights -> int8 with dynamically quantised activations; "
                "embedding tables (word/position/token_type) -> uint8 with per-tensor "
                "scales; LayerNorm weights and all biases stay float32"
            ),
            "fp32_artifact_bytes": fp32_path.stat().st_size,
            "int8_artifact_bytes": int8_path.stat().st_size,
        },
        "parity": {
            "fp32_onnx_vs_pytorch_max_abs_diff": round(fp32_diff, 9),
            "n_fixture_texts": len(fixtures),
            "int8_vs_fp32_fixture_drift_max": round(float(drift.max()), 6),
            "int8_vs_fp32_fixture_drift_mean": round(float(drift.mean()), 6),
        },
        "test_set_float32_onnx": m_fp32,
        "test_set_int8_shipped": m_int8,
        "delta_int8_minus_float32": delta,
        "external_holdout_int8": holdout_report,
        "verdict_disagreements_fp32_vs_int8": {
            "count": disagreements,
            "rate": round(disagreements / len(y_test), 6),
        },
    }
    (RESULTS / "quantization_report.json").write_text(
        json.dumps(quant_report, indent=2), encoding="utf-8")
    print(f"test set: fp32 acc {m_fp32['accuracy']} -> int8 acc {m_int8['accuracy']} "
          f"(delta {delta['accuracy']:+})")
    print(f"artifacts -> {EXT_MODEL}; fixtures -> {VERIFICATION / 'fixtures.json'}")


if __name__ == "__main__":
    main()
