"""Export: fp32 ONNX -> int8, parity checks, test + holdout impact, fixtures,
browser artifacts (model.onnx, vocab.txt, meta.json), quantization_report.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DIR = Path(__file__).resolve().parent
DATA = DIR / "data" / "processed"
MDIR = DIR / "out"
EXT = DIR.parent / "extension" / "model"
RESULTS = DIR.parent / "results"
VERIF = DIR.parent / "verification"

FIX = [("[url] http://paypal-secure-login.tk/verify-account?id=1", "url only"),
       ("[url] https://example.com/", "url only"),
       ("[url] https://github.com/features/actions", "url only"),
       ("[url] http://192.168.0.1/login", "url only"),
       ("[url] https://github.com/features/actions\n[title] GitHub Actions", "url + title"),
       ("[url] https://www.southbankmosaics.com\n[title] Southbank Mosaics", "url + title"),
       ("[url] https://en.wikipedia.org/wiki/Phishing\n[body] Phishing is a form of social engineering.", "url + body"),
       ("[body] Your account will be suspended within 24 hours. Verify your password immediately.", "body only"),
       ("[from] security@paypal-secure-login.tk\n[body] Dear customer, confirm your billing information.", "from + body"),
       ("[url] http://paypal-secure-login.tk/verify\n[title] Sign in\n[from] security@paypal-secure-login.tk\n[body] Verify your password immediately.", "full"),
       ("[from] billing@meridian.com\n[body] Invoice #48217 follow-up\nI am following up on invoice #INV-48217 for $4,200.00. Please confirm the billing details using the secure portal below.", "polite BEC")]
N_SAMPLE, BATCH = 13, 256  # 11 crafted + 13 sampled = 24 fixtures


def m(y, p):
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    pred = (p >= 0.5).astype(int)
    return {"n": int(len(y)), "accuracy": round(float((pred == y).mean()), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4)}


def probs_torch(model, tok, texts, ml):
    b = tok(texts, return_tensors="pt", truncation=True, max_length=ml, padding="max_length")
    with torch.no_grad():
        l = model(**b).logits.numpy()
    e = np.exp(l - l.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True))[:, 1]


def probs_onnx(sess, tok, texts, ml):
    b = tok(texts, return_tensors="np", truncation=True, max_length=ml, padding="max_length")
    feeds = {k: b[k].astype(np.int64) for k in ("input_ids", "attention_mask", "token_type_ids")}
    l = sess.run(["logits"], feeds)[0]
    e = np.exp(l - l.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True))[:, 1]


def batched(sess, tok, texts, ml):
    return np.concatenate([probs_onnx(sess, tok, texts[i:i + BATCH], ml)
                           for i in range(0, len(texts), BATCH)])


def main() -> None:
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rep = json.loads((RESULTS / "train_report.json").read_text(encoding="utf-8"))
    ml = int(rep["training"]["max_len"])
    EXT.mkdir(parents=True, exist_ok=True)
    VERIF.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(MDIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MDIR)).eval()

    fp32 = MDIR / "model.fp32.onnx"
    d = tok(FIX[0][0], return_tensors="pt", truncation=True, max_length=ml)
    with torch.no_grad():
        torch.onnx.export(model, (d["input_ids"], d["attention_mask"], d["token_type_ids"]),
                          str(fp32), input_names=["input_ids", "attention_mask", "token_type_ids"],
                          output_names=["logits"],
                          dynamic_axes={n: {0: "batch", 1: "sequence"} for n in
                                        ("input_ids", "attention_mask", "token_type_ids")} | {"logits": {0: "batch"}},
                          opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"fp32: {fp32.stat().st_size / 1e6:.1f} MB")

    crafted = [t for t, _ in FIX]
    ref = probs_torch(model, tok, crafted, ml)
    s32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    p32 = probs_onnx(s32, tok, crafted, ml)
    diff = float(np.abs(ref - p32).max())
    assert diff <= 1e-4, f"fp32 diverges: {diff:.2e}"

    int8 = EXT / "model.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    s8 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    drift = np.abs(p32 - probs_onnx(s8, tok, crafted, ml))
    print(f"int8: {int8.stat().st_size / 1e6:.1f} MB drift max {drift.max():.4f} mean {drift.mean():.4f}")

    test = pd.read_parquet(DATA / "test.parquet")
    yt = test["label"].to_numpy()
    texts = test["text"].tolist()
    pf, pi = batched(s32, tok, texts, ml), batched(s8, tok, texts, ml)
    mf, mi = m(yt, pf), m(yt, pi)
    delta = {k: round(mi[k] - mf[k], 6) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}
    disag = int(((pf >= 0.5) != (pi >= 0.5)).sum())

    hold = pd.read_parquet(DATA / "external_holdout.parquet")
    yh = hold["label"].to_numpy()
    ph = batched(s8, tok, hold["text"].tolist(), ml)
    mh = m(yh, ph)
    mh = {"source": "external:spamassassin", "never_trained_on": True, "rows": int(len(hold)), **mh,
          "false_positive_rate": round(float(((ph >= 0.5) & (yh == 0)).sum()) / max(int((yh == 0).sum()), 1), 4)}

    rng = np.random.default_rng(42)
    sampled = rng.choice(len(texts), size=N_SAMPLE, replace=False)
    fixtures = [{"text": t, "shape": s} for t, s in FIX]
    fixtures += [{"text": texts[int(i)], "shape": "test sample"} for i in sampled]
    ef = probs_torch(model, tok, [f["text"] for f in fixtures], ml)
    ei = batched(s8, tok, [f["text"] for f in fixtures], ml)
    for i, fx in enumerate(fixtures):
        fx["expected_ids"] = [int(x) for x in
                              tok(fx["text"], truncation=True, max_length=ml)["input_ids"]]
        fx["expected_fp32"] = round(float(ef[i]), 6)
        fx["expected_int8"] = round(float(ei[i]), 6)
    (VERIF / "fixtures.json").write_text(json.dumps(fixtures, indent=2), encoding="utf-8")

    vocab = tok.get_vocab()
    ordered = [None] * (max(vocab.values()) + 1)
    for t, idx in vocab.items():
        ordered[idx] = t
    (EXT / "vocab.txt").write_text("\n".join("" if t is None else t for t in ordered) + "\n",
                                  encoding="utf-8")
    (EXT / "meta.json").write_text(json.dumps(
        {"backbone": rep["model"]["backbone"], "max_length": ml, "threshold": 0.5,
         "labels": ["legitimate", "phishing"], "quantized_int8": True,
         "parameters_total": rep["model"]["params"]}, indent=2), encoding="utf-8")
    (RESULTS / "quantization_report.json").write_text(json.dumps(
        {"quantization": {"method": "dynamic QInt8", "fp32_bytes": fp32.stat().st_size,
                          "int8_bytes": int8.stat().st_size},
         "parity": {"fp32_vs_torch": round(diff, 9), "drift_max": round(float(drift.max()), 6),
                    "drift_mean": round(float(drift.mean()), 6)},
         "test_fp32": mf, "test_int8": mi, "delta": delta,
         "disagreements": {"count": disag, "rate": round(disag / len(yt), 6)},
         "external_holdout_int8": mh}, indent=2), encoding="utf-8")
    print(f"test fp32 {mf['accuracy']} -> int8 {mi['accuracy']} (delta {delta['accuracy']:+})")


if __name__ == "__main__":
    main()
