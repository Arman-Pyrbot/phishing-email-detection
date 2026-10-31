"""Fine-tune prajjwal1/bert-small (29M params, 6.6x bert-tiny) with AMP,
grad accumulation, label smoothing and cosine schedule. Early-stops on val F1,
evals on test, writes results/train_report.json."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

DIR = Path(__file__).resolve().parent
DATA = DIR / "data" / "processed"
OUT = DIR / "out"
RESULTS = DIR.parent / "results"

SEED = 42
BACKBONE = "prajjwal1/bert-small"  # 4 layers x 512 hidden, ~29M params
TOK_SRC = "bert-base-uncased"  # same WordPiece vocab as the bert-small family
MAX_LEN = 256  # vs 192 before: keeps full bodies, fewer truncations
EPOCHS, PATIENCE = 5, 2
BATCH, ACCUM, EVAL_BATCH = 32, 2, 256  # eff. train batch 64 via accumulation
LR, WD, WARMUP, SMOOTH = 3e-5, 0.01, 0.10, 0.05
LOG = RESULTS / "training_log.csv"


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rep = classification_report(y, pred, labels=[0, 1],
                                target_names=["legitimate", "phishing"],
                                output_dict=True, zero_division=0)
    return {"n": int(len(y)), "threshold": 0.5,
            "accuracy": round(float((pred == y).mean()), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "per_class": {n: {k: round(float(rep[n][k]), 4)
                              for k in ("precision", "recall", "f1-score", "support")}
                          for n in ("legitimate", "phishing")}}


class Texts(Dataset):
    def __init__(self, texts, labels, tok):
        self.texts, self.labels, self.tok = texts, labels, tok

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        e = self.tok(self.texts[i], truncation=True, max_length=MAX_LEN)
        return {**{k: torch.tensor(v) for k, v in e.items()},
                "labels": torch.tensor(int(self.labels[i]))}


def collate(pad: int):
    def f(batch):
        m = max(len(b["input_ids"]) for b in batch)
        out = {}
        for k in ("input_ids", "attention_mask", "token_type_ids"):
            if k in batch[0]:
                fill = pad if k == "input_ids" else 0
                out[k] = torch.stack([torch.nn.functional.pad(b[k], (0, m - len(b[k])), value=fill)
                                      for b in batch])
        out["labels"] = torch.stack([b["labels"] for b in batch])
        return out
    return f


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, lgs, loss, n = [], [], 0.0, 0
    ce = torch.nn.CrossEntropyLoss()
    for b in loader:
        y = b.pop("labels")
        b = {k: v.to(device) for k, v in b.items()}
        logits = model(**b).logits.float().cpu()
        loss += ce(logits, y).item() * len(y)
        n += len(y)
        ys.append(y.numpy())
        lgs.append(logits)
    e = torch.softmax(torch.cat(lgs), -1).numpy()
    return np.concatenate(ys), e[:, 1], loss / max(n, 1)


def main() -> None:
    seed_all(SEED)
    RESULTS.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.unlink(missing_ok=True)
    amp = torch.cuda.is_available()
    device = "cuda" if amp else "cpu"
    print("device:", device)
    tr = pd.read_parquet(DATA / "train.parquet")
    va = pd.read_parquet(DATA / "val.parquet")
    te = pd.read_parquet(DATA / "test.parquet")
    print(f"train {len(tr):,} val {len(va):,} test {len(te):,}")

    tok = AutoTokenizer.from_pretrained(TOK_SRC)
    cfg = BertConfig.from_pretrained(BACKBONE, num_labels=2,
                                     id2label={0: "legitimate", 1: "phishing"},
                                     label2id={"legitimate": 0, "phishing": 1})
    model = BertForSequenceClassification.from_pretrained(BACKBONE, config=cfg).to(device)
    tuned = sum(p.numel() for p in model.parameters())
    print(f"params: {tuned:,}")

    col = collate(tok.pad_token_id or 0)
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(Texts(tr["text"].tolist(), tr["label"].to_numpy(), tok),
                              batch_size=BATCH, shuffle=True, collate_fn=col, generator=g)
    val_loader = DataLoader(Texts(va["text"].tolist(), va["label"].to_numpy(), tok),
                            batch_size=EVAL_BATCH, collate_fn=col)
    test_loader = DataLoader(Texts(te["text"].tolist(), te["label"].to_numpy(), tok),
                             batch_size=EVAL_BATCH, collate_fn=col)

    decay = [p for n, p in model.named_parameters()
             if p.requires_grad and not (n.endswith("bias") or "LayerNorm" in n)]
    nodecay = [p for n, p in model.named_parameters()
               if p.requires_grad and (n.endswith("bias") or "LayerNorm" in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                             {"params": nodecay, "weight_decay": 0.0}], lr=LR)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=SMOOTH)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    total = max(len(train_loader) * EPOCHS // ACCUM, 1)
    warm = max(int(total * WARMUP), 1)

    def factor(step):
        if step < warm:
            return step / warm
        pr = (step - warm) / max(total - warm, 1)
        return max(1e-6, 0.5 * (1 + math.cos(math.pi * pr)))

    t0, step, best, bad, best_ep, best_state, hist = time.time(), 0, -1.0, 0, 0, None, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        run, seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(train_loader):
            y = b.pop("labels").to(device)
            b = {k: v.to(device) for k, v in b.items()}
            for grp in opt.param_groups:
                grp["lr"] = LR * factor(step)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = ce(model(**b).logits, y) / ACCUM
            scaler.scale(loss).backward()
            run += loss.item() * len(y) * ACCUM
            seen += len(y)
            if (i + 1) % ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
        yv, pv, vl = evaluate(model, val_loader, device)
        vm = metrics(yv, pv)
        row = {"epoch": epoch, "train_loss": round(run / seen, 4),
               "val_loss": round(vl, 4), "val_f1": vm["f1"], "val_acc": vm["accuracy"],
               "val_auc": vm["roc_auc"], "lr": float(f"{opt.param_groups[0]['lr']:.3g}"),
               "seconds": round(time.time() - t0, 1)}
        hist.append(row)
        with LOG.open("a") as fh:
            if epoch == 1:
                fh.write(",".join(row) + "\n")
            fh.write(",".join(str(row[k]) for k in row) + "\n")
        print(f"epoch {epoch}: train {run / seen:.4f} val F1 {vm['f1']:.4f} "
              f"acc {vm['accuracy']:.4f} AUC {vm['roc_auc']:.4f} ({row['seconds']:.0f}s)")
        if vm["f1"] > best:
            best, best_ep, bad = vm["f1"], epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad > PATIENCE:
                print(f"early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    yt, pt, _ = evaluate(model, test_loader, device)
    tm = metrics(yt, pt)
    model.save_pretrained(str(OUT))
    tok.save_pretrained(str(OUT))
    report = {"model": {"backbone": BACKBONE, "params": tuned, "labels": ["legitimate", "phishing"]},
              "training": {"seed": SEED, "lr": LR, "batch_eff": BATCH * ACCUM, "max_len": MAX_LEN,
                           "epochs_max": EPOCHS, "epochs_run": epoch, "best_epoch": best_ep,
                           "minutes": round((time.time() - t0) / 60, 2), "amp": amp,
                           "label_smoothing": SMOOTH, "per_epoch": hist,
                           "train_rows": len(tr), "val_rows": len(va)},
              "preprocessing": {"tokenizer": "BERT WordPiece uncased", "max_len": MAX_LEN},
              "evaluation_test_set": tm,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "device": device, "date_utc": pd.Timestamp.now("UTC").isoformat()}}
    (RESULTS / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(tm, indent=2))


if __name__ == "__main__":
    main()
