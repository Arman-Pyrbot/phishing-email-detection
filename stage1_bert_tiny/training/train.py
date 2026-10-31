"""Fine-tune prajjwal1/bert-tiny for binary phishing classification.

A plain PyTorch loop (stable across transformers releases), hyperparameters as
named constants at the top. Logs per-epoch metrics to results/training_log.csv,
applies early stopping on validation F1, evaluates the best checkpoint on the
held-out test set, and writes results/train_report.json plus the checkpoint to
training/out/.
"""

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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

TRAINING_DIR = Path(__file__).resolve().parent
ROOT = TRAINING_DIR.parent
DATA = TRAINING_DIR / "data" / "processed"
OUT = TRAINING_DIR / "out"
RESULTS = ROOT / "results"

SEED = 42
BACKBONE = "prajjwal1/bert-tiny"  # 4.4M params, 2 layers x 128 hidden
# prajjwal1/bert-tiny ships only vocab.txt; transformers 5.x needs a tokenizer.json.
# Its vocab.txt is byte-identical (sha256-verified) to bert-base-uncased's, which does.
TOKENIZER_SOURCE = "bert-base-uncased"
MAX_LENGTH = 192
EPOCHS = 3
PATIENCE = 1  # stop after this many epochs without a new best val F1
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 256
LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.06
LOG_CSV = RESULTS / "training_log.csv"
LOG_HEADER = "epoch,train_loss,val_loss,val_accuracy,val_precision,val_recall,val_f1,val_roc_auc,lr,seconds"


def set_seed(seed: int) -> None:
    """Seed every RNG that affects weights, shuffling and dropout."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Threshold 0.5; class 1 = phishing. Includes per-class and confusion matrix."""
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    report = classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=["legitimate", "phishing"],
        output_dict=True, zero_division=0,
    )
    per_class = {
        name: {k: round(float(report[name][k]), 4)
               for k in ("precision", "recall", "f1-score", "support")}
        for name in ("legitimate", "phishing")
    }
    return {
        "n": int(len(y_true)),
        "threshold": 0.5,
        "accuracy": round(float((y_pred == y_true).mean()), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_class": per_class,
    }


class TextDataset(Dataset):
    """Tokenises lazily so the corpus is never held in memory twice."""

    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer):
        self.texts, self.labels, self.tokenizer = texts, labels, tokenizer

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        enc = self.tokenizer(self.texts[i], truncation=True, max_length=MAX_LENGTH)
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return item


def make_collator(pad_id: int):
    """Pad each batch to its own longest sequence, not to MAX_LENGTH."""

    def collate(batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        out = {}
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key not in batch[0]:
                continue
            fill = pad_id if key == "input_ids" else 0
            out[key] = torch.stack([
                torch.nn.functional.pad(b[key], (0, max_len - len(b[key])), value=fill)
                for b in batch
            ])
        out["labels"] = torch.stack([b["labels"] for b in batch])
        return out

    return collate


@torch.no_grad()
def evaluate(model, loader, device: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (labels, P(phishing), mean cross-entropy loss) for a loader."""
    model.eval()
    labels, logits_all, loss_sum, count = [], [], 0.0, 0
    for batch in loader:
        y = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch, labels=y.to(device))
        loss_sum += out.loss.item() * y.size(0)
        count += y.size(0)
        labels.append(y.numpy())
        logits_all.append(out.logits.float().cpu())
    probs = torch.softmax(torch.cat(logits_all), dim=-1)[:, 1].numpy()
    return np.concatenate(labels), probs, loss_sum / max(count, 1)


def lr_factor(step: int, total: int, warmup: int) -> float:
    """Linear warmup then cosine decay to ~0."""
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))


def load_split(name: str) -> tuple[list[str], np.ndarray]:
    df = pd.read_parquet(DATA / f"{name}.parquet")
    return df["text"].tolist(), df["label"].to_numpy(dtype=np.int64)


def main() -> None:
    set_seed(SEED)
    RESULTS.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_CSV.unlink(missing_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    train_texts, train_labels = load_split("train")
    val_texts, val_labels = load_split("val")
    test_texts, test_labels = load_split("test")
    print(f"train {len(train_texts):,}  val {len(val_texts):,}  test {len(test_texts):,}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SOURCE)
    # bert-tiny's config.json predates the `model_type` key, so AutoConfig
    # rejects it; BertConfig carries model_type="bert" as a class attribute.
    config = BertConfig.from_pretrained(
        BACKBONE, num_labels=2,
        id2label={"0": "legitimate", "1": "phishing"},
        label2id={"legitimate": 0, "phishing": 1},
    )
    model = BertForSequenceClassification.from_pretrained(
        BACKBONE, config=config,
    ).to(device)

    collate = make_collator(tokenizer.pad_token_id or 0)
    train_loader = DataLoader(
        TextDataset(train_texts, train_labels, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        TextDataset(val_texts, val_labels, tokenizer),
        batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate,
    )
    test_loader = DataLoader(
        TextDataset(test_texts, test_labels, tokenizer),
        batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate,
    )

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if name.endswith("bias") or "LayerNorm" in name else decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LR,
    )

    total_steps = max(len(train_loader) * EPOCHS, 1)
    warmup_steps = max(int(total_steps * WARMUP_FRACTION), 1)
    started = time.time()
    step = 0
    best_f1, best_state, best_epoch, bad_epochs = -1.0, None, 0, 0
    per_epoch: list[dict] = []
    epochs_run = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            for group in optimizer.param_groups:
                group["lr"] = LR * lr_factor(step, total_steps, warmup_steps)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            step += 1

        train_loss = running / max(seen, 1)
        y_val, p_val, val_loss = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(y_val, p_val)
        current_lr = optimizer.param_groups[0]["lr"]
        seconds = round(time.time() - started, 1)
        epochs_run = epoch
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "lr": float(f"{current_lr:.3g}"),
            "seconds": seconds,
        }
        per_epoch.append(row)
        with LOG_CSV.open("a", encoding="utf-8") as fh:
            if epoch == 1:
                fh.write(LOG_HEADER + "\n")
            fh.write(",".join(str(row[k]) for k in LOG_HEADER.split(",")) + "\n")
        print(
            f"epoch {epoch}: train loss {train_loss:.4f} | val loss {val_loss:.4f} "
            f"F1 {val_metrics['f1']:.4f} acc {val_metrics['accuracy']:.4f} "
            f"AUC {val_metrics['roc_auc']:.4f} lr {current_lr:.2e} ({seconds:.0f}s)",
            flush=True,
        )

        if val_metrics["f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = val_metrics["f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs > PATIENCE:
                print(f"early stopping after epoch {epoch} (no val F1 gain for {PATIENCE})")
                break

    minutes = round((time.time() - started) / 60, 2)
    if best_state is not None:
        model.load_state_dict(best_state)
    y_test, p_test, _ = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(y_test, p_test)

    model.save_pretrained(str(OUT))
    tokenizer.save_pretrained(str(OUT))
    parameters_total = sum(p.numel() for p in model.parameters())
    report = {
        "model": {
            "base_checkpoint": BACKBONE,
            "architecture": "BERT-tiny (2 layers, 128 hidden, 2 attention heads) + 2-class linear head",
            "parameters_total": parameters_total,
            "parameters_trainable": parameters_total,
            "num_labels": 2,
            "labels": ["legitimate", "phishing"],
            "id2label": {"0": "legitimate", "1": "phishing"},
        },
        "training": {
            "seed": SEED,
            "optimizer": "AdamW",
            "lr_schedule": f"linear warmup ({WARMUP_FRACTION}) + cosine decay",
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "epochs_max": EPOCHS,
            "epochs_run": epochs_run,
            "early_stopping": {"monitor": "val_f1", "patience": PATIENCE, "best_epoch": best_epoch},
            "max_sequence_length": MAX_LENGTH,
            "train_rows": len(train_texts),
            "val_rows": len(val_texts),
            "minutes": minutes,
            "per_epoch": per_epoch,
        },
        "preprocessing": {
            "tokenizer": "BERT WordPiece uncased (bert-base-uncased vocab)",
            "truncation": True,
            "max_sequence_length": MAX_LENGTH,
        },
        "evaluation_test_set": test_metrics,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": device,
            "hardware": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
            "cpu_count": os.cpu_count(),
            "date_utc": pd.Timestamp.now("UTC").isoformat(),
        },
    }
    (RESULTS / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(test_metrics, indent=2))
    print(f"best epoch {best_epoch}, checkpoint -> {OUT}")


if __name__ == "__main__":
    main()
