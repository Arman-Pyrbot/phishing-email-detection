"""Data: load raw + external corpora, add synthetic BEC, clean, dedupe,
group-disjoint 80/10/10 split, then 1x label-preserving augmentation on TRAIN
only. Writes data/processed/{train,val,test,external_holdout}.parquet."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parent
RAW = DIR / "data" / "raw"
EXT = DIR / "data" / "external"
OUT = DIR / "data" / "processed"
RESULTS = DIR.parent / "results"
SEED = 42
MIN_LEN = 12
N_SYNTH_PHISH = 2500
N_SYNTH_LEGIT = 2500

EXT_TRAIN = {"Nazario.csv": "external:nazario", "Nigerian_Fraud.csv": "external:nigerian_fraud"}
EXT_HOLD = {"SpamAssasin.csv": "external:spamassassin"}

CTL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
WS = re.compile(r"[ \t\r\f\v]+")


def clean(t) -> str:
    return WS.sub(" ", CTL.sub(" ", str(t or ""))).strip()


def read_ext(files: dict) -> pd.DataFrame:
    frames = []
    for fn, src in files.items():
        df = pd.read_csv(EXT / fn, dtype=str).fillna("")
        key = df["sender"].str.strip() + "|" + df["subject"].str.strip()
        frames.append(pd.DataFrame({
            "group": "h:" + key.map(lambda k: hashlib.sha1(k.encode()).hexdigest()[:16]),
            "text": "[from] " + df["sender"].str.strip() + "\n[body] "
                    + df["subject"].str.strip() + "\n" + df["body"].str.strip(),
            "label": df["label"].astype(int), "source": src}))
    return pd.concat(frames, ignore_index=True)


def synthetic(rng: random.Random) -> pd.DataFrame:
    """Polite BEC phishing + ordinary business replies — the style the old
    4.4M model missed (it scored courteous invoice fraud at ~0.07)."""
    names = ["James Carter", "Maria Santos", "David Kim", "Anna Novak", "Omar Haddad"]
    cos = ["Meridian Logistics", "BluePeak Supplies", "Northgate Trading", "Alba Foods"]
    phish_t = [
        "I am following up on invoice #{inv} for ${amt} from {co}. Please confirm the billing details using the secure portal below:\n{url}\nRegards, {name}",
        "Hello, our finance team flagged invoice #{inv} (${amt}) as pending. Kindly review and approve the payment via this link:\n{url}\nThank you, {name} ({co})",
        "Dear vendor, please find attached remittance advice for invoice #{inv}. To release the ${amt} transfer, verify your account here:\n{url}\nBest regards, {name}",
    ]
    legit_t = [
        "Hi {name}, thanks for invoice #{inv} for ${amt}. We have forwarded it to accounts payable and will update you once it is processed. Regards, Finance ({co})",
        "Hello, confirming receipt of invoice #{inv} (${amt}). Payment is scheduled per our standard terms. Let us know if anything else is needed. Thanks, {name}",
        "Dear {name}, invoice #{inv} is under review. We will notify you before the due date. No action required at this time. Kind regards, {co} billing team",
    ]
    rows = []
    for i in range(N_SYNTH_PHISH):
        rows.append(("synthetic:bec:%d" % i, "[from] billing@%s.com\n[body] Invoice #%s follow-up\n" % (
            rng.choice(cos).split()[0].lower(), i) + rng.choice(phish_t).format(
            inv=rng.randint(10000, 99999), amt="%.2f" % rng.uniform(500, 50000),
            co=rng.choice(cos), name=rng.choice(names),
            url=rng.choice(["http://%s-billing.com/verify" % rng.choice(cos).split()[0].lower(),
                            "https://secure-%s-pay.com/invoice" % rng.choice(cos).split()[0].lower()])), 1, "synthetic:bec"))
    for i in range(N_SYNTH_LEGIT):
        rows.append(("synthetic:legit:%d" % i, "[from] accounts@%s.com\n[body] Re: invoice #%s\n" % (
            rng.choice(cos).split()[0].lower(), i) + rng.choice(legit_t).format(
            inv=rng.randint(10000, 99999), amt="%.2f" % rng.uniform(500, 50000),
            co=rng.choice(cos), name=rng.choice(names)), 0, "synthetic:legit"))
    return pd.DataFrame(rows, columns=["group", "text", "label", "source"])


def augment(text: str, rng: random.Random) -> str:
    """One label-preserving variant: URL obfuscation shapes + case/whitespace jitter."""
    t = text
    if "[url]" in t and "[title]" in t and rng.random() < 0.30:
        # field dropout: the extension's thin-page fallback sends URL-only docs
        # (no title), so the model must judge bare URLs, not just url+title pairs.
        t = "\n".join(ln for ln in t.split("\n") if not ln.startswith("[title]"))
    elif "[url]" in t and rng.random() < 0.6:
        t = re.sub(r"https?://(www\.)?", lambda m: ("https://" if rng.random() < 0.5 else "http://")
                   + ("www." if rng.random() < 0.5 and not m.group(1) else ""), t, count=1)
        if rng.random() < 0.3 and not t.rstrip().endswith("/"):
            t = t.replace("\n", "/", 1) if "\n" in t else t + "/"
    words = t.split(" ")
    if len(words) > 3 and rng.random() < 0.5:
        j = rng.randrange(len(words))
        words[j] = words[j].upper() if rng.random() < 0.5 else words[j].lower()
        t = " ".join(words)
    if rng.random() < 0.3:
        t = re.sub(r"\n", " \n " if rng.random() < 0.5 else " ", t, count=1)
    greets = ["Hello, ", "Dear customer, ", ""]
    if t.startswith("[body]") and rng.random() < 0.25:
        t = t.replace("[body] ", "[body] " + rng.choice(greets), 1)
    t = WS.sub(" ", CTL.sub(" ", t)).strip()
    if t == text and len(words) > 3:  # guarantee a change: flip one word's case
        j = max(range(len(words)), key=lambda i: len(words[i]))
        words[j] = words[j].upper() if words[j].islower() else words[j].lower()
        t = " ".join(words)
    return t


def gsplit(df: pd.DataFrame, seed: int) -> dict:
    groups = list(df["group"].unique())
    random.Random(seed).shuffle(groups)
    ordered = df.set_index("group").loc[groups].reset_index()
    n, seen, c1, c2 = len(ordered), 0, None, None
    for size in ordered.groupby("group", sort=False).size():
        seen += int(size)
        if c1 is None and seen >= n * 0.8:
            c1 = seen
        elif c1 is not None and c2 is None and seen >= n * 0.9:
            c2 = seen
    assert c1 and c2, "split failed"
    return {"train": ordered.iloc[:c1], "val": ordered.iloc[c1:c2], "test": ordered.iloc[c2:]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    frames = []
    for name in ("train", "val", "test"):
        d = pd.read_parquet(RAW / f"{name}.parquet")
        frames.append(d)
    frames.append(read_ext(EXT_TRAIN))
    frames.append(synthetic(rng))
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].map(clean)
    df = df.dropna(subset=["label", "group"])
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() >= MIN_LEN]
    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    dedup = before - len(df)

    hold = read_ext(EXT_HOLD)
    hold["text"] = hold["text"].map(clean)
    hold = hold[hold["text"].str.len() >= MIN_LEN]
    hold = hold[~hold["text"].isin(set(df["text"]))].reset_index(drop=True)
    hold.to_parquet(OUT / "external_holdout.parquet", index=False)

    splits = gsplit(df, SEED)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (set(splits[a]["text"]) & set(splits[b]["text"])), f"{a}/{b} text leak"
        assert not (set(splits[a]["group"]) & set(splits[b]["group"])), f"{a}/{b} group leak"

    # 1x augmentation on TRAIN only (inherits group so it can never leak).
    trng = random.Random(SEED + 1)
    seen = set(df["text"]) | set(hold["text"])
    aug_rows = []
    for _, r in splits["train"].iterrows():
        v = augment(r["text"], trng)
        if len(v) >= MIN_LEN and v not in seen:
            seen.add(v)
            aug_rows.append({"group": r["group"] + "/aug", "text": v,
                             "label": r["label"], "source": r["source"] + "+aug"})
    if aug_rows:
        splits["train"] = pd.concat(
            [splits["train"], pd.DataFrame(aug_rows)], ignore_index=True)
    for name, part in splits.items():
        part.to_parquet(OUT / f"{name}.parquet", index=False)

    stats = {"seed": SEED, "raw_rows": int(len(df)),
             "removed_duplicates": int(dedup),
             "synthetic_rows": N_SYNTH_PHISH + N_SYNTH_LEGIT,
             "augmented_train_rows": len(aug_rows),
             "sizes": {k: int(len(v)) for k, v in splits.items()},
             "phishing_rate": {k: round(float(v["label"].mean()), 4) for k, v in splits.items()},
             "holdout_rows": int(len(hold)), "max_len_note": "tokenized at 256 (see train.py)"}
    (RESULTS / "data_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
