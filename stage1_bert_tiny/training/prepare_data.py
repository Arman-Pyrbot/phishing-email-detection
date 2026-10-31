"""Load the original phish-sentinel datasets plus the extra email corpora,
clean, deduplicate, re-split.

Reads the three unified parquet splits copied verbatim from
phish-sentinel/data/unified/ (columns: text, label, group, source) and the three
extra CSVs downloaded by fetch_external.py, removes invalid and duplicate rows,
then writes a group-disjoint 80/10/10 split with seed 42 to data/processed/ and
the data record to results/data_stats.json.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd

TRAINING_DIR = Path(__file__).resolve().parent
ROOT = TRAINING_DIR.parent
RAW = TRAINING_DIR / "data" / "raw"
EXTERNAL = TRAINING_DIR / "data" / "external"
PROCESSED = TRAINING_DIR / "data" / "processed"
RESULTS = ROOT / "results"
SEED = 42
MIN_TEXT_LENGTH = 12

# Extra corpora fetched by fetch_external.py. Nazario and Nigerian Fraud join
# the training pool; SpamAssassin is held out entirely so it stays a genuine
# external test set, exactly as in the project this one is derived from. That
# project measured a 17.6 % false-positive rate on SpamAssassin against 1.5 %
# in-distribution, and noted that much of its "legitimate" class is promotional
# mail a phishing detector may reasonably flag - training on it would both
# destroy the benchmark and teach the model that spam is phishing.
EXTERNAL_TRAIN = {
    "Nazario.csv": "external:nazario",
    "Nigerian_Fraud.csv": "external:nigerian_fraud",
}
EXTERNAL_HOLDOUT = {"SpamAssasin.csv": "external:spamassassin"}

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")


def clean(text: str | None) -> str:
    """Drop ASCII control characters and collapse runs of whitespace."""
    if not text:
        return ""
    text = _CONTROL_CHARS.sub(" ", str(text))
    return _WHITESPACE.sub(" ", text).strip()


def read_external(files: dict[str, str]) -> pd.DataFrame:
    """Map the extra email corpora onto the (group, text, label, source) schema.

    The text layout matches the phish-sentinel corpus - `[from] <sender>` then
    `[body] <subject>` then the body - so both halves look identical to the
    model. Groups are derived from sender plus subject, which is the closest
    proxy for "same campaign" these corpora offer.
    """
    frames = []
    for filename, source in files.items():
        path = EXTERNAL / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run fetch_external.py first")
        df = pd.read_csv(path, dtype=str).fillna("")
        sender = df["sender"].str.strip()
        subject = df["subject"].str.strip()
        body = df["body"].str.strip()
        keys = sender + "|" + subject
        frames.append(pd.DataFrame({
            "group": "h:" + keys.map(
                lambda key: hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()[:16]
            ),
            "text": "[from] " + sender + "\n[body] " + subject + "\n" + body,
            "label": df["label"].astype(int),
            "source": source,
            "original_split": "external",
        }))
    return pd.concat(frames, ignore_index=True)


def group_disjoint_split(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Shuffle groups, order rows by group, cut at the 80% / 90% row targets.

    Whole groups land in exactly one split, so no registrable domain (or hash
    group) can leak between train, val and test.
    """
    groups = list(df["group"].unique())
    random.Random(seed).shuffle(groups)
    ordered = df.set_index("group").loc[groups].reset_index()

    n = len(ordered)
    cut1 = cut2 = None
    seen = 0
    for size in ordered.groupby("group", sort=False).size():
        seen += int(size)
        if cut1 is None and seen >= n * 0.8:
            cut1 = seen
        elif cut1 is not None and cut2 is None and seen >= n * 0.9:
            cut2 = seen
    if cut1 is None or cut2 is None:
        raise AssertionError("could not find 80%/90% group boundaries")

    return {
        "train": ordered.iloc[:cut1],
        "val": ordered.iloc[cut1:cut2],
        "test": ordered.iloc[cut2:],
    }


def assert_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    """Fail loudly if any text or group appears in two different splits."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            text_overlap = set(splits[a]["text"]) & set(splits[b]["text"])
            group_overlap = set(splits[a]["group"]) & set(splits[b]["group"])
            if text_overlap or group_overlap:
                raise AssertionError(
                    f"{a}/{b} overlap: {len(text_overlap)} texts, "
                    f"{len(group_overlap)} groups"
                )


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    frames, raw_sizes, raw_text_sets = [], {}, {}
    for name in ("train", "val", "test"):
        df = pd.read_parquet(RAW / f"{name}.parquet")
        df["original_split"] = name
        raw_sizes[name] = int(len(df))
        raw_text_sets[name] = set(df["text"].astype(str))
        frames.append(df)
    external = read_external(EXTERNAL_TRAIN)
    external_sizes = external["source"].value_counts().to_dict()
    frames.append(external)
    df = pd.concat(frames, ignore_index=True)
    raw_phishing_rate = round(float(df["label"].mean()), 6)

    # Duplicates that straddled the original train/val/test files (the old
    # project documented ~2.7k of these from its curated source).
    cross_split_texts = sum(
        len(raw_text_sets[a] & raw_text_sets[b])
        for i, a in enumerate(raw_text_sets)
        for b in list(raw_text_sets)[i + 1 :]
    )

    df["text"] = df["text"].map(clean)
    df = df.dropna(subset=["label", "group"])
    df["label"] = df["label"].astype(int)
    invalid_mask = df["text"].str.len() < MIN_TEXT_LENGTH
    removed_invalid = int(invalid_mask.sum())
    df = df[~invalid_mask]
    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    removed_duplicates = before - len(df)

    # The held-out corpus is cleaned the same way and then stripped of any row
    # whose text survived into the training pool, so the external test set
    # cannot overlap what the model has seen.
    holdout = read_external(EXTERNAL_HOLDOUT)
    holdout["text"] = holdout["text"].map(clean)
    holdout = holdout[holdout["text"].str.len() >= MIN_TEXT_LENGTH]
    holdout = holdout[~holdout["text"].isin(set(df["text"]))].reset_index(drop=True)
    holdout.to_parquet(PROCESSED / "external_holdout.parquet", index=False)
    holdout_stats = {
        "source": "external:spamassassin",
        "rows": int(len(holdout)),
        "phishing_rate": round(float(holdout["label"].mean()), 6),
        "never_trained_on": True,
        "note": (
            "SpamAssassin is a spam corpus and much of its legitimate class is "
            "promotional mail, so its false-positive rate is an upper bound "
            "rather than a like-for-like measurement"
        ),
    }

    splits = group_disjoint_split(df, SEED)
    assert_disjoint(splits)
    for name, part in splits.items():
        part.to_parquet(PROCESSED / f"{name}.parquet", index=False)

    original_stats = json.loads((RAW / "original_stats.json").read_text(encoding="utf-8"))
    stats = {
        "dataset": {
            "name": "phish-sentinel unified corpus plus three extra email corpora",
            "copied_verbatim_from": "phish-sentinel/data/unified/{train,val,test}.parquet",
            "original_builder": "phish-sentinel ml/build_dataset.py",
            "original_sources": original_stats["sources"],
            "raw_split_sizes": raw_sizes,
            "raw_class_balance": {"phishing_rate_overall": raw_phishing_rate},
            "extra_corpora": {
                "fetched_by": "training/fetch_external.py",
                "manifest": "training/data/external/external_sources.json",
                "mirror_repository": "https://github.com/rokibulroni/Phishing-Email-Dataset",
                "row_counts": external_sizes,
                "grouping": "h:<sha1(sender|subject)> - a campaign proxy",
                "text_layout": "[from] <sender>\\n[body] <subject>\\n<body>",
                "held_out": holdout_stats,
            },
        },
        "deduplication": {
            "method": "exact match on cleaned text, first occurrence kept",
            "removed_invalid_rows": removed_invalid,
            "removed_duplicate_rows": removed_duplicates,
            "cross_split_duplicate_texts_in_raw_files": cross_split_texts,
        },
        "split": {
            "seed": SEED,
            "method": "group-disjoint; shuffled groups, cut at 80%/90% of rows",
            "sizes": {k: int(len(v)) for k, v in splits.items()},
            "phishing_rate": {
                k: round(float(v["label"].mean()), 6) for k, v in splits.items()
            },
            "asserted_no_text_overlap": True,
            "asserted_no_group_overlap": True,
        },
        "preprocessing": [
            "strip ASCII control characters",
            "collapse runs of whitespace to single spaces",
            f"drop rows with text shorter than {MIN_TEXT_LENGTH} characters",
            "drop exact duplicate texts (keep first occurrence)",
        ],
    }
    (RESULTS / "data_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"raw rows: {raw_sizes} (cross-split duplicate texts: {cross_split_texts})")
    print(f"extra corpora added to training: {external_sizes}")
    print(f"external holdout (never trained on): {holdout_stats['rows']} rows "
          f"from {holdout_stats['source']}")
    print(f"removed: {removed_invalid} invalid, {removed_duplicates} duplicates")
    for name, part in splits.items():
        print(f"  {name:5s} {len(part):7d} rows  phishing rate {part['label'].mean():.3f}")
    print(f"wrote {PROCESSED} and {RESULTS / 'data_stats.json'}")


if __name__ == "__main__":
    main()
