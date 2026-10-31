"""Download the extra phishing-email corpora used alongside the phish-sentinel data.

Three public sources, no credentials or API keys required:

  Nazario         real phishing emails collected by Jose Nazario (2005-2021)
  Nigerian_Fraud  419 / advance-fee-fraud emails
  SpamAssasin     spam and ham; the ham half supplies realistic legitimate mail

All three are distributed in the per-source CSV form of the "Phishing Email
Dataset" mirror at github.com/rokibulroni/Phishing-Email-Dataset, which is the
widely used Kaggle "Phishing Email Dataset" (naserabdullahalam) split by source.
Rows are labelled as that source labels them; the SpamAssassin caveat is
recorded in external_sources.json.

Downloads land in training/data/external/ and each file's SHA-256 is verified,
so a re-run either reproduces the same bytes or fails loudly rather than
silently training on something different.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
EXTERNAL = TRAINING_DIR / "data" / "external"
MANIFEST = EXTERNAL / "external_sources.json"

REPO = "https://github.com/rokibulroni/Phishing-Email-Dataset"
RAW = f"{REPO}/raw/main"

SOURCES = {
    "Nazario.csv": {
        "name": "nazario",
        "url": f"{RAW}/Nazario.csv",
        "sha256": "b8fbc4158fbdfaff1ed98584c43d72e283c6352d7ade4d457d34c1d79488d184",
        "origin": "Jose Nazario's phishing corpus, 2005-2021",
        "rows": 1565,
        "label_note": "all rows are phishing",
    },
    "Nigerian_Fraud.csv": {
        "name": "nigerian_fraud",
        "url": f"{RAW}/Nigerian_Fraud.csv",
        "sha256": "f5d6930763cae6feeae6484131c79655a7440abbb97d6f51eab757aea4e6ab4b",
        "origin": "419 / advance-fee-fraud emails",
        "rows": 3332,
        "label_note": "all rows are fraudulent",
    },
    "SpamAssasin.csv": {
        "name": "spamassassin",
        "url": f"{RAW}/SpamAssasin.csv",
        "sha256": "3bfe8f8abff89f69a98456be50413c2fcb20a476141116dd84ee1507b980c00e",
        "origin": "SpamAssassin public mail corpus (ham + spam)",
        "rows": 5809,
        "label_note": (
            "label 0 = ham, label 1 = spam. Spam is not strictly phishing; "
            "this source's own labelling is kept, which makes the positive class "
            "broader than phishing alone"
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, spec: dict) -> None:
    """Fetch one source unless the pinned copy is already present."""
    target = EXTERNAL / name
    if target.exists() and sha256(target) == spec["sha256"]:
        print(f"{name}: present, hash matches")
        return
    print(f"{name}: downloading {spec['url']}")
    with urllib.request.urlopen(spec["url"], timeout=120) as response:
        target.write_bytes(response.read())
    actual = sha256(target)
    if actual != spec["sha256"]:
        raise AssertionError(f"{name}: sha256 {actual} != pinned {spec['sha256']}")


def main() -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    for name, spec in SOURCES.items():
        download(name, spec)

    manifest = {
        "note": (
            "Extra email corpora, downloaded by training/fetch_external.py. "
            "The per-source CSV form comes from the public GitHub mirror; the "
            "same collection is published as the Kaggle 'Phishing Email Dataset'."
        ),
        "mirror_repository": REPO,
        "kaggle_equivalent": "https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset",
        "sources": {
            spec["name"]: {k: v for k, v in spec.items() if k != "name"}
            for spec in SOURCES.values()
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
