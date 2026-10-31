# Datasets

Train/validation/test split for all three phases. Split is zipped to remain within the 100MB file limit of GitHub.

| Folder | Files inside the zips | Split | Rows (train / val / test) |
|---|---|---|---|
| `stage1/` | `.parquet` | grouped by domain and campaign, 80/10/10, seed 42 | 123,479 / 15,379 / 15,428 |
| `stage2/` | `.parquet` | same as Stage 1, plus augmentation and 5,000 synthetic emails | 245,073 / 15,897 / 15,928 |
| `stage1/`, `stage2/` `external_holdout.zip` | `.parquet` | SpamAssassin, never trained on | 5,809 |
| `stage3_text/` | `train_text.csv`, ... | stratified 70/15/15, seed 42 | 97,651 / 20,925 / 20,926 |
| `stage3_url/url_dataset_full.zip` | all 235,370 URLs and their 25 PhiUSIIL features | `ml/scripts/train_xgboost.py` makes the stratified 70/10/20 split from this file (seed 42) | 164,759 / 23,537 / 47,074 |
| `stage3_url/train.zip`, `val.zip`, `test.zip` | URLs and features | saved 80/10/10 split, used by `train_xgboost_improved.py` | 188,296 / 23,537 / 23,537 |
| `stage3_url_domain_split/` | URLs | no domain shared between splits | 188,296 / 23,537 / 23,537 |

Label: 1 = phishing, 0 = legitimate.

## Sources

Stages 1 and 2:

| Dataset | Link | Rows used |
|---|---|---|
| CEAS-08 (Al-Subaiey et al.) | https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset | 37,154 |
| Seven-corpus email collection (Champa et al.) | https://huggingface.co/datasets/puyang2025/seven-phishing-email-datasets (also on [Figshare](https://figshare.com/articles/dataset/Seven_Phishing_Email_Datasets/25432108)) | 60,000 |
| PhiUSIIL URL and page records | https://archive.ics.uci.edu/dataset/967 | 60,000 |
| PhreshPhish (revision `eabec4b7a66324b7`) | https://huggingface.co/datasets/phreshphish/phreshphish | 11,995 |
| Nazario | https://github.com/rokibulroni/Phishing-Email-Dataset (CSV mirror of the Kaggle set) | 1,565 |
| Nigerian fraud | same mirror | 3,332 |
| SpamAssassin (holdout only) | same mirror | 5,809 |

Stage 3:

| Dataset | Link | Rows |
|---|---|---|
| Curated email datasets, 11 corpora (Champa et al., ICMI 2024) | https://zenodo.org/records/8339691 | 139,502 after merging with the two sets below and removing duplicates |
| SpaPhish | https://doi.org/10.17632/hz2d6gz7pc.3 | 1,395 |
| Short-text phishing set (`phishing_dataset_1000.xlsx`) | source link not recorded | 909 |
| PhiUSIIL (Prasad and Chandra, 2024) | https://archive.ics.uci.edu/dataset/967 | 235,370 after removing duplicates |

The raw data is NOT included. It is the property of their respective authors. `stage1_bert_tiny/training/fetch_external.py` downloads Nazario, Nigerian fraud and SpamAssassin automatically.
