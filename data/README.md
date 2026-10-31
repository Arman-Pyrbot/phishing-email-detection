# Datasets

Train, validation and test splits for all three stages. Each split is zipped to stay under GitHub's 100 MB file limit.

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

- Stages 1 and 2: CEAS-08, a seven-corpus email collection, PhiUSIIL URL and page records, PhreshPhish, Nazario, Nigerian fraud, and SpamAssassin (holdout only).
- Stage 3 text: the curated email datasets of Champa et al. (2024), SpaPhish, and a short-text set.
- Stage 3 URLs: PhiUSIIL (Prasad and Chandra, 2024).

The raw corpora belong to their original authors and are not included. `ml/scripts/` and `stage1_bert_tiny/training/fetch_external.py` show where each one comes from.
