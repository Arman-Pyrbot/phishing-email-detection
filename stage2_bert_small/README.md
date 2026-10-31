# Stage 2: BERT-small in the browser

Same extension as Stage 1 with a bigger backbone, `prajjwal1/bert-small` (28.8M parameters, 29.0 MB int8). Training added one augmentation per row and 5,000 synthetic emails (polite BEC phishing and ordinary business mail). It also fixes a tokenizer bug with Chinese, Japanese and Korean characters.

Test F1: 0.9890 (fp32), 0.9889 (int8). Median inference 125.8 ms. All numbers are in `results/metrics.json`.

The fp32 checkpoint (115 MB) is too large for GitHub, so only the int8 model in `extension/model/` is included. `python training/run.py` rebuilds the full checkpoint.

## Load the extension

1. Open `chrome://extensions` and turn on Developer mode.
2. Click "Load unpacked" and pick the `extension` folder.
3. Open an email in Gmail and click the extension icon.

## Retrain

```
cd training
pip install -r requirements.txt
python run.py
```

The splits we used are in `../data/stage2/`. Unzip them into `training/data/processed/` to reuse them.
