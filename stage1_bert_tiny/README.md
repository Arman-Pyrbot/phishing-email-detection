# Stage 1: BERT-tiny in the browser

First local model. A finetuned `prajjwal1/bert-tiny` (4.4M params) converted to ONNX, int8 quant (4.48 MB). Executed in Chrome extension using ONNX Runtime Web - nothing goes out of browser.

Test F1: 0.9797 (fp32), 0.9785 (int8), median inference speed 5.2 ms. See `results/metrics.json` for all the numbers.

## Folders

- `extension/` Chrome extension (Manifest V3) using the int8 model on `extension/model/`
- `training/` data preparation, training, export and quantization
- `results/` metrics, benchmarks and the training log
- `verification/` tokenizer parity, browser benchmark and offline checks
- `demo/` local test pages

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

`run.py` does the data prep, train, export to int8, and checks. `../data/stage1/` is the train/val/test we used. Unzip them to `training/data/processed/` to continue using the same ones.
