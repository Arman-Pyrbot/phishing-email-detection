# Stage 1: BERT-tiny in the browser

First local model. A fine-tuned `prajjwal1/bert-tiny` (4.4M parameters), exported to ONNX and quantized to int8 (4.48 MB). It runs inside the Chrome extension with ONNX Runtime Web, so nothing leaves the browser.

Test F1: 0.9797 (fp32), 0.9785 (int8). Median inference 5.2 ms. All numbers are in `results/metrics.json`.

## Folders

- `extension/` Chrome extension (Manifest V3) with the int8 model in `extension/model/`
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

`run.py` prepares the data, trains, exports the int8 model and runs the checks. The train/val/test splits we used are in `../data/stage1/`. Unzip them into `training/data/processed/` to reuse the same splits.
