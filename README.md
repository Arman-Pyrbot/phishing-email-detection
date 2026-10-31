# Fish Phishing Detection

Local phishing detection system with trained DeBERTa text classifier, trained XGBoost URL classifier and Gmail browser extension to find phishing emails and malicious URLs.

The main detection endpoint is local. It has no keyword matching. No external LLM API. (Gemini is still an optional legacy endpoint but not necessary for the extension workflow).

## Repository layout

This repo holds the whole project, built in three stages:

| Folder | What it is |
|---|---|
| `stage1_bert_tiny/` | Stage 1: BERT-tiny running fully in the browser |
| `stage2_bert_small/` | Stage 2: BERT-small running fully in the browser |
| `backend/`, `extension/`, `frontend/`, `ml/` | Stage 3 (final system): DeBERTa-v3 text model and XGBoost URL model on a local Flask service |
| `ml/results/` | Stage 3 test results and training histories |
| `data/` | Train, validation and test splits for all three stages (see `data/README.md`) |
| `paper/` | Research paper (PDF and LaTeX source) |
| `presentation/` | Presentation slides |

The Stage 3 model weights aren't in the repo because they're too big. See "Important: Trained Model Files" below.

---

# Project Overview

Fake emails, malicious links (URLs), counterfeit login pages, time-sensitive messages, social engineering, etc. are often used in phishing attacks to trick people.

The first version of the project had the email classification powered by an external API (a large language model, LLM).

The current detection pipeline uses:

- **DeBERTa-v3-base** for email/text phishing detection
- **XGBoost** for URL phishing detection
- URL handcrafted features
- A 25-feature handcrafted URL representation
- A weighted fusion of text and URL probabilities
- Chrome browser extension integration

The deployed system analyzes:

1. Email text
2. URLs
3. Multiple URLs inside an email
4. Combined text + URL evidence

---

# Project Goals

The main goals of the project are:

- Detect phishing emails using a self-trained Deep Learning model.
- Detect malicious/phishing URLs using a Machine Learning model.
- Don't just check one type or the other but both text and URLs.
- Reduce dependency on external LLM APIs.
- Run trained models as part of a browser extension.
- Provide phishing probability/risk information.
- Compare individual models with a combined multimodal approach.
- Establish a reproducible ML workflow for dataset preparation, training and evaluation.

---

# Actual Model Inference

The extension uses the `POST /api/ml-detect` endpoint in `backend/routes/llm.py`.

1. The Gmail content script pulls out the subject, sender, body and URLs.
2. DeBERTa checkpoint at `ml/models/deberta_phishing/best` is used for phishing probability of the entire email text.
3. All the URLs are converted into the 25 handcrafted features in `ml/scripts/url_features.py`.
4. The trained XGBoost model on `ml/models/xgboost_url/url_xgboost.json` generates the URL phishing probability.
5. The final risk probability is calculated as:

```text
final risk = 0.60 × text probability + 0.40 × URL probability
```

6. Returns final classification, risk score, model probabilities, number of URLs and the extracted URLs.

If trained-model dependencies or artifacts are missing, the API returns an error. It doesn't quietly fall back to keyword matching.

The files in `ml/data/processed/multimodal_predictions` and the SVD files are needed for training and evaluation (not in the current live API URL inference pipeline).

---

# How People Use This Project

## Students and Researchers

Through this project, students can understand the whole process of building an entire system for detecting phishing. They could get to see everything from preparation of the dataset, text classification, URL feature extraction, model evaluation, building the Flask API, as well as browser extension.

The training scripts and evaluation files allow comparing detection based on text, detection based on URLs, and the combination of detection based on text and URL.

## Gmail Users

If you use Gmail, you can load the Chrome extension locally and let it check an opened email before you click links or share anything sensitive. It reads the subject, sender, body and any URLs, and shows the phishing result and risk info in the popup.

## Developers

Developers can call the Flask API as a local service for testing their phishing detection workflows. Additionally, you can modify the frontend to support other email providers or add new UI elements or better model responses.

## Security Demonstrations

This project could be used for demonstrating it in a classroom, for presenting it in projects. Or for controlled security awareness exercises. For example, a demonstrator could show how urgent language, suspicious login requests and unsafe URLs affect the final classification.

## Email Testing Workflow

1. Start the local Flask backend.
2. Build and load the Chrome extension.
3. Open Gmail and select an email.
4. Open the Phishing Detector popup.
5. Review the extracted subject, sender, and message preview.
6. Select **Scan this email**.
7. Review the classification, risk score, and detected URLs.

You should only use it to look at emails and datasets you are allowed to access. This is only meant as a tool for education, research, and controlled testing. It is not meant to replace any enterprise email security solutions.

---

# Commands to Run the Project

Simple: clone it. Just open PowerShell in a folder where you want to store the project and do:

```powershell
git clone https://github.com/Arman-Pyrbot/phishing-email-detection.git
cd phishing-email-detection
```

The project root is now the directory containing `backend`, `frontend`, `extension`.

If you downloaded a ZIP file, start PowerShell from the extracted project root directory. If the project is in a subfolder, don't run the commands from the parent directory.

## Important: Trained Model Files

Trained model files size around 751MB and this GitHub repository doesn't contain the files due to GitHub file size limit. Trained model detector will need following folders:

```text
ml/models/deberta_phishing/best/
ml/models/xgboost_url/url_xgboost.json
```

Copy the `ml/models` folder from the project package or get it from the project owner before starting the backend. Check that the files are there with:

```powershell
Test-Path ml\models\deberta_phishing\best\config.json
Test-Path ml\models\xgboost_url\url_xgboost.json
```

Both commands must return `True`.

## Train the Models

Run these from the project root, before starting the backend or the browser extension. If you see `ModuleNotFoundError: No module named 'numpy'` etc (package errors) then first install the project dependencies into the active virtual environment:

```powershell
# activate the local virtual environment
.\.venv-1\Scripts\Activate.ps1

# install Python dependencies needed for training
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r backend\requirements.txt

# train the models
python ml/scripts/train_deberta.py
python ml/scripts/train_xgboost.py
python ml/scripts/train_fusion.py
```

So the text model is trained on the processed email data and writes out to `ml/models/deberta_phishing`. The URL model is trained on the processed URL data and writes out to `ml/models/xgboost_url`, and the fusion model writes out to `ml/models/fusion`.

## 1. Start the Backend

Open the first terminal:

```powershell
cd backend
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python app.py
```

Keep this terminal running. The Flask API starts at:

```text
http://127.0.0.1:5000
```

## 2. Check the Backend

Open a second terminal and run:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/health
```

Expected response:

```text
status
------
ok
```

## 3. Build the Chrome Extension

In the second terminal:

```powershell
cd frontend\phishing-extension
npm install
npm run lint
npm run build
```

## 4. Load the Extension in Chrome

1. Open `chrome://extensions`.
2. Turn on **Developer mode**.
3. Click **Load unpacked**.
4. Select the `extension` folder inside the downloaded project folder.

5. Click **Reload** after rebuilding the extension.
6. Open Gmail and refresh the page.
7. Open an email, open the **Phishing Detector** popup and choose **Scan this email**.

## Quick Start After Installation

Backend terminal:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
python app.py
```

Frontend terminal:

```powershell
cd frontend\phishing-extension
npm run build
```

After the build is complete, reload the extension in `chrome://extensions` and refresh Gmail.

---

# System Architecture

## Current Architecture

```text
                         EMAIL / TEXT
                              |
                              v
                    +-------------------+
                    |    DeBERTa-v3     |
                    |      Base         |
                    +-------------------+
                              |
                              v
                    Text Phishing Score
                              |
                              |
                              |
                              v
                       +-------------+
                       |   Fusion    |
                       |    Layer    |
                       +-------------+
                              ^
                              |
                              |
                       URL Phishing Score
                              ^
                              |
                    +-------------------+
                    |     XGBoost       |
                    |   URL Detector   |
                    +-------------------+
                              ^
                              |
                    +-------------------+
                    | URL Representation|
                    +-------------------+
                       /             \
                      /               \
                     v                 v
          Handcrafted Features      URL TF-IDF
                                      |
                                      v
                                  SVD (300)
```
