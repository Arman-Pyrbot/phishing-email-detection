# Fish Phishing Detection

A local phishing detection system that combines a trained DeBERTa text classifier, a trained XGBoost URL classifier, and a Gmail browser extension to detect phishing emails and malicious URLs.

The main detection endpoint runs locally and does not use keyword matching or an external LLM API. Gemini remains only as an optional legacy endpoint and is not required for the extension workflow.

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

The Stage 3 model weights are not in the repo because of their size. See "Important: Trained Model Files" below.

---

# Project Overview

Phishing attacks commonly use fraudulent emails, malicious URLs, fake login pages, urgency-based messages, and social engineering techniques to deceive users.

The original version of this project used an external Large Language Model (LLM) API to classify emails.

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
- Analyze both text and URLs rather than relying only on one signal.
- Reduce dependency on external LLM APIs.
- Run trained models as part of a browser extension.
- Provide phishing probability/risk information.
- Compare individual models with a combined multimodal approach.
- Create a reproducible ML pipeline for dataset preparation, training, and evaluation.

---

# 🤖 Actual Model Inference

The extension uses the `POST /api/ml-detect` endpoint in `backend/routes/llm.py`.

1. The Gmail content script extracts the subject, sender, body, and URLs.
2. The DeBERTa checkpoint at `ml/models/deberta_phishing/best` produces the phishing probability for the full email text.
3. URLs are converted into the 25 handcrafted features defined in `ml/scripts/url_features.py`.
4. The trained XGBoost model at `ml/models/xgboost_url/url_xgboost.json` produces the URL phishing probability.
5. The final risk probability is calculated as:

```text
final risk = 0.60 × text probability + 0.40 × URL probability
```

6. The API returns the final classification, risk score, model probabilities, URL count, and extracted URLs.

If the trained-model dependencies or artifacts are missing, the API returns an error instead of silently falling back to keyword matching.

The `ml/data/processed/multimodal_predictions` and SVD-related files are used by training and evaluation experiments; they are not required by the current live API URL inference path.

---

# How People Use This Project

## Students and Researchers

Students can use this project to learn how a complete phishing detection system is developed. It demonstrates dataset preparation, text classification, URL feature extraction, model evaluation, Flask API development, and browser-extension integration in one project.

Researchers can use the training scripts and evaluation files to compare text-based detection, URL-based detection, and combined text-plus-URL detection approaches.

## Gmail Users

Gmail users can load the Chrome extension locally and scan an opened email before clicking links or sharing sensitive information. The extension extracts the email subject, sender, body, and available URLs, then displays a phishing result and risk information in the popup.

## Developers

Developers can use the Flask API as a local service for testing phishing detection workflows. The frontend can also be modified to support additional email providers, new UI features, or improved model responses.

## Security Demonstrations

The project can be used in classroom demonstrations, project presentations, and controlled security-awareness exercises. A demonstrator can show how urgent language, suspicious login requests, and unsafe-looking URLs affect the final classification.

## Email Testing Workflow

1. Start the local Flask backend.
2. Build and load the Chrome extension.
3. Open Gmail and select an email.
4. Open the Phishing Detector popup.
5. Review the extracted subject, sender, and message preview.
6. Select **Scan this email**.
7. Review the classification, risk score, and detected URLs.

The project should only be used with emails and datasets that the user is authorized to inspect. It is intended for education, research, and controlled testing, not as a replacement for enterprise email security systems.

---

# Commands to Run the Project

The easiest method is to clone the repository. Open PowerShell in any folder where you want to keep the project and run:

```powershell
git clone https://github.com/AnupamUdeniya/Fish_phishing.git
cd Fish_phishing
```

The project root is now the folder that contains `backend`, `frontend`, and `extension`.

If you downloaded a ZIP file instead, open PowerShell in the extracted project root. Do not run the commands from the parent folder if the project is inside another folder.

## Important: Trained Model Files

The trained model files are approximately 751 MB and are not stored in this GitHub repository because of GitHub file-size limits. The trained detector requires these folders:

```text
ml/models/deberta_phishing/best/
ml/models/xgboost_url/url_xgboost.json
```

Copy the `ml/models` folder from the project package or download it from the project owner before starting the backend. Check that the files exist with:

```powershell
Test-Path ml\models\deberta_phishing\best\config.json
Test-Path ml\models\xgboost_url\url_xgboost.json
```

Both commands must return `True`.

## Train the Models

Run these commands from the project root, before starting the backend or using the browser extension. If you see `ModuleNotFoundError: No module named 'numpy'` or similar package errors, install the project dependencies first in the active virtual environment:

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

The text model trains on the processed email dataset and saves output to `ml/models/deberta_phishing`, the URL model trains on the processed URL dataset and saves to `ml/models/xgboost_url`, and the fusion model saves to `ml/models/fusion`.

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
7. Open an email, open the **Phishing Detector** popup, and select **Scan this email**.

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

After the build finishes, reload the extension in `chrome://extensions` and refresh Gmail.

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
