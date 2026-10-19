# SMS Spam vs Ham Classifier

**Author:** Miles Robinson · **Year:** 2026  
**Stack:** scikit-learn · Hugging Face Transformers · DistilBERT · Gradio

Binary text classification on SMS messages (`ham` vs `spam`), framed as a lightweight ML engineering project with a light cybersecurity angle (spam / phishing-adjacent message detection).

---

## Problem statement

SMS spam is common, short, and noisy. The goal is to build a **reproducible** classifier that:

1. Establishes a strong classical baseline (TF-IDF + Logistic Regression)
2. Improves on it with a fine-tuned transformer (DistilBERT)
3. Ships a simple Gradio demo for interactive inference

This is **Project 1** in an AI engineering portfolio aimed at big-tech / applied ML roles: clean code, real metrics, honest constraints (CPU training, documented subset).

---

## Dataset

| Field | Value |
|-------|--------|
| Source | Hugging Face [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam) (UCI SMS Spam Collection) |
| Size | ~5,574 labeled SMS messages |
| Classes | `ham` (≈86%) · `spam` (≈14%) — imbalanced |
| Split | 70% train / 15% val / 15% test, stratified, seed=42 |

Labels: `0 = ham`, `1 = spam`.

---

## Approach

1. **Baseline** — TF-IDF (1–2 grams, 10k features) + Logistic Regression with `class_weight="balanced"`.
2. **Fine-tune** — `distilbert-base-uncased` for sequence classification (2 labels).
   - Trained on a **stratified subset of 3,000** train examples for **2 epochs** so the pipeline runs on CPU in reasonable time.
   - Hyperparams: lr=2e-5, batch size=8, max length=128, warmup ratio=0.1.
3. **Eval** — Accuracy, precision, recall, F1, and confusion matrices on the held-out test set.

---

## Results (test set)

> Numbers below come from `python train.py` on this repo (CPU). Re-run to regenerate `results/metrics.json`.

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| TF-IDF + Logistic Regression | 0.9809 | 0.9211 | 0.9375 | 0.9292 |
| DistilBERT (fine-tuned, 3k subset, 2 epochs) | **0.9928** | **0.9732** | **0.9732** | **0.9732** |

Confusion matrices are written to `results/confusion_matrix_baseline.png` and `results/confusion_matrix_distilbert.png` after training.

---

## Project structure

```
text-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── train.py              # baseline + DistilBERT fine-tune + metrics
├── app.py                # Gradio demo
├── explore.py            # quick EDA (class balance + a few examples)
├── results/
│   ├── metrics.json      # filled by train.py
│   └── *.png             # confusion matrices (gitignored)
└── models/
    └── distilbert-sms-spam/   # saved after train.py (gitignored)
```

---

## How to run

### 1. Setup

```bash
cd text-classifier
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Explore the data (optional)

```bash
python explore.py
```

### 3. Train

```bash
python train.py
```

This will:

- Download the dataset via Hugging Face `datasets`
- Train the TF-IDF baseline on the full train split
- Fine-tune DistilBERT on a 3k stratified subset (2 epochs, CPU-friendly)
- Write `results/metrics.json`, confusion-matrix PNGs, and save the model under `models/distilbert-sms-spam/`

**Note:** First run downloads model weights and the dataset (~300MB). DistilBERT on CPU typically takes ~20–40 minutes for 2 epochs on 3k examples (depends on machine).

### 4. Demo

```bash
python app.py
```

Open the Gradio URL (default `http://127.0.0.1:7860`). If the model folder is missing, the UI tells you to run `train.py` first.

---

## Requirements

See `requirements.txt` (Transformers, Datasets, scikit-learn, PyTorch CPU, Gradio, evaluate, accelerate, matplotlib/seaborn, pandas/numpy).

Tested on Linux with Python 3.11–3.13, CPU-only. GPU is optional and will be used automatically if available.

---

## Design notes / honesty

- DistilBERT is trained on a **subset** so portfolio demos stay runnable without a GPU. Metrics are still on the full held-out test split.
- Class imbalance is handled in the baseline via `class_weight="balanced"`; DistilBERT relies on the natural distribution plus F1 as the selection metric.
- Model weights are **not** committed (see `.gitignore`). Always produce them with `train.py`.
- This is educational spam classification — not a production phishing detector.

---

## License

Code in this repo is provided for portfolio / educational use. The SMS Spam Collection dataset follows its original UCI / Hugging Face licensing terms.

---

*Miles Robinson · Cybersecurity + AI Engineering · CPCC → UNC Charlotte · 2026*
