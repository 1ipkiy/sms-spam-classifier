# SMS Spam vs Ham Classifier

**Author:** Miles Robinson · **Year:** 2026  
**Portfolio:** Project 1 of an AI engineering series  
**Stack:** scikit-learn · Hugging Face Transformers · DistilBERT · Gradio

Binary text classification on SMS messages (`ham` vs `spam`), framed as a lightweight ML engineering project with a cybersecurity angle (spam / phishing-adjacent detection). Clean train → eval → demo loop with real metrics on a held-out test set.

Also see [RAG Q&A with Citations](https://github.com/1ipkiy/rag-qa-with-citations) (Project 2), [Prompt Injection Detector](https://github.com/1ipkiy/prompt-injection-detector) (Project 6), and [Multimodal Phishing Vision](https://github.com/1ipkiy/multimodal-phishing-vision) (Project 7).

---

## Problem statement

SMS spam is common, short, and noisy. The goal is a **reproducible** classifier that:

1. Establishes a strong classical baseline (TF-IDF + Logistic Regression)
2. Improves on it with a fine-tuned transformer (DistilBERT)
3. Ships a simple Gradio demo for interactive inference

Built for applied ML / AI engineering interviews: clean code, honest CPU constraints, documented subset training, and metrics you can regenerate.

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
4. **Demo** — Gradio UI loads the saved DistilBERT checkpoint for live inference.

---

## Results

> Numbers from `python train.py` on this repo (CPU). See `results/metrics.json`.

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| TF-IDF + Logistic Regression | 0.9809 | 0.9211 | 0.9375 | 0.9292 |
| DistilBERT (fine-tuned, 3k subset, 2 epochs) | **0.9928** | **0.9732** | **0.9732** | **0.9732** |

Confusion matrices are written to `results/confusion_matrix_baseline.png` and `results/confusion_matrix_distilbert.png` after training.

**Honesty note:** DistilBERT is trained on a 3k stratified subset so portfolio demos stay runnable without a GPU. Metrics above are still on the full held-out test split (837 examples).

---

## Project structure

```
text-classifier/
├── README.md
├── HOW_I_BUILT_IT.md     # first-person build story
├── BREAK_IT.md           # intentional what-ifs lab
├── requirements.txt
├── .gitignore
├── train.py              # baseline + DistilBERT fine-tune + metrics
├── app.py                # Gradio demo
├── explore.py            # quick EDA (class balance + examples)
├── scripts/
│   └── break_experiments.py  # safe baseline-only failure demos
├── results/
│   ├── metrics.json      # filled by train.py
│   └── *.png             # confusion matrices (gitignored)
└── models/
    └── distilbert-sms-spam/   # saved after train.py (gitignored)
```

---

## Setup & run

```bash
cd text-classifier
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# optional EDA
python explore.py

# train baseline + DistilBERT, write results/metrics.json
python train.py

# Gradio demo (needs models/distilbert-sms-spam/ from train.py)
python app.py
```

First run downloads dataset + model weights (~300MB). DistilBERT on CPU typically takes ~20–40 minutes for 2 epochs on 3k examples. Open the Gradio URL (default `http://127.0.0.1:7860`).

Safe break-it demos (baseline only; never deletes the DistilBERT folder):

```bash
python scripts/break_experiments.py --list
python scripts/break_experiments.py --all
```

Tested on Linux with Python 3.11–3.13, CPU-only. GPU is used automatically if available.

---

## Learn by breaking it

- [HOW_I_BUILT_IT.md](HOW_I_BUILT_IT.md) — first-person build story
- [BREAK_IT.md](BREAK_IT.md) — intentional what-ifs, why it broke, how to fix
- [scripts/break_experiments.py](scripts/break_experiments.py) — safe baseline-only demos (`--tiny-subset`, `--no-class-weight`, `--all`, …)

---

## License / notes

Educational portfolio project. Not a production phishing detector. Model weights are not committed (see `.gitignore`); always produce them with `train.py`. The SMS Spam Collection dataset follows its original UCI / Hugging Face licensing terms.

*Miles Robinson · Cybersecurity + AI Engineering · CPCC → UNC Charlotte · 2026*
