# How I Built This SMS Spam Classifier

**Author:** Miles Robinson · **2026**

This is my walkthrough of Project 1 — how I went from "I want a binary text classifier with a cyber angle" to a trained DistilBERT model and a Gradio demo. Written so I can re-read it later when something breaks.

---

## 1. Why SMS spam?

I wanted a **binary classification** problem that feels cyber-adjacent without needing production network data.

- SMS spam / phishing-style messages are short, noisy, and real.
- Labels are clear: `ham` (legit) vs `spam`.
- It's a classic NLP task, so I can practice the full ML engineering loop: data → baseline → transformer → eval → demo.

This is **not** a production phishing detector. It's a portfolio project that teaches the pipeline.

---

## 2. Dataset choice and class imbalance

I used Hugging Face [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam) (UCI SMS Spam Collection).

From `results/metrics.json` (my actual run):

| | Count | Share |
|---|------:|------:|
| Ham | 4,827 | ~86.6% |
| Spam | 747 | ~13.4% |
| **Total** | **5,574** | |

Split (seed=42, stratified): train 3,901 / val 836 / test 837.

**Why imbalance matters:** a dumb model that always predicts "ham" gets ~87% accuracy and looks smart. That's why I care about **spam precision, recall, and F1** — not accuracy alone.

Labels in code: `0 = ham`, `1 = spam` (see `LABELS` and `load_sms_data()` in `train.py`).

---

## 3. Why baseline first (TF-IDF + LogReg) before DistilBERT

Before I touch transformers, I want a **cheap, strong baseline**.

My baseline (`train_baseline` in `train.py`):

- **TF-IDF** with 1–2 grams, 10k features, English stop words, `sublinear_tf=True`
- **Logistic Regression** with `class_weight="balanced"`, `max_iter=1000`, `C=1.0`

Why this order:

1. Baseline trains in seconds and tells me the problem is solvable.
2. If DistilBERT can't beat TF-IDF+LogReg, something is wrong with my fine-tune setup.
3. Interviewers like seeing you didn't jump straight to a 66M-param model.

My baseline test metrics:

| Metric | Value |
|--------|------:|
| Accuracy | 0.9809 |
| Precision (spam) | 0.9211 |
| Recall (spam) | 0.9375 |
| F1 | 0.9292 |

That's already strong. DistilBERT has to earn its keep.

---

## 4. What "fine-tuning" means (plain words)

**Pretrained DistilBERT** already knows English from a huge corpus. Fine-tuning means:

1. Load `distilbert-base-uncased` with a new 2-class head (`num_labels=2`).
2. Feed it my SMS texts + labels.
3. Run a few training epochs so the weights adjust for *this* task.

I'm not training a language model from scratch. I'm adapting a general model to spam vs ham. That's why a small learning rate (`2e-5`) and few epochs work.

---

## 5. Why subset (3000) + 2 epochs on CPU

Full train split is 3,901 messages. DistilBERT on CPU is slow, so I capped training data:

```python
DISTILBERT_MAX_SAMPLES = 3000
DISTILBERT_EPOCHS = 2
DISTILBERT_BATCH_SIZE = 8
DISTILBERT_MAX_LENGTH = 128
learning_rate = 2e-5
warmup_ratio = 0.1
```

I take a **stratified** subset of 3,000 train examples so ham/spam ratios stay honest. Eval still runs on the **full held-out test set** (837 messages) — I don't cheat by testing on the subset.

Tradeoff I'm honest about in the README: subset training for portfolio speed; metrics still on real held-out data.

My DistilBERT test metrics:

| Metric | Value |
|--------|------:|
| Accuracy | **0.9928** |
| Precision (spam) | **0.9732** |
| Recall (spam) | **0.9732** |
| F1 | **0.9732** |
| Train subset | 3000 |
| Epochs | 2 |

It beat the baseline on every spam-oriented metric. Good.

---

## 6. How evaluation works

`compute_metrics` in `train.py` reports accuracy, precision, recall, F1 (sklearn defaults → positive class = spam = 1).

On ~86% ham data:

| Metric | What it answers for spam |
|--------|--------------------------|
| **Accuracy** | Overall % correct — can look great while missing spam |
| **Precision** | Of messages I called spam, how many really were? |
| **Recall** | Of real spam, how many did I catch? |
| **F1** | Balance of precision and recall |

I also save confusion matrices under `results/` so I can see false positives vs false negatives.

Trainer picks the best checkpoint by **F1** (`metric_for_best_model="f1"`), not accuracy.

---

## 7. How the Gradio demo loads the model

`app.py` does this on startup:

1. Look for `models/distilbert-sms-spam/`.
2. If missing/empty → set `_load_error` and show a "run train.py" message in the UI.
3. Else load tokenizer + `AutoModelForSequenceClassification`, set `eval()` mode.
4. On classify: tokenize (`max_length=128`), softmax logits, map `probs[0]→ham`, `probs[1]→spam`.

Empty input is handled — it returns a prompt instead of crashing.

```bash
python app.py
# → http://127.0.0.1:7860 (listens on 0.0.0.0:7860)
```

---

## 8. File map

| Path | What it does |
|------|----------------|
| `train.py` | Load data, split, train baseline + DistilBERT, write metrics/plots/model |
| `app.py` | Gradio UI; loads saved DistilBERT for live inference |
| `explore.py` | Quick EDA — class counts, length stats, a few ham/spam examples |
| `requirements.txt` | Pinned deps (transformers, datasets, sklearn, torch, gradio, …) |
| `results/metrics.json` | Canonical test metrics from my last `train.py` run |
| `results/confusion_matrix_*.png` | Confusion matrices (gitignored PNGs; regenerated by train) |
| `models/distilbert-sms-spam/` | Saved fine-tuned weights + tokenizer (gitignored; from `train.py`) |
| `scripts/break_experiments.py` | Safe baseline-only break demos — never deletes the DistilBERT folder |
| `HOW_I_BUILT_IT.md` | This file — how I made it |
| `BREAK_IT.md` | Lab guide — how I intentionally break it and fix it |

---

## 9. The loop I actually ran

```bash
cd text-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python explore.py          # see imbalance with my own eyes
python train.py            # baseline → DistilBERT → metrics.json
python app.py              # try real messages in the browser
```

If something fails, I open `BREAK_IT.md` and pick an experiment that matches the symptom.

---

*Miles Robinson · Cybersecurity + AI Engineering · CPCC → UNC Charlotte · 2026*
