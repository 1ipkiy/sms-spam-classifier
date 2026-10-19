"""
SMS Spam vs Ham text classifier
Baseline: TF-IDF + Logistic Regression
Fine-tune: DistilBERT (distilbert-base-uncased)

Author: Miles Robinson (2026)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models" / "distilbert-sms-spam"

# DistilBERT on CPU: use a stratified subset so training finishes in reasonable time.
# Full SMS spam corpus is ~5.5k labeled messages; subset keeps class balance.
DISTILBERT_MAX_SAMPLES = 3000
DISTILBERT_EPOCHS = 2
DISTILBERT_BATCH_SIZE = 8
DISTILBERT_MAX_LENGTH = 128
LABELS = ["ham", "spam"]  # 0 = ham, 1 = spam


def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def load_sms_data() -> tuple[list[str], list[int]]:
    """Load SMS spam dataset from Hugging Face (ucirvine/sms_spam)."""
    print("Loading dataset: ucirvine/sms_spam ...")
    ds = load_dataset("ucirvine/sms_spam", split="train")
    texts = [row["sms"].strip() for row in ds]
    labels = [int(row["label"]) for row in ds]  # 0=ham, 1=spam
    print(f"  Loaded {len(texts)} messages "
          f"(ham={labels.count(0)}, spam={labels.count(1)})")
    return texts, labels


def split_data(
    texts: list[str], labels: list[int]
) -> tuple[list, list, list, list, list, list]:
    """70/15/15 train/val/test with fixed seed and stratification."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    print(
        f"Split → train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def plot_confusion(y_true, y_pred, title: str, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved confusion matrix → {out_path}")


# ---------------------------------------------------------------------------
# Baseline: TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------
def train_baseline(X_train, y_train, X_test, y_test) -> dict:
    print("\n=== Baseline: TF-IDF + Logistic Regression ===")
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=SEED, C=1.0
    )
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    metrics = compute_metrics(y_test, y_pred)
    print(classification_report(y_test, y_pred, target_names=LABELS, digits=4))
    plot_confusion(
        y_test,
        y_pred,
        "TF-IDF + Logistic Regression",
        RESULTS_DIR / "confusion_matrix_baseline.png",
    )
    return metrics


# ---------------------------------------------------------------------------
# DistilBERT fine-tune
# ---------------------------------------------------------------------------
def train_distilbert(
    X_train, y_train, X_val, y_val, X_test, y_test
) -> dict:
    print("\n=== Fine-tune: DistilBERT (distilbert-base-uncased) ===")

    # Stratified subset for CPU-friendly training
    n = min(DISTILBERT_MAX_SAMPLES, len(X_train))
    if n < len(X_train):
        idx, _ = train_test_split(
            np.arange(len(X_train)),
            train_size=n,
            random_state=SEED,
            stratify=y_train,
        )
        X_tr = [X_train[i] for i in idx]
        y_tr = [y_train[i] for i in idx]
        print(f"  Using stratified subset: {n} / {len(X_train)} train examples")
    else:
        X_tr, y_tr = X_train, y_train
        print(f"  Using full train set: {len(X_tr)} examples")

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=DISTILBERT_MAX_LENGTH,
            padding=False,
        )

    from datasets import Dataset

    train_ds = Dataset.from_dict({"text": X_tr, "label": y_tr}).map(
        tokenize, batched=True, remove_columns=["text"]
    )
    val_ds = Dataset.from_dict({"text": X_val, "label": y_val}).map(
        tokenize, batched=True, remove_columns=["text"]
    )
    test_ds = Dataset.from_dict({"text": X_test, "label": y_test}).map(
        tokenize, batched=True, remove_columns=["text"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label={0: "ham", 1: "spam"}, label2id={"ham": 0, "spam": 1}
    )

    use_cuda = torch.cuda.is_available()
    print(f"  Device: {'cuda' if use_cuda else 'cpu'}")

    args = TrainingArguments(
        output_dir=str(ROOT / "models" / "checkpoints"),
        num_train_epochs=DISTILBERT_EPOCHS,
        per_device_train_batch_size=DISTILBERT_BATCH_SIZE,
        per_device_eval_batch_size=DISTILBERT_BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        seed=SEED,
        report_to="none",
        fp16=False,
        save_total_limit=1,
        dataloader_num_workers=0,
    )

    def hf_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_metrics(labels, preds)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=hf_metrics,
    )

    print(f"  Training for {DISTILBERT_EPOCHS} epoch(s)...")
    trainer.train()

    print("  Evaluating on held-out test set...")
    pred_out = trainer.predict(test_ds)
    y_pred = np.argmax(pred_out.predictions, axis=-1)
    metrics = compute_metrics(y_test, y_pred)
    print(classification_report(y_test, y_pred, target_names=LABELS, digits=4))
    plot_confusion(
        y_test,
        y_pred,
        "DistilBERT Fine-tuned",
        RESULTS_DIR / "confusion_matrix_distilbert.png",
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"  Saved model → {MODEL_DIR}")

    metrics["train_subset_size"] = n
    metrics["epochs"] = DISTILBERT_EPOCHS
    return metrics


def print_comparison(baseline: dict, distilbert: dict) -> None:
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON (test set)")
    print("=" * 60)
    header = f"{'Metric':<12} {'Baseline':>12} {'DistilBERT':>12}"
    print(header)
    print("-" * 60)
    for key in ("accuracy", "precision", "recall", "f1"):
        print(
            f"{key:<12} {baseline[key]:>12.4f} {distilbert[key]:>12.4f}"
        )
    print("=" * 60)


def main() -> None:
    set_all_seeds(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)

    texts, labels = load_sms_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(texts, labels)

    baseline_metrics = train_baseline(X_train, y_train, X_test, y_test)
    distilbert_metrics = train_distilbert(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    print_comparison(baseline_metrics, distilbert_metrics)

    payload = {
        "dataset": "ucirvine/sms_spam",
        "seed": SEED,
        "split": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "class_balance_full": {
            "ham": labels.count(0),
            "spam": labels.count(1),
            "total": len(labels),
        },
        "baseline": {
            "model": "TF-IDF + LogisticRegression",
            **baseline_metrics,
        },
        "distilbert": {
            "model": "distilbert-base-uncased (fine-tuned)",
            **distilbert_metrics,
        },
    }
    metrics_path = RESULTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {metrics_path}")


if __name__ == "__main__":
    main()
