#!/usr/bin/env python3
"""
Safe "break it" demos for the SMS spam classifier.

Runs quick TF-IDF + Logistic Regression experiments only.
NEVER deletes or overwrites models/distilbert-sms-spam/.

Usage (from repo root, venv active):
  python scripts/break_experiments.py --list
  python scripts/break_experiments.py --tiny-subset
  python scripts/break_experiments.py --all

Author: Miles Robinson (2026)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
PROTECTED_MODEL = ROOT / "models" / "distilbert-sms-spam"

EXPERIMENTS = {
    "tiny-subset": "Train on only 50 examples → unstable / weak spam F1",
    "no-class-weight": "Drop class_weight=balanced → spam recall often tanks",
    "bad-lr": "Extreme LogReg C (proxy for huge transformer LR) → unstable metrics",
    "swapped-labels": "Flip ham/spam labels → metrics invert meaning / UI would lie",
    "short-maxlen": "Truncate each SMS to 8 chars before TF-IDF → loses spam cues",
    "train-eval": "Score on train set → leakage illusion (too-good metrics)",
}


def assert_model_intact() -> None:
    """Refuse to run if somehow asked to touch the DistilBERT folder."""
    # This script never writes there; this is a safety banner for the reader.
    if PROTECTED_MODEL.exists():
        print(f"[safe] Leaving trained model untouched: {PROTECTED_MODEL}")
    else:
        print(
            f"[note] {PROTECTED_MODEL} not present "
            "(run train.py later for the Gradio demo)."
        )


def load_split():
    ds = load_dataset("ucirvine/sms_spam", split="train")
    texts = [row["sms"].strip() for row in ds]
    labels = [int(row["label"]) for row in ds]
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    return X_train, X_test, y_train, y_test


def metrics_dict(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def fmt(m: dict) -> str:
    return (
        f"acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
        f"rec={m['recall']:.4f}  f1={m['f1']:.4f}"
    )


def run_baseline(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    class_weight="balanced",
    C: float = 1.0,
    max_iter: int = 1000,
    eval_on_train: bool = False,
) -> dict:
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    Xtr = vectorizer.fit_transform(X_train)
    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=SEED,
        C=C,
    )
    clf.fit(Xtr, y_train)
    if eval_on_train:
        y_pred = clf.predict(Xtr)
        return metrics_dict(y_train, y_pred)
    Xte = vectorizer.transform(X_test)
    y_pred = clf.predict(Xte)
    return metrics_dict(y_test, y_pred)


def print_pair(title: str, before: dict, after: dict, why: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)
    print(f"  BEFORE (sane baseline): {fmt(before)}")
    print(f"  AFTER  (broken):        {fmt(after)}")
    print(f"  WHY: {why}")


def exp_tiny_subset(X_train, y_train, X_test, y_test) -> None:
    before = run_baseline(X_train, y_train, X_test, y_test)
    idx, _ = train_test_split(
        np.arange(len(X_train)),
        train_size=50,
        random_state=SEED,
        stratify=y_train,
    )
    X_tiny = [X_train[i] for i in idx]
    y_tiny = [y_train[i] for i in idx]
    after = run_baseline(X_tiny, y_tiny, X_test, y_test)
    print_pair(
        "Experiment: --tiny-subset (50 train examples)",
        before,
        after,
        "Too few spam examples → high variance / weak generalization on full test.",
    )


def exp_no_class_weight(X_train, y_train, X_test, y_test) -> None:
    before = run_baseline(X_train, y_train, X_test, y_test, class_weight="balanced")
    after = run_baseline(X_train, y_train, X_test, y_test, class_weight=None)
    print_pair(
        "Experiment: --no-class-weight",
        before,
        after,
        "Without balancing, the model favors ham; spam recall often drops "
        "while accuracy still looks fine.",
    )


def exp_bad_lr(X_train, y_train, X_test, y_test) -> None:
    before = run_baseline(X_train, y_train, X_test, y_test, C=1.0)
    # Extreme C makes LogReg overconfident/unstable; mirrors "LR too high" lesson
    # for DistilBERT (do NOT set learning_rate=1e-2 on the transformer here).
    after = run_baseline(X_train, y_train, X_test, y_test, C=1e6, max_iter=50)
    print_pair(
        "Experiment: --bad-lr (LogReg C=1e6 proxy; transformer lesson)",
        before,
        after,
        "Huge effective step size / overfit proxy. For DistilBERT, learning_rate=1e-2 "
        "(vs 2e-5) can diverge or NaN — never overwrite models/distilbert-sms-spam/ "
        "to demo that; change train.py on a branch instead.",
    )


def exp_swapped_labels(X_train, y_train, X_test, y_test) -> None:
    before = run_baseline(X_train, y_train, X_test, y_test)
    y_train_f = [1 - y for y in y_train]
    y_test_f = [1 - y for y in y_test]
    after = run_baseline(X_train, y_train_f, X_test, y_test_f)
    # Also show what happens if we train flipped but score against TRUE labels
    vectorizer = TfidfVectorizer(
        max_features=10_000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True
    )
    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=SEED, C=1.0
    )
    clf.fit(vectorizer.fit_transform(X_train), y_train_f)
    y_pred_vs_true = clf.predict(vectorizer.transform(X_test))
    mismatch = metrics_dict(y_test, y_pred_vs_true)
    print_pair(
        "Experiment: --swapped-labels",
        before,
        after,
        "If EVERY label is flipped consistently, sklearn F1 can still look fine "
        "(you just renamed the classes). The real break is train/serve desync.",
    )
    print(f"  VS TRUE labels after flipped train: {fmt(mismatch)}")
    print(
        "  → That row is what a Gradio UI hardcoding ham=0/spam=1 would show "
        "if training maps were reversed."
    )


def exp_short_maxlen(X_train, y_train, X_test, y_test) -> None:
    before = run_baseline(X_train, y_train, X_test, y_test)
    X_train_s = [t[:8] for t in X_train]
    X_test_s = [t[:8] for t in X_test]
    after = run_baseline(X_train_s, y_train, X_test_s, y_test)
    print_pair(
        "Experiment: --short-maxlen (8-char truncation proxy for max_length=8)",
        before,
        after,
        "Truncation deletes spam cues. Same idea as DistilBERT max_length=8.",
    )


def exp_train_eval(X_train, y_train, X_test, y_test) -> None:
    before = run_baseline(X_train, y_train, X_test, y_test, eval_on_train=False)
    after = run_baseline(X_train, y_train, X_test, y_test, eval_on_train=True)
    print_pair(
        "Experiment: --train-eval (score on train = leakage illusion)",
        before,
        after,
        "Train-set scores memorize; always report the held-out test split.",
    )


RUNNERS = {
    "tiny-subset": exp_tiny_subset,
    "no-class-weight": exp_no_class_weight,
    "bad-lr": exp_bad_lr,
    "swapped-labels": exp_swapped_labels,
    "short-maxlen": exp_short_maxlen,
    "train-eval": exp_train_eval,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Safe baseline-only break demos (never touches DistilBERT weights)."
    )
    p.add_argument("--list", action="store_true", help="List experiments and exit")
    p.add_argument("--all", action="store_true", help="Run every safe experiment")
    for key, desc in EXPERIMENTS.items():
        p.add_argument(f"--{key}", action="store_true", help=desc)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        print("Safe experiments (baseline only):\n")
        for key, desc in EXPERIMENTS.items():
            print(f"  --{key:<16} {desc}")
        print("\n  --all              Run all of the above")
        print(f"\nProtected path (never deleted): {PROTECTED_MODEL}")
        return 0

    selected = [k for k in EXPERIMENTS if getattr(args, k.replace("-", "_"))]
    if args.all:
        selected = list(EXPERIMENTS.keys())
    if not selected:
        print("No experiment selected. Try --list or --tiny-subset or --all")
        return 1

    assert_model_intact()
    print("Loading dataset + split (same seed/stratify as train.py)...")
    X_train, X_test, y_train, y_test = load_split()
    print(f"  train={len(X_train)}  test={len(X_test)}")

    for key in selected:
        RUNNERS[key](X_train, y_train, X_test, y_test)

    assert_model_intact()
    print("\nDone. Read BREAK_IT.md for the DistilBERT / Gradio what-ifs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
