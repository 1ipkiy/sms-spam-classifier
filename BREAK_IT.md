# Break It: Intentional Failure Lab

**Author:** Miles Robinson · **2026**

I learn this project best by **breaking it on purpose**, writing down what I expected vs what happened, then fixing it. Every experiment below maps to real lines in `train.py` / `app.py`.

**Rules I follow:**

1. Work on a git branch or stash before big edits — don't trash a working commit.
2. Never delete `models/distilbert-sms-spam/` just to "see what happens" unless I can retrain (CPU DistilBERT takes a while). Prefer the safe script below.
3. After each fix, re-check with the commands in "How to fix it."

---

## Safe automated demos (baseline only)

```bash
source .venv/bin/activate
python scripts/break_experiments.py --list
python scripts/break_experiments.py --tiny-subset
python scripts/break_experiments.py --no-class-weight
python scripts/break_experiments.py --bad-lr          # LogReg C analog / note on transformer LR
python scripts/break_experiments.py --swapped-labels
python scripts/break_experiments.py --short-maxlen    # TF-IDF char truncation proxy
python scripts/break_experiments.py --train-eval      # leakage illusion
python scripts/break_experiments.py --all             # run every safe demo
```

These **only** train quick TF-IDF baselines. They **never** write to or delete `models/distilbert-sms-spam/`.

---

## Experiment 1 — Wrong / swapped ham–spam labels

### What I change
In `train.py`, inside `load_sms_data()` (~line 69):

```python
# BROKEN: flip every label
labels = [1 - int(row["label"]) for row in ds]
```

Or swap the model maps (~line 200):

```python
id2label={0: "spam", 1: "ham"}, label2id={"spam": 0, "ham": 1}
```

while leaving `app.py` hardcoded as `probs[0]=ham`, `probs[1]=spam` (~lines 64–65).

### What I expect vs what happens
- **Expect:** Model still "learns," metrics look weird or the UI says the opposite of reality.
- **Happens:** Training can still report high F1 if *everything* is consistently flipped — but the Gradio demo labels are backwards. Or if only `id2label` flips and `app.py` doesn't, the UI lies.

### Why it broke
The numeric label `1` means spam everywhere in sklearn metrics and in `app.py`'s softmax index. Changing the map in one place without the other silently desyncs training semantics from the UI.

### How to fix it
Restore:

```python
labels = [int(row["label"]) for row in ds]  # 0=ham, 1=spam
# and
id2label={0: "ham", 1: "spam"}, label2id={"ham": 0, "spam": 1}
```

Keep `app.py` as `conf = {"ham": probs[0], "spam": probs[1]}`.

### What I learned
Label contracts must be identical in data load → train → serve. One swapped map is enough to ship a confident wrong product.

**Safe demo:** `python scripts/break_experiments.py --swapped-labels`

---

## Experiment 2 — Tiny train subset (e.g. 50 examples)

### What I change
In `train.py` config (~line 50):

```python
DISTILBERT_MAX_SAMPLES = 50   # was 3000
```

Or for a fast baseline-only demo, use the script flag below.

### What I expect vs what happens
- **Expect:** "Small data still works somehow."
- **Happens:** Metrics jump around; spam F1 collapses or looks randomly great on a lucky split; model memorizes a handful of phrases (overfit / useless).

### Why it broke
With ~14% spam, 50 stratified examples might hold only ~7 spam messages. The model never sees enough variety. Variance explodes.

### How to fix it
Put it back:

```python
DISTILBERT_MAX_SAMPLES = 3000
```

### What I learned
Subset size is a real hyperparameter. Too small ≠ "faster learning" — it's garbage signal.

**Safe demo:** `python scripts/break_experiments.py --tiny-subset`

---

## Experiment 3 — Drop `class_weight` / ignore imbalance

### What I change
In `train.py` `train_baseline` (~line 135–137):

```python
clf = LogisticRegression(
    max_iter=1000, random_state=SEED, C=1.0  # removed class_weight="balanced"
)
```

### What I expect vs what happens
- **Expect:** Accuracy stays high (~86–98%).
- **Happens:** Accuracy can still look fine; **spam recall drops** — the model prefers predicting ham because it's safer for the loss.

### Why it broke
Without balancing, the optimizer is rewarded for the majority class. Imbalanced binary problems hide failure behind accuracy.

### How to fix it
Restore `class_weight="balanced"`.

### What I learned
On 86% ham data, always check spam recall/F1. Accuracy alone is a trap (see Experiment 9).

**Safe demo:** `python scripts/break_experiments.py --no-class-weight`

---

## Experiment 4 — Learning rate way too high (e.g. `1e-2`)

### What I change
In `train.py` `TrainingArguments` (~line 216):

```python
learning_rate=1e-2,   # was 2e-5
```

### What I expect vs what happens
- **Expect:** Faster convergence.
- **Happens:** Loss explodes, metrics become NaN/garbage, F1 tanks. Fine-tuning a pretrained net with a huge LR destroys useful weights.

### Why it broke
Pretrained transformers need small LRs (order `1e-5`–`5e-5`). `1e-2` is ~500× too large and overwrites features learned during pretraining.

### How to fix it
Restore `learning_rate=2e-5` (and keep `warmup_ratio=0.1`).

### What I learned
"Higher LR = faster" is false for fine-tuning. Start small.

**Note:** DistilBERT retrain is slow on CPU. The safe script demos an analogous baseline failure with extreme `C` / unstable settings and prints the transformer lesson — it does **not** overwrite your saved DistilBERT model.

**Safe demo:** `python scripts/break_experiments.py --bad-lr`

---

## Experiment 5 — `max_length` too short (e.g. 8)

### What I change
In `train.py` (~line 53) and the tokenizer call (~line 183):

```python
DISTILBERT_MAX_LENGTH = 8   # was 128
```

Also change `app.py` tokenize `max_length=8` (~line 57) if you want train/serve consistent.

### What I expect vs what happens
- **Expect:** Faster training, similar quality (SMS are short!).
- **Happens:** Many messages get truncated to a few tokens. Model can't see the spam cues ("FREE", "claim", phone numbers later in the text). F1 drops.

### Why it broke
`max_length` is the token budget. Truncation throws away the rest. SMS *can* be short, but 8 tokens is not enough for a full spam pitch.

### How to fix it
Restore `DISTILBERT_MAX_LENGTH = 128` and `max_length=128` in `app.py`.

### What I learned
Token budget is part of the model contract between train and serve.

**Safe demo:** `python scripts/break_experiments.py --short-maxlen` (truncates raw text before TF-IDF as a fast proxy)

---

## Experiment 6 — Evaluate on the train set only (leakage illusion)

### What I change
In `train_baseline`, predict on train instead of test (~line 139):

```python
y_pred = clf.predict(X_train_tfidf)
metrics = compute_metrics(y_train, y_pred)  # was y_test
```

(Same idea: point DistilBERT `trainer.predict` at `train_ds`.)

### What I expect vs what happens
- **Expect:** "Wow, 99.9% — I'm done."
- **Happens:** Metrics look amazing and **lie**. Real users / held-out SMS will be worse.

### Why it broke
You're scoring memorization, not generalization. That's classic data leakage / eval malpractice.

### How to fix it
Always evaluate on the held-out **test** split that was never used for fitting.

### What I learned
If a number feels too good, ask: "Did this example ever touch training?"

**Safe demo:** `python scripts/break_experiments.py --train-eval`

---

## Experiment 7 — Wrong model path / missing `models/` folder

### What I change
Temporarily rename the folder:

```bash
mv models/distilbert-sms-spam models/distilbert-sms-spam.bak
python app.py
```

Or in `app.py` (~line 20):

```python
MODEL_DIR = ROOT / "models" / "wrong-name"
```

### What I expect vs what happens
- **Expect:** Crash with a scary stack trace.
- **Happens:** App still launches; UI shows *"Model not available — run train.py first"* and confidence stays at zeros. (`load_model` sets `_load_error` when the dir is missing/empty.)

### Why it broke
`app.py` checks `MODEL_DIR.exists()` and contents before loading. Weights are gitignored — a fresh clone has no model until `train.py` runs.

### How to fix it

```bash
mv models/distilbert-sms-spam.bak models/distilbert-sms-spam
# or re-run: python train.py
```

Restore `MODEL_DIR = ROOT / "models" / "distilbert-sms-spam"`.

### What I learned
Demo code should fail loudly *in the UI*, not silently classify with a random init. Mine does the right thing — this experiment proves it.

---

## Experiment 8 — Empty string or non-English gibberish in the demo

### What I change
No code change. In the Gradio UI:

1. Click Classify with an empty box.
2. Paste: `asdf qwer zxcv 🚀🚀🚀 !!!`
3. Paste: `これは日本語のテストです`

### What I expect vs what happens
- **Empty:** Message *"Enter an SMS message above."* and zero confidences (handled in `predict`).
- **Gibberish / other languages:** Model still returns ham/spam probabilities. Often high ham confidence — DistilBERT was fine-tuned on English SMS, not on random noise or Japanese.

### Why it broke (or "didn't")
Empty input is guarded. OOD text isn't an exception — the model *always* outputs a distribution. Low-quality / OOD inputs are a product problem: confidence ≠ correctness.

### How to fix it
For empty: already fixed. For OOD: optional guardrails (min length, charset checks, confidence threshold + "not sure" label). I haven't shipped those yet on purpose — honesty over fake certainty.

### What I learned
A classifier never says "I don't know" unless I build that path.

---

## Experiment 9 — Claim accuracy alone is enough on 86% ham data

### What I change
Conceptual. Pretend my success criterion is:

```python
assert metrics["accuracy"] > 0.90
```

and ignore precision/recall/F1.

### What I expect vs what happens
- **Expect:** Ship the model.
- **Happens:** A constant-"ham" classifier gets ~86.6% accuracy and passes my bar while catching **zero** spam. Useless for the cyber goal.

### Why it broke
Class imbalance makes accuracy a vanity metric. Spam recall is the metric that matches "did I catch the bad messages?"

### How to fix it
Use the same metrics `train.py` already prints — especially spam F1 — and keep confusion matrices. Trainer already selects checkpoints by F1.

### What I learned
Pick metrics that match the threat model, not the easiest number.

---

## Experiment 10 — Forget to activate venv / missing dependency

### What I change
From a fresh shell in the project root:

```bash
deactivate 2>/dev/null
python train.py
# or: python -c "import gradio"
```

### What I expect vs what happens
- **Expect:** Same as last time.
- **Happens:** `ModuleNotFoundError: No module named 'transformers'` (or torch/gradio/datasets).

### Why it broke
Dependencies live in `.venv`. System Python doesn't have them. `requirements.txt` only helps after `pip install -r requirements.txt` *inside* the venv.

### How to fix it

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

### What I learned
Environment issues look like code bugs. Check `which python` before rewriting training logic.

---

## Extra Experiment 11 — Hardcoded UI probs vs flipped `id2label`

### What I change
Only in `train.py` when creating the model (~line 200):

```python
id2label={0: "spam", 1: "ham"}, label2id={"spam": 0, "ham": 1}
```

Retrain, then run `app.py` **unchanged** (`probs[0]` still labeled ham).

### What I expect vs what happens
- **Expect:** Hugging Face metadata keeps the UI honest.
- **Happens:** Softmax index 0 is whatever class the head learned as label-0. `app.py` ignores `id2label` and hardcodes names → UI can show the wrong word for the right probability mass.

### Why it broke
Serve code duplicated the label contract instead of reading `model.config.id2label`.

### How to fix it
Keep train maps as `0=ham, 1=spam`, **or** change `app.py` to:

```python
id2label = _model.config.id2label
conf = {id2label[0]: float(probs[0]), id2label[1]: float(probs[1])}
```

### What I learned
One source of truth for label names. Prefer `config.id2label` at serve time.

---

## Extra Experiment 12 — Non-stratified split

### What I change
In `split_data` (~lines 79–84), drop `stratify=`:

```python
X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.30, random_state=SEED  # no stratify
)
```

### What I expect vs what happens
- **Expect:** "Random is fine with 5k rows."
- **Happens:** Test spam count wobbles; metrics become unstable across seeds; rare-class evaluation gets noisy.

### Why it broke
With only ~747 spam messages, unlucky splits starve train or test of spam.

### How to fix it
Restore `stratify=labels` / `stratify=y_temp` (and keep stratification on the DistilBERT subset split).

### What I learned
Stratify whenever classes are imbalanced — especially before claiming a metric improvement.

---

## Extra Experiment 13 — Point Gradio at checkpoints instead of the final save

### What I change
In `app.py`:

```python
MODEL_DIR = ROOT / "models" / "checkpoints" / "checkpoint-XXX"
```

(or a half-written path after a killed training run).

### What I expect vs what happens
- **Expect:** Latest training folder is always loadable.
- **Happens:** Missing tokenizer files, incomplete checkpoint, or an early epoch with worse F1 than the final `save_model` output.

### Why it broke
`train.py` writes the **best** model to `models/distilbert-sms-spam/` via `trainer.save_model` after training. Intermediate `models/checkpoints/` are ephemeral (`save_total_limit=1`) and not the demo contract.

### How to fix it
Always load `models/distilbert-sms-spam/`. If missing, run `python train.py` to completion.

### What I learned
Demo path ≠ Trainer `output_dir`. Document one canonical artifact folder.

---

## Quick recovery checklist

| Symptom | First thing I check |
|---------|---------------------|
| `ModuleNotFoundError` | venv activated? `pip install -r requirements.txt` |
| Gradio "Model not available" | Does `models/distilbert-sms-spam/` exist and have files? |
| Great accuracy, bad spam recall | `class_weight`? Looking at F1? |
| NaNs / collapsed F1 after fine-tune | Learning rate still `2e-5`? |
| UI labels feel backwards | `id2label` vs `app.py` probs mapping |
| Metrics too good to be true | Evaluating on test or train? |

---

*Miles Robinson · Break it, write the what-if, fix it · 2026*
