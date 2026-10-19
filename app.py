"""
Gradio demo: SMS Spam vs Ham classifier
Loads the fine-tuned DistilBERT model from models/distilbert-sms-spam/

Run:  python app.py
Then open the local URL Gradio prints.

Author: Miles Robinson (2026)
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "distilbert-sms-spam"

_model = None
_tokenizer = None
_load_error: str | None = None


def load_model() -> None:
    global _model, _tokenizer, _load_error
    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        _load_error = (
            "Model not found. Run `python train.py` first to fine-tune "
            f"DistilBERT and save weights to:\n  {MODEL_DIR}"
        )
        return
    try:
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        _model.eval()
        _load_error = None
    except Exception as exc:  # noqa: BLE001 — surface any load failure in UI
        _load_error = f"Failed to load model from {MODEL_DIR}:\n{exc}"


def predict(text: str) -> tuple[str, dict]:
    """Return human label and class-probability dict for Gradio Label."""
    if _load_error or _model is None or _tokenizer is None:
        return (
            "⚠️ Model not available — run train.py first",
            {"ham": 0.0, "spam": 0.0},
        )

    text = (text or "").strip()
    if not text:
        return "Enter an SMS message above.", {"ham": 0.0, "spam": 0.0}

    inputs = _tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128
    )
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    # Model id2label: 0=ham, 1=spam
    conf = {"ham": float(probs[0]), "spam": float(probs[1])}
    label = "🚨 SPAM" if conf["spam"] >= conf["ham"] else "✅ HAM (not spam)"
    return label, conf


def build_ui() -> gr.Blocks:
    examples = [
        ["Hey, are we still meeting for lunch at noon?"],
        ["FREE entry to win $1000 cash! Call now 09061701461. Claim code: XYZ"],
        ["Your package will arrive tomorrow. Tracking: 1Z999AA10123456784"],
        ["URGENT! You have won a £500 prize. Text CLAIM to 81010 now."],
        ["Can you pick up milk on the way home? Thanks!"],
    ]

    with gr.Blocks(title="SMS Spam Classifier") as demo:
        gr.Markdown(
            """
            # SMS Spam vs Ham Classifier
            Fine-tuned **DistilBERT** for binary SMS classification.

            Cybersecurity angle: spam / phishing-adjacent message detection
            as a lightweight text-classification demo.
            """
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(
                    label="SMS message",
                    placeholder="Paste or type an SMS here...",
                    lines=4,
                )
                btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                out_label = gr.Textbox(label="Prediction")
                out_conf = gr.Label(label="Confidence", num_top_classes=2)

        btn.click(fn=predict, inputs=inp, outputs=[out_label, out_conf])
        inp.submit(fn=predict, inputs=inp, outputs=[out_label, out_conf])
        gr.Examples(examples=examples, inputs=inp)

        if _load_error:
            gr.Markdown(f"**Setup needed:** {_load_error}")
        else:
            gr.Markdown(f"*Model loaded from `{MODEL_DIR.name}/`*")

    return demo


if __name__ == "__main__":
    load_model()
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)
