"""Streamlit app: type a message, get the predicted emotion.

Uses the fine-tuned DistilBERT (int8 ONNX, served with onnxruntime) when it is available,
and falls back to the lightweight TF-IDF + logistic regression model otherwise.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
MAX_LENGTH = 64
ONNX_PATH = Path("models/distilbert-emotion-int8.onnx")
TOKENIZER_PATH = Path("models/tokenizer.json")

st.set_page_config(page_title="Mood Detector", layout="centered")
st.title("Mood Detector")


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class TransformerModel:
    name = "DistilBERT, fine-tuned (int8 ONNX)"

    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        self.tokenizer.enable_truncation(MAX_LENGTH)
        self.tokenizer.enable_padding()
        self.session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    def predict_proba(self, texts):
        enc = self.tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        logits = self.session.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]
        return softmax(logits)

    @property
    def classes_(self):
        return LABELS


class TfidfModel:
    name = "TF-IDF + logistic regression (fallback)"

    def __init__(self):
        import joblib

        if not Path("mood_model.joblib").exists():
            import train
            with st.spinner("Training the fallback model on 20,000 labeled sentences..."):
                train.main([])
        self.model = joblib.load("mood_model.joblib")

    def predict_proba(self, texts):
        return self.model.predict_proba(list(texts))

    @property
    def classes_(self):
        return list(self.model.classes_)


@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    try:
        return TransformerModel(), None
    except Exception as exc:  # model files missing, onnxruntime not installed, etc.
        return TfidfModel(), f"{type(exc).__name__}: {exc}"


model, fallback_reason = load_model()
labels = model.classes_
st.caption(f"Model: {model.name}")
if fallback_reason:
    st.caption(f"DistilBERT unavailable, using the fallback. Reason: {fallback_reason}")

st.write("Type a message. Click Detect.")

text = st.text_area("Message", "I can't believe this happened!")
top_k = st.slider("Top probabilities to show", 3, len(labels), 5)
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Detect"):
        if text.strip():
            probs = model.predict_proba([text])[0]
            order = np.argsort(probs)[::-1][:top_k]
            top = [(labels[i], float(probs[i])) for i in order]
            st.success(f"Predicted mood: **{top[0][0]}**")
            st.write(pd.DataFrame(top, columns=["label", "probability"]))
        else:
            st.warning("Enter some text first.")

with col2:
    st.write("Examples")
    ex = [
        "I’m thrilled we did it!",
        "I feel completely alone.",
        "This made my whole day.",
        "I’m terrified of failing.",
        "No way that just happened!",
    ]
    st.write(pd.DataFrame({"try me": ex}))
