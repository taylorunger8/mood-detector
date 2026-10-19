import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Mood Detector", layout="centered")
st.title("Mood Detector")

@st.cache_resource
def load_model():
    """Load the trained model, or train it on first run (about 15 seconds)."""
    if Path("mood_model.joblib").exists():
        return joblib.load("mood_model.joblib")
    import train
    with st.spinner("Training the model on 20,000 labeled sentences..."):
        train.main([])
    return joblib.load("mood_model.joblib")

model = load_model()
labels = model.classes_

st.write("Type a message. Click Detect.")

text = st.text_area("Message", "I can't believe this happened!")
top_k = st.slider("Top probabilities to show", 3, min(5, len(labels)), 5)
col1, col2 = st.columns([1,1])
with col1:
    if st.button("Detect"):
        if text.strip():
            pred = model.predict([text])[0]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([text])[0]
            else:
                dec = model.decision_function([text])[0]
                dec = dec - dec.min() + 1e-9
                probs = dec / dec.sum()

            order = np.argsort(probs)[::-1][:top_k]
            top = [(labels[i], float(probs[i])) for i in order]

            st.success(f"Predicted mood: **{pred}**")
            st.write(pd.DataFrame(top, columns=["label","probability"]))
        else:
            st.warning("Enter some text first.")

with col2:
    st.write("Examples")
    ex = [
        "I’m thrilled we did it!",
        "I feel completely alone.",
        "This made my whole day.",
        "I’m terrified of failing.",
        "No way that just happened!"           
    ]
    st.write(pd.DataFrame({"try me": ex}))