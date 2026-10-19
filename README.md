# Mood Detector

An NLP app that reads a short message and predicts the emotion behind it:
**anger, fear, joy, love, sadness, or surprise**.

Built as a team project by **Calvin Rutherford** and **Taylor Unger**
(original repo: [calvin-rutherford/mood-detector](https://github.com/calvin-rutherford/mood-detector)).

## How it works

1. **Training** (`train_baseline.py`): a scikit-learn pipeline turns text into
   TF-IDF features (single words and word pairs) and classifies them with
   logistic regression. Classes are weighted so each emotion counts equally.
2. **Evaluation**: stratified 5-fold cross-validation plus a held-out test set,
   reported as macro F1 and a per-emotion classification report.
3. **App** (`app.py`): a Streamlit page where you type a message and see the
   predicted emotion with the top probabilities.

## Run it

```bash
pip install -r requirements.txt
python train_baseline.py      # trains and saves mood_baseline.joblib
streamlit run app.py
```

## Current results and next steps

The baseline is trained on `data.csv`, about 300 hand-labeled sentences
(50 per emotion). It scores roughly 0.40 macro F1 in cross-validation, which
beats chance (about 0.17 for six classes) but leaves a lot of room to improve.

Planned improvements:
- Train on a larger labeled emotion dataset (thousands of examples per class).
- Compare the baseline against a fine-tuned transformer model.
- Add a confusion matrix and per-class metrics to the README.
