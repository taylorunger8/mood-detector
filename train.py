"""Train the mood detector on the dair-ai/emotion dataset (20,000 labeled sentences).

Usage:
    python train.py                    # downloads the dataset from Hugging Face
    python train.py --data-dir data/   # or use local train.txt / val.txt / test.txt
                                       # files in "text;label" format

Outputs:
    mood_model.joblib              trained pipeline used by app.py
    results/metrics.json           validation + test scores
    results/confusion_matrix.png   per-emotion test results
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]  # dair-ai label order
HF_REPO = "dair-ai/emotion"
HF_REVISION = "cab853a1dbdf4c42c2b3ef2173804746df8825fe"  # pinned so results are reproducible
SPLITS = {"train": "train", "val": "validation", "test": "test"}


def load_split(name, data_dir=None):
    """Return one split as a DataFrame with columns text, label (label = emotion name)."""
    if data_dir:
        return pd.read_csv(Path(data_dir) / f"{name}.txt", sep=";", names=["text", "label"], quoting=3)
    from huggingface_hub import hf_hub_download  # cached in ~/.cache/huggingface after the first run

    path = hf_hub_download(HF_REPO, f"split/{SPLITS[name]}-00000-of-00001.parquet",
                           repo_type="dataset", revision=HF_REVISION)
    df = pd.read_parquet(path)
    df["label"] = df["label"].map(dict(enumerate(LABELS)))
    return df[["text", "label"]]


def build_model():
    # Same pipeline design as the original baseline, tuned on the validation set.
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),   # single words + word pairs
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",  # surprise and love are rare; weight them up
            C=5.0,                    # picked from C in {2, 5, 10, 20} on validation
            random_state=42,
        )),
    ])


def scores(y_true, y_pred):
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4)}


def plot_confusion(y_true, y_pred, path):
    order = ["joy", "sadness", "anger", "fear", "love", "surprise"]
    cm = confusion_matrix(y_true, y_pred, labels=order, normalize="true")
    cmap = LinearSegmentedColormap.from_list("blue", ["#f7fafe", "#9ec5f4", "#3987e5", "#0d366b"])
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(order)):
        for j in range(len(order)):
            v = cm[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.5 else "#1f1f1f")
    ax.set_xticks(range(len(order)), order, fontsize=9, color="#3d3d3a")
    ax.set_yticks(range(len(order)), order, fontsize=9, color="#3d3d3a")
    ax.set_xlabel("Predicted emotion", color="#3d3d3a")
    ax.set_ylabel("Actual emotion", color="#3d3d3a")
    ax.set_title("Test set: share of each actual emotion predicted as...", fontsize=10, color="#1f1f1f")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="folder with train.txt, val.txt, test.txt")
    args = ap.parse_args(argv)

    train, val, test = (load_split(s, args.data_dir) for s in ("train", "val", "test"))
    print(f"Loaded {len(train)} train / {len(val)} validation / {len(test)} test sentences")

    model = build_model().fit(train["text"], train["label"])
    val_scores = scores(val["label"], model.predict(val["text"]))
    print("Validation:", val_scores)

    # Final model: train on train + validation, then score once on the untouched test set.
    model = build_model().fit(pd.concat([train["text"], val["text"]]), pd.concat([train["label"], val["label"]]))
    pred = model.predict(test["text"])
    test_scores = scores(test["label"], pred)
    print("Test:", test_scores)
    print(classification_report(test["label"], pred, digits=3))

    Path("results").mkdir(exist_ok=True)
    report = classification_report(test["label"], pred, output_dict=True)
    json.dump({"validation": val_scores, "test": test_scores,
               "test_per_class_f1": {k: round(report[k]["f1-score"], 4) for k in LABELS}},
              open("results/metrics.json", "w"), indent=2)
    plot_confusion(test["label"], pred, "results/confusion_matrix.png")
    joblib.dump(model, "mood_model.joblib")
    print("Saved mood_model.joblib and results/")


if __name__ == "__main__":
    main()
