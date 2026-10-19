"""Compare the TF-IDF model and the fine-tuned DistilBERT on the same test set.

Run after train.py and train_transformer.py. Accuracy and F1 come from results/*.json;
this script measures what those files don't: model size on disk and prediction speed,
all on the same machine.

Output: results/comparison.json
"""
import json
import os
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from train import load_split

N_LATENCY = 200  # sentences timed one at a time
MAX_LENGTH = 64


def dir_mb(path):
    path = Path(path)
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return round(sum(os.path.getsize(p) for p in files) / 1e6, 1)


def time_it(fn, texts):
    fn(texts[:5])  # warm-up
    single = []
    for t in texts[:N_LATENCY]:
        start = time.perf_counter()
        fn([t])
        single.append((time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    for i in range(0, len(texts), 64):
        fn(texts[i:i + 64])
    total = time.perf_counter() - start
    return {"median_ms_per_sentence_one_at_a_time": round(statistics.median(single), 2),
            "sentences_per_second_batched": round(len(texts) / total, 1)}


def main():
    texts = list(load_split("test")["text"])
    out = {"hardware": f"CPU, {torch.get_num_threads()} threads", "test_sentences": len(texts),
           "latency_sample": N_LATENCY}

    tfidf = joblib.load("mood_model.joblib")
    out["tfidf"] = {"size_mb": dir_mb("mood_model.joblib"), **time_it(tfidf.predict_proba, texts)}

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("models/distilbert-emotion")
    model = AutoModelForSequenceClassification.from_pretrained("models/distilbert-emotion").eval()

    @torch.no_grad()
    def torch_predict(batch):
        enc = tok(batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
        return model(**enc).logits.softmax(-1).numpy()

    weights = Path("models/distilbert-emotion/model.safetensors")
    out["distilbert_fp32_pytorch"] = {"size_mb": dir_mb(weights), **time_it(torch_predict, texts)}

    import onnxruntime as ort
    from tokenizers import Tokenizer
    fast_tok = Tokenizer.from_file("models/tokenizer.json")
    fast_tok.enable_truncation(MAX_LENGTH)
    fast_tok.enable_padding()
    sess = ort.InferenceSession("models/distilbert-emotion-int8.onnx", providers=["CPUExecutionProvider"])

    def onnx_predict(batch):
        enc = fast_tok.encode_batch(batch)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        return sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]

    out["distilbert_int8_onnx"] = {"size_mb": dir_mb("models/distilbert-emotion-int8.onnx"),
                                   **time_it(onnx_predict, texts)}

    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/comparison.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
