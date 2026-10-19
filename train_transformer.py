"""Fine-tune DistilBERT on dair-ai/emotion and compare it with the TF-IDF model.

Same splits as train.py (16,000 train / 2,000 validation / 2,000 test). Sized for a CPU:
sentences are truncated to 64 tokens and training runs for 2 epochs by default.

Usage:
    python train_transformer.py                 # full run (about 35-50 minutes on a 2-core CPU)
    python train_transformer.py --limit 256     # quick smoke test on a few sentences

Protocol:
    * train on the train split only
    * after each epoch, score the validation split; keep the epoch with the best validation macro F1
    * score that model once on the untouched test split
    * export it to ONNX, quantize the weights to int8, and score the quantized model on the
      same test split (this is the version app.py serves)

Outputs:
    models/distilbert-emotion/           fine-tuned PyTorch model + tokenizer (not committed, ~270 MB)
    models/distilbert-emotion-int8.onnx  quantized model used by app.py
    models/tokenizer.json                tokenizer used by app.py
    results/transformer_metrics.json     validation + test scores, training time
    results/transformer_confusion_matrix.png
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          get_linear_schedule_with_warmup)

from train import LABELS, load_split, plot_confusion, scores

BASE_MODEL = "distilbert-base-uncased"
MODEL_DIR = Path("models")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def encode(tokenizer, df, max_length):
    enc = tokenizer(list(df["text"]), truncation=True, max_length=max_length)
    label_ids = [LABELS.index(label) for label in df["label"]]
    return [{"input_ids": ids, "attention_mask": mask, "labels": y}
            for ids, mask, y in zip(enc["input_ids"], enc["attention_mask"], label_ids)]


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds = []
    for batch in loader:
        batch.pop("labels", None)
        preds.extend(model(**batch).logits.argmax(-1).tolist())
    return [LABELS[i] for i in preds]


def export_onnx(model, tokenizer, max_length):
    """Export to ONNX, then quantize the weights to int8 so the app needs only onnxruntime."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    fp32_path = MODEL_DIR / "distilbert-emotion-fp32.onnx"
    int8_path = MODEL_DIR / "distilbert-emotion-int8.onnx"
    model = model.eval()
    dummy = tokenizer(["export example"], return_tensors="pt", padding="max_length", max_length=max_length)
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]), str(fp32_path),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    fp32_path.unlink()
    tokenizer.backend_tokenizer.save(str(MODEL_DIR / "tokenizer.json"))
    return int8_path


def predict_onnx(onnx_path, tokenizer, texts, max_length, batch_size=64):
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    preds = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=max_length,
                        padding=True, return_tensors="np")
        logits = session.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                          "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
        preds.extend(logits.argmax(-1).tolist())
    return [LABELS[i] for i in preds]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="folder with train.txt, val.txt, test.txt")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="use only N sentences per split (smoke test; results not saved)")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train, val, test = (load_split(s, args.data_dir) for s in ("train", "val", "test"))
    if args.limit:
        train, val, test = (d.sample(n=min(args.limit, len(d)), random_state=args.seed) for d in (train, val, test))
    log(f"Loaded {len(train)} train / {len(val)} validation / {len(test)} test sentences")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})

    lengths = [len(ids) for ids in tokenizer(list(test["text"]))["input_ids"]]
    truncated = float(np.mean([n > args.max_length for n in lengths]))
    log(f"Test sentences longer than {args.max_length} tokens (truncated): {truncated:.2%}")

    collate = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(encode(tokenizer, train, args.max_length), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(encode(tokenizer, val, args.max_length), batch_size=64, collate_fn=collate)
    test_loader = DataLoader(encode(tokenizer, test, args.max_length), batch_size=64, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best = {"macro_f1": -1}
    history = []
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 50 == 0 or step == len(train_loader):
                elapsed = time.time() - t_start
                done = (epoch - 1) * len(train_loader) + step
                log(f"epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f} "
                    f"elapsed {elapsed / 60:.1f} min, ~{elapsed / done * (total_steps - done) / 60:.1f} min left")
        val_scores = scores(val["label"], predict(model, val_loader))
        history.append({"epoch": epoch, "train_loss": round(running / len(train_loader), 4), "validation": val_scores})
        log(f"epoch {epoch} validation {val_scores}")
        if val_scores["macro_f1"] > best["macro_f1"]:
            best = {"epoch": epoch, **val_scores, "state": copy.deepcopy(model.state_dict())}
    train_minutes = (time.time() - t_start) / 60

    model.load_state_dict(best.pop("state"))
    pred = predict(model, test_loader)
    test_scores = scores(test["label"], pred)
    log(f"Test (PyTorch fp32, epoch {best['epoch']}): {test_scores}")
    print(classification_report(test["label"], pred, digits=3), flush=True)

    if args.limit:
        log("Smoke test finished; nothing saved.")
        return

    MODEL_DIR.mkdir(exist_ok=True)
    model.save_pretrained(MODEL_DIR / "distilbert-emotion")
    tokenizer.save_pretrained(MODEL_DIR / "distilbert-emotion")
    onnx_path = export_onnx(model, tokenizer, args.max_length)
    pred_int8 = predict_onnx(onnx_path, tokenizer, list(test["text"]), args.max_length)
    int8_scores = scores(test["label"], pred_int8)
    agreement = float(np.mean([a == b for a, b in zip(pred, pred_int8)]))
    log(f"Test (ONNX int8): {int8_scores}; same prediction as fp32 on {agreement:.2%} of test sentences")

    report = classification_report(test["label"], pred, output_dict=True)
    report_int8 = classification_report(test["label"], pred_int8, output_dict=True)
    Path("results").mkdir(exist_ok=True)
    json.dump({
        "base_model": BASE_MODEL,
        "settings": {k: v for k, v in vars(args).items() if k not in ("data_dir", "limit")},
        "selected_epoch": best["epoch"],
        "history": history,
        "validation": {"accuracy": best["accuracy"], "macro_f1": best["macro_f1"]},
        "test": test_scores,
        "test_per_class_f1": {k: round(report[k]["f1-score"], 4) for k in LABELS},
        "test_onnx_int8": int8_scores,
        "test_onnx_int8_per_class_f1": {k: round(report_int8[k]["f1-score"], 4) for k in LABELS},
        "int8_fp32_agreement": round(agreement, 4),
        "test_truncated_share": round(truncated, 4),
        "train_minutes": round(train_minutes, 1),
        "hardware": f"CPU, {torch.get_num_threads()} threads",
    }, open("results/transformer_metrics.json", "w"), indent=2)
    plot_confusion(test["label"], pred_int8, "results/transformer_confusion_matrix.png")
    log("Saved models/ and results/transformer_*")


if __name__ == "__main__":
    main()
