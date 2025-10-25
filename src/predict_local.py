#!/usr/bin/env python3
"""
Local CPU inference script that:
- Loads Topic and Sub-Topic models (LoRA adapters saved by train script)
- Scores a CSV and writes predictions with probabilities and labels

Usage:
  python src/predict_local.py \
    --data_csv .\Datasets\KCC_MarMay2025_combined.csv \
    --model_topic C:\path\to\models_mar_may_2025\topic \
    --model_subtopic C:\path\to\models_mar_may_2025\subtopic \
    --text_col QueryText \
    --out_csv .\Datasets\KCC_MarMay2025_scored.csv
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_head(model_dir: Path):
    mdl_dir = model_dir / "hf_model"
    labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
    thresholds = json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(mdl_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(mdl_dir)
    mdl.eval()
    return tok, mdl, labels, thresholds

def predict_batch(tokenizer, model, texts, max_length=160, batch_size=32):
    probs_all = []
    device = "cpu"
    model.to(device)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
            probs = torch.sigmoid(logits).cpu().numpy()
        probs_all.append(probs)
    return np.vstack(probs_all)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--model_topic", required=True)
    ap.add_argument("--model_subtopic", required=True)
    ap.add_argument("--text_col", default="QueryText")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv, encoding="utf-8-sig")
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column: {args.text_col}")
    texts = df[args.text_col].fillna("").astype(str).tolist()

    # Load models
    tok_t, mdl_t, labels_t, thr_t = load_head(Path(args.model_topic))
    tok_s, mdl_s, labels_s, thr_s = load_head(Path(args.model_subtopic))

    # Predict
    probs_t = predict_batch(tok_t, mdl_t, texts)
    probs_s = predict_batch(tok_s, mdl_s, texts)

    preds_t = (probs_t >= np.array(thr_t)).astype(int)
    preds_s = (probs_s >= np.array(thr_s)).astype(int)

    # Attach to dataframe
    for j, name in enumerate(labels_t):
        df[f"prob_topic::{name}"] = probs_t[:, j]
    for j, name in enumerate(labels_s):
        df[f"prob_sub::{name}"] = probs_s[:, j]

    def names_from(pred_row, label_list):
        return ";".join([lbl for lbl, v in zip(label_list, pred_row) if v == 1]) or ""

    df["pred_topic"] = [names_from(row, labels_t) for row in preds_t]
    df["pred_sub_topic"] = [names_from(row, labels_s) for row in preds_s]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote: {args.out_csv}")

if __name__ == "__main__":
    main()