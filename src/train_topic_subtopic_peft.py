#!/usr/bin/env python3
"""
Train a multi-label classifier with PEFT/LoRA on top of a base transformer (e.g., xlm-roberta-base).
Trains ONE label column at a time (run twice: once for topic, once for sub_topic).

- Uses BCEWithLogitsLoss via problem_type="multi_label_classification"
- Splits train/val/test (80/10/10)
- Tunes per-class thresholds on the validation set to maximize F1
- Saves: model (with LoRA), tokenizer, label binarizer classes, thresholds.json, metrics.json

Example (Colab):
  !python src/train_topic_subtopic_peft.py \
      --data_csv "/content/drive/MyDrive/Kisaan/Datasets/KCC_MarMay2025_combined.csv" \
      --out_dir  "/content/drive/MyDrive/Kisaan/models_mar_may_2025/topic" \
      --label_col topic --text_col QueryText --base_model xlm-roberta-base
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import Dataset
import torch
from peft import LoraConfig, get_peft_model, TaskType

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label_col", required=True, help="topic or sub_topic")
    ap.add_argument("--text_col", default="QueryText")
    ap.add_argument("--base_model", default="xlm-roberta-base")
    ap.add_argument("--max_length", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def sanitize_text(s):
    if pd.isna(s):
        return ""
    return str(s).strip()

def load_dataframe(path, text_col, label_col):
    df = pd.read_csv(path, encoding="utf-8-sig")
    # normalize column headers
    df = df.rename(columns=lambda c: c.strip().replace(" ", ""))
    # ensure text col present
    if text_col not in df.columns:
        raise ValueError(f"Missing text column: {text_col}")
    # fallbacks for labels if missing
    if label_col not in df.columns:
        if label_col.lower() == "topic" and "QueryType" in df.columns:
            df[label_col] = df["QueryType"]
        else:
            df[label_col] = "Other"
    # clean
    df[text_col] = df[text_col].apply(sanitize_text)
    df = df[df[text_col].str.len() > 0].copy()
    return df

def prepare_labels(series):
    # Accept either single labels or multi-label strings like "A;B"
    labels = []
    for v in series.fillna("Other"):
        v = str(v).strip()
        if ";" in v:
            parts = [p.strip() for p in v.split(";") if p.strip()]
            labels.append(parts if parts else ["Other"])
        else:
            labels.append([v] if v else ["Other"])
    return labels

def tune_thresholds(y_true_bin, y_prob, grid=None):
    # Per-class threshold maximizing F1 on val
    if grid is None:
        grid = np.linspace(0.2, 0.8, 13)  # 0.2..0.8
    thresholds = []
    for j in range(y_true_bin.shape[1]):
        best_f1, best_t = 0.5, 0.5
        yt = y_true_bin[:, j]
        yp = y_prob[:, j]
        for t in grid:
            pred = (yp >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(yt, pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)
    return thresholds

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(args.data_csv, args.text_col, args.label_col)
    y_multi = prepare_labels(df[args.label_col])
    texts = df[args.text_col].tolist()

    # Split (stratify by the first label for stability)
    primary = [ys[0] for ys in y_multi]
    X_train, X_tmp, y_train, y_tmp = train_test_split(texts, y_multi, test_size=0.2, random_state=args.seed, stratify=primary)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=args.seed)

    # Binarize
    mlb = MultiLabelBinarizer(sparse_output=False)
    y_train_bin = mlb.fit_transform(y_train)
    y_val_bin   = mlb.transform(y_val)
    y_test_bin  = mlb.transform(y_test)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    # Build HF Datasets
    ds_train = Dataset.from_dict({"text": X_train, "labels": list(y_train_bin)})
    ds_val   = Dataset.from_dict({"text": X_val,   "labels": list(y_val_bin)})
    ds_test  = Dataset.from_dict({"text": X_test,  "labels": list(y_test_bin)})

    ds_train = ds_train.map(tokenize_batch, batched=True, remove_columns=["text"])
    ds_val   = ds_val.map(tokenize_batch,   batched=True, remove_columns=["text"])
    ds_test  = ds_test.map(tokenize_batch,  batched=True, remove_columns=["text"])

    # Model + PEFT LoRA
    base = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(mlb.classes_),
        problem_type="multi_label_classification",
    )
    # Enable gradient checkpointing for memory savings (works on both GPU and CPU; bigger win on GPU)
    try:
        base.gradient_checkpointing_enable()
    except Exception:
        pass

    # Select LoRA target modules compatible with XLM-R/Roberta/BERT attention blocks.
    # For Roberta-like models, attention linear layers are typically named: query, key, value, dense
    # This avoids using q_proj/k_proj/v_proj/out_proj which are common in LLaMA-style architectures.
    lora_targets = ["query", "key", "value", "dense"]

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=lora_targets,
    )
    model = get_peft_model(base, lora_cfg)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        pred = (probs >= 0.5).astype(int)  # provisional; thresholds tuned later
        # Micro / Macro F1
        return {
            "micro_f1":  f1_score(labels, pred, average="micro", zero_division=0),
            "macro_f1":  f1_score(labels, pred, average="macro", zero_division=0),
            "micro_p":   precision_score(labels, pred, average="micro", zero_division=0),
            "micro_r":   recall_score(labels, pred, average="micro", zero_division=0),
            "macro_p":   precision_score(labels, pred, average="macro", zero_division=0),
            "macro_r":   recall_score(labels, pred, average="macro", zero_division=0),
        }

    # Prefer bf16 on newer GPUs, else fp16 if CUDA available
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    args_train = TrainingArguments(
        output_dir=str(out_dir / "hf_runs"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="macro_f1",
        load_best_model_at_end=True,
        logging_steps=50,
        save_total_limit=2,
        gradient_accumulation_steps=1,
        fp16=(torch.cuda.is_available() and not use_bf16),
        bf16=use_bf16,
        report_to="none",
        seed=args.seed
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    # Evaluate and tune thresholds on val
    val_logits = np.vstack(trainer.predict(ds_val).predictions)
    val_probs  = 1 / (1 + np.exp(-val_logits))
    thresholds = tune_thresholds(y_val_bin, val_probs)

    # Final test metrics with tuned thresholds
    test_logits = np.vstack(trainer.predict(ds_test).predictions)
    test_probs  = 1 / (1 + np.exp(-test_logits))
    preds = (test_probs >= np.array(thresholds)).astype(int)

    results = {
        "labels": list(mlb.classes_),
        "thresholds": thresholds,
        "test_micro_f1":  float(f1_score(y_test_bin, preds, average="micro", zero_division=0)),
        "test_macro_f1":  float(f1_score(y_test_bin, preds, average="macro", zero_division=0)),
        "test_micro_p":   float(precision_score(y_test_bin, preds, average="micro", zero_division=0)),
        "test_micro_r":   float(recall_score(y_test_bin, preds, average="micro", zero_division=0)),
        "test_macro_p":   float(precision_score(y_test_bin, preds, average="macro", zero_division=0)),
        "test_macro_r":   float(recall_score(y_test_bin, preds, average="macro", zero_division=0)),
    }
    print("Test metrics:", results)

    # Save artifacts
    (out_dir / "hf_model").mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir / "hf_model")
    tokenizer.save_pretrained(out_dir / "hf_model")
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(list(mlb.classes_), f, ensure_ascii=False, indent=2)
    with open(out_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()