# Project Plan (Now → Next)

## Milestones
- M0 Data ready (done): Merge months 3–5; UTF-8; dedup; CreatedOn parsed.
- M1 Taxonomy + Labels v1: topic/sub_topic columns added; fill gaps with QueryType→topic and "Other" for sub_topic.
- M2 Training (Colab): Train LoRA models (topic, sub-topic) on xlm-roberta-base with threshold tuning; save artifacts to Drive.
- M3 Inference (Local): Score combined CSV on CPU; export predictions; run error analysis and iterate.

## Colab Training (GPU)
1) Runtime: GPU; Install deps:
   - transformers, datasets, accelerate, peft, sentencepiece, scikit-learn, pandas, numpy, tqdm, pyarrow
2) Data path:
   - /content/drive/MyDrive/Kisaan/Datasets/KCC_MarMay2025_combined.csv
3) Train:
   - Topic:
     - epochs=4, batch_size=16 (8 if OOM), max_length=160, lr=2e-5
   - Sub-topic: same settings
4) Save:
   - hf_model/, labels.json, thresholds.json, metrics.json per head

## Local Inference (CPU)
- Load both heads + thresholds; score CSV; write prob_topic::*, prob_sub::*, pred_topic, pred_sub_topic.

## Evaluation
- Report micro/macro Precision/Recall/F1.
- Per-class F1 (Top 20 classes).
- Confusion/error review on misclassified frequent classes.

## Risks & Mitigations
- Imbalance: class weights not used (LoRA) → manage via taxonomy merge + threshold tuning.
- Noisy text: keep domain terms; minimal normalization.
- Limited labels: bootstrap from QueryType, then iterative annotation.

## Stretch (later)
- Named Entity Extraction (crops, pests, chemicals).
- Routing with priority scores.
- Trend dashboards by district/crop/time.
- FAQ retrieval / answer suggestion.

## Next 24 Hours
- [ ] Freeze taxonomy v1 and add `topic`/`sub_topic` columns (fill with fallbacks where needed).
- [ ] Push combined CSV + updated columns to Drive/Git.
- [ ] Run Colab training for Topic, then Sub-topic; download artifacts.
- [ ] Run local prediction; review per-class F1 and 20 misclassified samples for quick fixes.