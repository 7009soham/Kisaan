# Kisaan Query Topic + Sub-Topic Classification

## Objective
Build a multilingual (Marathi/English) classifier that assigns:
- Topic (e.g., Government Scheme, Pest Management, Fertilizer Use, Market Information, Weather)
- Sub-topic under the selected topic (e.g., PM-Kisan Installment Status, e-KYC, Mobile/OTP Issue, Crop Price)

## Users & Value
- Call center ops: auto-route and prioritize queries.
- Analysts: trend analysis by district/crop/time.
- Product: better self-serve advisories and dashboards.

## Input → Output
- Input: QueryText (free text), optional metadata (crop, district, month).
- Output:
  - topic: one or more labels (multi-label allowed)
  - sub_topic: one or more labels, constrained to the topic(s)
  - probabilities per label (for thresholding and dashboards)

## Taxonomy v1 (illustrative)
- Government Scheme
  - PM-Kisan Installment Status
  - PM-Kisan e-KYC
  - Mobile/OTP Issue
  - Scheme Helpline/Contact
  - Registration/Eligibility
  - NAMO Scheme Installment Status
  - Government Scheme – Other
- Pest Management
  - General Pest/Disease (v1)
- Fertilizer Use
  - Nutrient Management
- Market Information
  - Crop Price
- Weather
  - Forecast/Advisory
- Other
  - Other

Note: Merge or expand sub-topics based on sample counts (keep ≥100 per class if possible).

## Constraints
- Mixed language (Marathi + English).
- Class imbalance (long tail sub-topics).
- Compute: train on Colab GPU; local CPU inference.

## Non-goals (v1)
- Extraction of entities (crops/pests) as structured fields.
- Summarization/answer generation.

## Acceptance Criteria
- Macro F1 ≥ baseline (TF-IDF + linear) and stable across months.
- Per-class thresholds tuned on validation; probabilities exported.
- Reproducible training and inference scripts; no label leakage from KccAns.

## Definition of Done
- Two trained heads (topic, sub-topic) saved with tokenizer, labels.json, thresholds.json, metrics.json.
- Scored CSV for March–May with prob_* columns and pred_* columns.
- Brief error analysis on lowest-F1 classes and a plan to improve.