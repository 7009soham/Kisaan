# Run Guide: Kisaan Topic/Sub-topic Classification (No local GPU)

This guide shows how to train on Google Colab (GPU) and run inference locally on CPU.

## Option A: Train on Google Colab (recommended)

1) Upload or sync this repository folder to Google Drive at `MyDrive/Kisaan`.
2) Open the notebook: `notebooks/NoGPU_Run_and_Colab_Options.ipynb` in Colab.
3) In Colab: Runtime -> Change runtime type -> GPU.
4) Run cells in order. The "Optional: Kisaan Colab Training" section will:
   - Train Topic and Sub-topic heads with LoRA on `xlm-roberta-base`
   - Save artifacts under `MyDrive/Kisaan/models/topic` and `MyDrive/Kisaan/models/subtopic`
   - Produce `MyDrive/Kisaan/Datasets/KCC_MarMay2025_scored.csv`

Notes
- Data path expected: `MyDrive/Kisaan/Datasets/KCC_MarMay2025_combined.csv`.
- If missing, merge files using `Datasets/combine_kcc_join_m3_4_5_2025.py` locally and upload the resulting CSV.

## Option B: Local CPU inference (Windows PowerShell)

Use this after training on Colab and downloading model folders to your local machine.

```
# Activate your venv
./kisaanev/Scripts/Activate.ps1

# Install local inference deps
pip install -r requirements-local.txt

# Run inference (paths are examples; adjust accordingly)
python src/predict_local.py \
  --data_csv .\Datasets\KCC_MarMay2025_combined.csv \
  --model_topic C:\\path\\to\\models\\topic \
  --model_subtopic C:\\path\\to\\models\\subtopic \
  --text_col QueryText \
  --out_csv .\Datasets\KCC_MarMay2025_scored.csv
```

The output CSV will include columns `prob_topic::*`, `prob_sub::*`, and `pred_topic`, `pred_sub_topic`.

## Option C: Kaggle Notebook (GPU)

- Create a new Kaggle Notebook with GPU (T4/V100), then upload or link your Drive data.
- Install the same dependencies as in Colab.
- Run the same training commands pointing to the dataset path.

## Option D: Hugging Face Spaces (Demo UI)

For a simple public inference UI (after you have trained models):
- Build a small Gradio app that loads your saved `hf_model`, `labels.json`, and `thresholds.json`.
- Create a new Space and push your app. See notes in the notebook (Section 3) for helper steps.

## Troubleshooting
- If you see errors about LoRA target modules during training, update to the latest commit (we fixed target modules for XLM-R/Roberta).
- For out-of-memory on Colab, reduce batch size to 8 and/or set `max_length=128`.
- Ensure CSV encoding is UTF-8 (the merge script writes `utf-8-sig`).
