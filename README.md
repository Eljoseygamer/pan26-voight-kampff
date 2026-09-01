# PAN 2026 — Voight-Kampff AI Detection

Submission by Jose Alejandro Perez Dominguez
Master en Inteligencia Artificial — Universidad Europea de Valencia

## Models

| Directory | Model | Notes |
|---|---|---|
| tfidf_lr/ | TF-IDF + Logistic Regression | Baseline, trains at inference time |
| tfidf_svm/ | TF-IDF + LinearSVC (calibrated) | Baseline, trains at inference time |
| roberta_finetuned/ | RoBERTa-base fine-tuned | Main model. Weights on HuggingFace |
| qwen3_lora/ | Qwen3-0.6B + QLoRA | Advanced model. Base + adapters on HuggingFace |

## Usage

Each model is invoked as:
  python model_dir/predict.py /path/to/dataset.jsonl /path/to/output_dir

Output: predictions.jsonl with {"id": "...", "score": 0.XXXX} per line.
Score > 0.5 = AI-generated, score < 0.5 = human-written.

## Training data

data/train.jsonl — PAN 2026 Voight-Kampff training set (access via TIRA/Zenodo).
