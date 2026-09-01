# PAN 2026 Voight-Kampff AI Detection Submission

This repository contains the submission of **José Alejandro Pérez Domínguez**
(Universidad Europea de Valencia) for the **PAN 2026 Voight-Kampff** shared
task on generative AI text detection.

Each subdirectory is a self-contained, Dockerized model that reads an input
JSONL file of texts and writes a JSONL file of AI-generation scores.

## Repository structure

```
data/               train/validation splits used for model development
tfidf_svm/           TF-IDF features + Support Vector Machine classifier
tfidf_lr/            TF-IDF features + Logistic Regression classifier
gpt2_perplexity/      GPT-2 perplexity-based detector
roberta_zeroshot/     Pretrained RoBERTa detector used zero-shot
roberta_finetuned/    RoBERTa fine-tuned on the task's training data
qwen3_lora/           Qwen3 base model with a LoRA adapter fine-tuned for detection
```

## Models

| Model | Directory | Description |
|---|---|---|
| TF-IDF + SVM | [`tfidf_svm/`](tfidf_svm/) | Classical baseline: TF-IDF vectorization with an SVM classifier. |
| TF-IDF + Logistic Regression | [`tfidf_lr/`](tfidf_lr/) | Classical baseline: TF-IDF vectorization with a logistic regression classifier. |
| GPT-2 Perplexity | [`gpt2_perplexity/`](gpt2_perplexity/) | Detects AI-generated text using GPT-2 perplexity as a signal. |
| RoBERTa Zero-Shot | [`roberta_zeroshot/`](roberta_zeroshot/) | Off-the-shelf pretrained RoBERTa AI-text detector, used without task-specific fine-tuning. |
| RoBERTa Fine-Tuned | [`roberta_finetuned/`](roberta_finetuned/) | RoBERTa fine-tuned on the task's train/validation data. |
| Qwen3 + LoRA | [`qwen3_lora/`](qwen3_lora/) | Qwen3 base model with a LoRA adapter fine-tuned for AI-text detection. |

## Input / output format

Each model's `predict.py` follows the same interface:

```
python predict.py <input.jsonl> <output_dir>
```

- **Input**: a JSONL file where each line is a JSON object with at least an
  `id` and `text` field.
- **Output**: `<output_dir>/predictions.jsonl`, one JSON object per line:

  ```json
  {"id": "<id>", "score": 0.0}
  ```

  where `score` is a float in `[0, 1]`. A score `> 0.5` indicates the text is
  predicted **AI-generated**; a score `< 0.5` indicates **human-written**.

## Running with Docker

Each model directory includes a `Dockerfile` (based on `python:3.10-slim`).
To build and run a given model, e.g. `tfidf_svm`:

```bash
cd tfidf_svm
docker build -t pan26-tfidf-svm .
docker run --rm -v /path/to/data:/data pan26-tfidf-svm /data/input.jsonl /data/output
```

## Model artifacts

- `roberta_finetuned/model/` holds the fine-tuned RoBERTa checkpoint (not
  committed; populate before running inference).
- `qwen3_lora/adapters/` holds the trained LoRA adapter weights (not
  committed; populate before running inference).
