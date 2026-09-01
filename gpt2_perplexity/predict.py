"""
GPT-2 perplexity-based detector for PAN 2026 Voight-Kampff AI Detection.

Usage:
    python predict.py <input.jsonl> <output_dir>

Input:  JSONL file where each line has at least an "id" and "text" field.
Output: <output_dir>/predictions.jsonl, one JSON object per line:
        {"id": <id>, "score": <float in [0, 1]>}
        score > 0.5 -> predicted AI-generated
        score < 0.5 -> predicted human-written

Approach: lower perplexity under GPT-2 tends to correlate with machine-generated
text; the raw perplexity is mapped to a [0, 1] score (e.g. via a fitted
sigmoid/threshold) before being written out.
"""
import json
import os
import sys

MODEL_NAME = "gpt2"


def load_model():
    # TODO: load GPT-2 tokenizer/model (e.g. via transformers) for perplexity scoring
    raise NotImplementedError("load_model() not yet implemented")


def predict_scores(model, texts):
    # TODO: compute per-text perplexity and map to a [0, 1] AI-likelihood score
    raise NotImplementedError("predict_scores() not yet implemented")


def main():
    if len(sys.argv) != 3:
        print("Usage: python predict.py <input.jsonl> <output_dir>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "predictions.jsonl")

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    model = load_model()
    scores = predict_scores(model, [r["text"] for r in records])

    with open(output_path, "w", encoding="utf-8") as f:
        for record, score in zip(records, scores):
            f.write(json.dumps({"id": record["id"], "score": float(score)}) + "\n")


if __name__ == "__main__":
    main()
