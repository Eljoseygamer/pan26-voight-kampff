import json
import re
import unicodedata
import hashlib


def preprocess(text):
    text = unicodedata.normalize('NFC', str(text))
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return re.sub(r'[ \t]+', ' ', text).strip()


def load_input(path):
    items = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                items.append((d['id'], preprocess(d['text'])))
    return items


def load_train(path):
    texts, labels = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                texts.append(preprocess(d['text']))
                labels.append(int(d['label']))
    seen = set()
    clean_texts, clean_labels = [], []
    for t, l in zip(texts, labels):
        h = hashlib.md5(' '.join(t.lower().split()).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            clean_texts.append(t)
            clean_labels.append(l)
    return clean_texts, clean_labels


def write_predictions(output_dir, ids, scores):
    import os
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'predictions.jsonl')
    with open(out_path, 'w') as f:
        for id_, score in zip(ids, scores):
            f.write(json.dumps({'id': id_, 'score': round(float(score), 4)}) + '\n')
    print(f'Written {len(ids)} predictions to {out_path}')
