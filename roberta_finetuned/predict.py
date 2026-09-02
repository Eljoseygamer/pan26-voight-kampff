import sys
import os
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
from utils import load_input, write_predictions

MODEL_NAME = os.environ.get('MODEL_PATH', 'eljosey40/roberta-finetuned-pan26-voightkampff')
BATCH_SIZE = 32
MAX_LEN = 512


def predict(texts, model, tokenizer, device):
    scores = []
    model.eval()
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc='Predicting'):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        scores.extend(probs.tolist())
    return scores


def main():
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Loading model from {MODEL_NAME}...')
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    print('Loading test data...')
    items = load_input(input_file)
    ids = [x[0] for x in items]
    texts = [x[1] for x in items]
    scores = predict(texts, model, tokenizer, device)
    write_predictions(output_dir, ids, scores)


if __name__ == '__main__':
    main()
