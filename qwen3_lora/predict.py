import sys
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
from utils import load_input, write_predictions

BASE_MODEL = 'Qwen/Qwen3-0.6B'
ADAPTER_PATH = os.environ.get('ADAPTER_PATH', 'eljosey40/qwen3-lora-pan26-voightkampff')
BATCH_SIZE = 16
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
        probs = torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy()
        scores.extend(probs.tolist())
    return scores


def main():
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    print(f'Loading Qwen3 base model: {BASE_MODEL}')
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, quantization_config=bnb, device_map='auto'
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f'Loading LoRA adapters from {ADAPTER_PATH}')
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=False)
    model.eval()
    print('Loading test data...')
    items = load_input(input_file)
    ids = [x[0] for x in items]
    texts = [x[1] for x in items]
    scores = predict(texts, model, tokenizer, device)
    write_predictions(output_dir, ids, scores)


if __name__ == '__main__':
    main()
