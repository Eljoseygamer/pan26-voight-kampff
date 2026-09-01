FROM python:3.10-slim
WORKDIR /app
COPY shared/ shared/
COPY data/train.jsonl data/train.jsonl
COPY tfidf_lr/requirements.txt tfidf_lr/requirements.txt
RUN pip install --no-cache-dir -r tfidf_lr/requirements.txt
COPY tfidf_lr/predict.py tfidf_lr/predict.py
ENTRYPOINT ["python", "tfidf_lr/predict.py"]
