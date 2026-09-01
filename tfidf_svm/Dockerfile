FROM python:3.10-slim
WORKDIR /app
COPY shared/ shared/
COPY data/train.jsonl data/train.jsonl
COPY tfidf_svm/requirements.txt tfidf_svm/requirements.txt
RUN pip install --no-cache-dir -r tfidf_svm/requirements.txt
COPY tfidf_svm/predict.py tfidf_svm/predict.py
ENTRYPOINT ["python", "tfidf_svm/predict.py"]
