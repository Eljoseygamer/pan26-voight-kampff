FROM python:3.10-slim
WORKDIR /app
COPY shared/ shared/
COPY roberta_finetuned/requirements.txt roberta_finetuned/requirements.txt
RUN pip install --no-cache-dir -r roberta_finetuned/requirements.txt
COPY roberta_finetuned/predict.py roberta_finetuned/predict.py
ENTRYPOINT ["python", "roberta_finetuned/predict.py"]
