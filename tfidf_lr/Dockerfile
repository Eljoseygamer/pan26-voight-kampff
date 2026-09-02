FROM python:3.10-slim
WORKDIR /app
COPY shared/ shared/
ADD https://raw.githubusercontent.com/Eljoseygamer/pan26-voight-kampff/main/data/train.jsonl data/train.jsonl
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY predict.py predict.py
ENTRYPOINT ["python", "predict.py"]
