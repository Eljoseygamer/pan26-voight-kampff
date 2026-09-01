FROM python:3.10-slim
WORKDIR /app
COPY shared/ shared/
COPY qwen3_lora/requirements.txt qwen3_lora/requirements.txt
RUN pip install --no-cache-dir -r qwen3_lora/requirements.txt
COPY qwen3_lora/predict.py qwen3_lora/predict.py
ENTRYPOINT ["python", "qwen3_lora/predict.py"]
