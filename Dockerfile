FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PORT=8080
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY src ./src
COPY data ./data
ENV PYTHONPATH=/app/src CORPUS_DIR=/app/data/corpus
CMD exec uvicorn ragobs.app:app --host 0.0.0.0 --port ${PORT}
