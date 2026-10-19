FROM python:3.10-slim

# ffmpeg is needed to decode the HLS audio segments
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Code/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Code/ ./Code/
COPY models/ ./models/
COPY core/ ./core/

# Cloud Run sets PORT
EXPOSE 8080

CMD ["sh", "-c", "uvicorn Code.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
