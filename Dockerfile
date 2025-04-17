FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# This is optional and just documents the port intention
EXPOSE ${PORT:-8000}

# Use the PORT environment variable from Cloud Run
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
