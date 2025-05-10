FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models/checkpoints logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Add Procfile for Heroku
RUN echo "web: gunicorn src.api.app:app --workers 4 --preload" > Procfile

# Run gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT src.api.app:app --workers 4 --preload 