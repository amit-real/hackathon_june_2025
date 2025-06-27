FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_minimal.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_minimal.txt

# Copy application code
COPY app_web_simple.py app_web.py
COPY templates/ templates/
COPY .env.example .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "app_web:app"]