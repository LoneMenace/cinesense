FROM python:3.10-slim

# Prevent Python from writing pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (layer caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Streamlit config (no browser, bind to all interfaces)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
