# Use Python 3.13 base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (for better Docker caching)
COPY requirements.txt .

# Install all project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else (your main.py, services/, db/, knowledge_base/, etc.)
COPY . .

# Prevent Python output from being buffered
ENV PYTHONUNBUFFERED=1

# Expose the FastAPI port
EXPOSE 8000

# Run your FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
