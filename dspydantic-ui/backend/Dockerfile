FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for pdf2image and healthcheck
RUN apt-get update && apt-get install -y \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dspydantic from parent directory first
COPY pyproject.toml /tmp/pyproject.toml
COPY README.md /tmp/README.md
COPY src /tmp/src
WORKDIR /tmp
RUN pip install --no-cache-dir -e .

# Install backend requirements
WORKDIR /app
COPY dspydantic-ui/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY dspydantic-ui/backend/. .

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

CMD ["/app/entrypoint.sh", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
