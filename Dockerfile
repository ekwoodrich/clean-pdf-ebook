# Use an official Python base image
FROM python:3.12-slim

# Install system dependencies
# tesseract-ocr for OCR, libgl1 for OpenCV, etc.
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    ocrmypdf \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy the script and requirements
COPY clean_pdf.py pyproject.toml .

# Install dependencies using uv
RUN uv sync

# Create directories for volumes
RUN mkdir -p /input /output /archive

# Copy the entrypoint script
COPY docker-run.sh /docker-run.sh
RUN chmod +x /docker-run.sh

# Set the entrypoint
ENTRYPOINT ["/docker-run.sh"]
