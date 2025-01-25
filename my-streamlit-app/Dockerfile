# Use an official Python runtime as a parent image.
FROM python:3.10-slim

# Set environment variable to disable buffering (logs show in real time).
ENV PYTHONUNBUFFERED=1

# Install system dependencies.
# Poppler (for PDF processing) and Tesseract OCR.
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download language data if necessary (e.g., for Tesseract English data).
RUN mkdir -p /usr/share/tessdata && \
    wget -O /usr/share/tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata

# Set TESSDATA_PREFIX so that Tesseract finds its language data.
ENV TESSDATA_PREFIX=/usr/share/tessdata

# Create and set working directory inside the container.
WORKDIR /app

# Copy requirements file first to leverage Docker cache.
COPY requirements.txt /app/

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . /app/

# Expose the port on which Streamlit will run.
EXPOSE 8501

# Run the Streamlit application using Python module syntax.
CMD ["python", "-m", "streamlit", "run", "Api_all_File.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
