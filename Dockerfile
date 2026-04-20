FROM python:3.10-slim

# System dependencies for scipy/scikit-image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Use non-interactive matplotlib backend (no display in container)
ENV MPLBACKEND=Agg

# Create non-root user
RUN useradd --create-home appuser
RUN chown -R appuser:appuser /app
USER appuser

# Default: run the main segmentation script
ENTRYPOINT ["python", "segment3d_2.py"]
