# Dockerfile - Hybrid: Your working version + performance optimizations
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# ENHANCED: Performance-focused environment variables + Clean logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    DEBUG_LOGS=false \
    PYTHONWARNINGS=ignore \
    LOG_LEVEL=WARNING \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 \
    OPENBLAS_CORETYPE=HASWELL \
    MKL_ENABLE_INSTRUCTIONS=AVX2 \
    MALLOC_ARENA_MAX=2

# Install system dependencies required for PySAM (your proven list)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    python3-dev \
    build-essential \
    cmake \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (your explicit method)
COPY app.py .
COPY config.py .
COPY cloud_storage.py .
COPY pysam_form_integration2.py .
COPY dashboard_module.py .
COPY pysam_main_optimized.py .
COPY logging_config.py .

# Copy directories including all subdirectories
COPY templates/ ./templates/
COPY static/ ./static/
COPY data/ ./data/
COPY FCAS/ ./FCAS/
COPY 1.input/ ./1.input/
COPY 2.pysam/ ./2.pysam/
COPY 3.output/ ./3.output/

# Create necessary directories with proper permissions
RUN mkdir -p logs tmp && \
    mkdir -p 1.input/json_updated && \
    mkdir -p 2.pysam/results && \
    mkdir -p 3.output && \
    chmod 777 2.pysam/results && \
    chmod 777 1.input/json_updated && \
    chmod 777 tmp

# SKIP user creation (as you discovered it causes permission issues)
# Run as root to avoid permission problems

# Expose port (keep your working setup)
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# OPTIMIZED: Enhanced gunicorn config for Cloud Run performance
CMD exec gunicorn --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 8 \
    --timeout 300 \
    --worker-class gthread \
    --worker-tmp-dir /tmp \
    --preload \
    --access-logfile - \
    --error-logfile - \
    app:app