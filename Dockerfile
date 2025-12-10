# -----------------------------
# 1️⃣ Base Image: Slim + Python 3.10
# -----------------------------
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# -----------------------------
# 2️⃣ System Dependencies
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Fix OpenMP error for MacOS/Linux if needed
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV PYTHONUNBUFFERED=1

# -----------------------------
# 3️⃣ Copy & Install Python Dependencies
# -----------------------------
COPY requirements.txt .
# Install with no cache and pin critical versions
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -----------------------------
# 4️⃣ Copy Project Files
# -----------------------------
# Only copy code, configs, scripts (ignore heavy data/models)
COPY . .

# -----------------------------
# 5️⃣ Setup Models (optional: pull from S3 inside container)
# -----------------------------
RUN python scripts/setup_models.py

# -----------------------------
# 6️⃣ Expose API Port
# -----------------------------
EXPOSE 8000

# -----------------------------
# 7️⃣ Entry Point: FastAPI Server
# -----------------------------
CMD ["uvicorn", "src/app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
