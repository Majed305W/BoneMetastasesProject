# Dockerfile for Bone Metastases EPR Django App

# 1. Use a lightweight Python base image
FROM python:3.11-slim

# 2. Environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. Set working directory
WORKDIR /app

# 4. Install system dependencies (OpenCV, glib, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy all project files into the container
COPY . /app/

# 6. Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 7. Run the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
