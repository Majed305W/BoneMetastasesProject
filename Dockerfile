# Dockerfile for BoneMetastasesProject

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# System dependencies (needed for OpenCV, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Install Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Collect static, apply migrations, start gunicorn
CMD ["sh", "-c", "python manage.py migrate && python manage.py collectstatic --noinput && gunicorn aipipeline.wsgi:application --bind 0.0.0.0:8000"]
