FROM python:3.11-slim

# Install system dependencies for C-extensions and Google SDK
RUN apt-get update && apt-get install -y \
    libffi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# CRITICAL: Upgrade pip first to handle complex dependency resolutions
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

COPY ./app /code/app
ENV PYTHONPATH=/code

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]