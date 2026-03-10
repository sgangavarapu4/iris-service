# 1. Use a lightweight Python image
FROM python:3.11

WORKDIR /code

# Copy requirements first to keep builds fast (caching)
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the entire app folder (This now includes your model AND pipeline yaml)
COPY ./app /code/app

# Set PYTHONPATH so python can find the 'app' module easily
ENV PYTHONPATH=/code

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]