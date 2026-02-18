# 1. Use a lightweight Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /code

# 3. Copy requirements from the ROOT to the container
# We do this FIRST so Docker caches the 'pip install' layer
COPY ./requirements.txt /code/requirements.txt

# 4. Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copy the 'app' folder (code + .pkl) into the container
COPY ./app /code/app

# 6. Run the app
# Note: we use 'app.main:app' because main.py is now inside the 'app' folder
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]