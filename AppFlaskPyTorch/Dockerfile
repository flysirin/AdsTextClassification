# First layer
FROM python:3.12-slim AS builder

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Final image
FROM python:3.12-slim

# Copy only important extentions
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy important models
COPY config /app/config
COPY models /app/models
COPY saved /app/saved
COPY utils /app/utils

COPY app.py /app/app.py

# Go to work path
WORKDIR /app

# Open port for Flask
EXPOSE 5000

# Run
CMD ["python", "app.py"]
