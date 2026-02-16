# Use slim base image
FROM python:3.10-slim

# Create non-root user & group
RUN useradd -m appuser

# Set working directory
WORKDIR /app

# Copy & install requirements as root (needed for system deps)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Run as non-root
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]