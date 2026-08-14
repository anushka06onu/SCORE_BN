FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    sed -i 's/def is_torch_available():/def is_torch_available(): return True\n    def old_is_torch_available():/' /usr/local/lib/python3.10/site-packages/transformers/utils/import_utils.py

# Copy the backend code and models
COPY . /app

# Expose port (Hugging Face Spaces uses 7860, Render uses whatever PORT env var)
ENV PORT=7860
EXPOSE 7860

# Start the FastAPI server
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
