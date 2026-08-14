FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import transformers.utils.import_utils as iu; p = iu.__file__; code = open(p).read(); open(p, 'w').write(code + '\n\n_torch_available = True\n\ndef is_torch_available(): return True\n')"

# Copy the backend code and models
COPY . /app

# Expose port (Hugging Face Spaces uses 7860, Render uses whatever PORT env var)
ENV PORT=7860
EXPOSE 7860

# Start the FastAPI server
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
