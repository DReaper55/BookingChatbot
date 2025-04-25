FROM python:3.11.1

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy your app
COPY . .

# Expose the port FastAPI will run on
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
