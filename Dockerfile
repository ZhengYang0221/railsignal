FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the API. Override CMD in docker-compose for the dashboard.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
