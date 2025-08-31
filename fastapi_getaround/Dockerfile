FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# code
COPY . .

# port requis par HF
EXPOSE 7860

# lance FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
