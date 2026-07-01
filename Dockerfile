FROM python:3.9-slim 

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]