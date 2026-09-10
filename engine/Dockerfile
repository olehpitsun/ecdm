FROM python:3.12-slim

WORKDIR /engine

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY reference_policy.json .

EXPOSE 8100

CMD ["python", "app.py"]
