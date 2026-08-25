FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY static ./static
COPY run.py .
RUN mkdir -p /app/data

ENV SUMMARY_HOST=0.0.0.0 SUMMARY_PORT=28120 SUMMARY_DATA_DIR=/app/data
EXPOSE 28120
VOLUME ["/app/data"]
CMD ["python", "run.py"]
