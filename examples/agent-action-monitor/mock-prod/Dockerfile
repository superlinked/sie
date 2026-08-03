# mock-prod: dummy downstream controller/DB target for the DUSK example.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 1000 mockprod \
    && chown -R mockprod:mockprod /app

USER mockprod

COPY app.py .

ENV MOCK_PROD_HOST=0.0.0.0 \
    MOCK_PROD_PORT=9000

EXPOSE 9000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:9000/health', timeout=3)"]

CMD ["python", "app.py"]
