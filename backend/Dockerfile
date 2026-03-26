FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# F-025: Run as non-root user (security hardening)
RUN useradd -r -u 1001 -s /sbin/nologin app \
    && chown -R app:app /app
USER app

EXPOSE 8000

# F-026: Health check so Docker/PM2 can detect unhealthy containers
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
