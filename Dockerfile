FROM python:3.12-slim-bookworm

ARG APP_VERSION=0.6.0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_VERSION=${APP_VERSION}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl gosu \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 analyzer \
    && useradd --uid 1000 --gid analyzer --create-home analyzer

WORKDIR /opt/veo-analyzer
COPY pyproject.toml ./
COPY app ./app
RUN pip install .

RUN mkdir -p /config /exports \
    && chown -R analyzer:analyzer /opt/veo-analyzer /config /exports

EXPOSE 8080
VOLUME ["/config", "/exports"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 CMD curl -fsS http://127.0.0.1:8080/health || exit 1
COPY docker/entrypoint.sh /usr/local/bin/veo-analyzer-entrypoint
RUN chmod 0755 /usr/local/bin/veo-analyzer-entrypoint
ENTRYPOINT ["/usr/local/bin/veo-analyzer-entrypoint"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1"]
