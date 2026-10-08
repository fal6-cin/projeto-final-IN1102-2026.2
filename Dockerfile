FROM python:3.12-slim

# Extras a instalar, separados por espaço. Exemplos:
#   docker build -t projeto-am .                                  (baselines)
#   docker build --build-arg EXTRAS="baselines tabicl" -t projeto-am .
#   docker build --build-arg EXTRAS="baselines aplr" -t projeto-am .
ARG EXTRAS="baselines"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
COPY data/ data/

RUN uv sync --frozen --group dev $(for e in $EXTRAS; do printf -- "--extra %s " "$e"; done)

COPY . .

ENV PATH="/workspace/.venv/bin:$PATH"

CMD ["pytest"]
