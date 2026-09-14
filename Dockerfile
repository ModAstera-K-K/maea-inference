FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN groupadd --system maea \
    && useradd --system --gid maea --create-home maea

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY inference.py ./

RUN python -m pip install --no-cache-dir \
      --index-url "${TORCH_INDEX_URL}" \
      torch==2.7.0 torchvision==0.22.0 \
    && python -m pip install --no-cache-dir .

USER maea

ENTRYPOINT ["maea-infer"]
CMD ["--help"]
