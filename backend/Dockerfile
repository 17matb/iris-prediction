FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /uvx /bin/

RUN apt-get update && apt-get install -y curl

ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked

COPY . .

EXPOSE 8000

CMD sh -c "uv run python -m ml.train && uv run fastapi run app/app.py"
