FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /uvx /bin/

ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked

COPY . .

EXPOSE 8501

CMD sh -c "uv run python -m streamlit run app.py"
