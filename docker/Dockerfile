FROM python:3.12-slim

WORKDIR /app

# install uv
RUN pip install uv

# copy dependency files first (for Docker layer caching)
COPY pyproject.toml uv.lock .python-version ./

# install only production dependencies
RUN uv sync --frozen --no-dev

# copy source code and model
COPY src/ ./src/
COPY model/ ./model/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
