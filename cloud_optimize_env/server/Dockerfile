FROM python:3.12-slim

WORKDIR /app

COPY cloud_optimize_env/pyproject.toml /app/cloud_optimize_env/pyproject.toml
RUN pip install --no-cache-dir openenv-core fastapi uvicorn pydantic

COPY cloud_optimize_env/ /app/cloud_optimize_env/

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "cloud_optimize_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
