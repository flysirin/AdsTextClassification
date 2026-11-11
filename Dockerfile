FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-default-groups --extra cpu
COPY AppFlaskPyTorch/app.py AppFlaskPyTorch/app.py
COPY models/model.py models/model.py
ENV PATH="/app/.venv/bin:$PATH" MODEL_PATH=/models
EXPOSE 5000
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:5000", "AppFlaskPyTorch.app:app"]
