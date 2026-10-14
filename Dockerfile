# One container serves both the API (/api) and the built React app (/).

# ---- 1. build the frontend ------------------------------------------------
FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. runtime -------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    STATIC_DIR=/app/static
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/signallens ./signallens
COPY --from=web /web/dist ./static
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
# Render / Railway / Fly inject $PORT; Hugging Face Spaces expects 7860 (set PORT=7860).
CMD ["sh", "-c", "uvicorn signallens.api.app:app --host 0.0.0.0 --port ${PORT}"]
