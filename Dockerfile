FROM python:3.11-slim AS build
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
# NOTE: the container needs network access to Ollama. If Ollama runs on the
# host machine (not in this container), point OLLAMA_URL at host.docker.internal
# instead of localhost when you run this image.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
