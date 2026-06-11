# Ink2Index Backend

FastAPI backend for uploading newspaper TIFF files, running the article-separation pipeline, and retrieving document, article, visualization, metrics, and export artifacts.

## Run Locally

From this directory:

```bash
uv run uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## API Docs

Interactive FastAPI docs:

```text
http://127.0.0.1:8000/docs
```

OpenAPI schema:

```text
http://127.0.0.1:8000/openapi.json
```

Project API reference:

- [docs/api.md](docs/api.md)
