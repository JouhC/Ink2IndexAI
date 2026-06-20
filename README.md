# Ink2IndexAI

## Docker

Build the image from the project root:

```bash
docker build -t ink2indexai .
```

Run the app:

```bash
docker run --rm -p 8000:8000 -p 5173:5173 ink2indexai
```

Open the frontend at http://localhost:5173 and the backend API at
http://localhost:8000. The API health check is available at
http://localhost:8000/health.
