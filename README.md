# Ink2IndexAI

Ink2IndexAI is a full-stack document intelligence application for working with scanned documents. The project pairs a frontend interface with a backend API so document processing workflows can be developed, tested, and served from one Dockerized application.

## Highlights

- Full-stack document AI application with separate frontend and backend surfaces.
- Dockerized runtime for repeatable local setup.
- Frontend served on port `5173` and backend API served on port `8000`.
- Health endpoint available for quick backend verification.
- Companion repositories track research and legacy implementation work.

## Tech Stack

- Frontend web application
- Backend API
- Docker
- Document processing and AI workflow tooling

## Run With Docker

Build the image from the project root:

```bash
docker build -t ink2indexai .
```

Run the app:

```bash
docker run --rm -p 8000:8000 -p 5173:5173 ink2indexai
```

Open the frontend at:

```text
http://localhost:5173
```

Open the backend API at:

```text
http://localhost:8000
```

Check backend health:

```text
http://localhost:8000/health
```

## Related Repositories

- `Ink2IndexAI-Research` contains research and model-development work.
- `Ink2IndexAI-legacy` preserves earlier implementation work.

## Portfolio Note

This project is featured in my portfolio as a representative document-intelligence system that connects AI workflow development with a deployable application surface.
