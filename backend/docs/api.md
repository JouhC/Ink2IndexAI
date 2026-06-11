# Ink2Index API Reference

Base URL for local development:

```text
http://127.0.0.1:8000
```

Interactive docs are served by FastAPI at `/docs`. The raw OpenAPI schema is available at `/openapi.json`.

## Typical Flow

1. Upload a `.tif` or `.tiff` file with `POST /documents`.
2. Start processing with `POST /documents/{document_id}/process`.
3. Poll job status with `GET /jobs/{job_id}`.
4. Fetch processed outputs from the document endpoints.
5. Export articles with `GET /documents/{document_id}/export`.

## Health

### `GET /health`

Returns service health.

Example response:

```json
{
  "status": "ok"
}
```

## Documents

### `POST /documents`

Uploads a newspaper TIFF document.

Request type: `multipart/form-data`

Fields:

| Name | Type | Required | Notes |
| --- | --- | --- | --- |
| `file` | file | yes | Must end in `.tif` or `.tiff`. |
| `source_name` | string | no | Optional source label. |
| `publication_date` | string | no | Optional publication date. |

Example response:

```json
{
  "document_id": "doc_abc123def456",
  "status": "uploaded",
  "file_uri": "documents/doc_abc123def456/original.tif"
}
```

### `GET /documents/{document_id}`

Returns stored document metadata.

### `POST /documents/{document_id}/process`

Queues processing for an uploaded document.

Request body:

```json
{
  "run_yolo": true,
  "run_pairwise": true,
  "run_grouping": true,
  "model_version": "pairwise-v1",
  "run_ocr": false,
  "yolo_confidence": 0.25,
  "yolo_image_size": null
}
```

Example response:

```json
{
  "job_id": "job_abc123def456",
  "document_id": "doc_abc123def456",
  "status": "queued"
}
```

### `GET /documents/{document_id}/pages`

Lists generated page images for a processed document.

### `GET /documents/{document_id}/detections`

Lists detected layout blocks with page IDs, labels, confidence scores, and bounding boxes.

### `GET /documents/{document_id}/candidate-pairs`

Lists candidate block pairs and their feature values.

### `GET /documents/{document_id}/pairwise-results`

Lists pairwise model predictions for candidate block pairs.

### `GET /documents/{document_id}/articles`

Lists predicted articles for a document.

Example response item:

```json
{
  "article_id": "0",
  "confidence": 0.9134,
  "detection_ids": ["block_001", "block_002"],
  "page_ids": ["page_001"]
}
```

### `GET /documents/{document_id}/visualization`

Returns a document-level visualization payload containing pages, detections, article assignments, and pair predictions.

### `GET /documents/{document_id}/metrics`

Returns processing metrics derived from the pipeline manifest.

Example response:

```json
{
  "candidate_pairs": 120,
  "positive_pairs": 35,
  "negative_pairs": 85,
  "articles_found": 12,
  "processing_time_seconds": null
}
```

### `GET /documents/{document_id}/export`

Exports article results.

Query parameters:

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `format` | string | no | `json` | Supported values: `json`, `csv`, `xml`. |

Example response:

```json
{
  "download_url": "/exports/doc_abc123def456_articles.json"
}
```

## Jobs

### `GET /jobs/{job_id}`

Returns processing job status, progress, request settings, timestamps, and any persisted error.

Possible stages:

| Stage | Progress |
| --- | ---: |
| `uploaded` | 0 |
| `processing` | 10 |
| `completed` | 100 |
| `failed` | 100 |

## Articles

### `GET /articles/{article_id}`

Finds an article across processed documents and returns the article with its detections and pages.

## Exports

### `GET /exports/{filename}`

Downloads a generated export file.

The filename must be a plain filename, not a path.

## Common Errors

| Status | Cause |
| ---: | --- |
| `400` | Unsupported upload file type, unsupported export format, or invalid export filename. |
| `404` | Document, job, article, pipeline output, or export file was not found. |
