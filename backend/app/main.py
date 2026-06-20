from __future__ import annotations

import csv
import hashlib
import json
import logging
import subprocess
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .pipeline.config import PipelineConfig
from .pipeline.run_pipeline import run_pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
JOBS_DIR = DATA_DIR / "jobs"
EXPORTS_DIR = DATA_DIR / "exports"
MODELS_DIR = BASE_DIR / "models"

YOLO_MODEL_PATH = MODELS_DIR / "yolo26l-doclaynet.pt"
CLASSIFIER_DIR = MODELS_DIR / "xgboost_ocr_model"
CLASSIFIER_MODEL_PATH = CLASSIFIER_DIR / "pairwise_xgboost_ocr.json"
FEATURE_COLUMNS_PATH = CLASSIFIER_DIR / "feature_columns.json"
CLASSIFIER_METRICS_PATH = CLASSIFIER_DIR / "metrics.json"
logger = logging.getLogger("uvicorn.error")

STAGES = {
    "uploaded": 0,
    "queued": 3,
    "processing": 8,
    "extract_pages": 12,
    "load_cached_yolo": 24,
    "detect_blocks": 24,
    "candidate_pairs": 42,
    "ocr_features": 58,
    "pairwise_features": 68,
    "pairwise_prediction": 78,
    "clustering": 88,
    "write_outputs": 96,
    "completed": 100,
    "failed": 100,
}

STAGE_LABELS = {
    "uploaded": "Uploaded",
    "queued": "Queued",
    "processing": "Preparing pipeline",
    "extract_pages": "Extracting page images",
    "load_cached_yolo": "Loading cached YOLO detections",
    "detect_blocks": "Detecting layout blocks",
    "candidate_pairs": "Building candidate pairs",
    "ocr_features": "Extracting OCR features",
    "pairwise_features": "Computing pairwise features",
    "pairwise_prediction": "Predicting same-article pairs",
    "clustering": "Clustering articles",
    "write_outputs": "Writing outputs",
    "completed": "Completed",
    "failed": "Failed",
}

app = FastAPI(
    title="Ink2Index API",
    description="Upload newspaper TIFF/PNG files, run article separation, and retrieve visualization/export artifacts.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_origin_regex=r"^http://(127\.0\.0\.1|localhost):\d+$",
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    run_yolo: bool = True
    run_pairwise: bool = True
    run_grouping: bool = True
    model_version: str = "pairwise-v1"
    run_ocr: bool = True
    clustering_method: str = Field(default="union_find", pattern="^(union_find|leiden)$")
    leiden_resolution: float = Field(default=1.0, gt=0.0)
    leiden_seed: int = 13
    cluster_validation_enabled: bool = False
    strong_pair_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    medium_pair_min_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    medium_pair_max_probability: float = Field(default=0.9199, ge=0.0, le=1.0)
    cluster_validation_threshold: float = Field(default=0.9, ge=0.0, le=1.5)
    same_column_top_k: int = Field(default=3, ge=0)
    adjacent_column_top_k: int = Field(default=2, ge=0)
    cross_column_top_k: int = Field(default=1, ge=0)
    pairwise_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    reuse_yolo_cache: bool = True
    yolo_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    yolo_image_size: int | None = Field(default=None, gt=0)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_data_dirs() -> None:
    for directory in [DOCUMENTS_DIR, JOBS_DIR, EXPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def document_dir(document_id: str) -> Path:
    return DOCUMENTS_DIR / document_id


def metadata_path(document_id: str) -> Path:
    return document_dir(document_id) / "metadata.json"


def job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def read_json(path: Path, not_found: str) -> Any:
    if not path.exists():
        raise HTTPException(status_code=404, detail=not_found)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def update_document(document_id: str, **updates: Any) -> dict[str, Any]:
    metadata = read_json(metadata_path(document_id), "Document not found")
    metadata.update(updates)
    metadata["updated_at"] = now_iso()
    write_json(metadata_path(document_id), metadata)
    return metadata


def update_job(job_id: str, **updates: Any) -> dict[str, Any]:
    job = read_json(job_path(job_id), "Job not found")
    job.update(updates)
    job["updated_at"] = now_iso()
    stage = str(job.get("stage", "uploaded"))
    job["progress"] = int(job.get("progress", STAGES.get(stage, 0)))
    job["stage_label"] = str(job.get("stage_label") or STAGE_LABELS.get(stage, stage.replace("_", " ").title()))
    write_json(job_path(job_id), job)
    return job


def require_document(document_id: str) -> dict[str, Any]:
    return read_json(metadata_path(document_id), "Document not found")


def output_path(document_id: str, filename: str) -> Path:
    path = document_dir(document_id) / "outputs" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} is not available for this document")
    return path


def safe_document_file(document_id: str, relative_path: str, not_found: str) -> Path:
    base = document_dir(document_id).resolve()
    path = (base / relative_path).resolve()
    if base not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail=not_found)
    return path


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_yolo_cache(document_id: str, file_hash: str | None) -> tuple[str, Path] | None:
    if not file_hash:
        return None
    for metadata_file in sorted(DOCUMENTS_DIR.glob("doc_*/metadata.json")):
        cached_document_id = metadata_file.parent.name
        if cached_document_id == document_id:
            continue
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if metadata.get("file_sha256") != file_hash:
            continue
        blocks_path = metadata_file.parent / "outputs" / "blocks.csv"
        if blocks_path.is_file():
            return cached_document_id, blocks_path
    return None


def coerce_number(value: Any) -> Any:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else number


def build_articles(document_id: str) -> list[dict[str, Any]]:
    clusters = read_csv_records(output_path(document_id, "block_clusters.csv"))
    articles: dict[str, dict[str, Any]] = {}
    confidences: dict[str, list[float]] = {}
    for row in clusters:
        article_id = row["predicted_cluster_id"]
        article = articles.setdefault(
            article_id,
            {
                "article_id": article_id,
                "confidence": 0.0,
                "detection_ids": [],
                "page_ids": [],
            },
        )
        article["detection_ids"].append(row["block_id"])
        page_id = f"page_{int(float(row['image_id'])) + 1:03d}"
        if page_id not in article["page_ids"]:
            article["page_ids"].append(page_id)
        confidence = coerce_number(row.get("confidence"))
        if isinstance(confidence, float | int):
            confidences.setdefault(article_id, []).append(float(confidence))

    for article_id, article in articles.items():
        values = confidences.get(article_id, [])
        article["confidence"] = round(sum(values) / len(values), 4) if values else 0.0
    return list(articles.values())


def process_document(document_id: str, job_id: str, request: ProcessRequest) -> None:
    try:
        logger.info("Processing started document_id=%s job_id=%s request=%s", document_id, job_id, model_to_dict(request))
        metadata = update_document(document_id, status="processing")
        update_job(
            job_id,
            status="running",
            stage="processing",
            stage_label=STAGE_LABELS["processing"],
            progress=STAGES["processing"],
            message="Preparing pipeline inputs",
            started_at=now_iso(),
        )
        yolo_cache = find_yolo_cache(document_id, metadata.get("file_sha256")) if request.reuse_yolo_cache else None
        if yolo_cache:
            cache_document_id, cached_blocks_path = yolo_cache
            logger.info(
                "Reusing YOLO detections document_id=%s cache_document_id=%s path=%s",
                document_id,
                cache_document_id,
                cached_blocks_path,
            )
        else:
            cache_document_id, cached_blocks_path = None, None
        config = PipelineConfig(
            input_tif=Path(metadata["file_path"]),
            output_dir=document_dir(document_id) / "outputs",
            yolo_model_path=YOLO_MODEL_PATH,
            classifier_model_path=CLASSIFIER_MODEL_PATH,
            classifier_feature_columns_path=FEATURE_COLUMNS_PATH,
            classifier_metrics_path=CLASSIFIER_METRICS_PATH,
            newspaper_id=document_id,
            cached_yolo_blocks_path=cached_blocks_path,
            yolo_cache_document_id=cache_document_id,
            yolo_confidence=request.yolo_confidence,
            yolo_image_size=request.yolo_image_size,
            same_column_top_k=request.same_column_top_k,
            adjacent_column_top_k=request.adjacent_column_top_k,
            cross_column_top_k=request.cross_column_top_k,
            pairwise_threshold=request.pairwise_threshold,
            run_ocr=request.run_ocr,
            clustering_method=request.clustering_method,
            leiden_resolution=request.leiden_resolution,
            leiden_seed=request.leiden_seed,
            cluster_validation_enabled=request.cluster_validation_enabled,
            strong_pair_threshold=request.strong_pair_threshold,
            medium_pair_min_probability=request.medium_pair_min_probability,
            medium_pair_max_probability=request.medium_pair_max_probability,
            cluster_validation_threshold=request.cluster_validation_threshold,
        )
        def update_progress(stage: str, progress: int, message: str) -> None:
            update_job(
                job_id,
                status="running",
                stage=stage,
                stage_label=STAGE_LABELS.get(stage, stage.replace("_", " ").title()),
                progress=progress,
                message=message,
            )
            logger.info(
                "Processing progress document_id=%s job_id=%s stage=%s progress=%s message=%s",
                document_id,
                job_id,
                stage,
                progress,
                message,
            )

        manifest = run_pipeline(config, progress_callback=update_progress)
        counts = manifest.get("counts", {})
        update_document(
            document_id,
            status="processed",
            page_count=counts.get("pages", 0),
            article_count=counts.get("clusters", 0),
            model_version=request.model_version,
            yolo_cache_document_id=cache_document_id,
            manifest_path=manifest.get("paths", {}).get("manifest_json"),
        )
        update_job(
            job_id,
            status="completed",
            stage="completed",
            stage_label=STAGE_LABELS["completed"],
            progress=STAGES["completed"],
            message="Processing complete",
            completed_at=now_iso(),
        )
        logger.info("Processing completed document_id=%s job_id=%s counts=%s", document_id, job_id, counts)
    except Exception as exc:  # The background task must persist failures for the client.
        logger.exception("Processing failed document_id=%s job_id=%s", document_id, job_id)
        update_document(document_id, status="failed", error=str(exc))
        update_job(
            job_id,
            status="failed",
            stage="failed",
            stage_label=STAGE_LABELS["failed"],
            progress=STAGES["failed"],
            message="Processing failed",
            error=str(exc),
            completed_at=now_iso(),
        )


def launch_process_document(document_id: str, job_id: str, request_payload: dict[str, Any]) -> None:
    log_path = JOBS_DIR / f"{job_id}.log"
    try:
        with log_path.open("ab") as log_file:
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "app.pipeline_worker",
                    document_id,
                    job_id,
                    json.dumps(request_payload),
                ],
                cwd=str(BASE_DIR.parent),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
    except Exception as exc:
        logger.exception("Failed to launch pipeline worker document_id=%s job_id=%s", document_id, job_id)
        update_document(document_id, status="failed", error=str(exc))
        update_job(
            job_id,
            status="failed",
            stage="failed",
            stage_label=STAGE_LABELS["failed"],
            progress=STAGES["failed"],
            message="Failed to launch processing worker",
            error=str(exc),
            completed_at=now_iso(),
        )


@app.on_event("startup")
def startup() -> None:
    ensure_data_dirs()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", status_code=201)
def upload_document(
    file: UploadFile = File(...),
    source_name: str | None = Form(default=None),
    publication_date: str | None = Form(default=None),
) -> dict[str, Any]:
    ensure_data_dirs()
    logger.info(
        "Upload received filename=%s content_type=%s source_name=%s publication_date=%s",
        file.filename,
        file.content_type,
        source_name,
        publication_date,
    )
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".tif", ".tiff", ".png"}:
        raise HTTPException(status_code=400, detail="Only .tif, .tiff, and .png files are supported")

    document_id = f"doc_{uuid.uuid4().hex[:12]}"
    doc_dir = document_dir(document_id)
    doc_dir.mkdir(parents=True, exist_ok=False)
    original_path = doc_dir / f"original{suffix}"
    with original_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    file_size_bytes = original_path.stat().st_size
    file_hash = file_sha256(original_path)

    metadata = {
        "document_id": document_id,
        "filename": file.filename,
        "file_size_bytes": file_size_bytes,
        "file_sha256": file_hash,
        "source_name": source_name,
        "publication_date": publication_date,
        "status": "uploaded",
        "page_count": 0,
        "article_count": 0,
        "file_uri": f"documents/{document_id}/{original_path.name}",
        "file_path": str(original_path),
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    write_json(metadata_path(document_id), metadata)
    logger.info("Upload saved document_id=%s path=%s sha256=%s", document_id, original_path, file_hash)
    return {
        "document_id": document_id,
        "status": metadata["status"],
        "file_uri": metadata["file_uri"],
        "file_sha256": file_hash,
    }


@app.post("/documents/{document_id}/process")
def start_processing(document_id: str, request: ProcessRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    require_document(document_id)
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    request_payload = model_to_dict(request)
    logger.info("Processing queued document_id=%s job_id=%s request=%s", document_id, job_id, request_payload)
    write_json(
        job_path(job_id),
        {
            "job_id": job_id,
            "document_id": document_id,
            "status": "queued",
            "stage": "queued",
            "stage_label": STAGE_LABELS["queued"],
            "progress": STAGES["queued"],
            "message": "Waiting for background worker",
            "request": request_payload,
            "created_at": now_iso(),
            "updated_at": now_iso(),
        },
    )
    background_tasks.add_task(launch_process_document, document_id, job_id, request_payload)
    return {"job_id": job_id, "document_id": document_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return read_json(job_path(job_id), "Job not found")


@app.get("/documents/{document_id}")
def get_document(document_id: str) -> dict[str, Any]:
    return require_document(document_id)


@app.get("/documents/{document_id}/pages")
def list_pages(document_id: str) -> list[dict[str, Any]]:
    require_document(document_id)
    pages = []
    for row in read_csv_records(output_path(document_id, "pages.csv")):
        page_number = int(float(row["page_number"]))
        pages.append(
            {
                "page_id": f"page_{page_number:03d}",
                "page_number": page_number,
                "image_uri": f"documents/{document_id}/outputs/pages/{row['page_filename']}",
            }
        )
    return pages


@app.get("/documents/{document_id}/outputs/pages/{filename}")
def get_page_image(document_id: str, filename: str) -> FileResponse:
    require_document(document_id)
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid page image filename")
    path = safe_document_file(document_id, f"outputs/pages/{filename}", "Page image not found")
    return FileResponse(path)


@app.get("/documents/{document_id}/detections")
def get_detections(document_id: str) -> list[dict[str, Any]]:
    require_document(document_id)
    detections = []
    for row in read_csv_records(output_path(document_id, "blocks.csv")):
        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])
        detections.append(
            {
                "detection_id": row["block_id"],
                "page_id": f"page_{int(float(row['image_id'])) + 1:03d}",
                "label": row["class_name"],
                "confidence": coerce_number(row["confidence"]),
                "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
            }
        )
    return detections


@app.get("/documents/{document_id}/candidate-pairs")
def get_candidate_pairs(document_id: str) -> list[dict[str, Any]]:
    require_document(document_id)
    pairs = []
    for row in read_csv_records(output_path(document_id, "candidate_block_pairs.csv")):
        pairs.append(
            {
                "pair_id": row["pair_id"],
                "source_detection_id": row["left_block_id"],
                "target_detection_id": row["right_block_id"],
                "features": {key: coerce_number(value) for key, value in row.items() if key not in {"pair_id", "left_block_id", "right_block_id"}},
            }
        )
    return pairs


@app.get("/documents/{document_id}/pairwise-results")
def get_pairwise_results(document_id: str) -> list[dict[str, Any]]:
    metadata = require_document(document_id)
    results = []
    for row in read_csv_records(output_path(document_id, "pair_predictions.csv")):
        results.append(
            {
                "pair_id": row["pair_id"],
                "prediction": bool(int(float(row["prediction"]))),
                "probability": coerce_number(row["probability_same_article"]),
                "model_version": metadata.get("model_version", "pairwise-v1"),
                "cluster_validation_status": row.get("cluster_validation_status"),
                "cluster_validation_score": coerce_number(row.get("cluster_validation_score")),
                "cluster_validation_reason": row.get("cluster_validation_reason"),
            }
        )
    return results


@app.get("/documents/{document_id}/articles")
def get_articles(document_id: str) -> list[dict[str, Any]]:
    require_document(document_id)
    return build_articles(document_id)


@app.get("/articles/{article_id}")
def get_article(article_id: str) -> dict[str, Any]:
    for metadata_file in DOCUMENTS_DIR.glob("doc_*/metadata.json"):
        document_id = metadata_file.parent.name
        try:
            articles = build_articles(document_id)
        except HTTPException:
            continue
        match = next((article for article in articles if article["article_id"] == article_id), None)
        if match:
            detections = [row for row in get_detections(document_id) if row["detection_id"] in set(match["detection_ids"])]
            pages = [row for row in list_pages(document_id) if row["page_id"] in set(match["page_ids"])]
            return {**match, "detections": detections, "pages": pages}
    raise HTTPException(status_code=404, detail="Article not found")


@app.get("/documents/{document_id}/visualization")
def get_visualization(document_id: str) -> dict[str, Any]:
    require_document(document_id)
    raw_payload = read_json(output_path(document_id, "visualization_payload.json"), "Visualization is not available")
    detections = get_detections(document_id)
    articles = build_articles(document_id)
    pages = []
    for page in list_pages(document_id):
        page_id = page["page_id"]
        pages.append(
            {
                **page,
                "detections": [detection for detection in detections if detection["page_id"] == page_id],
                "articles": [article for article in articles if page_id in article["page_ids"]],
            }
        )
    return {
        "document_id": document_id,
        "pages": pages,
        "pair_predictions": raw_payload.get("pair_predictions", []),
    }


@app.get("/documents/{document_id}/metrics")
def get_metrics(document_id: str) -> dict[str, Any]:
    require_document(document_id)
    manifest = read_json(output_path(document_id, "manifest.json"), "Metrics are not available")
    counts = manifest.get("counts", {})
    return {
        "candidate_pairs": counts.get("candidate_pairs", 0),
        "positive_pairs": counts.get("predicted_positive_pairs", 0),
        "negative_pairs": max(counts.get("candidate_pairs", 0) - counts.get("predicted_positive_pairs", 0), 0),
        "articles_found": counts.get("clusters", 0),
        "processing_time_seconds": None,
    }


@app.get("/documents/{document_id}/export")
def export_articles(document_id: str, format: str = "json") -> dict[str, str]:
    require_document(document_id)
    if format not in {"json", "csv", "xml"}:
        raise HTTPException(status_code=400, detail="Supported export formats: json, csv, xml")
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    export_path = EXPORTS_DIR / f"{document_id}_articles.{format}"
    articles = build_articles(document_id)
    if format == "json":
        write_json(export_path, articles)
    elif format == "csv":
        with export_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["article_id", "confidence", "detection_ids", "page_ids"])
            writer.writeheader()
            for article in articles:
                writer.writerow(
                    {
                        **article,
                        "detection_ids": "|".join(article["detection_ids"]),
                        "page_ids": "|".join(article["page_ids"]),
                    }
                )
    else:
        root = ElementTree.Element("articles")
        for article in articles:
            article_node = ElementTree.SubElement(
                root,
                "article",
                id=article["article_id"],
                confidence=str(article["confidence"]),
            )
            detections_node = ElementTree.SubElement(article_node, "detections")
            for detection_id in article["detection_ids"]:
                ElementTree.SubElement(detections_node, "detection", id=detection_id)
            pages_node = ElementTree.SubElement(article_node, "pages")
            for page_id in article["page_ids"]:
                ElementTree.SubElement(pages_node, "page", id=page_id)
        ElementTree.ElementTree(root).write(export_path, encoding="utf-8", xml_declaration=True)
    return {"download_url": f"/exports/{export_path.name}"}


@app.get("/exports/{filename}")
def download_export(filename: str) -> FileResponse:
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid export filename")
    path = EXPORTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export not found")
    return FileResponse(path)
