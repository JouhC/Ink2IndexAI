from __future__ import annotations

from pathlib import Path

import pandas as pd

from .images import PageImage

DEDUP_IOU_THRESHOLD = 0.85
DEDUP_CONTAINMENT_THRESHOLD = 0.95


def box_area(row) -> float:
    return max(float(row.x2) - float(row.x1), 0.0) * max(float(row.y2) - float(row.y1), 0.0)


def box_intersection_area(left, right) -> float:
    width = max(0.0, min(float(left.x2), float(right.x2)) - max(float(left.x1), float(right.x1)))
    height = max(0.0, min(float(left.y2), float(right.y2)) - max(float(left.y1), float(right.y1)))
    return width * height


def is_duplicate_detection(candidate, kept) -> bool:
    intersection = box_intersection_area(candidate, kept)
    if intersection <= 0:
        return False

    candidate_area = max(box_area(candidate), 1e-9)
    kept_area = max(box_area(kept), 1e-9)
    union = candidate_area + kept_area - intersection
    iou = intersection / max(union, 1e-9)
    candidate_containment = intersection / candidate_area
    kept_containment = intersection / kept_area

    return bool(
        iou >= DEDUP_IOU_THRESHOLD
        or candidate_containment >= DEDUP_CONTAINMENT_THRESHOLD
        or kept_containment >= DEDUP_CONTAINMENT_THRESHOLD
    )


def suppress_duplicate_blocks(blocks: pd.DataFrame) -> pd.DataFrame:
    if blocks.empty:
        return blocks

    required_columns = {"newspaper_id", "image_id", "page_filename", "class_name", "confidence", "x1", "y1", "x2", "y2"}
    missing_columns = required_columns - set(blocks.columns)
    if missing_columns:
        raise ValueError(f"Cannot suppress duplicate blocks; missing columns: {sorted(missing_columns)}")

    keep_indices = []
    page_keys = ["newspaper_id", "image_id", "page_filename"]
    for _, page in blocks.groupby(page_keys, sort=False):
        for _, same_class in page.groupby("class_name", sort=False):
            ordered = same_class.sort_values(["confidence", "width", "height"], ascending=[False, False, False])
            kept_rows = []
            for row in ordered.itertuples():
                if any(is_duplicate_detection(row, kept) for kept in kept_rows):
                    continue
                kept_rows.append(row)
                keep_indices.append(row.Index)

    return blocks.loc[sorted(keep_indices)].copy()


def detect_blocks(
    pages: list[PageImage],
    model_path: Path,
    confidence: float = 0.25,
    image_size: int | None = None,
) -> pd.DataFrame:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "The YOLO stage requires ultralytics. Install it and pass a trained YOLO weights file."
        ) from exc

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = YOLO(str(model_path))
    records = []

    for page in pages:
        kwargs = {"conf": confidence, "verbose": False}
        if image_size:
            kwargs["imgsz"] = image_size
        results = model.predict(str(page.image_path), **kwargs)
        names = getattr(model, "names", {}) or {}

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for idx, box in enumerate(boxes):
                xyxy = box.xyxy.detach().cpu().numpy()[0]
                class_id = int(box.cls.detach().cpu().numpy()[0]) if box.cls is not None else -1
                conf = float(box.conf.detach().cpu().numpy()[0]) if box.conf is not None else 0.0
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                width = max(x2 - x1, 0.0)
                height = max(y2 - y1, 0.0)
                records.append(
                    {
                        "newspaper_id": page.newspaper_id,
                        "image_id": page.image_id,
                        "page_number": page.page_number,
                        "page_filename": page.page_filename,
                        "image_path": str(page.image_path),
                        "image_path_in_zip": page.page_filename,
                        "page_width": page.page_width,
                        "page_height": page.page_height,
                        "block_id": f"{page.newspaper_id}__{page.image_id}__{idx}",
                        "class_id": class_id,
                        "class_name": str(names.get(class_id, class_id)),
                        "confidence": conf,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "bbox": [x1, y1, x2, y2],
                        "width": width,
                        "height": height,
                        "center_x": x1 + width / 2,
                        "center_y": y1 + height / 2,
                    }
                )

    return suppress_duplicate_blocks(pd.DataFrame(records))
