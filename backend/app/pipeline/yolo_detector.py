from __future__ import annotations

from pathlib import Path

import pandas as pd

from .images import PageImage


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

    return pd.DataFrame(records)

