from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    input_tif: Path
    output_dir: Path
    yolo_model_path: Path
    classifier_model_path: Path = Path(
        "new_datasets/pre-processed/pairwise_classification/xgboost_ocr_model/pairwise_xgboost_ocr.json"
    )
    classifier_feature_columns_path: Path = Path(
        "new_datasets/pre-processed/pairwise_classification/xgboost_ocr_model/feature_columns.json"
    )
    classifier_metrics_path: Path = Path(
        "new_datasets/pre-processed/pairwise_classification/xgboost_ocr_model/metrics.json"
    )
    newspaper_id: str | None = None
    yolo_confidence: float = 0.25
    yolo_image_size: int | None = None
    run_ocr: bool = False
    tesseract_lang: str = "eng"
    tesseract_psm: str = "6"

    @classmethod
    def from_args(cls, args) -> "PipelineConfig":
        return cls(
            input_tif=Path(args.input_tif),
            output_dir=Path(args.output_dir),
            yolo_model_path=Path(args.yolo_model),
            classifier_model_path=Path(args.classifier_model),
            classifier_feature_columns_path=Path(args.feature_columns),
            classifier_metrics_path=Path(args.classifier_metrics),
            newspaper_id=args.newspaper_id,
            yolo_confidence=args.yolo_confidence,
            yolo_image_size=args.yolo_image_size,
            run_ocr=args.run_ocr,
            tesseract_lang=args.tesseract_lang,
            tesseract_psm=args.tesseract_psm,
        )

