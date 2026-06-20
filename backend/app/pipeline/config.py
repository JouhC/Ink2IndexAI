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
    cached_yolo_blocks_path: Path | None = None
    yolo_cache_document_id: str | None = None
    yolo_confidence: float = 0.25
    yolo_image_size: int | None = None
    same_column_top_k: int = 3
    adjacent_column_top_k: int = 2
    cross_column_top_k: int = 1
    pairwise_threshold: float | None = None
    run_ocr: bool = True
    clustering_method: str = "union_find"
    leiden_resolution: float = 1.0
    leiden_seed: int = 13
    cluster_validation_enabled: bool = False
    strong_pair_threshold: float = 0.92
    medium_pair_min_probability: float = 0.5
    medium_pair_max_probability: float = 0.9199
    cluster_validation_threshold: float = 0.9
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
            cached_yolo_blocks_path=Path(args.cached_yolo_blocks) if args.cached_yolo_blocks else None,
            yolo_cache_document_id=args.yolo_cache_document_id,
            yolo_confidence=args.yolo_confidence,
            yolo_image_size=args.yolo_image_size,
            same_column_top_k=args.same_column_top_k,
            adjacent_column_top_k=args.adjacent_column_top_k,
            cross_column_top_k=args.cross_column_top_k,
            pairwise_threshold=args.pairwise_threshold,
            run_ocr=args.run_ocr,
            clustering_method=args.clustering_method,
            leiden_resolution=args.leiden_resolution,
            leiden_seed=args.leiden_seed,
            cluster_validation_enabled=args.cluster_validation_enabled,
            strong_pair_threshold=args.strong_pair_threshold,
            medium_pair_min_probability=args.medium_pair_min_probability,
            medium_pair_max_probability=args.medium_pair_max_probability,
            cluster_validation_threshold=args.cluster_validation_threshold,
            tesseract_lang=args.tesseract_lang,
            tesseract_psm=args.tesseract_psm,
        )
