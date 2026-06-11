from __future__ import annotations

import argparse
from pathlib import Path

from .candidates import build_candidate_pairs
from .classifier import predict_pairs
from .clustering import cluster_blocks
from .config import PipelineConfig
from .images import explode_tif_pages, pages_to_frame
from .ocr_features import build_block_ocr, compute_pairwise_ocr_features
from .storage import write_outputs
from .yolo_detector import detect_blocks


def run_pipeline(config: PipelineConfig) -> dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = explode_tif_pages(config.input_tif, output_dir, config.newspaper_id)
    pages_df = pages_to_frame(pages)

    blocks = detect_blocks(
        pages,
        config.yolo_model_path,
        confidence=config.yolo_confidence,
        image_size=config.yolo_image_size,
    )
    blocks, columns, pairs = build_candidate_pairs(blocks)

    block_ocr = build_block_ocr(
        blocks,
        output_dir / "block_ocr.csv",
        run_ocr=config.run_ocr,
        tesseract_lang=config.tesseract_lang,
        tesseract_psm=config.tesseract_psm,
    )
    pair_ocr_features = compute_pairwise_ocr_features(pairs, block_ocr)
    pairs_with_ocr = pairs.merge(pair_ocr_features, on="pair_id", how="left")
    for column in ["ocr_cosine_similarity", "text_similarity", "entity_overlap", "shared_keywords"]:
        pairs_with_ocr[column] = pairs_with_ocr[column].fillna(0.0)

    pair_predictions = predict_pairs(
        pairs_with_ocr,
        config.classifier_model_path,
        config.classifier_feature_columns_path,
        config.classifier_metrics_path,
    )
    clusters = cluster_blocks(blocks, pair_predictions)

    return write_outputs(
        output_dir=output_dir,
        pages=pages_df,
        blocks=blocks,
        columns=columns,
        pairs=pairs,
        pair_ocr_features=pair_ocr_features,
        pair_predictions=pair_predictions,
        clusters=clusters,
        config_summary={
            "input_tif": str(config.input_tif),
            "yolo_model_path": str(config.yolo_model_path),
            "classifier_model_path": str(config.classifier_model_path),
            "classifier_feature_columns_path": str(config.classifier_feature_columns_path),
            "classifier_metrics_path": str(config.classifier_metrics_path),
            "run_ocr": config.run_ocr,
            "yolo_confidence": config.yolo_confidence,
            "yolo_image_size": config.yolo_image_size,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run newspaper TIFF to article-cluster inference pipeline.")
    parser.add_argument("input_tif", help="Input newspaper .tif/.tiff file, including multipage TIFFs.")
    parser.add_argument("--output-dir", required=True, help="Directory where pipeline outputs will be written.")
    parser.add_argument("--yolo-model", required=True, help="Path to trained YOLO weights, for example best.pt.")
    parser.add_argument("--classifier-model", default=str(PipelineConfig.classifier_model_path))
    parser.add_argument("--feature-columns", default=str(PipelineConfig.classifier_feature_columns_path))
    parser.add_argument("--classifier-metrics", default=str(PipelineConfig.classifier_metrics_path))
    parser.add_argument("--newspaper-id", default=None)
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-image-size", type=int, default=None)
    parser.add_argument("--run-ocr", action="store_true", help="Run tesseract OCR on text-like YOLO blocks.")
    parser.add_argument("--tesseract-lang", default="eng")
    parser.add_argument("--tesseract-psm", default="6")
    return parser


def main() -> None:
    config = PipelineConfig.from_args(build_arg_parser().parse_args())
    manifest = run_pipeline(config)
    print(f"Saved pipeline outputs to {manifest['paths']['manifest_json']}")


if __name__ == "__main__":
    main()

