from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd

from .candidates import build_candidate_pairs
from .classifier import predict_pairs
from .clustering import cluster_blocks
from .config import PipelineConfig
from .images import explode_tif_pages, pages_to_frame
from .ocr_features import build_block_ocr, compute_pairwise_ocr_features
from .storage import write_outputs
from .yolo_detector import detect_blocks

ProgressCallback = Callable[[str, int, str], None]


def load_cached_blocks(cached_blocks_path: Path, pages_df: pd.DataFrame, newspaper_id: str | None) -> pd.DataFrame:
    blocks = pd.read_csv(cached_blocks_path)
    page_lookup = pages_df.set_index("image_id").to_dict("index")
    blocks = blocks[blocks["image_id"].isin(page_lookup)].copy()
    target_newspaper_id = newspaper_id or str(blocks["newspaper_id"].iloc[0])

    blocks["newspaper_id"] = target_newspaper_id
    blocks["_block_index"] = blocks.groupby("image_id", sort=False).cumcount()
    blocks["block_id"] = blocks.apply(lambda row: f"{target_newspaper_id}__{int(row.image_id)}__{int(row._block_index)}", axis=1)

    for image_id, page in page_lookup.items():
        mask = blocks["image_id"].eq(image_id)
        blocks.loc[mask, "page_number"] = page["page_number"]
        blocks.loc[mask, "page_filename"] = page["page_filename"]
        blocks.loc[mask, "image_path"] = page["image_path"]
        blocks.loc[mask, "image_path_in_zip"] = page["page_filename"]
        blocks.loc[mask, "page_width"] = page["page_width"]
        blocks.loc[mask, "page_height"] = page["page_height"]

    return blocks.drop(columns=["_block_index"])


def run_pipeline(config: PipelineConfig, progress_callback: ProgressCallback | None = None) -> dict:
    def progress(stage: str, percent: int, message: str) -> None:
        if progress_callback:
            progress_callback(stage, percent, message)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress("extract_pages", 12, "Extracting page images")
    pages = explode_tif_pages(config.input_tif, output_dir, config.newspaper_id)
    pages_df = pages_to_frame(pages)

    if config.cached_yolo_blocks_path:
        progress("load_cached_yolo", 24, "Loading cached YOLO detections")
        blocks = load_cached_blocks(config.cached_yolo_blocks_path, pages_df, config.newspaper_id)
    else:
        progress("detect_blocks", 24, "Detecting layout blocks with YOLO")
        blocks = detect_blocks(
            pages,
            config.yolo_model_path,
            confidence=config.yolo_confidence,
            image_size=config.yolo_image_size,
        )
    progress("candidate_pairs", 42, "Building candidate block pairs")
    blocks, columns, pairs = build_candidate_pairs(
        blocks,
        same_column_top_k=config.same_column_top_k,
        adjacent_column_top_k=config.adjacent_column_top_k,
        cross_column_top_k=config.cross_column_top_k,
    )

    progress("ocr_features", 58, "Extracting OCR text features")
    block_ocr = build_block_ocr(
        blocks,
        output_dir / "block_ocr.csv",
        run_ocr=config.run_ocr,
        tesseract_lang=config.tesseract_lang,
        tesseract_psm=config.tesseract_psm,
    )
    progress("pairwise_features", 68, "Computing pairwise OCR features")
    pair_ocr_features = compute_pairwise_ocr_features(pairs, block_ocr)
    pairs_with_ocr = pairs.merge(pair_ocr_features, on="pair_id", how="left")
    for column in ["ocr_cosine_similarity", "text_similarity", "entity_overlap", "shared_keywords"]:
        pairs_with_ocr[column] = pairs_with_ocr[column].fillna(0.0)

    progress("pairwise_prediction", 78, "Predicting same-article pairs")
    pair_predictions = predict_pairs(
        pairs_with_ocr,
        config.classifier_model_path,
        config.classifier_feature_columns_path,
        config.classifier_metrics_path,
        threshold_override=config.pairwise_threshold,
    )
    progress("clustering", 88, "Clustering blocks into articles")
    clusters = cluster_blocks(
        blocks,
        pair_predictions,
        method=config.clustering_method,
        leiden_resolution=config.leiden_resolution,
        leiden_seed=config.leiden_seed,
    )

    progress("write_outputs", 96, "Writing visualization outputs")
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
            "cached_yolo_blocks_path": str(config.cached_yolo_blocks_path) if config.cached_yolo_blocks_path else None,
            "yolo_cache_document_id": config.yolo_cache_document_id,
            "classifier_model_path": str(config.classifier_model_path),
            "classifier_feature_columns_path": str(config.classifier_feature_columns_path),
            "classifier_metrics_path": str(config.classifier_metrics_path),
            "run_ocr": config.run_ocr,
            "same_column_top_k": config.same_column_top_k,
            "adjacent_column_top_k": config.adjacent_column_top_k,
            "cross_column_top_k": config.cross_column_top_k,
            "pairwise_threshold": config.pairwise_threshold,
            "clustering_method": config.clustering_method,
            "leiden_resolution": config.leiden_resolution,
            "leiden_seed": config.leiden_seed,
            "yolo_confidence": config.yolo_confidence,
            "yolo_image_size": config.yolo_image_size,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run newspaper TIFF/PNG to article-cluster inference pipeline.")
    parser.add_argument("input_tif", help="Input newspaper .tif/.tiff/.png file, including multipage TIFFs.")
    parser.add_argument("--output-dir", required=True, help="Directory where pipeline outputs will be written.")
    parser.add_argument("--yolo-model", required=True, help="Path to trained YOLO weights, for example best.pt.")
    parser.add_argument("--classifier-model", default=str(PipelineConfig.classifier_model_path))
    parser.add_argument("--feature-columns", default=str(PipelineConfig.classifier_feature_columns_path))
    parser.add_argument("--classifier-metrics", default=str(PipelineConfig.classifier_metrics_path))
    parser.add_argument("--newspaper-id", default=None)
    parser.add_argument("--cached-yolo-blocks", default=None)
    parser.add_argument("--yolo-cache-document-id", default=None)
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-image-size", type=int, default=None)
    parser.add_argument("--same-column-top-k", type=int, default=3)
    parser.add_argument("--adjacent-column-top-k", type=int, default=2)
    parser.add_argument("--cross-column-top-k", type=int, default=1)
    parser.add_argument("--pairwise-threshold", type=float, default=None)
    parser.add_argument("--run-ocr", action="store_true", default=True, help="Run tesseract OCR on text-like YOLO blocks.")
    parser.add_argument("--no-ocr", action="store_false", dest="run_ocr", help="Skip tesseract OCR feature extraction.")
    parser.add_argument("--clustering-method", choices=["union_find", "leiden"], default="union_find")
    parser.add_argument("--leiden-resolution", type=float, default=1.0)
    parser.add_argument("--leiden-seed", type=int, default=13)
    parser.add_argument("--tesseract-lang", default="eng")
    parser.add_argument("--tesseract-psm", default="6")
    return parser


def main() -> None:
    config = PipelineConfig.from_args(build_arg_parser().parse_args())
    manifest = run_pipeline(config)
    print(f"Saved pipeline outputs to {manifest['paths']['manifest_json']}")


if __name__ == "__main__":
    main()
