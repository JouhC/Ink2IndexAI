from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def write_outputs(
    output_dir: Path,
    pages: pd.DataFrame,
    blocks: pd.DataFrame,
    columns: pd.DataFrame,
    pairs: pd.DataFrame,
    pair_ocr_features: pd.DataFrame,
    pair_predictions: pd.DataFrame,
    clusters: pd.DataFrame,
    config_summary: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pages_csv": output_dir / "pages.csv",
        "blocks_csv": output_dir / "blocks.csv",
        "inferred_columns_csv": output_dir / "inferred_page_columns.csv",
        "candidate_pairs_csv": output_dir / "candidate_block_pairs.csv",
        "pairwise_ocr_features_csv": output_dir / "pairwise_ocr_features.csv",
        "pair_predictions_csv": output_dir / "pair_predictions.csv",
        "block_clusters_csv": output_dir / "block_clusters.csv",
        "visualization_payload_json": output_dir / "visualization_payload.json",
        "manifest_json": output_dir / "manifest.json",
    }

    pages.to_csv(paths["pages_csv"], index=False)
    blocks.to_csv(paths["blocks_csv"], index=False)
    columns.to_csv(paths["inferred_columns_csv"], index=False)
    pairs.to_csv(paths["candidate_pairs_csv"], index=False)
    pair_ocr_features.to_csv(paths["pairwise_ocr_features_csv"], index=False)
    pair_predictions.to_csv(paths["pair_predictions_csv"], index=False)
    clusters.to_csv(paths["block_clusters_csv"], index=False)

    payload = {
        "pages": pages.to_dict("records"),
        "blocks": blocks.to_dict("records"),
        "pair_predictions": pair_predictions.to_dict("records"),
        "clusters": clusters.to_dict("records"),
    }
    paths["visualization_payload_json"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config_summary,
        "counts": {
            "pages": int(len(pages)),
            "blocks": int(len(blocks)),
            "candidate_pairs": int(len(pairs)),
            "predicted_positive_pairs": int(pair_predictions.get("prediction", pd.Series(dtype=int)).sum()),
            "clusters": int(clusters["predicted_cluster_id"].nunique()) if len(clusters) else 0,
        },
        "paths": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

