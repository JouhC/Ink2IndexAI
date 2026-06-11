from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

OCR_FEATURE_COLUMNS = ["ocr_cosine_similarity", "text_similarity", "entity_overlap", "shared_keywords"]
NUMERIC_FEATURES = [
    "num_candidate_sources",
    "left_confidence",
    "right_confidence",
    "min_yolo_confidence",
    "mean_yolo_confidence",
    "left_x1",
    "left_y1",
    "left_x2",
    "left_y2",
    "right_x1",
    "right_y1",
    "right_x2",
    "right_y2",
    "left_column_id",
    "right_column_id",
    "column_delta",
    "abs_column_delta",
    "reading_order_delta",
    "global_reading_order_delta",
    "x_overlap_ratio",
    "y_overlap_ratio",
    "horizontal_gap_norm",
    "vertical_gap_norm",
    "center_dx_norm",
    "center_dy_norm",
    "abs_center_dx_norm",
    "abs_center_dy_norm",
    "center_distance_norm",
    "area_ratio",
    "width_ratio",
    "height_ratio",
    *OCR_FEATURE_COLUMNS,
]
CATEGORICAL_FEATURES = ["left_class_name", "right_class_name", "class_pair", "column_relation"]


def load_threshold(metrics_path: Path, default: float = 0.5) -> float:
    if not Path(metrics_path).exists():
        return default
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    return float(metrics.get("best_threshold", default))


def make_features(df: pd.DataFrame, fitted_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in NUMERIC_FEATURES:
        if column not in df.columns:
            df[column] = 0.0

    numeric = df[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    source_data = pd.DataFrame(index=df.index)
    candidate_sources = df.get("candidate_sources", pd.Series("", index=df.index)).fillna("").astype(str)
    for column in fitted_columns:
        if column.startswith("source__"):
            token = column.removeprefix("source__")
            source_data[column] = candidate_sources.str.contains(token, regex=False).astype(int)

    for column in CATEGORICAL_FEATURES:
        if column not in df.columns:
            df[column] = "__missing__"
    categoricals = pd.get_dummies(
        df[CATEGORICAL_FEATURES].fillna("__missing__").astype(str),
        columns=CATEGORICAL_FEATURES,
        prefix=CATEGORICAL_FEATURES,
        dtype=np.uint8,
    )
    features = pd.concat([numeric, source_data, categoricals], axis=1)
    return features.reindex(columns=fitted_columns, fill_value=0)


def predict_pairs(
    pairs: pd.DataFrame,
    model_path: Path,
    feature_columns_path: Path,
    metrics_path: Path,
) -> pd.DataFrame:
    if pairs.empty:
        return pairs.assign(probability_same_article=pd.Series(dtype=float), prediction=pd.Series(dtype=int))

    feature_columns = json.loads(Path(feature_columns_path).read_text(encoding="utf-8"))
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    threshold = load_threshold(metrics_path)
    X = make_features(pairs, feature_columns)
    probabilities = model.predict_proba(X)[:, 1]

    predictions = pairs.copy()
    predictions["probability_same_article"] = probabilities
    predictions["prediction"] = (probabilities >= threshold).astype(int)
    predictions["prediction_threshold"] = threshold
    return predictions

