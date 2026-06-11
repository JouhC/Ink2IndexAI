from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np
import pandas as pd

TEXTISH_CLASSES = {"Text", "List-item", "Caption", "Footnote"}
K_NEAREST_SPATIAL = 8
READING_WINDOW = 5
GLOBAL_READING_WINDOW = 3
SAME_COLUMN_WINDOW = 6
ADJACENT_COLUMN_MAX_PER_BLOCK = 4
MAX_ADJACENT_COLUMN_Y_DISTANCE = 0.14
MIN_ADJACENT_COLUMN_Y_OVERLAP = 0.08


def stable_hash_int(value: str) -> int:
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def prepare_blocks(blocks: pd.DataFrame) -> pd.DataFrame:
    blocks = blocks.copy()
    for column in [
        "image_id",
        "page_width",
        "page_height",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "center_x",
        "center_y",
        "confidence",
    ]:
        blocks[column] = pd.to_numeric(blocks[column], errors="coerce")
    blocks["area"] = blocks["width"].clip(lower=1) * blocks["height"].clip(lower=1)
    blocks["page_key"] = (
        blocks["newspaper_id"].astype(str)
        + "||"
        + blocks["image_id"].astype(str)
        + "||"
        + blocks["page_filename"].astype(str)
    )
    return blocks


def infer_columns(page_df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    page_width = float(page_df["page_width"].iloc[0])
    ref = page_df[
        page_df["class_name"].isin(TEXTISH_CLASSES)
        & (page_df["width"] <= page_width * 0.55)
        & (page_df["confidence"] >= 0.30)
    ].copy()
    if len(ref) < 4:
        ref = page_df[page_df["width"] <= page_width * 0.70].copy()
    if len(ref) == 0:
        columns = pd.DataFrame([{"column_id": 0, "x1": 0.0, "x2": page_width, "center_x": page_width / 2}])
        return pd.Series(0, index=page_df.index), columns

    intervals = ref[["x1", "x2"]].sort_values(["x1", "x2"]).to_numpy(float)
    median_width = float(np.median(np.maximum(intervals[:, 1] - intervals[:, 0], 1.0)))
    merge_gap = max(page_width * 0.025, median_width * 0.20)

    merged: list[list[float]] = []
    for x1, x2 in intervals:
        if not merged or x1 > merged[-1][1] + merge_gap:
            merged.append([x1, x2])
        else:
            merged[-1][1] = max(merged[-1][1], x2)

    if len(merged) <= 1 and ref["center_x"].nunique() > 1:
        centers = np.sort(ref["center_x"].to_numpy(float))
        gaps = np.diff(centers)
        threshold = max(
            page_width * 0.08,
            float(np.median(gaps[gaps > 0])) * 3 if np.any(gaps > 0) else page_width,
        )
        groups = []
        current = [centers[0]]
        for center, gap in zip(centers[1:], gaps):
            if gap > threshold:
                groups.append(current)
                current = [center]
            else:
                current.append(center)
        groups.append(current)
        merged = [[min(group) - median_width / 2, max(group) + median_width / 2] for group in groups]

    columns = pd.DataFrame(merged, columns=["x1", "x2"]).sort_values("x1").reset_index(drop=True)
    columns["column_id"] = np.arange(len(columns), dtype=int)
    columns["center_x"] = (columns["x1"] + columns["x2"]) / 2
    assigned = np.abs(
        page_df["center_x"].to_numpy(float)[:, None] - columns["center_x"].to_numpy(float)[None, :]
    ).argmin(axis=1)
    return pd.Series(assigned, index=page_df.index), columns[["column_id", "x1", "x2", "center_x"]]


def add_layout(blocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    page_frames = []
    column_records = []
    for page_key, page in blocks.groupby("page_key", sort=False):
        page = page.copy()
        column_ids, columns = infer_columns(page)
        page["column_id"] = column_ids.astype(int)
        page["reading_order_rank"] = (
            page.sort_values(["column_id", "y1", "x1", "block_id"])
            .reset_index()
            .reset_index()
            .set_index("index")["level_0"]
        )
        page["global_reading_rank"] = (
            page.sort_values(["y1", "x1", "block_id"])
            .reset_index()
            .reset_index()
            .set_index("index")["level_0"]
        )
        for record in columns.to_dict("records"):
            record["page_key"] = page_key
            record["newspaper_id"] = page["newspaper_id"].iloc[0]
            record["image_id"] = int(page["image_id"].iloc[0])
            record["page_filename"] = page["page_filename"].iloc[0]
            column_records.append(record)
        page_frames.append(page)
    return pd.concat(page_frames, ignore_index=False).sort_index(), pd.DataFrame(column_records)


def interval_overlap_ratio(a1: float, a2: float, b1: float, b2: float) -> float:
    overlap = max(0.0, min(a2, b2) - max(a1, b1))
    return overlap / max(min(a2 - a1, b2 - b1), 1e-9)


def gap_between(a1: float, a2: float, b1: float, b2: float) -> float:
    if a2 < b1:
        return b1 - a2
    if b2 < a1:
        return a1 - b2
    return 0.0


def ordered_pair(page_df: pd.DataFrame, i, j):
    left = page_df.loc[i]
    right = page_df.loc[j]
    left_key = (left["reading_order_rank"], left["y1"], left["x1"], left["block_id"])
    right_key = (right["reading_order_rank"], right["y1"], right["x1"], right["block_id"])
    return (i, j) if left_key <= right_key else (j, i)


def add_pair(pair_sources, page_df: pd.DataFrame, i, j, source: str) -> None:
    if i == j:
        return
    a, b = ordered_pair(page_df, i, j)
    pair_sources[(a, b)].add(source)


def build_page_candidate_pairs(page: pd.DataFrame):
    pair_sources = defaultdict(set)
    page_width = float(page["page_width"].iloc[0])
    page_height = float(page["page_height"].iloc[0])
    diag = math.hypot(page_width, page_height)
    indices = list(page.index)

    centers = page[["center_x", "center_y"]].to_numpy(float)
    if len(page) > 1:
        dists = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)) / max(diag, 1.0)
        for row_pos, idx in enumerate(indices):
            for neighbor_pos in np.argsort(dists[row_pos])[1 : K_NEAREST_SPATIAL + 1]:
                add_pair(pair_sources, page, idx, indices[neighbor_pos], "spatial_knn")

    for sort_cols, window, source in [
        (["column_id", "y1", "x1", "block_id"], READING_WINDOW, "reading_order"),
        (["y1", "x1", "block_id"], GLOBAL_READING_WINDOW, "global_reading_order"),
    ]:
        ordered = list(page.sort_values(sort_cols).index)
        for pos, idx in enumerate(ordered):
            for neighbor in ordered[pos + 1 : pos + 1 + window]:
                add_pair(pair_sources, page, idx, neighbor, source)

    for _, col_page in page.groupby("column_id", sort=True):
        ordered = list(col_page.sort_values(["y1", "x1", "block_id"]).index)
        for pos, idx in enumerate(ordered):
            for neighbor in ordered[pos + 1 : pos + 1 + SAME_COLUMN_WINDOW]:
                add_pair(pair_sources, page, idx, neighbor, "same_column_window")

    for col in sorted(page["column_id"].dropna().unique()):
        left_col = page[page["column_id"].eq(col)]
        right_col = page[page["column_id"].eq(col + 1)]
        if right_col.empty:
            continue
        for idx, block in left_col.iterrows():
            candidates = []
            for jdx, other in right_col.iterrows():
                y_overlap = interval_overlap_ratio(block.y1, block.y2, other.y1, other.y2)
                y_distance = abs(block.center_y - other.center_y) / max(page_height, 1.0)
                if y_overlap >= MIN_ADJACENT_COLUMN_Y_OVERLAP or y_distance <= MAX_ADJACENT_COLUMN_Y_DISTANCE:
                    candidates.append((y_distance, -y_overlap, jdx))
            for _, _, jdx in sorted(candidates)[:ADJACENT_COLUMN_MAX_PER_BLOCK]:
                add_pair(pair_sources, page, idx, jdx, "adjacent_column")
    return pair_sources


def pair_feature_record(page: pd.DataFrame, left_idx, right_idx, sources: set[str]) -> dict:
    a = page.loc[left_idx]
    b = page.loc[right_idx]
    page_width = float(a.page_width)
    page_height = float(a.page_height)
    diag = math.hypot(page_width, page_height)

    x_overlap = interval_overlap_ratio(a.x1, a.x2, b.x1, b.x2)
    y_overlap = interval_overlap_ratio(a.y1, a.y2, b.y1, b.y2)
    horizontal_gap = gap_between(a.x1, a.x2, b.x1, b.x2) / max(page_width, 1.0)
    vertical_gap = gap_between(a.y1, a.y2, b.y1, b.y2) / max(page_height, 1.0)
    center_dx = (b.center_x - a.center_x) / max(page_width, 1.0)
    center_dy = (b.center_y - a.center_y) / max(page_height, 1.0)
    center_distance = math.hypot(b.center_x - a.center_x, b.center_y - a.center_y) / max(diag, 1.0)
    column_delta = int(b.column_id - a.column_id)
    abs_column_delta = abs(column_delta)
    column_relation = "same_column" if abs_column_delta == 0 else "adjacent_column" if abs_column_delta == 1 else "cross_column"

    return {
        "pair_id": f"{a.block_id}__PAIR__{b.block_id}",
        "newspaper_id": a.newspaper_id,
        "image_id": int(a.image_id),
        "page_number": int(a.page_number) if not pd.isna(a.page_number) else None,
        "page_filename": a.page_filename,
        "image_path_in_zip": a.page_filename,
        "left_block_id": a.block_id,
        "right_block_id": b.block_id,
        "left_class_name": a.class_name,
        "right_class_name": b.class_name,
        "class_pair": f"{a.class_name}__{b.class_name}",
        "left_article_id": pd.NA,
        "right_article_id": pd.NA,
        "label": pd.NA,
        "label_reason": "production_unknown",
        "candidate_sources": "|".join(sorted(sources)),
        "num_candidate_sources": len(sources),
        "left_confidence": float(a.confidence),
        "right_confidence": float(b.confidence),
        "min_yolo_confidence": float(min(a.confidence, b.confidence)),
        "mean_yolo_confidence": float((a.confidence + b.confidence) / 2),
        "left_article_id_confidence": 0.0,
        "right_article_id_confidence": 0.0,
        "min_article_id_confidence": 0.0,
        "mean_article_id_confidence": 0.0,
        "left_article_ambiguous": False,
        "right_article_ambiguous": False,
        "left_x1": float(a.x1),
        "left_y1": float(a.y1),
        "left_x2": float(a.x2),
        "left_y2": float(a.y2),
        "right_x1": float(b.x1),
        "right_y1": float(b.y1),
        "right_x2": float(b.x2),
        "right_y2": float(b.y2),
        "left_column_id": int(a.column_id),
        "right_column_id": int(b.column_id),
        "column_delta": column_delta,
        "abs_column_delta": abs_column_delta,
        "column_relation": column_relation,
        "reading_order_delta": int(b.reading_order_rank - a.reading_order_rank),
        "global_reading_order_delta": int(b.global_reading_rank - a.global_reading_rank),
        "x_overlap_ratio": float(x_overlap),
        "y_overlap_ratio": float(y_overlap),
        "horizontal_gap_norm": float(horizontal_gap),
        "vertical_gap_norm": float(vertical_gap),
        "center_dx_norm": float(center_dx),
        "center_dy_norm": float(center_dy),
        "abs_center_dx_norm": float(abs(center_dx)),
        "abs_center_dy_norm": float(abs(center_dy)),
        "center_distance_norm": float(center_distance),
        "area_ratio": float(min(a.area, b.area) / max(a.area, b.area, 1.0)),
        "width_ratio": float(min(a.width, b.width) / max(a.width, b.width, 1.0)),
        "height_ratio": float(min(a.height, b.height) / max(a.height, b.height, 1.0)),
        "hard_negative": False,
    }


def build_candidate_pairs(blocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blocks = prepare_blocks(blocks)
    blocks_layout, columns = add_layout(blocks)
    records = []
    for _, page in blocks_layout.groupby("page_key", sort=False):
        for (left_idx, right_idx), sources in build_page_candidate_pairs(page).items():
            records.append(pair_feature_record(page, left_idx, right_idx, sources))
    return blocks_layout.reset_index(drop=True), columns, pd.DataFrame(records)

