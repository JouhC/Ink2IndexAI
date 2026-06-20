from __future__ import annotations

import hashlib
import re

import pandas as pd
import igraph as ig
import leidenalg


class UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, item):
        self.parent.setdefault(item, item)

    def find(self, item):
        self.add(item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


CLUSTER_VALIDATION_STATUS_COLUMN = "cluster_validation_status"
CLUSTER_VALIDATION_SCORE_COLUMN = "cluster_validation_score"
CLUSTER_VALIDATION_REASON_COLUMN = "cluster_validation_reason"
MAX_APPEND_CLUSTER_SIZE = 10
MIN_APPEND_TARGET_CLUSTER_SIZE = 3
MIN_APPEND_PAIR_PROBABILITY = 0.85
MAX_APPEND_VERTICAL_GAP_NORM = 0.02
MAX_APPEND_READING_ORDER_DELTA = 3

TOKEN_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "into",
    "have",
    "has",
    "had",
    "was",
    "were",
    "are",
    "but",
    "not",
    "his",
    "her",
    "its",
    "their",
    "they",
    "them",
    "you",
    "your",
    "our",
    "out",
    "who",
    "said",
    "will",
    "would",
    "could",
    "should",
    "been",
    "being",
    "after",
    "before",
    "last",
    "also",
    "page",
    "turn",
    "nan",
}


def page_key_from_row(row) -> tuple[str, int, str]:
    return (row.newspaper_id, int(row.image_id), row.page_filename)


def cluster_id_for(block_ids: list[str]) -> str:
    return "pred_" + hashlib.md5("|".join(block_ids).encode("utf-8")).hexdigest()[:10]


def text_tokens(text: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", str(text).lower())
        if len(token) >= 3 and token not in TOKEN_STOPWORDS
    }


def init_cluster_validation_columns(pair_predictions: pd.DataFrame) -> None:
    if CLUSTER_VALIDATION_STATUS_COLUMN not in pair_predictions.columns:
        pair_predictions[CLUSTER_VALIDATION_STATUS_COLUMN] = "not_evaluated"
    if CLUSTER_VALIDATION_SCORE_COLUMN not in pair_predictions.columns:
        pair_predictions[CLUSTER_VALIDATION_SCORE_COLUMN] = pd.NA
    if CLUSTER_VALIDATION_REASON_COLUMN not in pair_predictions.columns:
        pair_predictions[CLUSTER_VALIDATION_REASON_COLUMN] = ""


def block_text_token_lookup(block_ocr: pd.DataFrame | None) -> dict[str, set[str]]:
    if block_ocr is None or block_ocr.empty or "block_id" not in block_ocr.columns:
        return {}

    text_column = "ocr_text" if "ocr_text" in block_ocr.columns else None
    if text_column is None:
        return {}

    return {
        str(row.block_id): text_tokens(getattr(row, text_column, ""))
        for row in block_ocr.fillna("").itertuples(index=False)
    }


def component_members(uf: UnionFind) -> dict[str, list[str]]:
    components = {}
    for block_id in sorted(uf.parent):
        components.setdefault(uf.find(block_id), []).append(block_id)
    return components


def component_for_block(uf: UnionFind, block_id: str) -> list[str]:
    root = uf.find(block_id)
    return component_members(uf).get(root, [block_id])


def validate_cluster_to_block_edge(
    row: pd.Series,
    block_id: str,
    cluster_block_ids: list[str],
    token_lookup: dict[str, set[str]],
    validation_threshold: float,
) -> tuple[bool, float, str]:
    probability = float(row.probability_same_article)
    score = probability
    reasons = [f"base_probability={probability:.6f}", f"cluster_size={len(cluster_block_ids)}"]

    if len(cluster_block_ids) >= 2:
        score += 0.02
        reasons.append("cluster_size>=2=+0.02")

    source = str(getattr(row, "candidate_sources", ""))
    column_relation = str(getattr(row, "column_relation", ""))
    class_pair = str(getattr(row, "class_pair", ""))
    x_overlap = float(getattr(row, "x_overlap_ratio", 0.0) or 0.0)
    text_similarity = float(getattr(row, "text_similarity", 0.0) or 0.0)
    vertical_gap = float(getattr(row, "vertical_gap_norm", 0.0) or 0.0)

    if source == "same_column_window" or column_relation == "same_column":
        score += 0.04
        reasons.append("same_column_support=+0.04")
    elif column_relation == "adjacent_column" and x_overlap > 0.2:
        score += 0.02
        reasons.append("adjacent_with_overlap=+0.02")
    elif "cross_column_headline_continuation" in source:
        score -= 0.02
        reasons.append("cross_column_headline_continuation=-0.02")
    elif column_relation == "cross_column":
        score -= 0.05
        reasons.append("cross_column=-0.05")

    if "Section-header__Text" in class_pair or "Text__Section-header" in class_pair:
        score += 0.02
        reasons.append("header_text_class_pair=+0.02")
    elif "Section-header__Section-header" in class_pair:
        score -= 0.02
        reasons.append("header_header_pair=-0.02")
    elif "Picture" in class_pair:
        score -= 0.05
        reasons.append("picture_pair=-0.05")

    if text_similarity >= 0.2:
        score += 0.02
        reasons.append("text_similarity>=0.2=+0.02")
    elif text_similarity == 0:
        score -= 0.02
        reasons.append("text_similarity_zero=-0.02")

    if vertical_gap > 0.08:
        score -= 0.03
        reasons.append("vertical_gap_high=-0.03")

    block_tokens = token_lookup.get(block_id, set())
    cluster_tokens = set()
    for cluster_block_id in cluster_block_ids:
        cluster_tokens.update(token_lookup.get(cluster_block_id, set()))
    shared_tokens = sorted(block_tokens & cluster_tokens)

    if len(shared_tokens) >= 3:
        score += 0.08
        reasons.append(f"cluster_ocr_overlap_{len(shared_tokens)}=+0.08")
    elif len(shared_tokens) >= 2:
        score += 0.05
        reasons.append(f"cluster_ocr_overlap_{len(shared_tokens)}=+0.05")
    elif len(shared_tokens) == 1:
        score += 0.02
        reasons.append("cluster_ocr_overlap_1=+0.02")
    elif token_lookup:
        score -= 0.04
        reasons.append("cluster_ocr_overlap_0=-0.04")
    else:
        reasons.append("cluster_ocr_unavailable")

    block_class = (
        str(getattr(row, "left_class_name", ""))
        if str(getattr(row, "left_block_id", "")) == block_id
        else str(getattr(row, "right_class_name", ""))
    )
    if block_class in {"Section-header", "Title"} and len(shared_tokens) >= 2:
        score += 0.04
        reasons.append("headline_matches_cluster_topic=+0.04")

    accept = bool(probability >= 0.8 and score >= validation_threshold and len(shared_tokens) >= 2)
    if shared_tokens:
        reasons.append("shared_tokens=" + ",".join(shared_tokens[:8]))
    reasons.append(f"validation_threshold={validation_threshold:.3f}")
    return accept, score, "; ".join(reasons)


def validate_trailing_cluster_append_edge(
    row: pd.Series,
    append_block_ids: list[str],
    target_block_ids: list[str],
    validation_threshold: float,
) -> tuple[bool, float, str]:
    probability = float(row.probability_same_article)
    score = probability
    reasons = [
        f"base_probability={probability:.6f}",
        f"append_cluster_size={len(append_block_ids)}",
        f"target_cluster_size={len(target_block_ids)}",
    ]

    source = str(getattr(row, "candidate_sources", ""))
    column_relation = str(getattr(row, "column_relation", ""))
    x_overlap = float(getattr(row, "x_overlap_ratio", 0.0) or 0.0)
    vertical_gap = float(getattr(row, "vertical_gap_norm", 0.0) or 0.0)
    reading_order_delta = abs(int(getattr(row, "reading_order_delta", 999) or 999))

    if probability >= MIN_APPEND_PAIR_PROBABILITY:
        score += 0.02
        reasons.append(f"probability>={MIN_APPEND_PAIR_PROBABILITY:.2f}=+0.02")
    else:
        reasons.append(f"probability<{MIN_APPEND_PAIR_PROBABILITY:.2f}")

    if source == "same_column_window" or column_relation == "same_column":
        score += 0.04
        reasons.append("same_column_support=+0.04")
    else:
        score -= 0.06
        reasons.append("not_same_column=-0.06")

    if x_overlap >= 0.8:
        score += 0.03
        reasons.append("x_overlap>=0.8=+0.03")

    if vertical_gap <= MAX_APPEND_VERTICAL_GAP_NORM:
        score += 0.03
        reasons.append(f"vertical_gap<={MAX_APPEND_VERTICAL_GAP_NORM:.2f}=+0.03")
    else:
        score -= 0.05
        reasons.append(f"vertical_gap>{MAX_APPEND_VERTICAL_GAP_NORM:.2f}=-0.05")

    if reading_order_delta <= MAX_APPEND_READING_ORDER_DELTA:
        score += 0.02
        reasons.append(f"reading_order_delta<={MAX_APPEND_READING_ORDER_DELTA}=+0.02")
    else:
        score -= 0.03
        reasons.append(f"reading_order_delta>{MAX_APPEND_READING_ORDER_DELTA}=-0.03")

    accept = bool(
        probability >= MIN_APPEND_PAIR_PROBABILITY
        and len(append_block_ids) <= MAX_APPEND_CLUSTER_SIZE
        and len(target_block_ids) >= MIN_APPEND_TARGET_CLUSTER_SIZE
        and (source == "same_column_window" or column_relation == "same_column")
        and x_overlap >= 0.8
        and vertical_gap <= MAX_APPEND_VERTICAL_GAP_NORM
        and reading_order_delta <= MAX_APPEND_READING_ORDER_DELTA
        and score >= validation_threshold
    )
    reasons.append(f"validation_threshold={validation_threshold:.3f}")
    return accept, score, "; ".join(reasons)


def cluster_records(blocks: pd.DataFrame, clusters_by_page: dict[tuple[str, int, str], list[list[str]]]) -> pd.DataFrame:
    block_lookup = blocks.set_index("block_id", drop=False)
    records = []
    for key, components in clusters_by_page.items():
        for block_ids in sorted(components, key=lambda ids: (min(ids), len(ids))):
            cluster_id = cluster_id_for(block_ids)
            for block_id in block_ids:
                if block_id not in block_lookup.index:
                    continue
                block = block_lookup.loc[block_id]
                records.append(
                    {
                        "newspaper_id": key[0],
                        "image_id": key[1],
                        "page_filename": key[2],
                        "block_id": block_id,
                        "predicted_cluster_id": cluster_id,
                        "predicted_cluster_size": len(block_ids),
                        "class_name": block.class_name,
                        "confidence": block.confidence,
                        "x1": block.x1,
                        "y1": block.y1,
                        "x2": block.x2,
                        "y2": block.y2,
                        "page_width": block.page_width,
                        "page_height": block.page_height,
                        "image_path": block.image_path,
                    }
                )
    return pd.DataFrame(records)


def union_find_clusters(blocks: pd.DataFrame, pair_predictions: pd.DataFrame) -> dict[tuple[str, int, str], list[list[str]]]:
    uf_by_page = {}

    for block in blocks.itertuples(index=False):
        key = page_key_from_row(block)
        uf_by_page.setdefault(key, UnionFind()).add(block.block_id)

    for row in pair_predictions.itertuples(index=False):
        key = page_key_from_row(row)
        uf = uf_by_page.setdefault(key, UnionFind())
        uf.add(row.left_block_id)
        uf.add(row.right_block_id)
        if int(row.prediction) == 1:
            uf.union(row.left_block_id, row.right_block_id)

    clusters_by_page = {}
    for key, uf in uf_by_page.items():
        components = {}
        for block_id in sorted(uf.parent):
            components.setdefault(uf.find(block_id), []).append(block_id)
        clusters_by_page[key] = list(components.values())
    return clusters_by_page


def validated_union_find_clusters(
    blocks: pd.DataFrame,
    pair_predictions: pd.DataFrame,
    block_ocr: pd.DataFrame | None,
    strong_threshold: float,
    medium_min_probability: float,
    medium_max_probability: float,
    validation_threshold: float,
) -> dict[tuple[str, int, str], list[list[str]]]:
    init_cluster_validation_columns(pair_predictions)
    uf_by_page = {}

    for block in blocks.itertuples(index=False):
        key = page_key_from_row(block)
        uf_by_page.setdefault(key, UnionFind()).add(block.block_id)

    for index, row in pair_predictions.iterrows():
        key = page_key_from_row(row)
        uf = uf_by_page.setdefault(key, UnionFind())
        left = str(row.left_block_id)
        right = str(row.right_block_id)
        uf.add(left)
        uf.add(right)
        probability = float(row.probability_same_article)
        if probability >= strong_threshold:
            uf.union(left, right)
            pair_predictions.at[index, CLUSTER_VALIDATION_STATUS_COLUMN] = "strong_accept"
            pair_predictions.at[index, CLUSTER_VALIDATION_SCORE_COLUMN] = probability
            pair_predictions.at[index, CLUSTER_VALIDATION_REASON_COLUMN] = (
                f"probability>={strong_threshold:.3f}; accepted_by_pairwise_threshold"
            )
        elif probability < medium_min_probability or probability > medium_max_probability:
            pair_predictions.at[index, CLUSTER_VALIDATION_STATUS_COLUMN] = "outside_medium_band"

    token_lookup = block_text_token_lookup(block_ocr)
    medium_pairs = pair_predictions[
        pair_predictions["probability_same_article"].astype(float).ge(medium_min_probability)
        & pair_predictions["probability_same_article"].astype(float).le(medium_max_probability)
    ].sort_values("probability_same_article", ascending=False)

    for index, row in medium_pairs.iterrows():
        key = page_key_from_row(row)
        uf = uf_by_page.setdefault(key, UnionFind())
        left = str(row.left_block_id)
        right = str(row.right_block_id)
        uf.add(left)
        uf.add(right)
        if uf.find(left) == uf.find(right):
            pair_predictions.at[index, CLUSTER_VALIDATION_STATUS_COLUMN] = "already_connected"
            continue

        left_component = component_for_block(uf, left)
        right_component = component_for_block(uf, right)
        if len(left_component) == 1 and len(right_component) > 1:
            block_id = left
            cluster_ids = right_component
        elif len(right_component) == 1 and len(left_component) > 1:
            block_id = right
            cluster_ids = left_component
        else:
            smaller_component, larger_component = (
                (left_component, right_component)
                if len(left_component) <= len(right_component)
                else (right_component, left_component)
            )
            accept, score, reason = validate_trailing_cluster_append_edge(
                row,
                append_block_ids=smaller_component,
                target_block_ids=larger_component,
                validation_threshold=validation_threshold,
            )
            pair_predictions.at[index, CLUSTER_VALIDATION_SCORE_COLUMN] = score
            pair_predictions.at[index, CLUSTER_VALIDATION_REASON_COLUMN] = reason
            if accept:
                uf.union(left, right)
                pair_predictions.at[index, CLUSTER_VALIDATION_STATUS_COLUMN] = "validated_append_cluster_accept"
            else:
                pair_predictions.at[index, CLUSTER_VALIDATION_STATUS_COLUMN] = "rejected_not_singleton_to_cluster"
            continue

        accept, score, reason = validate_cluster_to_block_edge(
            row,
            block_id=block_id,
            cluster_block_ids=cluster_ids,
            token_lookup=token_lookup,
            validation_threshold=validation_threshold,
        )
        pair_predictions.at[index, CLUSTER_VALIDATION_SCORE_COLUMN] = score
        pair_predictions.at[index, CLUSTER_VALIDATION_REASON_COLUMN] = reason
        if accept:
            uf.union(left, right)
            pair_predictions.at[index, CLUSTER_VALIDATION_STATUS_COLUMN] = "validated_accept"
        else:
            pair_predictions.at[index, CLUSTER_VALIDATION_STATUS_COLUMN] = "validated_reject"

    clusters_by_page = {}
    for key, uf in uf_by_page.items():
        clusters_by_page[key] = list(component_members(uf).values())
    return clusters_by_page


def leiden_clusters(
    blocks: pd.DataFrame,
    pair_predictions: pd.DataFrame,
    resolution: float,
    seed: int,
) -> dict[tuple[str, int, str], list[list[str]]]:
    clusters_by_page = {}

    for block in blocks.itertuples(index=False):
        key = page_key_from_row(block)
        clusters_by_page.setdefault(key, [[block.block_id]])

    positive_pairs = pair_predictions[pair_predictions["prediction"].astype(int).eq(1)]
    pairs_by_page = positive_pairs.groupby(["newspaper_id", "image_id", "page_filename"], sort=False)
    blocks_by_page = blocks.groupby(["newspaper_id", "image_id", "page_filename"], sort=False)

    for key, page_blocks in blocks_by_page:
        block_ids = sorted(page_blocks["block_id"].astype(str).unique())
        id_to_index = {block_id: index for index, block_id in enumerate(block_ids)}
        page_pairs = pairs_by_page.get_group(key) if key in pairs_by_page.groups else pd.DataFrame()
        edges = []
        weights = []
        for row in page_pairs.itertuples(index=False):
            left = str(row.left_block_id)
            right = str(row.right_block_id)
            if left not in id_to_index or right not in id_to_index:
                continue
            edges.append((id_to_index[left], id_to_index[right]))
            weights.append(max(float(row.probability_same_article), 1e-6))

        if not edges:
            clusters_by_page[key] = [[block_id] for block_id in block_ids]
            continue

        graph = ig.Graph(n=len(block_ids), edges=edges, directed=False)
        graph.es["weight"] = weights
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=seed,
        )
        clusters_by_page[key] = [[block_ids[index] for index in community] for community in partition]
    return clusters_by_page


def cluster_blocks(
    blocks: pd.DataFrame,
    pair_predictions: pd.DataFrame,
    block_ocr: pd.DataFrame | None = None,
    method: str = "union_find",
    leiden_resolution: float = 1.0,
    leiden_seed: int = 13,
    cluster_validation_enabled: bool = False,
    strong_pair_threshold: float = 0.92,
    medium_pair_min_probability: float = 0.5,
    medium_pair_max_probability: float = 0.9199,
    cluster_validation_threshold: float = 0.9,
) -> pd.DataFrame:
    if cluster_validation_enabled:
        init_cluster_validation_columns(pair_predictions)

    if method == "union_find" and cluster_validation_enabled:
        clusters_by_page = validated_union_find_clusters(
            blocks,
            pair_predictions,
            block_ocr,
            strong_pair_threshold,
            medium_pair_min_probability,
            medium_pair_max_probability,
            cluster_validation_threshold,
        )
    elif method == "union_find":
        clusters_by_page = union_find_clusters(blocks, pair_predictions)
    elif method == "leiden":
        clusters_by_page = leiden_clusters(blocks, pair_predictions, leiden_resolution, leiden_seed)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    return cluster_records(blocks, clusters_by_page)
