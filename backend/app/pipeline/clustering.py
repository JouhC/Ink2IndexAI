from __future__ import annotations

import hashlib

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


def page_key_from_row(row) -> tuple[str, int, str]:
    return (row.newspaper_id, int(row.image_id), row.page_filename)


def cluster_id_for(block_ids: list[str]) -> str:
    return "pred_" + hashlib.md5("|".join(block_ids).encode("utf-8")).hexdigest()[:10]


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
    method: str = "union_find",
    leiden_resolution: float = 1.0,
    leiden_seed: int = 13,
) -> pd.DataFrame:
    if method == "union_find":
        clusters_by_page = union_find_clusters(blocks, pair_predictions)
    elif method == "leiden":
        clusters_by_page = leiden_clusters(blocks, pair_predictions, leiden_resolution, leiden_seed)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    return cluster_records(blocks, clusters_by_page)
