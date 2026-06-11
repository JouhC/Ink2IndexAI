from __future__ import annotations

import hashlib

import pandas as pd


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


def cluster_blocks(blocks: pd.DataFrame, pair_predictions: pd.DataFrame) -> pd.DataFrame:
    block_lookup = blocks.set_index("block_id", drop=False)
    uf_by_page = {}

    for block in blocks.itertuples(index=False):
        key = (block.newspaper_id, int(block.image_id), block.page_filename)
        uf_by_page.setdefault(key, UnionFind()).add(block.block_id)

    for row in pair_predictions.itertuples(index=False):
        key = (row.newspaper_id, int(row.image_id), row.page_filename)
        uf = uf_by_page.setdefault(key, UnionFind())
        uf.add(row.left_block_id)
        uf.add(row.right_block_id)
        if int(row.prediction) == 1:
            uf.union(row.left_block_id, row.right_block_id)

    records = []
    for key, uf in uf_by_page.items():
        components = {}
        for block_id in sorted(uf.parent):
            components.setdefault(uf.find(block_id), []).append(block_id)
        for block_ids in sorted(components.values(), key=lambda ids: (min(ids), len(ids))):
            cluster_id = "pred_" + hashlib.md5("|".join(block_ids).encode("utf-8")).hexdigest()[:10]
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

