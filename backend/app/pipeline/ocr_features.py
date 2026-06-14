from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TEXTISH_CLASSES = {"Text", "Section-header", "Caption", "List-item", "Footnote", "Title"}
PAIRWISE_OCR_FEATURE_COLUMNS = [
    "pair_id",
    "ocr_cosine_similarity",
    "text_similarity",
    "entity_overlap",
    "shared_keywords",
]
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "not",
    "but",
    "you",
    "your",
    "his",
    "her",
    "their",
    "our",
    "its",
    "all",
    "can",
    "will",
    "one",
    "two",
    "new",
    "said",
    "page",
}


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_text(text))
        if len(token) >= 3 and token not in STOPWORDS
    }


def keyword_set(text: str, max_keywords: int = 12) -> set[str]:
    tokens = list(token_set(text))
    return set(sorted(tokens, key=lambda token: (-len(token), token))[:max_keywords])


def entity_set(text: str) -> set[str]:
    if pd.isna(text):
        return set()
    phrases = re.findall(r"\b[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,3}", str(text))
    return {normalize_text(phrase) for phrase in phrases if len(phrase.strip()) >= 3}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def sequence_similarity(left: str, right: str) -> float:
    left = normalize_text(left)
    right = normalize_text(right)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def crop_block(page_image: Image.Image, row, padding: int = 6) -> Image.Image | None:
    x1 = max(int(np.floor(row.x1)) - padding, 0)
    y1 = max(int(np.floor(row.y1)) - padding, 0)
    x2 = min(int(np.ceil(row.x2)) + padding, page_image.width)
    y2 = min(int(np.ceil(row.y2)) + padding, page_image.height)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return page_image.crop((x1, y1, x2, y2))


def preprocess_crop(crop: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(crop)
    if max(gray.size) < 1600:
        gray = gray.resize((gray.width * 2, gray.height * 2), Image.Resampling.LANCZOS)
    return ImageOps.autocontrast(gray).filter(ImageFilter.SHARPEN)


def run_tesseract(crop: Image.Image, tmp_dir: str, lang: str, psm: str) -> tuple[str, str]:
    input_path = Path(tmp_dir) / "crop.png"
    output_base = Path(tmp_dir) / "ocr"
    output_txt = Path(tmp_dir) / "ocr.txt"
    crop.save(input_path)
    result = subprocess.run(
        ["tesseract", str(input_path), str(output_base), "-l", lang, "--psm", psm],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "", result.stderr.strip()
    if not output_txt.exists():
        return "", "missing tesseract output"
    return re.sub(r"\s+", " ", output_txt.read_text(encoding="utf-8", errors="ignore")).strip(), ""


def build_block_ocr(
    blocks: pd.DataFrame,
    output_path: Path,
    run_ocr: bool,
    tesseract_lang: str = "eng",
    tesseract_psm: str = "6",
) -> pd.DataFrame:
    if not run_ocr:
        block_ocr = blocks[["block_id", "class_name"]].drop_duplicates("block_id").copy()
        block_ocr["ocr_text"] = ""
        block_ocr["ocr_error"] = "ocr_disabled"
        return block_ocr
    if shutil.which("tesseract") is None:
        raise RuntimeError("run_ocr=True but the tesseract command is not available")

    page_cache: OrderedDict[str, Image.Image] = OrderedDict()
    records = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for row in blocks.drop_duplicates("block_id").itertuples(index=False):
            if row.class_name not in TEXTISH_CLASSES:
                records.append({"block_id": row.block_id, "class_name": row.class_name, "ocr_text": "", "ocr_error": "non_text"})
                continue
            image_path = str(row.image_path)
            if image_path not in page_cache:
                page_cache[image_path] = Image.open(image_path).convert("RGB")
                if len(page_cache) > 8:
                    _, old = page_cache.popitem(last=False)
                    old.close()
            crop = crop_block(page_cache[image_path], row)
            if crop is None:
                text, error = "", "crop_too_small"
            else:
                text, error = run_tesseract(preprocess_crop(crop), tmp_dir, tesseract_lang, tesseract_psm)
            records.append({"block_id": row.block_id, "class_name": row.class_name, "ocr_text": text, "ocr_error": error})
    for image in page_cache.values():
        image.close()
    block_ocr = pd.DataFrame(records)
    block_ocr.to_csv(output_path, index=False)
    return block_ocr


def compute_pairwise_ocr_features(pairs: pd.DataFrame, block_ocr: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=PAIRWISE_OCR_FEATURE_COLUMNS)

    block_ocr = block_ocr.drop_duplicates("block_id").copy()
    text_by_block_id = dict(zip(block_ocr["block_id"], block_ocr["ocr_text"].fillna("").astype(str)))
    row_by_block_id = {block_id: i for i, block_id in enumerate(block_ocr["block_id"])}
    all_text = [normalize_text(text) for text in block_ocr["ocr_text"].fillna("").astype(str)]
    tfidf = None
    if any(text.strip() for text in all_text):
        try:
            tfidf = TfidfVectorizer(min_df=1, max_features=50000, ngram_range=(1, 2)).fit_transform(all_text)
        except ValueError:
            tfidf = None
    entities_by_block_id = {row.block_id: entity_set(row.ocr_text) for row in block_ocr.itertuples(index=False)}
    keywords_by_block_id = {row.block_id: keyword_set(row.ocr_text) for row in block_ocr.itertuples(index=False)}

    records = []
    for row in pairs[["pair_id", "left_block_id", "right_block_id"]].itertuples(index=False):
        left_text = text_by_block_id.get(row.left_block_id, "")
        right_text = text_by_block_id.get(row.right_block_id, "")
        left_row = row_by_block_id.get(row.left_block_id)
        right_row = row_by_block_id.get(row.right_block_id)
        if tfidf is None or left_row is None or right_row is None:
            ocr_cosine = 0.0
        else:
            ocr_cosine = float(cosine_similarity(tfidf[left_row], tfidf[right_row])[0, 0])
        records.append(
            {
                "pair_id": row.pair_id,
                "ocr_cosine_similarity": ocr_cosine,
                "text_similarity": float(sequence_similarity(left_text, right_text)),
                "entity_overlap": float(jaccard(entities_by_block_id.get(row.left_block_id, set()), entities_by_block_id.get(row.right_block_id, set()))),
                "shared_keywords": int(len(keywords_by_block_id.get(row.left_block_id, set()) & keywords_by_block_id.get(row.right_block_id, set()))),
            }
        )
    return pd.DataFrame(records, columns=PAIRWISE_OCR_FEATURE_COLUMNS)
