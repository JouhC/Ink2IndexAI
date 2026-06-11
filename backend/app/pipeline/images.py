from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, ImageSequence


@dataclass(frozen=True)
class PageImage:
    newspaper_id: str
    image_id: int
    page_number: int
    page_filename: str
    image_path: Path
    page_width: int
    page_height: int


def default_newspaper_id(input_tif: Path) -> str:
    digest = hashlib.md5(str(input_tif.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{input_tif.stem}__{digest}"


def explode_tif_pages(input_tif: Path, output_dir: Path, newspaper_id: str | None = None) -> list[PageImage]:
    input_tif = Path(input_tif)
    if not input_tif.exists():
        raise FileNotFoundError(input_tif)

    newspaper_id = newspaper_id or default_newspaper_id(input_tif)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    pages: list[PageImage] = []
    with Image.open(input_tif) as tif:
        for zero_index, frame in enumerate(ImageSequence.Iterator(tif)):
            page_number = zero_index + 1
            image = frame.convert("RGB")
            page_filename = f"{input_tif.stem}_page_{page_number:03d}.png"
            image_path = pages_dir / page_filename
            image.save(image_path)
            pages.append(
                PageImage(
                    newspaper_id=newspaper_id,
                    image_id=zero_index,
                    page_number=page_number,
                    page_filename=page_filename,
                    image_path=image_path,
                    page_width=image.width,
                    page_height=image.height,
                )
            )
    return pages


def pages_to_frame(pages: list[PageImage]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "newspaper_id": page.newspaper_id,
                "image_id": page.image_id,
                "page_number": page.page_number,
                "page_filename": page.page_filename,
                "image_path": str(page.image_path),
                "page_width": page.page_width,
                "page_height": page.page_height,
            }
            for page in pages
        ]
    )

