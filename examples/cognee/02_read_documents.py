"""Read an image or bounded PDF page by page using SIE OCR, then remember literal text."""

import argparse
import asyncio
import io
from collections.abc import Iterator
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from sie_cognee.documents import ocr_page

from common import ROOT, Demo, initialize, launch, require_input


def pages(path: Path, *, max_bytes: int, max_pages: int, max_pixels: int, scale: float) -> Iterator[tuple[int, bytes]]:
    if path.stat().st_size > max_bytes:
        raise ValueError("Document exceeds document_bytes; use a smaller file.")
    if path.suffix.lower() == ".pdf":
        with pdfium.PdfDocument(path) as document:
            if len(document) > max_pages:
                raise ValueError("PDF exceeds document_pages; select fewer pages explicitly.")
            for index in range(len(document)):
                page = document[index]
                try:
                    width, height = page.get_size()
                    if width * height * scale * scale > max_pixels:
                        raise ValueError("Rendered page exceeds page_pixels; reduce pdf_scale.")
                    bitmap = page.render(scale=scale)
                    try:
                        with io.BytesIO() as output:
                            bitmap.to_pil().save(output, format="PNG")
                            yield index + 1, output.getvalue()
                    finally:
                        bitmap.close()
                finally:
                    page.close()
    else:
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > max_pixels:
                raise ValueError("Image exceeds page_pixels.")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Multi-frame images are unsupported; supply each page separately.")
            image.verify()
        yield 1, data


async def main(demo: Demo) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=ROOT / "data/access-note.png")
    path = parser.parse_args().path
    if path == ROOT / "data/access-note.png":
        require_input(path)
    cognee = initialize(demo)
    with demo.native("ocr") as client:
        for page_number, image in pages(
            path,
            max_bytes=demo.limits["document_bytes"],
            max_pages=demo.limits["document_pages"],
            max_pixels=demo.limits["page_pixels"],
            scale=demo.limits["pdf_scale"],
        ):
            document = await asyncio.to_thread(
                ocr_page,
                client,
                model=demo.models["ocr"],
                image=image,
                source_id=f"{path.stem}:page-{page_number}",
                filename=path.name,
                page_number=page_number,
                max_bytes=demo.limits["document_bytes"],
                provision_timeout_s=demo.limits["provision_timeout_seconds"],
            )
            demo.show(f"page-{page_number}-actual-ocr", document)
            item = document.to_data_item()
            demo.show(f"page-{page_number}-provenance", item.external_metadata)
            await cognee.remember(item, dataset_name=demo.dataset, self_improvement=False)


if __name__ == "__main__":
    launch("02-documents", main)
