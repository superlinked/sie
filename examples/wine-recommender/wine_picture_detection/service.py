from __future__ import annotations

import traceback
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from .textract import (
    SIE_OCR_MODEL,
    ALLOWED_OCR_MODELS,
    extract_and_match_image,
    extract_blob,
)


@dataclass
class DetectedWine:
    wine_id: int | None
    match_score: float
    ocr_text: str = ""


def detect_wine_from_image_bytes(image_bytes: bytes) -> DetectedWine:
    if not image_bytes:
        return DetectedWine(wine_id=None, match_score=0.0)

    if SIE_OCR_MODEL not in ALLOWED_OCR_MODELS:
        raise ValueError(
            f"Unsupported OCR model '{SIE_OCR_MODEL}'. "
            f"Choose one of: {sorted(ALLOWED_OCR_MODELS)}"
        )

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError):
        return DetectedWine(wine_id=None, match_score=0.0)

    try:
        extracted_text, matched_labels = extract_and_match_image(image, top_n=1)
        extracted_blob = extract_blob(extracted_text)
        print("wine-image OCR output:", extracted_text)
        print(
            "wine-image detection:",
            {
                "mode": image.mode,
                "size": image.size,
                "ocr_entities": len(extracted_text.get("entities", [])),
                "ocr_blob": extracted_blob,
                "matches": len(matched_labels),
                "top_match": matched_labels[0] if matched_labels else None,
            },
        )
    except Exception as exc:
        print("wine-image detection error:", repr(exc))
        traceback.print_exc()
        return DetectedWine(wine_id=None, match_score=0.0)

    if not matched_labels:
        return DetectedWine(
            wine_id=None,
            match_score=0.0,
            ocr_text=extracted_blob,
        )

    best_match = matched_labels[0]
    return DetectedWine(
        wine_id=(
            int(best_match["wine_id"])
            if best_match.get("wine_id") is not None
            else None
        ),
        match_score=max(
            0.0, min(1.0, float(best_match.get("match_score", 0.0)) / 100.0)
        ),
        ocr_text=extracted_blob,
    )
