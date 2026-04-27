from __future__ import annotations

import argparse
import os
import re
import sqlite3
from pathlib import Path
from typing import Any
from rapidfuzz import fuzz
import cv2

import numpy as np
from dotenv import load_dotenv
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent

load_dotenv(ROOT_DIR / ".env")
load_dotenv(PACKAGE_DIR / ".env", override=True)

_database_path_value = os.getenv("DATABASE_PATH", "wine_flavor.db")
DATABASE_PATH = Path(_database_path_value)
if not DATABASE_PATH.is_absolute():
    DATABASE_PATH = ROOT_DIR / DATABASE_PATH

CLUSTER_URL = os.getenv("CLUSTER_URL")
API_KEY = os.getenv("API_KEY") or None
TOP_N = int(os.getenv("TOP_N", 5))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0))
SIE_OCR_MODEL = os.getenv("SIE_OCR_MODEL", "microsoft/Florence-2-base")
ALLOWED_OCR_MODELS = {
    "microsoft/Florence-2-base",
    "microsoft/Florence-2-large",
}
OCR_GPU = os.getenv("OCR_GPU", "l4-spot")
OCR_WAIT_FOR_CAPACITY = True
OCR_PROVISION_TIMEOUT_S = int(os.getenv("OCR_PROVISION_TIMEOUT_S", 900))
DB_FIELDS = {"wine_name": 1.0, "winery_name": 1.0}  # can change field weights here
DB_BONUS_FIELDS = ["region_name", "country_name"]

# Image quality check >>>


def _is_blurry(img: np.ndarray, threshold_pct: float = 10.0) -> bool:
    # Laplacian variance empirical range 0–1000
    return bool(cv2.Laplacian(img, cv2.CV_64F).var() < (threshold_pct / 100) * 1000)


def _check_exposure(
    img: np.ndarray, min_pct: float = 19.6, max_pct: float = 78.4
) -> str:
    mean = img.mean()
    if mean < (min_pct / 100) * 255:
        return "underexposed"
    if mean > (max_pct / 100) * 255:
        return "overexposed"
    return "ok"


def _is_noisy(img: np.ndarray, threshold_pct: float = 33.0) -> bool:
    # mean absolute difference empirical range 0–30
    denoised = cv2.GaussianBlur(img.astype(float), (5, 5), 0)
    return bool(
        np.mean(np.abs(img.astype(float) - denoised)) > (threshold_pct / 100) * 30
    )


def _is_low_contrast(img: np.ndarray, threshold_pct: float = 19.6) -> bool:
    # std dev on 0–255 scale
    return bool(img.std() < (threshold_pct / 100) * 255)


def _is_low_resolution(
    img: np.ndarray, min_resolution: tuple[int, int] = (640, 480)
) -> bool:
    h, w = img.shape
    return bool(w < min_resolution[0] or h < min_resolution[1])


def check_quality(image: Image.Image) -> dict:
    """validate that the image is good enough for OCR"""

    gray_image: np.ndarray = np.array(image.convert("L"))

    return {
        "blurry": _is_blurry(gray_image),
        "exposure": _check_exposure(gray_image),
        "noisy": _is_noisy(gray_image),
        "low_contrast": _is_low_contrast(gray_image),
        "low_resolution": _is_low_resolution(gray_image),
    }


# Image quality check <<<

# Text Extraction >>>


def _require_sie_sdk():
    from sie_sdk import Item, SIEClient

    return SIEClient, Item


def textract(image: Image.Image) -> dict[str, Any]:
    if not CLUSTER_URL:
        raise ValueError(
            "Missing SIE base URL. Set CLUSTER_URL in the environment or in .env."
        )

    SIE_CLIENT, Item = _require_sie_sdk()
    sie_client = SIE_CLIENT(CLUSTER_URL, api_key=API_KEY)

    return sie_client.extract(
        SIE_OCR_MODEL,
        Item(images=[{"data": image, "format": "png"}]),
        options={"task": "<OCR_WITH_REGION>"},
        gpu=OCR_GPU,
        wait_for_capacity=OCR_WAIT_FOR_CAPACITY,
        provision_timeout_s=OCR_PROVISION_TIMEOUT_S,
    )


# Text Extraction <<<

# Fuzzy Match >>>


def extract_blob(label_data: dict) -> str:
    """Join all entity texts into one string, stripping XML artifacts."""
    cleaned = (
        re.sub(r"</?\w+>", "", e["text"]).strip()
        for e in label_data.get("entities", [])
        if e.get("text")
    )
    return " ".join(t for t in cleaned if t)


def extract_vintage(blob: str) -> str | None:
    """Pull first 4-digit year in valid wine vintage range via regex."""
    match = re.search(r"\b(1[89]\d{2}|20[0-2]\d)\b", blob)
    return match.group(0) if match else None


def fetch_wines(db_path: str, vintage: str | None) -> list[dict]:
    """Pre-filter by vintage if available, otherwise fetch all."""
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    if vintage:
        cursor.execute(
            """
            SELECT wine_id, winery_name, wine_name, vintage_year,
                   rating_average, ratings_count, country_name, region_name
            FROM wines WHERE vintage_year = ? OR vintage_year IS NULL
        """,
            (vintage,),
        )
    else:
        cursor.execute(
            """
            SELECT wine_id, winery_name, wine_name, vintage_year,
                   rating_average, ratings_count, country_name, region_name
            FROM wines
        """
        )

    rows = [dict(row) for row in cursor.fetchall()]
    connection.close()
    return rows


def _normalize_separators(text: str):
    """Helps _field_score to ensure db fields are separated by spaces"""
    return text.lower().replace("-", " ")


def _field_score(query: str, target: str) -> float:
    """
    Score query against target. Rewards full word matches over partial overlap.
    Combines fuzzy ratio (handles OCR noise) with word coverage (penalizes missing words).
    """

    query = _normalize_separators(query)
    target = _normalize_separators(target)

    query_tokens = set(query.lower().split())
    target_tokens = set(target.lower().split())

    # Combination of the intersection + union penalizes both missing and extra words
    coverage = len(query_tokens & target_tokens) / len(query_tokens | target_tokens)

    # token_set_ratio scores the shared token intersection against each remainder
    ## better handles prefix noise from OCR adding extra words
    set_fuzzy_score = fuzz.token_set_ratio(query, target)

    # 60% coverage, 40% fuzzy — adjust if OCR quality is poor
    return (set_fuzzy_score * 0.4) + (coverage * 100 * 0.6)


def _subsequences(entity: str) -> list[str]:
    """Helps score_wine to generate all contiguous token sub-sequences of an entity string."""
    tokens = entity.split()
    return [
        " ".join(tokens[start:end])
        for start in range(len(tokens))
        for end in range(start + 1, len(tokens) + 1)
    ]


def score_wine(wine: dict, entities: list[str]) -> float:
    """
    For each entity, take its best sub-sequence score across all DB fields, then average across entities.
    Vintage and field matches scored separately as additive bonuses.
    """

    FIELD_BONUS = 0.0  # TODO: add to .env

    entity_scores = [
        max(
            _field_score(candidate, str(wine.get(field) or "")) * weight
            for field, weight in DB_FIELDS.items()
            # score all sub-sequences and take the best
            ## handles OCR bundling extra words
            for candidate in _subsequences(entity)
        )
        for entity in entities
    ]

    # small additive bonus if any entity matches a geographic field
    field_bonus = (
        FIELD_BONUS
        if any(
            _field_score(entity, str(wine.get(field) or "")) > 80
            for entity in entities
            for field in DB_BONUS_FIELDS
        )
        else 0.0
    )

    VINTAGE_BONUS = 0.0  # TODO: add to .env

    # The average of the per-entity scores
    ## how well the OCR text matched the wine's fields.
    average_field_match_score = (
        sum(entity_scores) / len(entity_scores) if entity_scores else 0.0
    )

    # add vintage bonus
    vintage_bonus = (
        VINTAGE_BONUS
        if (
            wine.get("vintage_year")  # Guard against null
            and any(entity == str(wine["vintage_year"]) for entity in entities)
        )
        else 0.0
    )

    # only using average_field_match_score for now (scores better)
    return average_field_match_score + vintage_bonus + field_bonus


def match_label(label_data: dict, top_n: int = TOP_N) -> list[dict]:
    """
    Match a raw label extraction dict against the wine database.
    Returns a list of up to top_n wine dicts sorted by match_score descending,
    each containing: wine_id, winery_name, wine_name, vintage_year,
    rating_average, rating_count, country_name, region_name, match_score (0-100).
    Returns an empty list if no text or wine fields could be extracted.
    """
    blob = extract_blob(label_data)

    if not blob:
        return []

    vintage = extract_vintage(blob)

    # Strip XML artifacts (e.g. </s>) leaked by the OCR model before scoring
    entities = [
        re.sub(r"</?\w+>", "", e["text"]).strip().lower()
        for e in label_data.get("entities", [])
        if e.get("text")
    ]

    wines = fetch_wines(DATABASE_PATH, vintage)
    scored = sorted(
        [{**wine, "match_score": score_wine(wine, entities)} for wine in wines],
        key=lambda wine: wine["match_score"],
        reverse=True,
    )

    return [wine for wine in scored if wine["match_score"] >= SCORE_THRESHOLD][:top_n]


def extract_and_match_image(
    image: Image.Image, top_n: int = TOP_N
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    extracted = textract(image)
    matches = match_label(extracted, top_n=top_n)
    return extracted, matches


# Fuzzy Match <<<

# DEBUG >>>


def debug_ranking(matched_labels: list[dict], target: str) -> None:

    rank = 0

    for dictionary in matched_labels:
        if dictionary["wine_name"] == target:
            print(f"Wine found (ranked {rank}th):\n{dictionary}\n")
            return

        rank += 1

    print(f"failed to find {target}")


# DEBUG <<<


def main():
    if SIE_OCR_MODEL not in ALLOWED_OCR_MODELS:
        raise ValueError(
            f"SIE_OCR_MODEL '{SIE_OCR_MODEL}' not in ALLOWED_OCR_MODELS: {ALLOWED_OCR_MODELS}"
        )

    parser = argparse.ArgumentParser(description="Run OCR against a wine label image.")
    parser.add_argument(
        "image",
        nargs="?",
        default=str(PACKAGE_DIR / "wine_test.webp"),
        help="Path to the image to OCR. Defaults to wine_picture_detection/wine_test.webp",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N,
        help="Number of fuzzy-match results to print.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    print(f"Image: {image_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    wine_label = Image.open(image_path)
    wine_label.load()
    print(
        f"Loaded: format={wine_label.format} mode={wine_label.mode} size={wine_label.size}"
    )

    # Check image quality
    image_quality = check_quality(wine_label)
    print(f"\nImage quality:\n{image_quality}")

    # Extract text and fuzzy match with known wines
    extracted_text, matched_labels = extract_and_match_image(
        wine_label, top_n=args.top_n
    )
    print(f"\nOCR output:\n{extracted_text}\n")
    # print(f"\nMatched Labels:")
    # [print(dictionary) for dictionary in matched_labels] # DEBUG

    print(f"\nTop {args.top_n} matches:")
    for match in matched_labels:
        print(match)

    # Debug wine ranking by score
    # debug_ranking(matched_labels, "walker-hill-vineyard-chardonnay")


if __name__ == "__main__":
    main()
