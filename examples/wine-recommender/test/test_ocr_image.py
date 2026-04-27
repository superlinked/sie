from pathlib import Path
import os
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wine_picture_detection import detect_wine_from_image_bytes

TEST_IMAGE = PROJECT_ROOT / "test" / "test_images" / "dutton_test.png"


def _run_detection() -> None:
    if not TEST_IMAGE.exists():
        raise FileNotFoundError(f"Test image not found: {TEST_IMAGE}")

    image_bytes = TEST_IMAGE.read_bytes()
    detection = detect_wine_from_image_bytes(image_bytes)

    print(f"Image: {TEST_IMAGE}")
    print(f"Detected wine id: {detection.wine_id}")
    print(f"Match score: {detection.match_score:.4f}")
    print(f"OCR text: {detection.ocr_text}")

    if detection.wine_id is None:
        raise SystemExit("OCR test failed: no wine was detected.")


def test_ocr_image_detects_wine() -> None:
    if not os.getenv("CLUSTER_URL"):
        pytest.skip("Set CLUSTER_URL to run the SIE OCR integration test")

    _run_detection()


def main() -> None:
    _run_detection()


if __name__ == "__main__":
    main()
