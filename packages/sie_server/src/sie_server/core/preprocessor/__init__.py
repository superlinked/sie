"""Preprocessor protocol and implementations for modality-specific preprocessing.

This module defines the Preprocessor protocol and concrete implementations
for text (tokenization) and images (PIL -> processor -> tensor).

See roadmap.md Project 10.4 Phase 2 for design details.

Performance optimizations:
- Parallel image processing with ThreadPoolExecutor (GIL released during I/O)
- pillow-simd detection for 4-6x faster image resizing (if installed)
- Efficient BytesIO handling to minimize copies
"""

from sie_server.core.preprocessor.base import (
    Preprocessor,
    check_pillow_features,
    get_image_executor,
)
from sie_server.core.preprocessor.image import ImagePreprocessor
from sie_server.core.preprocessor.text import CharCountPreprocessor, TextPreprocessor
from sie_server.core.preprocessor.vision import (
    DetectionPreprocessor,
    DonutPreprocessor,
    Florence2Preprocessor,
    LightOnOCRPreprocessor,
    NemoColEmbedPreprocessor,
    _dynamic_preprocess,
)

__all__ = [
    # Text preprocessors
    "CharCountPreprocessor",
    # Image preprocessors
    "DetectionPreprocessor",
    "DonutPreprocessor",
    "Florence2Preprocessor",
    "ImagePreprocessor",
    "LightOnOCRPreprocessor",
    "NemoColEmbedPreprocessor",
    # Protocol
    "Preprocessor",
    "TextPreprocessor",
    "_dynamic_preprocess",
    # Utilities
    "check_pillow_features",
    "get_image_executor",
]
